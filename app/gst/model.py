"""GST underwriting model.

Trains on gst/dataset.csv (6000 GST profiles). One train run fits FOUR real
Gradient-Boosting heads, each 3-fold cross-validated on the same corpus:

  1. GST Underwriting Score Model  — regressor  -> gst_underwriting_score (0-100)
  2. GST Risk Flag Model           — classifier -> gst_risk_flag (LOW/MEDIUM/HIGH)
  3. GST Turnover Trend Model      — regressor  -> turnover_growth_yoy (%)
  4. GST Filing Compliance Model   — classifier -> return filed on-time vs late

Heads 1 & 2 back predict()/predict_many(). Heads 3 & 4 are scored and stored too.

Artifacts are versioned under Backend/models/gst/ and SHA-256 + HMAC signed via
security.serialization. Old versions are never overwritten. An uploaded file is
appended to a growing corpus so each train run sees more data.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from app.gst.schema import CATEGORICAL, DROP, FLAG_TARGET, RISK_ORDER, SCORE_TARGET
from app.services.security import serialization

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent
_BUNDLED_CSV = _PKG_DIR / "dataset.csv"
_MODEL_DIR = _PKG_DIR.parents[1] / "models" / "gst"   # Backend/models/gst
_CORPUS_CSV = _MODEL_DIR / "corpus.csv"
_REGISTRY = _MODEL_DIR / "registry.json"
_DEPLOY_JSON = _MODEL_DIR / "deploy.json"

_cache: dict | None = None   # {artifact, meta}


# ── dataset ────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    frames = []
    if _BUNDLED_CSV.exists():
        frames.append(pd.read_csv(_BUNDLED_CSV))
    if _CORPUS_CSV.exists():
        try:
            frames.append(pd.read_csv(_CORPUS_CSV))
        except (OSError, ValueError):
            logger.warning("GST corpus.csv unreadable — ignoring it.")
    if not frames:
        raise FileNotFoundError("No GST dataset found (app/gst/dataset.csv).")
    df = pd.concat(frames, ignore_index=True)
    if "customer_id" in df.columns:
        df = df.drop_duplicates(subset=["customer_id"], keep="last")
    return df.reset_index(drop=True)


def dataset_rows() -> int:
    try:
        return len(load_dataset())
    except FileNotFoundError:
        return 0


def append_to_corpus(df: pd.DataFrame) -> int:
    """Add rows to the accumulating corpus. Returns the new corpus size."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(_CORPUS_CSV) if _CORPUS_CSV.exists() else None
    combined = pd.concat([existing, df], ignore_index=True) if existing is not None else df
    if "customer_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["customer_id"], keep="last")
    combined.to_csv(_CORPUS_CSV, index=False)
    return len(combined)


# ── feature frame ─────────────────────────────────────────────────────────
def _feature_frame(df: pd.DataFrame, *, exclude: set[str] | None = None) -> pd.DataFrame:
    drop = DROP | (exclude or set())
    cols = [c for c in df.columns if c not in drop]
    num_cols = [c for c in cols if c not in CATEGORICAL]
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for c in CATEGORICAL:
        if c in df.columns and c not in drop:
            dummies = pd.get_dummies(df[c].astype(str).str.upper(), prefix=c)
            X = pd.concat([X, dummies.astype(float)], axis=1)
    return X


# ── the 4 heads ──────────────────────────────────────────────────────────
# Each head is a genuinely separate Gradient-Boosting model trained on a real
# column of the GST corpus. `extraExclude` drops columns that would leak the
# target (e.g. turnover_decline_percentage is just -min(0, growth_yoy)).
_HEADS = [
    {
        "id": "gst_underwriting_score_model",
        "name": "GST Underwriting Score Model",
        "target": SCORE_TARGET, "kind": "regressor", "unit": "score", "extraExclude": [],
        "desc": "Predicts the 0-100 GST underwriting score from filing behaviour, "
                "turnover trend, ITC ratios and buyer concentration.",
    },
    {
        "id": "gst_risk_flag_model",
        "name": "GST Risk Flag Model",
        "target": FLAG_TARGET, "kind": "classifier", "unit": "", "extraExclude": [],
        "desc": "Classifies the GST risk band (LOW / MEDIUM / HIGH) from the same "
                "filing, turnover and ITC signals.",
    },
    {
        "id": "gst_loan_eligibility_model",
        "name": "GST Loan Eligibility Model",
        "target": "maximum_loan_by_gst_rule", "kind": "regressor", "unit": "inr",
        "extraExclude": ["gst_turnover_to_loan_ratio", "proposed_loan_amount"],
        "desc": "Predicts the maximum loan a business qualifies for under the GST "
                "turnover rule, from its filed turnover, ITC and vintage.",
    },
    {
        "id": "gst_filing_compliance_model",
        "name": "GST Filing Compliance Model",
        "target": "filing_regularity_percentage", "kind": "regressor", "unit": "pct",
        "extraExclude": [],
        "desc": "Predicts filing-regularity % — how consistently returns are filed "
                "on time — from filing delays, missed / late returns and turnover trend.",
    },
]


def _make_target(df: pd.DataFrame, spec: dict):
    col = df[spec["target"]]
    if spec["kind"] == "regressor":
        y = pd.to_numeric(col, errors="coerce")
        return y.fillna(y.median()).to_numpy(), None
    raw = col.astype(str).str.upper().to_numpy()
    seen = set(raw)
    if spec["target"] == FLAG_TARGET:
        classes = [c for c in RISK_ORDER if c in seen] or sorted(seen)
    else:
        classes = sorted(seen)
    cidx = {c: i for i, c in enumerate(classes)}
    return np.array([cidx.get(v, 0) for v in raw]), classes


ALGORITHMS = ("gradient_boosting", "xgboost", "random_forest", "logistic_regression", "svm")


def _reg(algorithm: str = "gradient_boosting"):
    alg = (algorithm or "gradient_boosting").lower()
    if alg == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9,
                            colsample_bytree=0.9, n_jobs=-1, random_state=42)
    if alg == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42)
    if alg in ("logistic_regression", "svm"):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07, random_state=42)


def _clf(algorithm: str = "gradient_boosting"):
    alg = (algorithm or "gradient_boosting").lower()
    if alg == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9,
                             colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=42)
    if alg == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    if alg == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=600, random_state=42)
    if alg == "svm":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, random_state=42)


def _train_one_head(df: pd.DataFrame, spec: dict, algorithm: str = "gradient_boosting") -> dict:
    """Fit + 3-fold CV one head. Returns fitted estimator, its scaler, feature
    list, classes and real CV metrics."""
    if spec["target"] not in df.columns:
        raise ValueError(f"Dataset missing head target '{spec['target']}'.")
    X = _feature_frame(df, exclude={spec["target"], *spec["extraExclude"]})
    feat = list(X.columns)
    Xs = StandardScaler().fit(X.to_numpy(float))
    Xv = Xs.transform(X.to_numpy(float))
    y, classes = _make_target(df, spec)

    if spec["kind"] == "regressor":
        pred = cross_val_predict(_reg(algorithm), Xv, y, cv=KFold(3, shuffle=True, random_state=42))
        mae = float(mean_absolute_error(y, pred))
        metrics = {"r2": round(float(r2_score(y, pred)), 4), "mae": round(mae, 3)}
        unit = spec.get("unit", "")
        if unit == "inr":
            mae_s = f"₹{mae:,.0f}"
        elif unit == "pct":
            mae_s = f"{mae:.1f}%"
        else:
            mae_s = f"{mae:.2f}"
        acc = f"R² {metrics['r2']:.2f}"
        line = f"3-fold CV · R² {metrics['r2']} · MAE {mae_s}"
    else:
        metrics = {}
        if len(classes) > 1 and np.bincount(y).min() >= 3:
            pred = cross_val_predict(
                _clf(algorithm), Xv, y, cv=StratifiedKFold(3, shuffle=True, random_state=42))
            metrics = {
                "accuracy": round(float(accuracy_score(y, pred)), 4),
                "f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
            }
        acc = f"{metrics.get('accuracy', 0) * 100:.1f}%"
        line = (f"3-fold CV · acc {metrics.get('accuracy', 0) * 100:.1f}% · "
                f"F1 {metrics.get('f1', 0):.2f} · {len(classes)} classes")

    est = _reg(algorithm) if spec["kind"] == "regressor" else _clf(algorithm)
    est.fit(Xv, y)
    Xdf = X  # the un-scaled feature frame — column means feed single-record predict
    imp = getattr(est, "feature_importances_", None)
    return {
        "id": spec["id"], "name": spec["name"], "desc": spec["desc"],
        "kind": spec["kind"], "target": spec["target"], "unit": spec.get("unit", ""),
        "est": est, "scaler": Xs, "feat": feat, "classes": classes,
        "means": {c: float(Xdf[c].mean()) for c in feat},
        "std": {c: float(Xdf[c].std() or 1.0) for c in feat},
        "importance": ({feat[i]: float(imp[i]) for i in range(len(feat))}
                       if imp is not None else {}),
        "metrics": metrics, "accuracyLabel": acc, "metricLine": line,
        "nFeatures": len(feat),
    }


def _feature_summary(df: pd.DataFrame) -> dict:
    """Headline GST aggregates over the training corpus — shown in the Model Hub
    'Real Data' panel instead of the bank-statement feature grid."""
    def _mean(col: str):
        if col not in df.columns:
            return None
        v = pd.to_numeric(df[col], errors="coerce")
        return None if v.dropna().empty else float(v.mean())

    on_time = None
    if "return_status" in df.columns:
        s = df["return_status"].astype(str).str.upper()
        on_time = float((s == "FILED").mean()) * 100 if len(s) else None

    itc_ratio = None
    if {"itc_claimed_amount", "itc_available_amount"} <= set(df.columns):
        num = pd.to_numeric(df["itc_claimed_amount"], errors="coerce")
        den = pd.to_numeric(df["itc_available_amount"], errors="coerce").replace(0, np.nan)
        r = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
        itc_ratio = float(r.mean()) if not r.empty else None

    return {
        "avgAnnualTurnover": _mean("annualised_gst_turnover"),
        "avgMonthlyTurnover": _mean("monthly_turnover"),
        "filingRegularityPct": _mean("filing_regularity_percentage"),
        "onTimeFilingPct": on_time,
        "turnoverGrowthYoY": _mean("turnover_growth_yoy"),
        "itcClaimRatio": itc_ratio,
        "topBuyerPct": _mean("top_buyer_sales_percentage"),
        "avgVintageYears": _mean("business_vintage_years"),
        "avgUnderwritingScore": _mean("gst_underwriting_score"),
    }


# ── registry ──────────────────────────────────────────────────────────────
def _read_registry() -> dict:
    if _REGISTRY.exists():
        try:
            return json.loads(_REGISTRY.read_text("utf-8"))
        except (OSError, ValueError):
            pass
    return {"versions": [], "active": None}


def _write_registry(reg: dict) -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REGISTRY.write_text(json.dumps(reg, indent=2), "utf-8")


# ── training ──────────────────────────────────────────────────────────────
def train(algorithm: str = "gradient_boosting") -> dict:
    global _cache
    algorithm = (algorithm or "gradient_boosting").lower()
    if algorithm not in ALGORITHMS:
        algorithm = "gradient_boosting"
    df = load_dataset()
    if len(df) < 200:
        raise ValueError(f"GST dataset needs >= 200 rows (have {len(df)}).")
    if SCORE_TARGET not in df.columns or FLAG_TARGET not in df.columns:
        raise ValueError(f"Dataset must contain '{SCORE_TARGET}' and '{FLAG_TARGET}'.")

    # Train every head that has its target column present in the corpus.
    heads: list[dict] = []
    for spec in _HEADS:
        if spec["target"] not in df.columns:
            logger.warning("GST: skipping head '%s' — no '%s' column.", spec["id"], spec["target"])
            continue
        heads.append(_train_one_head(df, spec, algorithm))
    if len(heads) < 2:
        raise ValueError("GST corpus is missing the score / flag target columns.")

    by_id = {h["id"]: h for h in heads}
    score_h = by_id["gst_underwriting_score_model"]
    flag_h = by_id["gst_risk_flag_model"]
    classes = flag_h["classes"]

    # Back-compat metrics block (score + flag) used by status()/predict callers.
    metrics = {
        "scoreR2": score_h["metrics"].get("r2"),
        "scoreMae": score_h["metrics"].get("mae"),
    }
    if flag_h["metrics"]:
        metrics["flagAccuracy"] = flag_h["metrics"]["accuracy"]
        metrics["flagF1"] = flag_h["metrics"]["f1"]

    # predict() uses the score + flag heads (they share the full feature frame).
    Xfull = _feature_frame(df)
    feat_names = list(Xfull.columns)
    artifact = {
        "scaler": score_h["scaler"], "regressor": score_h["est"], "classifier": flag_h["est"],
        "feat_names": feat_names, "classes": classes,
        "col_means": {c: float(Xfull[c].mean()) for c in feat_names},
        "col_std": {c: float(Xfull[c].std() or 1.0) for c in feat_names},
        "heads": {h["id"]: {k: h[k] for k in
                            ("est", "scaler", "feat", "classes", "kind", "target", "unit",
                             "means", "std", "importance", "name")}
                  for h in heads},
    }

    reg_json = _read_registry()
    version = max((v["version"] for v in reg_json["versions"]), default=0) + 1
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / f"gst_model_v{version}.joblib"
    import joblib
    joblib.dump(artifact, path)
    sha = serialization.sha256_file(path)

    head_meta = [
        {"id": h["id"], "name": h["name"], "desc": h["desc"], "kind": h["kind"],
         "target": h["target"], "metrics": h["metrics"], "accuracyLabel": h["accuracyLabel"],
         "metricLine": h["metricLine"], "nFeatures": h["nFeatures"],
         "classes": h["classes"]}
        for h in heads
    ]
    meta = {
        "version": version, "path": path.name, "sha256": sha,
        "signature": serialization.sign(sha),
        "nSamples": int(len(df)), "nFeatures": len(feat_names),
        "classes": classes, "metrics": metrics, "heads": head_meta,
        "algorithm": algorithm,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
    }
    reg_json["versions"].append(meta)
    reg_json["active"] = version
    _write_registry(reg_json)

    _cache = {"artifact": artifact, "meta": meta}
    logger.info("GST model v%d — %d heads on %d rows: %s",
                version, len(heads), len(df),
                {h["id"]: h["accuracyLabel"] for h in heads})
    return {
        "version": version, "nSamples": int(len(df)), "features": len(feat_names),
        "classes": classes, "metrics": metrics, "algorithm": algorithm,
        "models": head_meta,
        "featureSummary": _feature_summary(df),
    }


# ── loading ───────────────────────────────────────────────────────────────
def _load_active() -> dict | None:
    global _cache
    if _cache:
        return _cache
    reg = _read_registry()
    if not reg["versions"]:
        return None
    active = reg["active"] or reg["versions"][-1]["version"]
    meta = next((v for v in reg["versions"] if v["version"] == active), reg["versions"][-1])
    path = serialization.guard_path(_MODEL_DIR / meta["path"])
    try:
        serialization.verify_file(path, meta.get("sha256"), meta.get("signature"))
    except serialization.ModelIntegrityError as exc:
        logger.error("GST model integrity check failed: %s", exc)
        return None
    import joblib
    _cache = {"artifact": joblib.load(path), "meta": meta}
    return _cache


def is_trained() -> bool:
    return bool(_read_registry()["versions"])


# ── version registry + per-head deploy state (Model Hub deployment table) ──
HEAD_IDS = [h["id"] for h in _HEADS]
HEAD_NAMES = {h["id"]: h["name"] for h in _HEADS}


def list_versions() -> list[dict]:
    """Every trained GST bundle version, newest first — feeds the 'Select
    Version' dropdown. All 4 heads live in one artifact, so one version list."""
    reg = _read_registry()
    active = reg.get("active")
    out = [
        {
            "version": v["version"],
            "value": f"v{v['version']}",
            "label": f"v{v['version']}" + (" (Active)" if v["version"] == active else "")
                     + (" (Old)" if active is not None and v["version"] != active else ""),
            "active": v["version"] == active,
            "algorithm": v.get("algorithm", "gradient_boosting"),
            "trainedAt": v.get("trainedAt"),
            "nSamples": v.get("nSamples"),
            "metrics": v.get("metrics", {}),
        }
        for v in reg.get("versions", [])
    ]
    out.sort(key=lambda x: x["version"], reverse=True)
    return out


_TARGET_LABEL = {
    SCORE_TARGET: "GST underwriting score (0-100)",
    FLAG_TARGET: "GST risk band (LOW / MEDIUM / HIGH)",
    "maximum_loan_by_gst_rule": "Maximum loan under the GST turnover rule (₹)",
    "filing_regularity_percentage": "Filing-regularity % (on-time returns)",
}


def active_heads_meta() -> dict:
    """{head_id: {desc, metricLine, accuracyLabel, target, targetLabel, nFeatures,
    algorithm}} for the active bundle — used to explain each head's output."""
    reg = _read_registry()
    active = reg.get("active")
    versions = reg.get("versions", [])
    v = next((x for x in versions if x["version"] == active), versions[-1] if versions else None)
    if not v:
        return {}
    algo = v.get("algorithm", "gradient_boosting")
    out: dict = {}
    for h in v.get("heads", []):
        out[h["id"]] = {
            "desc": h.get("desc"),
            "metricLine": h.get("metricLine"),
            "accuracyLabel": h.get("accuracyLabel"),
            "target": h.get("target"),
            "targetLabel": _TARGET_LABEL.get(h.get("target"), h.get("target")),
            "nFeatures": h.get("nFeatures"),
            "algorithm": algo,
        }
    return out


def set_active_version(n: int) -> bool:
    """Roll the active GST bundle to version n (kept — old versions are never lost)."""
    global _cache
    reg = _read_registry()
    if not any(v["version"] == n for v in reg.get("versions", [])):
        return False
    reg["active"] = n
    _write_registry(reg)
    _cache = None
    return True


def deploy_state() -> dict:
    """{head_id: bool} — whether each GST head's output is live. Default: all on."""
    st = {hid: True for hid in HEAD_IDS}
    if _DEPLOY_JSON.exists():
        try:
            saved = json.loads(_DEPLOY_JSON.read_text("utf-8"))
            st.update({k: bool(v) for k, v in saved.items() if k in st})
        except (OSError, ValueError):
            pass
    return st


def set_deployed(head_id: str, value: bool) -> dict:
    if head_id not in HEAD_IDS:
        raise ValueError(f"Unknown GST model '{head_id}'.")
    st = deploy_state()
    st[head_id] = bool(value)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _DEPLOY_JSON.write_text(json.dumps(st, indent=2), "utf-8")
    return st


def registry_view() -> dict:
    reg = _read_registry()
    return {
        "versions": list_versions(),
        "active": reg.get("active"),
        "deployed": deploy_state(),
        "heads": [{"id": h["id"], "name": h["name"]} for h in _HEADS],
    }


_LABELS = {
    "filing_delay_days": "Filing Delay (days)", "gstr1_sales_value": "GSTR-1 Sales Value",
    "gstr3b_taxable_outward_supply": "GSTR-3B Taxable Supply",
    "gstr3b_net_tax_liability": "Net Tax Liability", "total_taxable_turnover": "Total Taxable Turnover",
    "b2b_sales_amount": "B2B Sales", "b2c_sales_amount": "B2C Sales",
    "b2b_sales_percentage": "B2B Sales %", "b2c_sales_percentage": "B2C Sales %",
    "export_sales_amount": "Export Sales", "sez_sales_amount": "SEZ Sales",
    "reverse_charge_sales_amount": "Reverse-Charge Sales", "igst_amount": "IGST",
    "cgst_amount": "CGST", "sgst_amount": "SGST", "cess_amount": "Cess",
    "itc_available_amount": "ITC Available", "itc_claimed_amount": "ITC Claimed",
    "itc_reversed_amount": "ITC Reversed", "net_itc_amount": "Net ITC",
    "unique_buyer_count": "Unique Buyers", "unique_b2b_buyer_count": "Unique B2B Buyers",
    "top_buyer_sales_percentage": "Top Buyer %", "monthly_turnover": "Monthly Turnover",
    "quarterly_turnover": "Quarterly Turnover", "turnover_growth_mom": "Turnover Growth MoM %",
    "turnover_growth_qoq": "Turnover Growth QoQ %", "turnover_growth_yoy": "Turnover Growth YoY %",
    "turnover_decline_percentage": "Turnover Decline %",
    "consecutive_declining_quarters": "Consecutive Declining Quarters",
    "filing_regularity_percentage": "Filing Regularity %", "missed_return_count": "Missed Returns",
    "late_return_count": "Late Returns", "business_vintage_years": "Business Vintage (yrs)",
    "annualised_gst_turnover": "Annualised GST Turnover", "proposed_loan_amount": "Proposed Loan",
    "gst_turnover_to_loan_ratio": "Turnover-to-Loan Ratio",
    "maximum_loan_by_gst_rule": "Max Loan (GST rule)", "gst_data_completeness_score": "Data Completeness",
}


def _label(col: str) -> str:
    if col in _LABELS:
        return _LABELS[col]
    for cat in CATEGORICAL:
        if col.startswith(cat + "_"):
            pretty = cat.replace("_", " ").title().replace("Gst", "GST")
            return f"{pretty} = {col[len(cat) + 1:]}"
    return col.replace("_", " ").title()


def feature_ranking(top_k: int | None = None) -> list[dict]:
    """The model's real input features ranked by variance on the training data
    (min-max scaled to [0,1] so magnitudes are comparable) — the same method the
    bank-statement pipeline's Stage 5 uses. Nothing hard-coded."""
    try:
        df = load_dataset()
    except FileNotFoundError:
        return []
    X = _feature_frame(df)
    rng = X.max() - X.min()
    scaled = (X - X.min()) / rng.where(rng != 0, 1.0)
    var = scaled.var().sort_values(ascending=False)
    # Keep the features that carry signal: variance-threshold selection, the same
    # idea as the bank pipeline's Stage 5 (drop near-constant columns).
    k = top_k if top_k is not None else max(1, round(len(var) * 0.6))
    keep = set(var.index[:k])
    return [
        {"feature": _label(col), "column": col,
         "variance": round(float(v), 6), "selected": col in keep}
        for col, v in var.items()
    ]


def status() -> dict:
    reg = _read_registry()
    if not reg["versions"]:
        return {"trained": False, "datasetRows": dataset_rows()}
    meta = next((v for v in reg["versions"] if v["version"] == reg["active"]), reg["versions"][-1])
    return {
        "trained": True, "version": meta["version"],
        "totalVersions": len(reg["versions"]), "metrics": meta["metrics"],
        "nSamples": meta["nSamples"], "nFeatures": meta["nFeatures"],
        "classes": meta["classes"], "trainedAt": meta["trainedAt"],
        "heads": meta.get("heads", []),
        "datasetRows": dataset_rows(),
    }


# ── prediction ────────────────────────────────────────────────────────────
def _norm_key(k: str) -> str:
    return str(k).strip().lower().replace(" ", "_").replace("-", "_")


def _row_for_feat(rec: dict, feat: list[str], means: dict) -> np.ndarray:
    """One feature row over `feat`, starting from training means, overriding with
    any numeric / categorical value present in the record."""
    idx = {c: i for i, c in enumerate(feat)}
    row = np.array([means.get(c, 0.0) for c in feat], dtype=float)
    for c in feat:
        if c in CATEGORICAL or c not in rec or rec[c] in (None, ""):
            continue
        try:
            row[idx[c]] = float(rec[c])
        except (TypeError, ValueError):
            pass
    for cat in CATEGORICAL:
        if cat in rec and rec[cat] not in (None, ""):
            val = str(rec[cat]).strip().upper()
            for c in feat:
                if c.startswith(cat + "_"):
                    row[idx[c]] = 1.0 if c == f"{cat}_{val}" else 0.0
    return row


def _predict_heads(record: dict, loaded: dict) -> dict:
    """Run every stored GST head on one record. Returns
    {head_id: {name, kind, unit, value|label, proba?}}."""
    heads = (loaded.get("artifact") or {}).get("heads") or {}
    if not heads:
        return {}
    rec = {_norm_key(k): v for k, v in (record or {}).items()}
    out: dict = {}
    for hid, h in heads.items():
        feat = h.get("feat") or []
        if not feat or "scaler" not in h or "est" not in h:
            continue
        means, stds = h.get("means") or {}, h.get("std") or {}
        row = _row_for_feat(rec, feat, means)
        xs = h["scaler"].transform(row.reshape(1, -1))
        entry = {"name": h.get("name", hid), "kind": h["kind"], "unit": h.get("unit", "")}

        # per-head drivers: rank this head's own numeric inputs by
        # (how much THIS model weights the feature) × (how far this profile's
        # value sits from the training-corpus mean). Falls back to |z| when the
        # estimator exposes no feature importances (HistGB).
        imp = h.get("importance") or {}
        factors = []
        for c in feat:
            if c in CATEGORICAL or c[:1] == "_" or c not in rec or rec[c] in (None, ""):
                continue
            try:
                val = float(rec[c])
            except (TypeError, ValueError):
                continue
            z = (val - means.get(c, 0.0)) / (stds.get(c, 1.0) or 1.0)
            weight = (imp.get(c, 0.0) if imp else 1.0)
            factors.append({"feature": c, "value": round(val, 4), "zScore": round(z, 2),
                            "_rank": abs(z) * (weight + (1e-9 if imp else 0.0))})
        factors.sort(key=lambda f: f["_rank"], reverse=True)
        entry["topFactors"] = [{k: v for k, v in f.items() if k != "_rank"} for f in factors[:6]]
        if h["kind"] == "regressor":
            v = float(h["est"].predict(xs)[0])
            if hid == "gst_underwriting_score_model":
                v = float(np.clip(v, 0, 100))
            elif h.get("unit") == "pct":
                v = float(np.clip(v, 0, 100))
            elif h.get("unit") == "inr":
                v = max(0.0, v)
            entry["value"] = round(v, 2)
        else:
            classes = h.get("classes") or []
            est = h["est"]
            if hasattr(est, "predict_proba") and classes:
                p = est.predict_proba(xs)[0]
                entry["label"] = classes[int(np.argmax(p))]
                entry["proba"] = {classes[i]: round(float(p[i]), 4) for i in range(len(classes))}
            else:
                pi = int(est.predict(xs)[0])
                entry["label"] = classes[pi] if pi < len(classes) else str(pi)
        out[hid] = entry
    return out


def predict(record: dict) -> dict:
    loaded = _load_active()
    if not loaded:
        return {"available": False,
                "message": "GST model not trained yet — call POST /api/gst/train."}
    art, meta = loaded["artifact"], loaded["meta"]
    feat_names = art["feat_names"]
    means, stds = art["col_means"], art["col_std"]

    rec = {_norm_key(k): v for k, v in (record or {}).items()}
    row = np.array([means.get(c, 0.0) for c in feat_names], dtype=float)
    idx = {c: i for i, c in enumerate(feat_names)}
    used = []

    for c in feat_names:
        if c in CATEGORICAL or c not in rec or rec[c] in (None, ""):
            continue
        try:
            row[idx[c]] = float(rec[c])
            used.append(c)
        except (TypeError, ValueError):
            pass
    for cat in CATEGORICAL:
        if cat in rec and rec[cat] not in (None, ""):
            val = str(rec[cat]).strip().upper()
            for c in feat_names:
                if c.startswith(cat + "_"):
                    row[idx[c]] = 1.0 if c == f"{cat}_{val}" else 0.0
            used.append(cat)

    xs = art["scaler"].transform(row.reshape(1, -1))
    score = float(np.clip(art["regressor"].predict(xs)[0], 0, 100))

    classes = art["classes"]
    proba = None
    if hasattr(art["classifier"], "predict_proba"):
        p = art["classifier"].predict_proba(xs)[0]
        proba = {classes[i]: round(float(p[i]), 4) for i in range(len(classes))}
        flag = classes[int(np.argmax(p))]
    else:
        flag = classes[int(art["classifier"].predict(xs)[0])]

    factors = []
    for c in used:
        if c in CATEGORICAL:
            continue
        z = (row[idx[c]] - means.get(c, 0.0)) / (stds.get(c, 1.0) or 1.0)
        factors.append({"feature": c, "value": round(row[idx[c]], 4), "zScore": round(z, 2)})
    factors.sort(key=lambda f: abs(f["zScore"]), reverse=True)

    return {
        "available": True,
        "underwritingScore": round(score, 2),
        "riskFlag": flag,
        "riskProbability": proba,
        "fieldsUsed": len(used),
        "topFactors": factors[:6],
        "modelVersion": meta["version"],
        "headScores": _predict_heads(record, loaded),
    }


def predict_many(records: list[dict]) -> dict:
    preds = [predict(r) for r in records]
    ok = [p for p in preds if p.get("available")]
    if not ok:
        return {"available": False,
                "message": preds[0].get("message") if preds else "no records",
                "count": 0, "predictions": []}
    from collections import Counter
    return {
        "available": True,
        "count": len(ok),
        "avgUnderwritingScore": round(sum(p["underwritingScore"] for p in ok) / len(ok), 2),
        "riskCounts": dict(Counter(p["riskFlag"] for p in ok)),
        "modelVersion": ok[0]["modelVersion"],
        "headSummary": _aggregate_head_scores(ok),
        "predictions": preds,
    }


def _aggregate_head_scores(preds: list[dict]) -> dict:
    """Average / tally each head across a batch of records — feeds the Model
    Testing GST result panel."""
    from collections import Counter
    by_head: dict = {}
    for p in preds:
        for hid, e in (p.get("headScores") or {}).items():
            by_head.setdefault(hid, {"name": e["name"], "kind": e["kind"],
                                     "unit": e.get("unit", ""), "_vals": [], "_labels": []})
            if e["kind"] == "regressor" and e.get("value") is not None:
                by_head[hid]["_vals"].append(e["value"])
            elif e.get("label"):
                by_head[hid]["_labels"].append(e["label"])
    out: dict = {}
    for hid, d in by_head.items():
        row = {"name": d["name"], "kind": d["kind"], "unit": d["unit"]}
        if d["_vals"]:
            row["value"] = round(sum(d["_vals"]) / len(d["_vals"]), 2)
        if d["_labels"]:
            row["distribution"] = dict(Counter(d["_labels"]))
            row["label"] = Counter(d["_labels"]).most_common(1)[0][0]
        out[hid] = row
    return out
