"""UPI Transaction Data Enrichment model — trains FOUR real Gradient-Boosting
heads on app.upi.schema.FEATURES (derived from analyze_upi()'s real, computed
per-applicant signals), each 3-fold cross-validated on the same corpus:

  1. UPI Transaction Risk Model     — classifier -> upi_transaction_risk_flag (LOW/MEDIUM/HIGH)
  2. Payment Reliability Score      — regressor  -> upi_payment_reliability_score (0-100)
  3. Spend Behaviour Model          — classifier -> upi_spend_behaviour (REGULAR/IRREGULAR)
  4. Network Stability Score        — regressor  -> upi_network_stability_score (0-100)

There's no ground-truth "did this applicant default" label for UPI spend
behaviour, so — same convention as app.bbps.model / app.aa.model — all four
targets are WEAK SUPERVISION: documented, non-ML formulas over the real
features. Each head deliberately weights a different subset of the same
features (see _compute_targets) so it learns a distinct signal instead of
four copies of one score:
  - Risk blends failed-transaction rate + high-risk MCC spend + thin payee
    network + short statement tenure into one composite band.
  - Reliability looks ONLY at the success/failed transaction ratio — pure
    payment execution quality, nothing else.
  - Behaviour looks at recurring-payee share and daily transaction cadence —
    a subscription-like regular spender vs. a sporadic one — independent of
    risk or reliability.
  - Stability looks at payee/payer network diversity and statement tenure —
    how established the applicant's UPI footprint is — independent of the
    other three.

Every real statement a user uploads gets appended to the corpus for real (its
FEATURES are 100% real; only the labels are policy formulas, recomputed fresh
from those features on every load — see _compute_targets), so all four models
keep learning from genuine data over time, exactly like BBPS's corpus.csv.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from app.upi.schema import (
    BEHAVIOUR_ORDER,
    BEHAVIOUR_TARGET,
    FEATURES,
    RELIABILITY_TARGET,
    RISK_ORDER,
    RISK_TARGET,
    STABILITY_TARGET,
)
from app.common.rng import create_rng, rand_int, rand_range
from app.common.security import serialization

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _PKG_DIR.parents[1] / "models" / "upi"   # Backend/models/upi
_CORPUS_CSV = _MODEL_DIR / "corpus.csv"
_REGISTRY = _MODEL_DIR / "registry.json"
_DEPLOY_JSON = _MODEL_DIR / "deploy.json"

_cache: dict | None = None
_N_FOLDS = 3

RISK_MODEL_ID = "upi_transaction_risk_model"
RELIABILITY_MODEL_ID = "upi_payment_reliability_score_model"
BEHAVIOUR_MODEL_ID = "upi_spend_behaviour_model"
STABILITY_MODEL_ID = "upi_network_stability_model"

_HEADS = [
    {
        "id": RISK_MODEL_ID, "name": "UPI Transaction Risk Model",
        "target": RISK_TARGET, "kind": "classifier", "unit": "",
        "desc": "Classifies overall UPI transaction risk (LOW/MEDIUM/HIGH) from failed-payment "
                "rate, high-risk-MCC spend share, payee-network thinness and statement tenure combined.",
    },
    {
        "id": RELIABILITY_MODEL_ID, "name": "Payment Reliability Score",
        "target": RELIABILITY_TARGET, "kind": "regressor", "unit": "score",
        "desc": "Predicts a 0-100 payment-reliability score from the success/failed transaction "
                "ratio alone — how consistently the applicant's UPI payments actually go through.",
    },
    {
        "id": BEHAVIOUR_MODEL_ID, "name": "Spend Behaviour Model",
        "target": BEHAVIOUR_TARGET, "kind": "classifier", "unit": "",
        "desc": "Classifies spend cadence as REGULAR (recurring payees, steady daily transaction "
                "rate) vs IRREGULAR (sporadic, one-off), independent of payment reliability or risk.",
    },
    {
        "id": STABILITY_MODEL_ID, "name": "Network Stability Score",
        "target": STABILITY_TARGET, "kind": "regressor", "unit": "score",
        "desc": "Predicts a 0-100 stability score from payee/payer network diversity and statement "
                "tenure — how established the applicant's UPI footprint is.",
    },
]
_HEAD_BY_ID = {h["id"]: h for h in _HEADS}


# ── weak-supervision targets ────────────────────────────────────────────────

def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Four documented formulas over the real features — see this module's
    docstring for what each head weights and why. Recomputed fresh on every
    load (never trusted from a stored column) so an improved formula applies
    retroactively to the whole corpus, not just new rows."""
    failed = pd.to_numeric(df["failed_ratio"], errors="coerce").fillna(0).clip(0, 1)
    success = pd.to_numeric(df["success_ratio"], errors="coerce").fillna(1).clip(0, 1)
    high_risk = pd.to_numeric(df["high_risk_mcc_spend_pct"], errors="coerce").fillna(0).clip(0, 100)
    payees = pd.to_numeric(df["unique_payees"], errors="coerce").fillna(0)
    payers = pd.to_numeric(df["unique_payers"], errors="coerce").fillna(0)
    recurring = pd.to_numeric(df["recurring_payee_count"], errors="coerce").fillna(0)
    span = pd.to_numeric(df["span_months"], errors="coerce").fillna(1).clip(lower=1)
    daily_avg = pd.to_numeric(df["daily_avg_transactions"], errors="coerce").fillna(0)

    recurring_share = (recurring / payees.clip(lower=1)).clip(0, 1)

    # 1. Risk — composite (deliberately uses most of FEATURES: leaving one out
    # entirely invites the model to fit noise on it).
    risk = (
        0.30 * failed
        + 0.30 * (high_risk / 100)
        + 0.20 * (payees < 2).astype(float)
        + 0.20 * (span < 3).astype(float)
    ).clip(0, 1)
    risk_band = np.where(risk < 0.25, 0, np.where(risk < 0.55, 1, 2))

    # 2. Reliability — success/failed ratio only.
    reliability = (success * 100 - failed * 30).clip(0, 100)

    # 3. Behaviour — recurring-payee share + a baseline daily cadence, nothing
    # about payment success or MCC risk.
    is_regular = ((recurring_share >= 0.4) & (daily_avg >= 0.1)).astype(int)

    # 4. Stability — network diversity + tenure only.
    stability = (
        100 * (0.35 * (payees + payers).clip(upper=20) / 20
               + 0.35 * span.clip(upper=12) / 12
               + 0.30 * recurring_share)
    ).clip(0, 100)

    return pd.DataFrame({
        RISK_TARGET: [RISK_ORDER[b] for b in risk_band],
        RELIABILITY_TARGET: reliability.round(1).to_numpy(),
        BEHAVIOUR_TARGET: np.where(is_regular == 1, "REGULAR", "IRREGULAR"),
        STABILITY_TARGET: stability.round(1).to_numpy(),
    })


def _synthetic_bootstrap(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Synthetic starting corpus — clearly labelled as such, same role as
    app.bbps.model's / app.aa.model's synthetic feature matrix. Real uploads
    accumulate into corpus.csv alongside this and eventually dominate the
    training mix.

    Rows are built the way feature_vector_from_analysis() actually derives
    them from a real file — recurring_payee_count never exceeds
    unique_payees, p2p_ratio + p2m_ratio == 1, daily_avg_transactions is
    derived from total_transactions and span_months rather than sampled
    independently. Generating features independently of each other puts most
    rows in a region of feature space no real applicant's data ever lands
    in — see app.bbps.model's docstring for why that generalizes poorly."""
    rng = create_rng(seed)
    rows = []
    for _ in range(n):
        span = rand_int(rng, 1, 12)
        total = rand_int(rng, 5, 400)
        p2p_share = rand_range(rng, 0.05, 0.6)
        p2p_count = round(total * p2p_share)
        p2m_count = total - p2p_count
        p2p_ratio = round(p2p_count / total, 4) if total else 0.0
        p2m_ratio = round(1 - p2p_ratio, 4)

        success = rand_range(rng, 0.75, 1.0)
        failed = rand_range(rng, 0.0, max(0.0, 1.0 - success))

        unique_payees = rand_int(rng, 0, max(0, min(p2m_count, 15))) if p2m_count > 0 else 0
        unique_payers = rand_int(rng, 0, max(0, min(p2p_count, 8))) if p2p_count > 0 else 0
        recurring = rand_int(rng, 0, unique_payees) if unique_payees > 0 else 0

        daily_avg = round(total / max(1, span * 30), 3)
        weekend_pct = round(rand_range(rng, 0, 60), 1)
        avg_ticket = round(rand_range(rng, 50, 5000), 2)
        is_risky_profile = rand_range(rng, 0, 1) < 0.15
        high_risk_pct = round(rand_range(rng, 10, 45) if is_risky_profile else rand_range(rng, 0, 5), 1)
        p2p_debit_vel = round(rand_range(rng, 0, 15000), 2)
        p2p_credit_vel = round(rand_range(rng, 0, 20000), 2)

        rows.append({
            "total_transactions": total,
            "span_months": span,
            "p2p_ratio": p2p_ratio,
            "p2m_ratio": p2m_ratio,
            "success_ratio": round(success, 4),
            "failed_ratio": round(failed, 4),
            "unique_payees": unique_payees,
            "unique_payers": unique_payers,
            "recurring_payee_count": recurring,
            "daily_avg_transactions": daily_avg,
            "weekend_spend_pct": weekend_pct,
            "avg_ticket_size": avg_ticket,
            "high_risk_mcc_spend_pct": high_risk_pct,
            "p2p_debit_velocity": p2p_debit_vel,
            "p2p_credit_velocity": p2p_credit_vel,
        })
    return pd.DataFrame(rows)


def load_dataset() -> pd.DataFrame:
    frames = [_synthetic_bootstrap()]
    if _CORPUS_CSV.exists():
        try:
            frames.append(pd.read_csv(_CORPUS_CSV)[FEATURES])
        except (OSError, ValueError, pd.errors.ParserError, KeyError) as exc:
            logger.warning("Could not read UPI corpus (%s) — bootstrap only.", exc)
    df = pd.concat(frames, ignore_index=True)
    return pd.concat([df, _compute_targets(df)], axis=1)


def dataset_rows() -> int:
    return len(load_dataset())


def append_to_corpus(row: dict) -> int:
    """One REAL statement's feature row. Labels are never stored — they're
    policy formulas recomputed fresh at load time (see _compute_targets)."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: row.get(k) for k in FEATURES}])
    existing = pd.read_csv(_CORPUS_CSV)[FEATURES] if _CORPUS_CSV.exists() else None
    combined = pd.concat([existing, df], ignore_index=True) if existing is not None else df
    combined.to_csv(_CORPUS_CSV, index=False)
    return len(combined)


# ── training ─────────────────────────────────────────────────────────────

def _reg(algorithm: str):
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42)


def _clf(algorithm: str):
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=42)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _importance(est, Xv: np.ndarray, y: np.ndarray, feat: list[str]) -> dict:
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        try:
            pi = permutation_importance(est, Xv, y, n_repeats=5, random_state=42, n_jobs=-1)
            imp = pi.importances_mean
        except Exception:  # noqa: BLE001 — importance is a display nicety, never blocks training
            imp = None
    return {feat[i]: max(0.0, float(imp[i])) for i in range(len(feat))} if imp is not None else {}


def _fold_status(fold_val: float, mean_val: float, tol: float) -> str:
    return "PASSED" if abs(fold_val - mean_val) <= tol else "REVIEW"


def _train_one_head(df: pd.DataFrame, X: pd.DataFrame, spec: dict, Xv: np.ndarray,
                     scaler: StandardScaler, feat: list[str], algorithm: str) -> dict:
    """Real 3-fold CV + final fit for one head. All four heads share the same
    feature matrix (Xv/scaler) — only the target column differs. Also builds
    `evalMetrics`/`cvFolds` in the same shape app.bbps.model._train_one_head
    produces, so this head plugs straight into the shared Model Evaluation
    panel on the Model Hub page."""
    eval_metrics: dict | None = None
    cv_folds: list[dict] = []

    if spec["kind"] == "regressor":
        y = pd.to_numeric(df[spec["target"]], errors="coerce").fillna(50.0).to_numpy()
        classes = None
        thr = float(np.median(y))
        kf = KFold(n_splits=_N_FOLDS, shuffle=True, random_state=42)
        r2s, mses, maes, precs, recs, f1s = [], [], [], [], [], []
        for tr, te in kf.split(Xv):
            m = _reg(algorithm).fit(Xv[tr], y[tr])
            p = m.predict(Xv[te])
            r2s.append(float(r2_score(y[te], p)))
            mses.append(float(mean_squared_error(y[te], p)))
            maes.append(float(mean_absolute_error(y[te], p)))
            precs.append(float(precision_score(y[te] > thr, p > thr, zero_division=0)))
            recs.append(float(recall_score(y[te] > thr, p > thr, zero_division=0)))
            f1s.append(float(f1_score(y[te] > thr, p > thr, zero_division=0)))
        m_r2, m_mae = float(np.mean(r2s)), float(np.mean(maes))
        metrics = {"r2": round(m_r2, 4), "mae": round(m_mae, 3)}
        accuracy_label = f"R² {metrics['r2']:.2f}"
        metric_line = f"{_N_FOLDS}-fold CV · R² {metrics['r2']} · MAE {m_mae:.2f}"
        eval_metrics = {
            "r2Score": f"{m_r2:.3f}", "mse": f"{np.mean(mses):.4f}",
            "precision": f"{np.mean(precs) * 100:.1f}%", "recall": f"{np.mean(recs) * 100:.1f}%",
            "mae": f"{m_mae:.4f}", "f1Score": f"{np.mean(f1s):.3f}",
            "metricMeta": {
                "r2Score": {"name": "R² SCORE", "sub": f"Variance explained ({_N_FOLDS}-fold CV)"},
                "mse": {"name": "MSE", "sub": "Mean squared error"},
                "mae": {"name": "MAE", "sub": "Mean absolute error"},
                "precision": {"name": "PRECISION", "sub": "Above-median band, positive predictive value"},
                "recall": {"name": "RECALL", "sub": "Above-median band, sensitivity"},
                "f1Score": {"name": "F1 SCORE", "sub": "Harmonic mean of P & R"},
                "cvTitle": f"{_N_FOLDS}-Fold Cross Validation — real refit per fold",
            },
        }
        cv_folds = [
            {
                "fold": f"Fold {i + 1}", "r2": f"{r2s[i]:.3f}", "mse": f"{mses[i]:.4f}",
                "precision": f"{precs[i] * 100:.1f}%", "recall": f"{recs[i] * 100:.1f}%",
                "mae": f"{maes[i]:.4f}", "status": _fold_status(r2s[i], m_r2, 0.08),
            }
            for i in range(_N_FOLDS)
        ]
        est = _reg(algorithm).fit(Xv, y)
    else:
        classes = RISK_ORDER if spec["target"] == RISK_TARGET else BEHAVIOUR_ORDER
        cidx = {c: i for i, c in enumerate(classes)}
        y = np.array([cidx.get(str(v).upper(), 0) for v in df[spec["target"]]])
        skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=42)
        accs, precs, recs, f1s = [], [], [], []
        for tr, te in skf.split(Xv, y):
            m = _clf(algorithm).fit(Xv[tr], y[tr])
            p = m.predict(Xv[te])
            accs.append(float(accuracy_score(y[te], p)))
            precs.append(float(precision_score(y[te], p, average="weighted", zero_division=0)))
            recs.append(float(recall_score(y[te], p, average="weighted", zero_division=0)))
            f1s.append(float(f1_score(y[te], p, average="weighted", zero_division=0)))
        m_acc, m_f1 = float(np.mean(accs)), float(np.mean(f1s))
        metrics = {"accuracy": round(m_acc, 4), "f1": round(m_f1, 4)}
        accuracy_label = f"{m_acc * 100:.1f}%"
        metric_line = f"{_N_FOLDS}-fold CV · acc {m_acc * 100:.1f}% · F1 {m_f1:.2f} · {len(classes)} classes"
        eval_metrics = {
            "r2Score": f"{m_acc:.3f}", "mse": "—",
            "precision": f"{np.mean(precs) * 100:.1f}%", "recall": f"{np.mean(recs) * 100:.1f}%",
            "mae": f"{1 - m_acc:.4f}", "f1Score": f"{np.mean(f1s):.3f}",
            "metricMeta": {
                "r2Score": {"name": "ACCURACY", "sub": f"Correct predictions ({_N_FOLDS}-fold CV)"},
                "mse": {"name": "CLASSES", "sub": f"{len(classes)} classes"},
                "mae": {"name": "ERROR RATE", "sub": "1 − accuracy"},
                "precision": {"name": "PRECISION", "sub": "Weighted positive predictive value"},
                "recall": {"name": "RECALL", "sub": "Weighted sensitivity"},
                "f1Score": {"name": "F1 SCORE", "sub": "Weighted harmonic mean of P & R"},
                "cvTitle": f"{_N_FOLDS}-Fold Stratified Cross Validation — real refit per fold",
            },
        }
        cv_folds = [
            {
                "fold": f"Fold {i + 1}", "r2": f"{accs[i]:.3f}", "mse": "—",
                "precision": f"{precs[i] * 100:.1f}%", "recall": f"{recs[i] * 100:.1f}%",
                "mae": f"{1 - accs[i]:.4f}", "status": _fold_status(accs[i], m_acc, 0.05),
            }
            for i in range(_N_FOLDS)
        ]
        est = _clf(algorithm).fit(Xv, y)

    imp = _importance(est, Xv, y, feat)
    return {
        "id": spec["id"], "name": spec["name"], "desc": spec["desc"], "kind": spec["kind"],
        "target": spec["target"], "unit": spec.get("unit", ""),
        "est": est, "scaler": scaler, "feat": feat, "classes": classes,
        "means": {c: float(X[c].mean()) for c in feat},
        "std": {c: float(X[c].std() or 1.0) for c in feat},
        "importance": imp,
        "metrics": metrics, "accuracyLabel": accuracy_label, "metricLine": metric_line,
        "evalMetrics": eval_metrics, "cvFolds": cv_folds,
        "nFeatures": len(feat),
    }


def _read_registry() -> dict:
    if _REGISTRY.exists():
        try:
            return json.loads(_REGISTRY.read_text("utf-8"))
        except (OSError, ValueError):
            pass
    return {"active": None, "versions": []}


def _write_registry(reg: dict) -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REGISTRY.write_text(json.dumps(reg, indent=2), "utf-8")


def train(algorithm: str = "gradient_boosting") -> dict:
    global _cache
    df = load_dataset()
    if len(df) < 50:
        raise ValueError(f"UPI dataset needs >= 50 rows (have {len(df)}).")

    X = _feature_frame(df)
    feat = list(X.columns)
    scaler = StandardScaler().fit(X.to_numpy(float))
    Xv = scaler.transform(X.to_numpy(float))

    heads = [_train_one_head(df, X, spec, Xv, scaler, feat, algorithm) for spec in _HEADS]
    by_id = {h["id"]: h for h in heads}
    risk_h = by_id[RISK_MODEL_ID]
    stability_h = by_id[STABILITY_MODEL_ID]

    # Back-compat top-level metrics (score = Stability head, flag = Risk head)
    # — what the Model Hub card / registry table renders.
    metrics = {
        "scoreR2": stability_h["metrics"].get("r2"),
        "scoreMae": stability_h["metrics"].get("mae"),
        "flagAccuracy": risk_h["metrics"].get("accuracy"),
        "flagF1": risk_h["metrics"].get("f1"),
    }

    reg_json = _read_registry()
    version = max((v["version"] for v in reg_json["versions"]), default=0) + 1
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / f"upi_model_v{version}.joblib"
    import joblib
    artifact = {
        "scaler": stability_h["scaler"], "regressor": stability_h["est"], "classifier": risk_h["est"],
        "feat_names": feat, "classes": RISK_ORDER,
        "col_means": {c: float(X[c].mean()) for c in feat},
        "col_std": {c: float(X[c].std() or 1.0) for c in feat},
        "heads": {h["id"]: {k: h[k] for k in
                            ("est", "scaler", "feat", "classes", "kind", "target", "unit",
                             "means", "std", "importance", "name")}
                  for h in heads},
    }
    joblib.dump(artifact, path)
    sha = serialization.sha256_file(path)

    head_meta = [
        {"id": h["id"], "name": h["name"], "desc": h["desc"], "kind": h["kind"],
         "target": h["target"], "unit": h.get("unit", ""), "metrics": h["metrics"],
         "accuracyLabel": h["accuracyLabel"], "metricLine": h["metricLine"],
         "nFeatures": h["nFeatures"], "classes": h["classes"],
         "evalMetrics": h.get("evalMetrics"), "cvFolds": h.get("cvFolds", [])}
        for h in heads
    ]
    meta = {
        "version": version, "path": path.name, "sha256": sha,
        "signature": serialization.sign(sha),
        "nSamples": int(len(df)), "nFeatures": len(feat),
        "classes": RISK_ORDER,
        "metrics": metrics,
        "heads": head_meta,
        "algorithm": algorithm,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
    }
    reg_json["versions"].append(meta)
    reg_json["active"] = version
    _write_registry(reg_json)
    _cache = {"artifact": artifact, "meta": meta}
    logger.info("UPI model v%d — %d rows, 4 heads: %s",
                version, len(df), {h["id"]: h["accuracyLabel"] for h in heads})
    return {
        "version": version, "nSamples": int(len(df)), "features": len(feat),
        "classes": RISK_ORDER, "metrics": metrics, "algorithm": algorithm,
        "models": head_meta,
    }


# ── loading / prediction ─────────────────────────────────────────────────

def is_trained() -> bool:
    return bool(_read_registry()["versions"])


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
        import joblib
        artifact = joblib.load(path)
    except (serialization.ModelIntegrityError, OSError, ValueError) as exc:
        logger.error("REFUSING to load UPI model v%s — %s", meta.get("version"), exc)
        return None
    _cache = {"artifact": artifact, "meta": meta}
    return _cache


def _predict_heads(record: dict, loaded: dict) -> dict:
    """Run every stored head on one record. Returns
    {head_id: {name, kind, unit, value|label, proba?, topFactors}}."""
    heads = (loaded.get("artifact") or {}).get("heads") or {}
    rec = record or {}
    out: dict = {}
    for hid, h in heads.items():
        feat = h["feat"]
        means, stds = h.get("means") or {}, h.get("std") or {}
        row = np.array([float(rec.get(c, means.get(c, 0.0)) or means.get(c, 0.0)) for c in feat])
        xs = h["scaler"].transform(row.reshape(1, -1))
        entry = {"name": h.get("name", hid), "kind": h["kind"], "unit": h.get("unit", "")}

        imp = h.get("importance") or {}
        factors = []
        for i, c in enumerate(feat):
            z = (row[i] - means.get(c, 0.0)) / (stds.get(c, 1.0) or 1.0)
            weight = imp.get(c, 1.0)
            factors.append({"feature": c, "value": round(float(row[i]), 4), "zScore": round(z, 2),
                            "_rank": abs(z) * (weight + 1e-9)})
        factors.sort(key=lambda f: f["_rank"], reverse=True)
        entry["topFactors"] = [{k: v for k, v in f.items() if k != "_rank"} for f in factors[:6]]

        if h["kind"] == "regressor":
            entry["value"] = round(float(np.clip(h["est"].predict(xs)[0], 0, 100)), 2)
        else:
            classes = h.get("classes") or []
            est = h["est"]
            if hasattr(est, "predict_proba") and classes:
                p = est.predict_proba(xs)[0]
                # est.classes_ may be a STRICT SUBSET of `classes` — a weak-
                # supervision band the training data never produced (e.g. no
                # row ever hit HIGH) leaves the classifier never having seen
                # it, so p has fewer columns than len(classes). Map by the
                # model's own seen-class indices instead of assuming full
                # coverage, and zero-fill whatever it never saw.
                seen = getattr(est, "classes_", range(len(p)))
                proba = {classes[int(idx)]: round(float(pv), 4)
                         for idx, pv in zip(seen, p) if int(idx) < len(classes)}
                for c in classes:
                    proba.setdefault(c, 0.0)
                entry["proba"] = proba
                entry["label"] = max(proba, key=proba.get)
            else:
                pi = int(est.predict(xs)[0])
                entry["label"] = classes[pi] if pi < len(classes) else str(pi)
        out[hid] = entry
    return out


def predict(record: dict) -> dict:
    loaded = _load_active()
    if not loaded:
        return {"available": False, "message": "UPI model not trained yet — call POST /api/upi/train."}
    meta = loaded["meta"]
    heads = _predict_heads(record, loaded)

    stability = heads.get(STABILITY_MODEL_ID, {})
    risk = heads.get(RISK_MODEL_ID, {})

    return {
        "available": True,
        "stabilityScore": stability.get("value"),
        "riskFlag": risk.get("label"),
        "riskProbability": risk.get("proba"),
        "topFactors": stability.get("topFactors", []),
        "modelVersion": meta["version"],
        "headScores": heads,
    }


def predict_many(records: list[dict]) -> dict:
    preds = [predict(r) for r in records]
    ok = [p for p in preds if p.get("available")]
    if not ok:
        return {"available": False, "message": preds[0].get("message") if preds else "no records",
                "count": 0, "predictions": []}
    from collections import Counter
    return {
        "available": True, "count": len(ok),
        "avgStabilityScore": round(sum(p["stabilityScore"] for p in ok) / len(ok), 2),
        "riskCounts": dict(Counter(p["riskFlag"] for p in ok)),
        "modelVersion": ok[0]["modelVersion"],
        "predictions": preds,
    }


# ── registry ─────────────────────────────────────────────────────────────

HEAD_IDS = [h["id"] for h in _HEADS]
HEAD_NAMES = {h["id"]: h["name"] for h in _HEADS}


def _active_version(reg: dict | None = None) -> dict | None:
    reg = reg or _read_registry()
    active = reg.get("active")
    versions = reg.get("versions", [])
    return next((x for x in versions if x["version"] == active),
                versions[-1] if versions else None)


def head_evaluations() -> dict:
    """{head_id: {name, kind, evalMetrics, cvFolds}} for the active bundle —
    feeds the Model Evaluation panel's UPI rows and per-head detail."""
    v = _active_version()
    if not v:
        return {}
    out: dict = {}
    for h in v.get("heads", []):
        if h.get("evalMetrics"):
            out[h["id"]] = {
                "name": h.get("name"),
                "kind": h.get("kind"),
                "evalMetrics": h["evalMetrics"],
                "cvFolds": h.get("cvFolds", []),
            }
    return out


def eval_context() -> dict:
    """Algorithm + trained-at for the active bundle (panel sub-header)."""
    v = _active_version()
    if not v:
        return {}
    return {"algorithm": v.get("algorithm", "gradient_boosting"), "trainedAt": v.get("trainedAt")}


def reevaluate() -> dict:
    """Recompute every head's CV metrics on the current corpus for the ACTIVE
    bundle's algorithm and write them back into registry.json in place — no new
    version, no re-signing of the artifact. Mirrors app.bbps.model.reevaluate."""
    reg = _read_registry()
    v = _active_version(reg)
    if not v:
        return {}
    algo = v.get("algorithm", "gradient_boosting")
    df = load_dataset()
    X = _feature_frame(df)
    feat = list(X.columns)
    scaler = StandardScaler().fit(X.to_numpy(float))
    Xv = scaler.transform(X.to_numpy(float))
    fresh = {h["id"]: h for h in
             (_train_one_head(df, X, spec, Xv, scaler, feat, algo) for spec in _HEADS)}
    for hm in v.get("heads", []):
        f = fresh.get(hm["id"])
        if not f:
            continue
        hm["evalMetrics"] = f.get("evalMetrics")
        hm["cvFolds"] = f.get("cvFolds", [])
        hm["metrics"] = f["metrics"]
        hm["accuracyLabel"] = f["accuracyLabel"]
        hm["metricLine"] = f["metricLine"]
    _write_registry(reg)
    return head_evaluations()


def list_versions() -> list[dict]:
    reg = _read_registry()
    active = reg.get("active")
    out = [
        {
            "version": v["version"], "active": v["version"] == active,
            "algorithm": v.get("algorithm", "gradient_boosting"),
            "trainedAt": v.get("trainedAt"), "nSamples": v.get("nSamples"),
            "metrics": v.get("metrics", {}),
            "heads": [{"id": h["id"], "name": h["name"]} for h in v.get("heads", [])],
        }
        for v in reg.get("versions", [])
    ]
    out.sort(key=lambda x: x["version"], reverse=True)
    return out


def deploy_state() -> dict:
    """{head_id: bool} — whether each UPI head's output is live. Default: all on."""
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
        raise ValueError(f"Unknown UPI model '{head_id}'.")
    st = deploy_state()
    st[head_id] = bool(value)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _DEPLOY_JSON.write_text(json.dumps(st, indent=2), "utf-8")
    return st


def registry_view() -> dict:
    reg = _read_registry()
    return {"versions": list_versions(), "active": reg.get("active"),
            "heads": [{"id": h["id"], "name": h["name"]} for h in _HEADS],
            "deployed": deploy_state()}


def set_active_version(n: int) -> bool:
    global _cache
    reg = _read_registry()
    if not any(v["version"] == n for v in reg.get("versions", [])):
        return False
    reg["active"] = n
    _write_registry(reg)
    _cache = None
    return True


def status() -> dict:
    reg = _read_registry()
    if not reg["versions"]:
        return {"trained": False, "datasetRows": dataset_rows()}
    active = reg.get("active") or reg["versions"][-1]["version"]
    meta = next((v for v in reg["versions"] if v["version"] == active), reg["versions"][-1])
    return {
        "trained": True, "version": meta["version"], "totalVersions": len(reg["versions"]),
        "metrics": meta["metrics"], "nSamples": meta["nSamples"], "nFeatures": meta["nFeatures"],
        "classes": meta["classes"], "trainedAt": meta["trainedAt"],
        "heads": meta.get("heads", []),
        "datasetRows": dataset_rows(),
    }
