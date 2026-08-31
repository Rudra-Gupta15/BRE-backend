"""GST underwriting model.

Trains on gst/dataset.csv (6000 GST profiles) and predicts, for any GST record:
  * gst_underwriting_score  (0-100 regression)
  * gst_risk_flag           (LOW / MEDIUM / HIGH classification)

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
def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in DROP]
    num_cols = [c for c in cols if c not in CATEGORICAL]
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for c in CATEGORICAL:
        if c in df.columns:
            dummies = pd.get_dummies(df[c].astype(str).str.upper(), prefix=c)
            X = pd.concat([X, dummies.astype(float)], axis=1)
    return X


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
def train() -> dict:
    global _cache
    df = load_dataset()
    if len(df) < 200:
        raise ValueError(f"GST dataset needs >= 200 rows (have {len(df)}).")
    if SCORE_TARGET not in df.columns or FLAG_TARGET not in df.columns:
        raise ValueError(f"Dataset must contain '{SCORE_TARGET}' and '{FLAG_TARGET}'.")

    X = _feature_frame(df)
    feat_names = list(X.columns)
    Xv = X.to_numpy(dtype=float)

    y_score = pd.to_numeric(df[SCORE_TARGET], errors="coerce").fillna(df[SCORE_TARGET].median()).to_numpy()
    y_flag_raw = df[FLAG_TARGET].astype(str).str.upper().to_numpy()
    classes = [c for c in RISK_ORDER if c in set(y_flag_raw)] or sorted(set(y_flag_raw))
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_flag = np.array([cls_idx.get(v, 0) for v in y_flag_raw])

    scaler = StandardScaler().fit(Xv)
    Xs = scaler.transform(Xv)

    def _reg():
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07, random_state=42)

    def _clf():
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, random_state=42)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    pred_s = cross_val_predict(_reg(), Xs, y_score, cv=kf)
    metrics = {
        "scoreR2": round(float(r2_score(y_score, pred_s)), 4),
        "scoreMae": round(float(mean_absolute_error(y_score, pred_s)), 3),
    }
    if len(classes) > 1 and np.bincount(y_flag).min() >= 3:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        pred_f = cross_val_predict(_clf(), Xs, y_flag, cv=skf)
        metrics["flagAccuracy"] = round(float(accuracy_score(y_flag, pred_f)), 4)
        metrics["flagF1"] = round(float(f1_score(y_flag, pred_f, average="weighted", zero_division=0)), 4)

    reg, clf = _reg(), _clf()
    reg.fit(Xs, y_score)
    clf.fit(Xs, y_flag)

    artifact = {
        "scaler": scaler, "regressor": reg, "classifier": clf,
        "feat_names": feat_names, "classes": classes,
        "col_means": {c: float(X[c].mean()) for c in feat_names},
        "col_std": {c: float(X[c].std() or 1.0) for c in feat_names},
    }

    reg_json = _read_registry()
    version = max((v["version"] for v in reg_json["versions"]), default=0) + 1
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / f"gst_model_v{version}.joblib"
    import joblib
    joblib.dump(artifact, path)
    sha = serialization.sha256_file(path)
    meta = {
        "version": version, "path": path.name, "sha256": sha,
        "signature": serialization.sign(sha),
        "nSamples": int(len(df)), "nFeatures": len(feat_names),
        "classes": classes, "metrics": metrics,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
    }
    reg_json["versions"].append(meta)
    reg_json["active"] = version
    _write_registry(reg_json)

    _cache = {"artifact": artifact, "meta": meta}
    logger.info("GST model v%d trained on %d rows — %s", version, len(df), metrics)
    return {"version": version, "nSamples": int(len(df)), "features": len(feat_names),
            "classes": classes, "metrics": metrics}


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
        "datasetRows": dataset_rows(),
    }


# ── prediction ────────────────────────────────────────────────────────────
def _norm_key(k: str) -> str:
    return str(k).strip().lower().replace(" ", "_").replace("-", "_")


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
        "predictions": preds,
    }
