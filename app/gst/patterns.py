"""GST anomaly & fraud pattern recognition — the twin of app.aa.patterns.

  compute()       — scans the whole GST training corpus for fraud signatures
                    (sales suppression, fake ITC, filing evasion, turnover
                    collapse …). Served by GET /api/gst/patterns.
  test_patterns() — the same detectors on the uploaded business(es), compared
                    to the corpus, with a verdict. GET /api/gst/pattern-match.
"""

from __future__ import annotations

import logging
from statistics import mean

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from app.common.security import pattern_baseline
from app.common.state.session import session_state
from app.gst import model as gst_model

logger = logging.getLogger(__name__)

# metric -> (label, typical-fallback median/MAD) for the fraud/anomaly compare
_FRAUD_METRICS: dict[str, tuple[str, tuple[float, float]]] = {
    "gstr1_vs_gstr3b_mismatch_pct":   ("GSTR-1 vs 3B sales mismatch %", (2.0, 4.0)),
    "missed_return_count":            ("Missed GST returns",            (0.0, 1.0)),
    "late_return_count":              ("Late GST returns",              (1.0, 2.0)),
    "itc_claim_ratio":                ("ITC claimed ÷ available",       (0.8, 0.2)),
    "turnover_decline_percentage":    ("Turnover decline %",            (0.0, 10.0)),
    "consecutive_declining_quarters": ("Consecutive declining quarters", (0.0, 1.0)),
    "turnover_growth_yoy":            ("Turnover growth YoY %",         (12.0, 20.0)),
    "top_buyer_sales_percentage":     ("Top-buyer sales concentration %", (30.0, 20.0)),
}
_STATIC_BASELINE = {k: v for k, (_, v) in _FRAUD_METRICS.items()}
_LABELS = {k: lbl for k, (lbl, _) in _FRAUD_METRICS.items()}



def _f(v, d: float = 0.0) -> float:
    try:
        f = float(str(v).replace(",", "").replace("₹", "").replace("%", "").strip())
        return f if f == f else d  # str(nan) parses back to nan without raising — catch it here
    except (TypeError, ValueError):
        return d



def _uploaded_profiles() -> list[dict]:
    # Test-time only — read the Model Testing upload (scope="testing"), the same
    # store every other GST tab on that page uses. (Was "hub", which is the
    # Model Hub store and is empty when the user only uploaded on Model Testing.)
    out: list[dict] = []
    for s in session_state.statements_for("gst_data", "testing"):
        for d in ((s or {}).get("gst") or {}).get("detail") or []:
            p = d.get("profile")
            if isinstance(p, dict):
                out.append(p)
    return out


def _agg(profiles: list[dict], key: str) -> float | None:
    vals = [_f(p[key]) for p in profiles if key in p and p[key] not in (None, "")]
    return mean(vals) if vals else None


# ── anomaly & fraud across the GST corpus (per business, aggregated) ────────

def _corpus_fraud_view() -> dict:
    """Run the GST fraud/anomaly detectors on every row of the training corpus,
    aggregate: how many businesses each typology fires on + per-metric spread."""
    try:
        df = gst_model.load_dataset()
    except FileNotFoundError:
        return {"files": 0, "typologies": [], "metrics": []}
    records = df.to_dict("records")
    per_metrics = [_profile_fraud_metrics({k: r.get(k) for k in df.columns}) for r in records]
    per_typ = [fraud_typologies(m) for m in per_metrics]
    n = len(per_metrics)
    if n == 0:
        return {"files": 0, "typologies": [], "metrics": []}

    base = per_typ[0]
    typ_rows = []
    for i, t0 in enumerate(base):
        matched = sum(1 for tl in per_typ if tl[i]["verdict"] == "match")
        elevated = sum(1 for tl in per_typ if tl[i]["verdict"] == "elevated")
        worst = next((tl[i]["evidence"] for tl in per_typ if tl[i]["verdict"] == "match"),
                     next((tl[i]["evidence"] for tl in per_typ if tl[i]["verdict"] == "elevated"), ""))
        typ_rows.append({"name": t0["name"], "desc": t0["desc"], "hard": t0["hard"],
                         "matched": matched, "elevated": elevated, "clear": n - matched - elevated,
                         "worstEvidence": worst})
    typ_rows.sort(key=lambda r: (r["matched"], r["elevated"]), reverse=True)

    metric_rows = []
    for k, (label, (smed, smad)) in _FRAUD_METRICS.items():
        vals = sorted(v for v in (mm.get(k) for mm in per_metrics) if v is not None)
        if not vals:
            continue
        med = vals[len(vals) // 2]
        thr = smed + smad / 0.6745 * 2.0
        metric_rows.append({"metric": k, "label": label, "median": round(med, 3),
                            "max": round(vals[-1], 3),
                            "flagged": sum(1 for v in vals if v > thr)})
    metric_rows.sort(key=lambda r: r["flagged"], reverse=True)
    return {"files": n, "typologies": typ_rows, "metrics": metric_rows}


# ── orchestrator ───────────────────────────────────────────────────────────

def compute() -> dict:
    if not gst_model.is_trained():
        return {"available": False, "message": "Train the GST model first."}
    ctx = gst_model.eval_context()
    return {
        "available": True,
        "source": "gst",
        "trainedAt": ctx.get("trainedAt"),
        "fraud": _corpus_fraud_view(),
        "modelSignals": [
            {"modelId": h["id"], "name": h["name"], "available": h.get("available", False),
             "note": h.get("note"),
             "features": [
                 {"name": f["column"], "label": f["label"],
                  "importance": f["importance"], "direction": "neutral"}
                 for f in h.get("features", [])
             ]}
            for h in gst_model.head_importances()
        ],
        "baseline": pattern_baseline.compare("gst", {}, static={}).get("basis"),
    }


# ── fraud / anomaly for GST ────────────────────────────────────────────────

def _itc_ratio(p: dict) -> float:
    claimed, avail = _f(p.get("itc_claimed_amount")), _f(p.get("itc_available_amount"))
    return (claimed / avail) if avail else 0.0


def _profile_fraud_metrics(p: dict) -> dict:
    return {
        "gstr1_vs_gstr3b_mismatch_pct":   _f(p.get("gstr1_vs_gstr3b_mismatch_pct")),
        "missed_return_count":            _f(p.get("missed_return_count")),
        "late_return_count":              _f(p.get("late_return_count")),
        "itc_claim_ratio":                _itc_ratio(p) or _f(p.get("itc_claim_ratio")),
        "turnover_decline_percentage":    _f(p.get("turnover_decline_percentage")),
        "consecutive_declining_quarters": _f(p.get("consecutive_declining_quarters")),
        "turnover_growth_yoy":            _f(p.get("turnover_growth_yoy")),
        "top_buyer_sales_percentage":     _f(p.get("top_buyer_sales_percentage")),
    }


def _agg_all_uploaded() -> dict:
    profs = _uploaded_profiles()
    keys = {k for p in profs for k in p}
    return {k: (_agg(profs, k) or 0.0) for k in keys}


def fraud_typologies(m: dict) -> list[dict]:
    def row(name, desc, matched, elevated, evidence, hard=False):
        return {"name": name, "desc": desc, "hard": hard, "evidence": evidence,
                "verdict": "match" if matched else "elevated" if elevated else "none"}

    mism = m.get("gstr1_vs_gstr3b_mismatch_pct") or 0.0
    itc = m.get("itc_claim_ratio") or 0.0
    return [
        row("Sales figures don't match",
            "The two GST returns report different sales for the same period",
            mism > 15, mism > 8, f"{mism:.1f}% gap between the two returns", hard=True),
        row("Claiming more tax credit than allowed",
            "The business claimed back more GST than its purchases support",
            itc > 1.10, itc > 1.02, f"claimed {itc:.2f}x what was available", hard=True),
        row("Skipped GST filings",
            "One or more GST returns were never filed",
            (m.get("missed_return_count") or 0) >= 2, (m.get("missed_return_count") or 0) >= 1,
            f"{m.get('missed_return_count') or 0:.0f} return(s) never filed", hard=True),
        row("Sales falling sharply",
            "Revenue has dropped steeply and kept dropping",
            (m.get("turnover_decline_percentage") or 0) > 40
            or (m.get("consecutive_declining_quarters") or 0) >= 3,
            (m.get("turnover_decline_percentage") or 0) > 20,
            f"down {m.get('turnover_decline_percentage') or 0:.0f}% over "
            f"{m.get('consecutive_declining_quarters') or 0:.0f} quarter(s)"),
        row("Sales jumped unrealistically",
            "Revenue grew far faster than a real business normally can",
            (m.get("turnover_growth_yoy") or 0) > 200, (m.get("turnover_growth_yoy") or 0) > 120,
            f"up {m.get('turnover_growth_yoy') or 0:.0f}% in a year"),
        row("Depends on a single customer",
            "Most sales go to just one buyer, so losing them would hit hard",
            (m.get("top_buyer_sales_percentage") or 0) > 70,
            (m.get("top_buyer_sales_percentage") or 0) > 50,
            f"one buyer = {m.get('top_buyer_sales_percentage') or 0:.0f}% of sales"),
    ]


def _verdict(typologies: list[dict], worst_band: str) -> tuple[str, str]:
    if any(t["hard"] and t["verdict"] == "match" for t in typologies):
        return "MATCHES FRAUD TYPOLOGY", "route to GST compliance review — a hard typology matched"
    soft = any(t["verdict"] in ("match", "elevated") for t in typologies)
    if worst_band == "extreme" or (soft and worst_band == "elevated"):
        return "ELEVATED ANOMALY SIGNATURE", "filing/turnover pattern deviates from the trained corpus — manual review"
    if soft or worst_band == "elevated":
        return "SOME ELEVATED SIGNALS", "minor deviation — proceed with a note"
    return "CONSISTENT WITH TRAINING", "GST fraud/anomaly patterns are in the normal band"


def save_training_baseline() -> dict | None:
    """Fraud-metric distribution over the current GST corpus. Called after
    /gst training."""
    try:
        df = gst_model.load_dataset()
    except FileNotFoundError:
        return None
    samples = [_profile_fraud_metrics({k: r.get(k) for k in df.columns})
               for r in df.to_dict("records")]
    return pattern_baseline.save_baseline("gst", samples)


# ── GST Fraud/Anomaly Pattern models (supervised, cross-validated) ─────────
PATTERN_MODEL_IDS = ("gst_pattern_fraud_model", "gst_pattern_anomaly_model")
_PATTERN_MODEL_NAMES = {
    "gst_pattern_fraud_model": "GST Fraud Pattern Model",
    "gst_pattern_anomaly_model": "GST Anomaly Pattern Model",
}
_FRAUD_FEATS = ["gstr1_vs_gstr3b_mismatch_pct", "itc_claim_ratio", "missed_return_count"]
_ANOMALY_FEATS = ["turnover_decline_percentage", "consecutive_declining_quarters",
                  "turnover_growth_yoy", "top_buyer_sales_percentage", "late_return_count"]

_PATTERN_MODELS: dict = {}   # {id: {est, features}}  — in memory, retrained on /gst train
_PATTERN_EVAL: dict = {}     # {id: {name, evalMetrics, cvFolds}}


def _fold_status(v: float, mean_v: float, tol: float) -> str:
    return "PASSED" if abs(v - mean_v) <= tol else "REVIEW"


def _cv_classifier(X, y) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc, prec, rec, f1 = [], [], [], []
    for tr, te in skf.split(X, y):
        est = GradientBoostingClassifier(random_state=42).fit(X[tr], y[tr])
        p = est.predict(X[te])
        acc.append(accuracy_score(y[te], p)); prec.append(precision_score(y[te], p, zero_division=0))
        rec.append(recall_score(y[te], p, zero_division=0)); f1.append(f1_score(y[te], p, zero_division=0))
    m_acc = float(np.mean(acc))
    folds = [{"fold": f"Fold {i + 1}", "r2": f"{acc[i]:.3f}", "mse": "—",
              "precision": f"{prec[i] * 100:.1f}%", "recall": f"{rec[i] * 100:.1f}%",
              "mae": f"{1 - acc[i]:.4f}", "status": _fold_status(acc[i], m_acc, 0.06)} for i in range(5)]
    return {"evalMetrics": {
        "r2Score": f"{m_acc:.3f}", "mse": "—",
        "precision": f"{np.mean(prec) * 100:.1f}%", "recall": f"{np.mean(rec) * 100:.1f}%",
        "mae": f"{1 - m_acc:.4f}", "f1Score": f"{np.mean(f1):.3f}",
        "metricMeta": {
            "r2Score": {"name": "ACCURACY", "sub": "Correct predictions (5-fold CV)"},
            "mse": {"name": "—", "sub": ""}, "mae": {"name": "ERROR RATE", "sub": "1 − accuracy"},
            "precision": {"name": "PRECISION", "sub": "Flagged patterns that were real"},
            "recall": {"name": "RECALL", "sub": "Real patterns that were caught"},
            "f1Score": {"name": "F1 SCORE", "sub": "Harmonic mean of P & R"},
            "cvTitle": "5-Fold Stratified Cross Validation — real refit per fold",
        }}, "cvFolds": folds}


def _synth_pattern_dataset(reals: list[dict], features: list[str], kind: str,
                           n: int = 700, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """A labeled population for one GST pattern model. Real corpus rows are the
    'clean' anchor; we perturb them for the negatives and synthesize deliberately
    suspicious rows for the positives, then label by the typology thresholds.
    Twin of app.aa.patterns._synth_pattern_dataset — guarantees both classes so
    the model always trains, even on a spotless corpus."""
    rng = np.random.default_rng(seed)
    anchor = np.array(
        [[float(r.get(f, 0.0) or 0.0) for f in features] for r in reals], dtype=float,
    ) if reals else np.array([[_STATIC_BASELINE[f][0] for f in features]], dtype=float)
    base = anchor.mean(axis=0)
    spread = np.maximum(anchor.std(axis=0) if len(anchor) > 1 else np.abs(base) * 0.5, 0.05)

    n_pos = n // 3
    neg = rng.normal(base, spread, size=(n - n_pos, len(features)))
    pos = rng.normal(base, spread, size=(n_pos, len(features)))
    for j, f in enumerate(features):
        if f == "gstr1_vs_gstr3b_mismatch_pct":
            pos[:, j] = rng.uniform(16, 45, n_pos)
        elif f == "itc_claim_ratio":
            pos[:, j] = rng.uniform(1.12, 1.7, n_pos)
        elif f == "missed_return_count":
            pos[:, j] = rng.integers(2, 7, n_pos)
        elif f == "turnover_decline_percentage":
            pos[:, j] = rng.uniform(25, 70, n_pos)
        elif f == "consecutive_declining_quarters":
            pos[:, j] = rng.integers(2, 6, n_pos)
        elif f == "late_return_count":
            pos[:, j] = rng.integers(2, 9, n_pos)
        elif f == "top_buyer_sales_percentage":
            pos[:, j] = rng.uniform(60, 95, n_pos)
    X = np.clip(np.vstack([neg, pos]), 0.0, None)

    y = np.zeros(len(X), dtype=int)
    for i, row in enumerate(X):
        m = {**{f: 0.0 for f in _FRAUD_METRICS}, **dict(zip(features, row))}
        if kind == "fraud":
            y[i] = 1 if any(t["hard"] and t["verdict"] == "match" for t in fraud_typologies(m)) else 0
        else:
            y[i] = 1 if (m["turnover_decline_percentage"] > 20
                         or m["consecutive_declining_quarters"] >= 2
                         or m["late_return_count"] >= 2) else 0
    flip = rng.random(len(y)) < 0.06
    y[flip] = 1 - y[flip]
    if y.sum() in (0, len(y)):        # degenerate — force a minority class
        y[: max(5, len(y) // 6)] = 1
        y[-max(5, len(y) // 6):] = 0
    return X, y


def train_pattern_models() -> dict:
    """Train + 5-fold CV the GST Fraud and Anomaly Pattern models. Real corpus
    rows anchor the 'clean' class; suspicious rows are synthesized for the
    positive class so both models always train (mirrors app.aa.patterns)."""
    global _PATTERN_MODELS, _PATTERN_EVAL
    try:
        df = gst_model.load_dataset()
        recs = [_profile_fraud_metrics(r) for r in df.to_dict("records")]
    except FileNotFoundError:
        recs = []

    out, models = {}, {}
    for mid, feats, kind in (
        ("gst_pattern_fraud_model", _FRAUD_FEATS, "fraud"),
        ("gst_pattern_anomaly_model", _ANOMALY_FEATS, "anomaly"),
    ):
        try:
            X, y = _synth_pattern_dataset(recs, feats, kind)
            ev = _cv_classifier(X, y)
            models[mid] = {"est": GradientBoostingClassifier(random_state=42).fit(X, y),
                           "features": feats}
            out[mid] = {"name": _PATTERN_MODEL_NAMES[mid], **ev}
        except Exception:  # noqa: BLE001
            logger.warning("GST pattern model %s failed", mid, exc_info=True)
    _PATTERN_MODELS, _PATTERN_EVAL = models, out
    return out


def pattern_evaluations() -> dict:
    return _PATTERN_EVAL


def _pattern_model_score(m: dict) -> dict:
    out = {}
    for mid, spec in _PATTERN_MODELS.items():
        try:
            row = np.array([[float(m.get(f) or 0.0) for f in spec["features"]]])
            pr = float(spec["est"].predict_proba(row)[0, 1])
            out[mid] = {"name": _PATTERN_MODEL_NAMES[mid], "probability": round(pr, 3), "flag": pr >= 0.5}
        except Exception:  # noqa: BLE001
            continue
    return out


def _zone_reason(reason_key: str, typ: list[dict], cmp: dict, fp: float, ap: float) -> dict:
    """Which of the four concern signals actually drove the verdict, in plain
    words — so the banner doesn't say "matches a known fraud pattern" when
    what really fired was a pure statistical outlier, or vice versa."""
    if reason_key == "typology":
        matched = [t for t in typ if t["hard"] and t["verdict"] == "match"]
        names = "; ".join(f"{t['name']} ({t['evidence']})" for t in matched) or "a known fraud typology"
        return {"kind": "typology",
                "line": f"Matched a specific, named fraud pattern: {names}."}
    if reason_key == "fraud_model":
        return {"kind": "fraud_model",
                "line": f"The trained GST Fraud Pattern Model scored this business at "
                        f"{fp * 100:.0f}% fraud probability, based on patterns learned from the training corpus "
                        "(not a specific named typology match)."}
    if reason_key in ("anomaly_model", "anomaly_metric"):
        worst = (cmp.get("perMetric") or [{}])[0]
        if worst.get("metric"):
            return {"kind": "anomaly",
                    "line": f"Statistically unusual, not a fraud-pattern match: {worst['label']} is "
                            f"{worst['value']} for this account vs. {worst['baseline']} normally "
                            f"(z={worst['z']}). Could be a legitimate but atypical business."}
        return {"kind": "anomaly",
                "line": f"Statistically unusual overall activity (anomaly probability {ap * 100:.0f}%), "
                        "not a specific fraud-pattern match."}
    return {"kind": "none", "line": "Nothing about this account stood out."}


def test_patterns() -> dict:
    profs = _uploaded_profiles()
    if not profs:
        return {"available": False,
                "message": "Upload a GST file on this page — pattern match runs on the scored business(es)."}
    agg = _agg_all_uploaded()
    m = _profile_fraud_metrics(agg)
    typ = fraud_typologies(m)
    cmp = pattern_baseline.compare("gst", m, static=_STATIC_BASELINE, labels=_LABELS)
    models = _pattern_model_score(m)
    verdict, note = _verdict(typ, cmp["worstBand"])
    if models.get("gst_pattern_fraud_model", {}).get("probability", 0) >= 0.6 and "MATCH" not in verdict:
        verdict, note = "MATCHES FRAUD TYPOLOGY", "the GST Fraud Pattern Model flags this business"

    fp = models.get("gst_pattern_fraud_model", {}).get("probability", 0.0)
    ap = models.get("gst_pattern_anomaly_model", {}).get("probability", 0.0)

    # The concern score used to be a bare max() of these four — now we track
    # WHICH one wins, so the UI can explain the actual reason instead of
    # always showing the same "known fraud pattern" copy.
    candidates = {
        "typology": 100.0 if "MATCH" in verdict else 0.0,
        "fraud_model": fp * 100.0,
        "anomaly_model": ap * 75.0,
        "anomaly_metric": min(100.0, cmp["worstZ"] / 3.5 * 60.0),
    }
    reason_key = max(candidates, key=candidates.get)
    concern = int(round(candidates[reason_key]))
    zone = "concern" if concern >= 65 else "review" if concern >= 35 else "good"
    zone_reason = _zone_reason(reason_key, typ, cmp, fp, ap) if zone != "good" else {
        "kind": "none", "line": "Nothing about this account stood out."}

    return {
        "available": True,
        "source": "gst",
        "businesses": len(profs),
        "verdict": verdict,
        "verdictNote": note,
        "concernScore": concern,
        "zone": zone,
        "zoneReason": zone_reason,
        "boxes": _gst_boxes(typ, models, cmp["worstBand"]),
        "patternAnomalyScore": min(100.0, round(cmp["worstZ"] / 3.5 * 60.0, 1)),
        "modelScores": models,
        "typologies": typ,
        "comparison": cmp,
    }


_GST_FRAUD_CANNED = {
    "clear": "The GST filings show no signs of hidden sales or inflated input-tax claims.",
    "review": "A couple of filing / tax figures look a little off — worth a manual check.",
    "alert": "One or more known GST fraud patterns show up (sales suppression, fake ITC, or missed returns).",
}
_GST_ANOM_CANNED = {
    "clear": "Turnover and filing behaviour look steady, like a healthy registered business.",
    "review": "Turnover or filing has shifted recently — not alarming, but worth checking.",
    "alert": "The business's GST activity looks quite different from a typical filer.",
}


def _gst_boxes(typ: list[dict], models: dict, worst_band: str) -> dict:
    hard = any(t["hard"] and t["verdict"] == "match" for t in typ)
    soft = any(t["verdict"] in ("match", "elevated") for t in typ)
    fp = models.get("gst_pattern_fraud_model", {}).get("probability", 0.0)
    ap = models.get("gst_pattern_anomaly_model", {}).get("probability", 0.0)
    fs = "alert" if (hard or fp >= 0.6) else "review" if (soft or fp >= 0.3) else "clear"
    ans = ("alert" if (ap >= 0.7 or worst_band == "extreme")
           else "review" if (ap >= 0.4 or worst_band == "elevated") else "clear")
    fh = {"clear": "No fraud detected", "review": "Worth a closer look", "alert": "Possible fraud pattern"}
    ah = {"clear": "Nothing unusual", "review": "Slightly unusual activity", "alert": "Unusual GST activity"}
    return {
        "fraud": {"status": fs, "headline": fh[fs], "detail": _GST_FRAUD_CANNED[fs], "probability": round(fp, 2)},
        "anomaly": {"status": ans, "headline": ah[ans], "detail": _GST_ANOM_CANNED[ans], "probability": round(ap, 2)},
    }
