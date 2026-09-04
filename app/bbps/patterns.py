"""BBPS anomaly & fraud pattern recognition — the twin of app.aa.patterns /
app.gst.patterns.

  compute()       — scans EVERY uploaded BBPS statement for known suspicious
                    utility-payment signatures (duplicate bills, rapid repeat
                    payments, one bill far above normal, …) and reports how
                    many statements each fires on, plus per-metric spread.
                    Served by GET /api/bbps/patterns.
  test_patterns() — the same detectors on the Model Testing upload, compared
                    to the corpus baseline, with an overall verdict. Served by
                    GET /api/bbps/pattern-match.

Metrics come from real transaction-level data (extract_bbps_transactions),
not the aggregate analyze_bbps() summary alone — duplicate/rapid-repeat/
round-amount detection needs the individual tagged bill payments.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from app.aa.scoring import _parse_tx_date
from app.bbps.analysis import extract_bbps_transactions
from app.common.security import pattern_baseline
from app.common.state.session import session_state

logger = logging.getLogger(__name__)

# metric -> (label, typical-fallback median/MAD) for the fraud/anomaly compare
_FRAUD_METRICS: dict[str, tuple[str, tuple[float, float]]] = {
    "duplicate_payment_count":      ("Duplicate bill payments",        (0.0, 1.0)),
    "rapid_repeat_count":           ("Same bill paid twice fast",      (0.0, 1.0)),
    "bill_spike_ratio":             ("Bill spike vs. own average",     (1.3, 0.5)),
    "single_utility_dominance_pct": ("Single utility's share of spend", (55.0, 20.0)),
    "round_amount_ratio":           ("Round-number payment share",     (0.15, 0.15)),
    "vague_narration_ratio":        ("Unclassified BBPS narrations",   (0.05, 0.1)),
    "missed_payment_count":         ("Missed utility payments",        (0.5, 1.0)),
}
_STATIC_BASELINE = {k: v for k, (_, v) in _FRAUD_METRICS.items()}
_LABELS = {k: lbl for k, (lbl, _) in _FRAUD_METRICS.items()}


# ── real per-statement fraud metrics ────────────────────────────────────────

def _profile_fraud_metrics(stmt: dict) -> dict | None:
    """One statement's real BBPS transactions -> the metrics above. None if
    the statement has no BBPS activity (nothing to score)."""
    bbps_result = (stmt or {}).get("bbps") or {}
    if not bbps_result.get("available"):
        return None
    txns = extract_bbps_transactions((stmt or {}).get("transactions") or [])
    if not txns:
        return None

    by_type_avg = {t["utilityType"]: t.get("averageBillAmount", 0.0)
                    for t in bbps_result.get("byType", [])}
    by_type_total = {t["utilityType"]: t.get("totalPaid", 0.0)
                      for t in bbps_result.get("byType", [])}

    seen: dict[tuple[str, float], list] = defaultdict(list)
    round_ct = 0
    spike_ratio = 0.0
    other_ct = 0
    for t in txns:
        amt = t.get("amount") or 0.0
        kind = t["utilityType"]
        d = _parse_tx_date(t.get("date"))
        seen[(kind, round(amt))].append(d)
        if amt > 0 and amt % 500 == 0:
            round_ct += 1
        if kind == "OTHER_BBPS":
            other_ct += 1
        avg = by_type_avg.get(kind) or 0.0
        if avg > 0:
            spike_ratio = max(spike_ratio, amt / avg)

    dup = 0
    rapid = 0
    for dates in seen.values():
        ds = sorted(d for d in dates if d)
        if len(ds) < 2:
            continue
        dup += len(ds) - 1
        for i in range(1, len(ds)):
            if (ds[i] - ds[i - 1]).days < 5:
                rapid += 1

    total_paid = sum(by_type_total.values()) or sum(t.get("amount") or 0.0 for t in txns)
    top_paid = max(by_type_total.values(), default=0.0)

    return {
        "duplicate_payment_count":      dup,
        "rapid_repeat_count":           rapid,
        "bill_spike_ratio":             round(spike_ratio, 2),
        "single_utility_dominance_pct": round((top_paid / total_paid * 100) if total_paid else 0.0, 1),
        "round_amount_ratio":           round(round_ct / len(txns), 3),
        "vague_narration_ratio":        round(other_ct / len(txns), 3),
        "missed_payment_count":         float(bbps_result.get("missedPaymentCount", 0)),
    }


def fraud_typologies(m: dict) -> list[dict]:
    """Named suspicious-payment signatures with a match verdict. `hard`
    typologies drive the overall verdict to MATCH; the rest can only reach
    ELEVATED."""
    def row(name, desc, matched, elevated, evidence, hard=False):
        return {"name": name, "desc": desc, "hard": hard, "evidence": evidence,
                "verdict": "match" if matched else "elevated" if elevated else "none"}

    dup = m.get("duplicate_payment_count") or 0
    rapid = m.get("rapid_repeat_count") or 0
    spike = m.get("bill_spike_ratio") or 0.0
    dom = m.get("single_utility_dominance_pct") or 0.0
    round_r = m.get("round_amount_ratio") or 0.0
    vague = m.get("vague_narration_ratio") or 0.0
    missed = m.get("missed_payment_count") or 0

    return [
        row("Duplicate bill payments",
            "The same utility bill, for the same amount, was paid more than once",
            dup >= 2, dup >= 1, f"{dup:.0f} duplicate payment(s)", hard=True),
        row("Same bill paid twice fast",
            "A bill for the same utility was paid again within days of the last one",
            rapid >= 2, rapid >= 1, f"{rapid:.0f} rapid repeat payment(s)", hard=True),
        row("One bill far above normal",
            "A single bill payment is much larger than that utility's own average",
            spike > 4.0, spike > 2.5, f"{spike:.1f}x that utility's average bill", hard=True),
        row("Almost all spend on one utility",
            "Nearly every rupee of BBPS spend goes to a single utility type — a thin profile",
            dom > 90, dom > 75, f"{dom:.0f}% of spend on one utility type"),
        row("Too many round-number payments",
            "Several bill payments are suspiciously exact round numbers — real bills rarely are",
            round_r > 0.5, round_r > 0.3, f"{round_r * 100:.0f}% of payments are round numbers"),
        row("Vague / unclassified bill narrations",
            "Several BBPS payments carry a generic narration that doesn't say which utility it was",
            vague > 0.4, vague > 0.2, f"{vague * 100:.0f}% of payments unclassified"),
        row("Missed utility payments piling up",
            "Several expected monthly bills were never paid",
            missed >= 4, missed >= 2, f"{missed:.0f} missed payment(s)"),
    ]


def _verdict(typologies: list[dict], worst_band: str) -> tuple[str, str]:
    if any(t["hard"] and t["verdict"] == "match" for t in typologies):
        return "MATCHES FRAUD TYPOLOGY", "route to fraud review — a hard typology matched"
    soft = any(t["verdict"] in ("match", "elevated") for t in typologies)
    if worst_band == "extreme" or (soft and worst_band == "elevated"):
        return "ELEVATED ANOMALY SIGNATURE", "utility-payment pattern deviates from the trained corpus — manual review"
    if soft or worst_band == "elevated":
        return "SOME ELEVATED SIGNALS", "minor deviation — proceed with a note"
    return "CONSISTENT WITH TRAINING", "BBPS fraud/anomaly patterns are in the normal band"


# ── training-time: build the BBPS fraud-pattern baseline ───────────────────

def _per_file_hub_statements() -> list[dict]:
    return [s for s in session_state.statements_for("bbps_utility", "hub") if s and s.get("bbps")]


def save_training_baseline() -> dict | None:
    """Compute fraud metrics for every BBPS statement on the Model Hub and
    persist their distribution. Called after BBPS training."""
    samples = [m for s in _per_file_hub_statements() if (m := _profile_fraud_metrics(s))]
    return pattern_baseline.save_baseline("bbps", samples)


# ── test-time: compare one applicant to the baseline ───────────────────────

def _uploaded_statement() -> dict | None:
    return session_state.merged_statement_for("bbps_utility", "testing")


def test_patterns() -> dict:
    stmt = _uploaded_statement()
    if not (stmt and stmt.get("transactions")):
        return {"available": False,
                "message": "Upload a BBPS file on this page — pattern match runs on real transactions."}

    from app.bbps.analysis import analyze_bbps
    bbps_result = analyze_bbps(stmt["transactions"])
    m = _profile_fraud_metrics({**stmt, "bbps": bbps_result})
    if m is None:
        return {"available": False, "message": "No BBPS / utility bill payments found in this statement."}

    typ = fraud_typologies(m)
    cmp = pattern_baseline.compare("bbps", m, static=_STATIC_BASELINE, labels=_LABELS)
    models = pattern_model_score(m)
    verdict, note = _verdict(typ, cmp["worstBand"])
    if models.get("bbps_pattern_fraud_model", {}).get("probability", 0) >= 0.6 and "MATCH" not in verdict:
        verdict, note = "MATCHES FRAUD TYPOLOGY", "the BBPS Fraud Pattern Model flags this applicant"

    fp = models.get("bbps_pattern_fraud_model", {}).get("probability", 0.0)
    ap = models.get("bbps_pattern_anomaly_model", {}).get("probability", 0.0)
    concern, zone = _concern_score(fp, ap, cmp["worstZ"], "MATCH" in verdict)

    return {
        "available": True,
        "source": "bbps",
        "verdict": verdict,
        "verdictNote": note,
        "concernScore": concern,
        "zone": zone,
        "boxes": _boxes(typ, models, cmp["worstBand"]),
        "patternAnomalyScore": min(100.0, round(cmp["worstZ"] / 3.5 * 60.0, 1)),
        "modelScores": models,
        "typologies": typ,
        "comparison": cmp,
    }


_FRAUD_HEADLINE = {"clear": "No fraud detected", "review": "Worth a closer look",
                   "alert": "Possible fraud pattern"}
_ANOM_HEADLINE = {"clear": "Nothing unusual", "review": "Slightly unusual activity",
                  "alert": "Unusual utility-payment activity"}
_FRAUD_CANNED = {
    "clear": "The BBPS payments show no signs of duplicated or manufactured bill activity.",
    "review": "A couple of things look a little off — a manual read of the utility payments is advised.",
    "alert": "One or more known BBPS fraud patterns show up in these payments.",
}
_ANOM_CANNED = {
    "clear": "Utility bill payments look steady and predictable, like a normal household.",
    "review": "Some bills look different from the rest — not alarming, but worth checking.",
    "alert": "The utility-payment activity behaves quite differently from a normal profile.",
}


def _boxes(typ: list[dict], models: dict, worst_band: str) -> dict:
    hard = any(t["hard"] and t["verdict"] == "match" for t in typ)
    soft = any(t["verdict"] in ("match", "elevated") for t in typ)
    fp = models.get("bbps_pattern_fraud_model", {}).get("probability", 0.0)
    ap = models.get("bbps_pattern_anomaly_model", {}).get("probability", 0.0)
    fraud_status = "alert" if (hard or fp >= 0.6) else "review" if (soft or fp >= 0.3) else "clear"
    anom_status = ("alert" if (ap >= 0.7 or worst_band == "extreme")
                   else "review" if (ap >= 0.4 or worst_band == "elevated") else "clear")
    return {
        "fraud": {"status": fraud_status, "headline": _FRAUD_HEADLINE[fraud_status],
                  "detail": _FRAUD_CANNED[fraud_status], "probability": round(fp, 2)},
        "anomaly": {"status": anom_status, "headline": _ANOM_HEADLINE[anom_status],
                    "detail": _ANOM_CANNED[anom_status], "probability": round(ap, 2)},
    }


def _concern_score(fraud_p: float, anom_p: float, worst_z: float, hard_match: bool) -> tuple[int, str]:
    score = max(
        min(100.0, worst_z / 3.5 * 60.0),
        fraud_p * 100.0,
        anom_p * 75.0,
        100.0 if hard_match else 0.0,
    )
    s = int(round(score))
    return s, ("concern" if s >= 65 else "review" if s >= 35 else "good")


# ── anomaly & fraud patterns across the training set (per file, aggregated) ─

def _corpus_fraud_view() -> dict:
    files = _per_file_hub_statements()
    per_metrics: list[dict] = []
    per_typ: list[list[dict]] = []
    for s in files:
        m = _profile_fraud_metrics(s)
        if m is None:
            continue
        per_metrics.append(m)
        per_typ.append(fraud_typologies(m))

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
        vals = sorted(float(m[k]) for m in per_metrics if m.get(k) is not None)
        if not vals:
            continue
        med = vals[len(vals) // 2]
        thr = smed + smad / 0.6745 * 2.0
        metric_rows.append({"metric": k, "label": label, "median": round(med, 3),
                            "max": round(vals[-1], 3),
                            "flagged": sum(1 for v in vals if v > thr)})
    metric_rows.sort(key=lambda r: r["flagged"], reverse=True)
    return {"files": n, "typologies": typ_rows, "metrics": metric_rows}


def compute() -> dict:
    from app.bbps import model as bbps_model
    if not bbps_model.is_trained():
        return {"available": False, "message": "Train the BBPS model first."}
    return {
        "available": True,
        "source": "bbps",
        "trainedAt": bbps_model.eval_context().get("trainedAt"),
        "fraud": _corpus_fraud_view(),
        "baseline": pattern_baseline.compare("bbps", {}, static={}).get("basis"),
    }


# ── BBPS Fraud/Anomaly Pattern models (supervised, cross-validated) ────────
PATTERN_MODEL_IDS = ("bbps_pattern_fraud_model", "bbps_pattern_anomaly_model")
_PATTERN_MODEL_NAMES = {
    "bbps_pattern_fraud_model": "BBPS Fraud Pattern Model",
    "bbps_pattern_anomaly_model": "BBPS Anomaly Pattern Model",
}
_FRAUD_FEATS = ["duplicate_payment_count", "rapid_repeat_count", "bill_spike_ratio", "round_amount_ratio"]
_ANOMALY_FEATS = ["single_utility_dominance_pct", "vague_narration_ratio",
                  "missed_payment_count", "bill_spike_ratio"]

_PATTERN_MODELS: dict = {}   # {id: {est, features}} — in memory, retrained on BBPS train
_PATTERN_EVAL: dict = {}     # {id: {name, evalMetrics, cvFolds}}


def _fold_status(v: float, mean_v: float, tol: float) -> str:
    return "PASSED" if abs(v - mean_v) <= tol else "REVIEW"


def _cv_classifier(X: np.ndarray, y: np.ndarray) -> dict:
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
    """A labeled population for one BBPS pattern model. Real corpus rows are
    the 'clean' anchor; perturbed for the negatives, deliberately-suspicious
    rows synthesized for the positives, then labeled by the typology
    thresholds. Twin of app.aa.patterns._synth_pattern_dataset."""
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
        if f in ("duplicate_payment_count", "rapid_repeat_count"):
            pos[:, j] = rng.integers(1, 5, n_pos)
        elif f == "bill_spike_ratio":
            pos[:, j] = rng.uniform(2.5, 8.0, n_pos)
        elif f == "single_utility_dominance_pct":
            pos[:, j] = rng.uniform(78, 100, n_pos)
        elif f == "round_amount_ratio":
            pos[:, j] = rng.uniform(0.35, 0.9, n_pos)
        elif f == "vague_narration_ratio":
            pos[:, j] = rng.uniform(0.25, 0.7, n_pos)
        elif f == "missed_payment_count":
            pos[:, j] = rng.integers(2, 8, n_pos)
    X = np.clip(np.vstack([neg, pos]), 0.0, None)

    y = np.zeros(len(X), dtype=int)
    for i, row in enumerate(X):
        m = {**{f: 0.0 for f in _FRAUD_METRICS}, **dict(zip(features, row))}
        typ = fraud_typologies(m)
        if kind == "fraud":
            y[i] = 1 if any(t["hard"] and t["verdict"] == "match" for t in typ) else 0
        else:
            y[i] = 1 if (m["single_utility_dominance_pct"] > 75 or m["vague_narration_ratio"] > 0.2
                         or m["missed_payment_count"] >= 2 or m["bill_spike_ratio"] > 2.5) else 0
    flip = rng.random(len(y)) < 0.06
    y[flip] = 1 - y[flip]
    if y.sum() in (0, len(y)):        # degenerate — force a minority class
        y[: max(5, len(y) // 6)] = 1
        y[-max(5, len(y) // 6):] = 0
    return X, y


def train_pattern_models() -> dict:
    """Train + 5-fold CV the BBPS Fraud and Anomaly Pattern models. Real
    per-statement rows anchor the 'clean' class; suspicious rows are
    synthesized for the positive class so both models always train."""
    global _PATTERN_MODELS, _PATTERN_EVAL
    reals = [m for s in _per_file_hub_statements() if (m := _profile_fraud_metrics(s))]

    out, models = {}, {}
    for mid, feats, kind in (
        ("bbps_pattern_fraud_model", _FRAUD_FEATS, "fraud"),
        ("bbps_pattern_anomaly_model", _ANOMALY_FEATS, "anomaly"),
    ):
        try:
            X, y = _synth_pattern_dataset(reals, feats, kind)
            ev = _cv_classifier(X, y)
            models[mid] = {"est": GradientBoostingClassifier(random_state=42).fit(X, y),
                           "features": feats}
            out[mid] = {"name": _PATTERN_MODEL_NAMES[mid], **ev}
        except Exception:  # noqa: BLE001
            logger.warning("BBPS pattern model %s failed", mid, exc_info=True)
    _PATTERN_MODELS, _PATTERN_EVAL = models, out
    return out


def pattern_evaluations() -> dict:
    return _PATTERN_EVAL


def pattern_model_score(metrics: dict) -> dict:
    out = {}
    for mid, spec in _PATTERN_MODELS.items():
        try:
            row = np.array([[float(metrics.get(f) or 0.0) for f in spec["features"]]])
            pr = float(spec["est"].predict_proba(row)[0, 1])
            out[mid] = {"name": _PATTERN_MODEL_NAMES[mid], "probability": round(pr, 3), "flag": pr >= 0.5}
        except Exception:  # noqa: BLE001
            continue
    return out
