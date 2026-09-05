"""UPI anomaly & fraud pattern recognition — the twin of app.aa.patterns /
app.gst.patterns / app.bbps.patterns.

  compute()       — scans EVERY uploaded UPI file for known suspicious
                    transaction signatures (rapid P2P pass-through, high-risk
                    MCC concentration, structuring, …) and reports how many
                    files each fires on, plus per-metric spread. Served by
                    GET /api/upi/patterns.
  test_patterns() — the same detectors on the Model Testing upload, compared
                    to the corpus baseline, with an overall verdict. Served by
                    GET /api/upi/pattern-match.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from app.common.dates import parse_tx_date as _parse_tx_date
from app.common.security import pattern_baseline
from app.common.state.session import session_state
from app.upi.schema import HIGH_RISK_MCC

logger = logging.getLogger(__name__)

# metric -> (label, typical-fallback median/MAD) for the fraud/anomaly compare
_FRAUD_METRICS: dict[str, tuple[str, tuple[float, float]]] = {
    "rapid_pass_through_count":     ("Rapid money pass-through events",  (0.0, 1.0)),
    "high_risk_mcc_spend_pct":      ("High-risk MCC spend share",       (3.0, 5.0)),
    "structuring_count":            ("Just-under-threshold transfers",  (0.0, 1.0)),
    "single_counterparty_dominance_pct": ("Single counterparty's share of P2P volume", (35.0, 20.0)),
    "sudden_spike_ratio":           ("Peak-day vs. average-day volume", (2.0, 1.0)),
    "failed_ratio":                 ("Failed transaction share",        (0.03, 0.05)),
    "round_amount_ratio":           ("Round-number transfer share",     (0.10, 0.1)),
}
_STATIC_BASELINE = {k: v for k, (_, v) in _FRAUD_METRICS.items()}
_LABELS = {k: lbl for k, (lbl, _) in _FRAUD_METRICS.items()}


# ── real per-file fraud metrics ─────────────────────────────────────────────

def _profile_fraud_metrics(stmt: dict) -> dict | None:
    """One file's real UPI transactions -> the metrics above. None if the
    file has no usable UPI activity."""
    upi_result = (stmt or {}).get("upi") or {}
    if not upi_result.get("available"):
        return None
    txns = (stmt or {}).get("upi", {}).get("rawTransactions") or []
    if not txns:
        return None

    dated = [(t, _parse_tx_date(t.get("date"))) for t in txns]
    dated = [(t, d) for t, d in dated if d]
    if not dated:
        return None

    # Rapid pass-through: a CREDIT followed by a DEBIT of a similar amount
    # within 2 days — money received then moved straight back out.
    dated.sort(key=lambda td: td[1])
    rapid = 0
    for i, (t, d) in enumerate(dated):
        if t.get("type") != "CREDIT":
            continue
        for t2, d2 in dated[i + 1:]:
            if (d2 - d).days > 2:
                break
            if t2.get("type") == "DEBIT" and t.get("amount") and abs(t2["amount"] - t["amount"]) / t["amount"] < 0.15:
                rapid += 1
                break

    p2m_debits = [(t, d) for t, d in dated if t.get("mode") == "P2M" and t.get("type") == "DEBIT"]
    p2m_total = sum(t["amount"] for t, _ in p2m_debits) or 1.0
    high_risk_amt = sum(t["amount"] for t, _ in p2m_debits if (t.get("mcc") or "") in HIGH_RISK_MCC)

    # Structuring: several P2P transfers clustered just under a round
    # threshold (₹2,000 / ₹10,000 / ₹50,000) — classic smurfing signature.
    structuring = sum(
        1 for t, _ in dated
        if t.get("mode") == "P2P" and t.get("type") == "DEBIT"
        and any(lo <= t["amount"] < hi for lo, hi in ((1800, 2000), (9000, 10000), (45000, 50000)))
    )

    p2p_debits = [(t, d) for t, d in dated if t.get("mode") == "P2P" and t.get("type") == "DEBIT"]
    p2p_total = sum(t["amount"] for t, _ in p2p_debits) or 1.0
    by_payee: dict = defaultdict(float)
    for t, _ in p2p_debits:
        vpa = t.get("payee_vpa") or "unknown"
        by_payee[vpa] += t["amount"]
    top_payee_amt = max(by_payee.values(), default=0.0)

    per_day: dict = defaultdict(int)
    for _, d in dated:
        per_day[d] += 1
    days = len(per_day) or 1
    avg_day = len(dated) / days
    peak_day = max(per_day.values(), default=0)

    failed = sum(1 for t, _ in dated if t.get("status") == "FAILED")
    round_ct = sum(1 for t, _ in dated if t.get("amount") and t["amount"] % 500 == 0)

    return {
        "rapid_pass_through_count":          rapid,
        "high_risk_mcc_spend_pct":           round((high_risk_amt / p2m_total) * 100, 1),
        "structuring_count":                 structuring,
        "single_counterparty_dominance_pct": round((top_payee_amt / p2p_total) * 100, 1),
        "sudden_spike_ratio":                round((peak_day / avg_day) if avg_day else 0.0, 2),
        "failed_ratio":                      round(failed / len(dated), 4),
        "round_amount_ratio":                round(round_ct / len(dated), 4),
    }


def fraud_typologies(m: dict) -> list[dict]:
    """Named suspicious-transaction signatures with a match verdict. `hard`
    typologies drive the overall verdict to MATCH; the rest can only reach
    ELEVATED."""
    def row(name, desc, matched, elevated, evidence, hard=False):
        return {"name": name, "desc": desc, "hard": hard, "evidence": evidence,
                "verdict": "match" if matched else "elevated" if elevated else "none"}

    rapid = m.get("rapid_pass_through_count") or 0
    high_risk = m.get("high_risk_mcc_spend_pct") or 0.0
    structuring = m.get("structuring_count") or 0
    dominance = m.get("single_counterparty_dominance_pct") or 0.0
    spike = m.get("sudden_spike_ratio") or 0.0
    failed_ratio = m.get("failed_ratio") or 0.0
    round_ratio = m.get("round_amount_ratio") or 0.0

    return [
        row("Money received then moved straight out",
            "A UPI credit was followed by a similar-sized debit within two days — a mule-account signature",
            rapid >= 3, rapid >= 1, f"{rapid:.0f} rapid pass-through event(s)", hard=True),
        row("Transfers just under a reporting threshold",
            "Several P2P transfers cluster just below ₹2,000/₹10,000/₹50,000 — a structuring pattern",
            structuring >= 3, structuring >= 1, f"{structuring:.0f} just-under-threshold transfer(s)", hard=True),
        row("High-risk merchant spend",
            "A meaningful share of merchant spend goes to gambling / quasi-cash / pawn-shop MCCs",
            high_risk > 20, high_risk > 8, f"{high_risk:.1f}% of merchant spend is high-risk MCC", hard=True),
        row("One counterparty dominates P2P flow",
            "Almost all P2P money moves to/from a single person — an undisclosed lending relationship",
            dominance > 85, dominance > 60, f"{dominance:.0f}% of P2P volume with one counterparty"),
        row("Sudden transaction spike",
            "One day's transaction count is far above the applicant's own normal daily rate",
            spike > 6, spike > 3, f"{spike:.1f}x the average daily transaction count"),
        row("Many failed transactions",
            "A higher-than-normal share of UPI payments never went through",
            failed_ratio > 0.15, failed_ratio > 0.08, f"{failed_ratio * 100:.0f}% of transactions failed"),
        row("Too many round-number transfers",
            "Several transfers are suspiciously exact round numbers — real spend rarely is",
            round_ratio > 0.4, round_ratio > 0.25, f"{round_ratio * 100:.0f}% of transfers are round numbers"),
    ]


def _verdict(typologies: list[dict], worst_band: str) -> tuple[str, str]:
    if any(t["hard"] and t["verdict"] == "match" for t in typologies):
        return "MATCHES FRAUD TYPOLOGY", "route to fraud review — a hard typology matched"
    soft = any(t["verdict"] in ("match", "elevated") for t in typologies)
    if worst_band == "extreme" or (soft and worst_band == "elevated"):
        return "ELEVATED ANOMALY SIGNATURE", "UPI transaction pattern deviates from the trained corpus — manual review"
    if soft or worst_band == "elevated":
        return "SOME ELEVATED SIGNALS", "minor deviation — proceed with a note"
    return "CONSISTENT WITH TRAINING", "UPI fraud/anomaly patterns are in the normal band"


# ── training-time: build the UPI fraud-pattern baseline ────────────────────

def _per_file_hub_statements() -> list[dict]:
    return [s for s in session_state.statements_for("upi_enrichment", "hub") if s and s.get("upi")]


def save_training_baseline() -> dict | None:
    """Compute fraud metrics for every UPI file on the Model Hub and persist
    their distribution. Called after UPI training."""
    samples = [m for s in _per_file_hub_statements() if (m := _profile_fraud_metrics(s))]
    return pattern_baseline.save_baseline("upi", samples)


# ── test-time: compare one applicant to the baseline ───────────────────────

def _uploaded_statement() -> dict | None:
    """UPI's real payload lives in each file's own `upi.rawTransactions`
    side-channel, not the generic `transactions` field session_state's
    merged_statement_for() merges — so it can't be used here (same reason
    app.gst.patterns reads statements_for() directly). Merges every uploaded
    file's transactions into one synthetic statement _profile_fraud_metrics
    can read."""
    txns: list[dict] = []
    for s in session_state.statements_for("upi_enrichment", "testing"):
        txns.extend(((s or {}).get("upi") or {}).get("rawTransactions") or [])
    if not txns:
        return None
    return {"upi": {"available": True, "rawTransactions": txns}}


def test_patterns() -> dict:
    stmt = _uploaded_statement()
    if not (stmt and stmt.get("upi")):
        return {"available": False,
                "message": "Upload a UPI file on this page — pattern match runs on real transactions."}

    m = _profile_fraud_metrics(stmt)
    if m is None:
        return {"available": False, "message": "No UPI transactions found in this file."}

    typ = fraud_typologies(m)
    cmp = pattern_baseline.compare("upi", m, static=_STATIC_BASELINE, labels=_LABELS)
    models = pattern_model_score(m)
    verdict, note = _verdict(typ, cmp["worstBand"])
    if models.get("upi_pattern_fraud_model", {}).get("probability", 0) >= 0.6 and "MATCH" not in verdict:
        verdict, note = "MATCHES FRAUD TYPOLOGY", "the UPI Fraud Pattern Model flags this applicant"

    fp = models.get("upi_pattern_fraud_model", {}).get("probability", 0.0)
    ap = models.get("upi_pattern_anomaly_model", {}).get("probability", 0.0)
    concern, zone = _concern_score(fp, ap, cmp["worstZ"], "MATCH" in verdict)

    return {
        "available": True,
        "source": "upi",
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
                  "alert": "Unusual UPI activity"}
_FRAUD_CANNED = {
    "clear": "The UPI transactions show no signs of mule-style pass-through or structured transfers.",
    "review": "A couple of things look a little off — a manual read of the UPI history is advised.",
    "alert": "One or more known UPI fraud patterns show up in these transactions.",
}
_ANOM_CANNED = {
    "clear": "UPI transaction activity looks steady and predictable, like a normal user.",
    "review": "Some days look different from the rest — not alarming, but worth checking.",
    "alert": "The UPI activity behaves quite differently from a normal profile.",
}


def _boxes(typ: list[dict], models: dict, worst_band: str) -> dict:
    hard = any(t["hard"] and t["verdict"] == "match" for t in typ)
    soft = any(t["verdict"] in ("match", "elevated") for t in typ)
    fp = models.get("upi_pattern_fraud_model", {}).get("probability", 0.0)
    ap = models.get("upi_pattern_anomaly_model", {}).get("probability", 0.0)
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
    from app.upi import model as upi_model
    if not upi_model.is_trained():
        return {"available": False, "message": "Train the UPI model first."}
    return {
        "available": True,
        "source": "upi",
        "trainedAt": upi_model.eval_context().get("trainedAt"),
        "fraud": _corpus_fraud_view(),
        "baseline": pattern_baseline.compare("upi", {}, static={}).get("basis"),
    }


# ── UPI Fraud/Anomaly Pattern models (supervised, cross-validated) ─────────
PATTERN_MODEL_IDS = ("upi_pattern_fraud_model", "upi_pattern_anomaly_model")
_PATTERN_MODEL_NAMES = {
    "upi_pattern_fraud_model": "UPI Fraud Pattern Model",
    "upi_pattern_anomaly_model": "UPI Anomaly Pattern Model",
}
_FRAUD_FEATS = ["rapid_pass_through_count", "structuring_count", "high_risk_mcc_spend_pct",
                "single_counterparty_dominance_pct"]
_ANOMALY_FEATS = ["sudden_spike_ratio", "failed_ratio", "round_amount_ratio",
                  "single_counterparty_dominance_pct"]

_PATTERN_MODELS: dict = {}   # {id: {est, features}} — in memory, retrained on UPI train
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
    """A labeled population for one UPI pattern model. Real corpus rows are
    the 'clean' anchor; perturbed for the negatives, deliberately-suspicious
    rows synthesized for the positives, then labeled by the typology
    thresholds. Twin of app.bbps.patterns._synth_pattern_dataset."""
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
        if f in ("rapid_pass_through_count", "structuring_count"):
            pos[:, j] = rng.integers(1, 6, n_pos)
        elif f == "high_risk_mcc_spend_pct":
            pos[:, j] = rng.uniform(15, 50, n_pos)
        elif f == "single_counterparty_dominance_pct":
            pos[:, j] = rng.uniform(65, 100, n_pos)
        elif f == "sudden_spike_ratio":
            pos[:, j] = rng.uniform(4, 10, n_pos)
        elif f == "failed_ratio":
            pos[:, j] = rng.uniform(0.12, 0.4, n_pos)
        elif f == "round_amount_ratio":
            pos[:, j] = rng.uniform(0.3, 0.8, n_pos)
    X = np.clip(np.vstack([neg, pos]), 0.0, None)

    y = np.zeros(len(X), dtype=int)
    for i, row in enumerate(X):
        m = {**{f: 0.0 for f in _FRAUD_METRICS}, **dict(zip(features, row))}
        typ = fraud_typologies(m)
        if kind == "fraud":
            y[i] = 1 if any(t["hard"] and t["verdict"] == "match" for t in typ) else 0
        else:
            y[i] = 1 if (m["sudden_spike_ratio"] > 3 or m["failed_ratio"] > 0.08
                         or m["round_amount_ratio"] > 0.25
                         or m["single_counterparty_dominance_pct"] > 60) else 0
    flip = rng.random(len(y)) < 0.06
    y[flip] = 1 - y[flip]
    if y.sum() in (0, len(y)):        # degenerate — force a minority class
        y[: max(5, len(y) // 6)] = 1
        y[-max(5, len(y) // 6):] = 0
    return X, y


def train_pattern_models() -> dict:
    """Train + 5-fold CV the UPI Fraud and Anomaly Pattern models. Real
    per-file rows anchor the 'clean' class; suspicious rows are synthesized
    for the positive class so both models always train."""
    global _PATTERN_MODELS, _PATTERN_EVAL
    reals = [m for s in _per_file_hub_statements() if (m := _profile_fraud_metrics(s))]

    out, models = {}, {}
    for mid, feats, kind in (
        ("upi_pattern_fraud_model", _FRAUD_FEATS, "fraud"),
        ("upi_pattern_anomaly_model", _ANOMALY_FEATS, "anomaly"),
    ):
        try:
            X, y = _synth_pattern_dataset(reals, feats, kind)
            ev = _cv_classifier(X, y)
            models[mid] = {"est": GradientBoostingClassifier(random_state=42).fit(X, y),
                           "features": feats}
            out[mid] = {"name": _PATTERN_MODEL_NAMES[mid], **ev}
        except Exception:  # noqa: BLE001
            logger.warning("UPI pattern model %s failed", mid, exc_info=True)
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
