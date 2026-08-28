"""Concept-drift monitoring.

Compares recent scored applicants (bre.inference_runs, last N) to a reference
window (the earliest runs, or the training-corpus feature distribution):

  * feature drift  - PSI per underwriting feature
  * prediction drift - shift in the credit-score / risk-grade distribution

Result is persisted to bre.drift_snapshots and surfaced on the Security page.
Crossing config.DRIFT_PSI_ALERT is the signal to retrain / cut a new
model_versions row.
"""
from __future__ import annotations

import logging

import numpy as np

from app import config

logger = logging.getLogger(__name__)

_FEATURES = [
    "account_age_days", "avg_monthly_inflow", "avg_monthly_debit",
    "nach_bounce_count_90d", "dscr_ratio", "cash_withdrawal_ratio",
    "balance_volatility", "transaction_volatility", "minimum_balance",
    "foir_ratio", "income_stability",
]


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 10:
        return 0.0
    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.clip(np.histogram(expected, bins=edges)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, bins=edges)[0] / len(actual), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _band(psi: float) -> str:
    if psi >= config.DRIFT_PSI_ALERT:
        return "alert"
    if psi >= config.DRIFT_PSI_WARN:
        return "warn"
    return "stable"


def compute(reference: list[dict], recent: list[dict],
            ref_scores: list[float] | None = None,
            recent_scores: list[float] | None = None) -> dict:
    """`reference` / `recent` are lists of feature-vector dicts."""
    if len(reference) < 20 or len(recent) < 10:
        return {
            "status": "insufficient-data",
            "referenceN": len(reference),
            "recentN": len(recent),
            "features": [],
            "overallPsi": 0.0,
        }

    feats = []
    worst = 0.0
    for f in _FEATURES:
        ref = np.array([_f(r.get(f)) for r in reference], dtype=float)
        rec = np.array([_f(r.get(f)) for r in recent], dtype=float)
        psi = _psi(ref[np.isfinite(ref)], rec[np.isfinite(rec)])
        worst = max(worst, psi)
        feats.append({
            "feature": f,
            "psi": round(psi, 3),
            "band": _band(psi),
            "referenceMean": _safe_mean(ref),
            "recentMean": _safe_mean(rec),
        })
    feats.sort(key=lambda x: x["psi"], reverse=True)

    prediction = None
    if ref_scores and recent_scores and len(ref_scores) >= 20 and len(recent_scores) >= 10:
        rs, cs = np.array(ref_scores, dtype=float), np.array(recent_scores, dtype=float)
        prediction = {
            "psi": round(_psi(rs, cs), 3),
            "referenceMean": round(float(np.mean(rs)), 1),
            "recentMean": round(float(np.mean(cs)), 1),
            "band": _band(_psi(rs, cs)),
        }

    return {
        "status": _band(worst),
        "referenceN": len(reference),
        "recentN": len(recent),
        "overallPsi": round(worst, 3),
        "features": feats,
        "prediction": prediction,
        "thresholds": {"warn": config.DRIFT_PSI_WARN, "alert": config.DRIFT_PSI_ALERT},
    }


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _safe_mean(a: np.ndarray):
    a = a[np.isfinite(a)]
    return round(float(a.mean()), 3) if len(a) else None


def _percentile(*args, **kwargs):  # kept for parity with outliers module
    return float(np.percentile(*args, **kwargs))
