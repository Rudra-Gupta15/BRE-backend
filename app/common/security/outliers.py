"""Outlier detection on the underwriting feature vector.

Separate from the product's transaction-level anomaly tab - this is an
*input-sanity* gate: is this applicant's 11-feature vector within the
distribution the model was calibrated on? A far-outside vector gets flagged so
the score is treated as low-confidence / routed to review rather than trusted.

Method: robust modified z-score (median / MAD) per feature against a profile
built from the population of already-scored applicants (bre.inference_runs),
persisted to models/outlier_profile.json. Falls back to static ranges until
enough real data has accumulated.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

_PROFILE_FILE = Path(__file__).resolve().parents[3] / "models" / "outlier_profile.json"
_MIN_SAMPLES = 30
_Z_FLAG = 3.5  # modified z-score beyond this on a feature = outlier

_FEATURES = [
    "account_age_days", "avg_monthly_inflow", "avg_monthly_debit",
    "nach_bounce_count_90d", "dscr_ratio", "cash_withdrawal_ratio",
    "balance_volatility", "transaction_volatility", "minimum_balance",
    "foir_ratio", "income_stability",
]

# Static fallback (median, MAD-like spread) - deliberately wide.
_STATIC = {
    "account_age_days":       (540.0, 400.0),
    "avg_monthly_inflow":     (55_000.0, 40_000.0),
    "avg_monthly_debit":      (48_000.0, 35_000.0),
    "nach_bounce_count_90d":  (0.0, 1.0),
    "dscr_ratio":             (1.25, 0.6),
    "cash_withdrawal_ratio":  (0.08, 0.06),
    "balance_volatility":     (0.45, 0.3),
    "transaction_volatility": (0.5, 0.3),
    "minimum_balance":        (12_000.0, 15_000.0),
    "foir_ratio":             (75.0, 25.0),
    "income_stability":       (0.8, 0.15),
}


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rebuild_profile(feature_vectors: list[dict]) -> dict | None:
    """Compute and persist the outlier profile from a list of feature vectors
    (typically every bre.inference_runs.feature_vector)."""
    rows = [fv for fv in feature_vectors if isinstance(fv, dict)]
    if len(rows) < _MIN_SAMPLES:
        return None

    profile: dict[str, dict] = {}
    for f in _FEATURES:
        vals = [v for v in (_to_float(r.get(f)) for r in rows) if v is not None]
        if len(vals) < _MIN_SAMPLES:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals]) or (statistics.pstdev(vals) or 1.0)
        profile[f] = {
            "median": round(med, 4),
            "mad": round(mad, 4),
            "p1": round(_percentile(vals, 1), 4),
            "p99": round(_percentile(vals, 99), 4),
        }
    payload = {"n_samples": len(rows), "features": profile}
    try:
        _PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PROFILE_FILE.write_text(json.dumps(payload, indent=2))
    except OSError:
        logger.warning("could not write outlier profile", exc_info=True)
    return payload


def _load_profile() -> dict:
    try:
        if _PROFILE_FILE.exists():
            return json.loads(_PROFILE_FILE.read_text())
    except (OSError, ValueError):
        pass
    return {}


def score(fv: dict) -> dict:
    """Returns {outlierScore: 0-100, isOutlier: bool, flags: [...], basis: str}.
    `outlierScore` is the max modified-z across features, scaled to 0-100."""
    prof = _load_profile()
    feats = prof.get("features") or {}
    basis = f"population n={prof.get('n_samples')}" if feats else "static fallback"

    flags: list[dict] = []
    worst_z = 0.0
    for f in _FEATURES:
        v = _to_float(fv.get(f))
        if v is None:
            continue
        if f in feats:
            med, mad = feats[f]["median"], max(feats[f]["mad"], 1e-9)
        else:
            med, mad = _STATIC[f]
        z = 0.6745 * abs(v - med) / mad
        worst_z = max(worst_z, z)
        if z >= _Z_FLAG:
            flags.append({
                "feature": f,
                "value": round(v, 4),
                "expected": round(med, 4),
                "z": round(z, 2),
            })

    out_score = min(100.0, round(worst_z / _Z_FLAG * 60.0, 1))  # z==flag -> 60
    return {
        "outlierScore": out_score,
        "isOutlier": bool(flags),
        "flags": flags,
        "basis": basis,
    }


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
