"""Fraud / anomaly pattern baselines — compare a tested applicant to training.

Generic, domain-agnostic. A domain (``"aa"`` / ``"gst"``) supplies:
  - at training time: a list of ``{metric: value}`` dicts (one per corpus
    entity) -> ``save_baseline`` computes a robust distribution per metric and
    persists it to ``models/<domain>_pattern_baseline.json``.
  - at test time: one ``{metric: value}`` dict -> ``compare`` returns the
    per-metric modified-z deviation and a band (within / elevated / extreme).

Same median/MAD method as security.outliers, just over arbitrary named
fraud-pattern metrics instead of the fixed 11-feature vector.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parents[3] / "models"
_MIN_SAMPLES = 8            # below this, a domain uses its static fallback
_Z_ELEVATED = 2.0
_Z_EXTREME = 3.5


def _f(x) -> float | None:
    try:
        v = float(x)
        return v if v == v else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _path(domain: str) -> Path:
    return _DIR / f"{domain}_pattern_baseline.json"


def save_baseline(domain: str, samples: list[dict]) -> dict | None:
    """Persist a per-metric {median, mad, p95} baseline for `domain`."""
    rows = [s for s in samples if isinstance(s, dict)]
    if len(rows) < _MIN_SAMPLES:
        logger.info("pattern_baseline[%s]: only %d samples — keeping static fallback.", domain, len(rows))
        return None
    metrics = sorted({k for r in rows for k in r})
    profile: dict[str, dict] = {}
    for m in metrics:
        vals = [v for v in (_f(r.get(m)) for r in rows) if v is not None]
        if len(vals) < _MIN_SAMPLES:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals]) or (statistics.pstdev(vals) or 1.0)
        profile[m] = {"median": round(med, 5), "mad": round(mad, 5),
                      "p95": round(_percentile(vals, 95), 5)}
    payload = {"domain": domain, "nSamples": len(rows), "metrics": profile}
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        _path(domain).write_text(json.dumps(payload, indent=2), "utf-8")
    except OSError:
        logger.warning("pattern_baseline[%s]: write failed", domain, exc_info=True)
    return payload


def _load(domain: str) -> dict:
    try:
        p = _path(domain)
        if p.exists():
            return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    return {}


def _band(z: float) -> str:
    return "extreme" if z >= _Z_EXTREME else "elevated" if z >= _Z_ELEVATED else "within"


def compare(domain: str, metrics: dict, *, static: dict | None = None,
            labels: dict | None = None) -> dict:
    """One tested applicant's metric dict vs the `domain` baseline.

    `static`: {metric: (median, mad)} fallback used per-metric when the trained
    baseline has nothing for it (or no baseline exists yet).
    Returns {basis, nSamples, perMetric: [...], worstZ, worstBand}.
    """
    prof = _load(domain)
    base = prof.get("metrics") or {}
    static = static or {}
    labels = labels or {}

    per: list[dict] = []
    worst_z = 0.0
    for m, raw in metrics.items():
        v = _f(raw)
        if v is None:
            continue
        if m in base:
            med, mad = base[m]["median"], max(base[m]["mad"], 1e-9)
            src = "training"
        elif m in static:
            med, mad = static[m][0], max(static[m][1], 1e-9)
            src = "typical"
        else:
            continue
        z = round(0.6745 * abs(v - med) / mad, 2)
        worst_z = max(worst_z, z)
        per.append({
            "metric": m,
            "label": labels.get(m, m.replace("_", " ")),
            "value": round(v, 4),
            "baseline": round(med, 4),
            "z": z,
            "band": _band(z),
            "basis": src,
        })
    per.sort(key=lambda d: d["z"], reverse=True)
    return {
        "basis": f"trained baseline (n={prof['nSamples']})" if base else "typical-statement fallback",
        "nSamples": prof.get("nSamples", 0),
        "perMetric": per,
        "worstZ": worst_z,
        "worstBand": _band(worst_z),
    }
