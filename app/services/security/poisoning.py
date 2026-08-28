"""Training-data poisoning defences.

The accumulated CSV corpus drives every future model, and the training endpoint
is the highest-value target. Layers:

  validate_dataframe()  - schema + per-column range checks; returns the clean
                          frame and a rejection report (bad rows are dropped,
                          not ingested)
  distribution_report() - compares an incoming batch to the existing corpus
                          (per-column mean shift / PSI) so a batch that would
                          move the distribution is visible before it lands
  promotion_guard()     - a freshly trained candidate may not regress the
                          currently-active model on the frozen golden set by
                          more than config.MODEL_PROMOTE_MAX_REGRESSION

The golden set is a small trusted slice frozen on first training
(models/golden_set.csv) - poisoned corpus data cannot touch it.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app import config

logger = logging.getLogger(__name__)

_GOLDEN_FILE = Path(__file__).resolve().parents[3] / "models" / "golden_set.csv"

# Plausible ranges for the numeric columns the dataset is expected to carry.
# Anything outside -> the row is quarantined (dropped from ingestion).
_RANGES: dict[str, tuple[float, float]] = {
    "bounce_count": (0, 100),
    "overdraft_count": (0, 100),
    "negative_balance_days": (0, 366),
    "emi_to_credit_ratio": (0, 5),
    "existing_debt_burden": (0, 5),
    "income_consistency_score": (0, 1),
    "credit_regularity_score": (0, 1),
    "cash_flow_stability_score": (0, 1),
    "monthly_net_cash_flow": (-5, 5),
    "credit_debit_ratio": (0, 100),
    "average_monthly_balance": (-1e8, 1e9),
    "monthly_income": (0, 1e9),
    "age": (18, 100),
}


def validate_dataframe(raw: bytes) -> tuple[pd.DataFrame, dict]:
    """Parse + clean an uploaded training CSV. Returns (clean_df, report).
    Raises ValueError only when nothing usable remains."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"not a readable CSV: {exc}") from exc

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    n0 = len(df)
    if n0 == 0:
        raise ValueError("CSV has no rows")

    reasons: dict[str, int] = {}
    mask = pd.Series(True, index=df.index)

    # 1. drop fully-empty and exact-duplicate rows
    empty = df.isna().all(axis=1)
    if empty.any():
        reasons["empty row"] = int(empty.sum())
        mask &= ~empty
    dups = df.duplicated()
    if dups.any():
        reasons["exact duplicate"] = int(dups.sum())
        mask &= ~dups

    # 2. numeric range checks
    for col, (lo, hi) in _RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s.notna() & ((s < lo) | (s > hi))
        inf = np.isinf(s.fillna(0))
        bad = bad | inf
        if bad.any():
            reasons[f"{col} out of [{lo}, {hi}]"] = int((bad & mask).sum())
            mask &= ~bad

    # 3. label sanity - if a ground-truth column exists it must be 0/1
    for label_col in ("default", "is_default", "target", "label"):
        if label_col in df.columns:
            s = pd.to_numeric(df[label_col], errors="coerce")
            bad = ~s.isin([0, 1])
            if bad.any():
                reasons[f"{label_col} not 0/1"] = int((bad & mask).sum())
                mask &= ~bad

    clean = df[mask].reset_index(drop=True)
    report = {
        "rowsIn": n0,
        "rowsAccepted": int(len(clean)),
        "rowsRejected": int(n0 - len(clean)),
        "reasons": reasons,
    }
    if len(clean) == 0:
        raise ValueError(f"every row failed validation: {reasons}")
    return clean, report


def distribution_report(new_df: pd.DataFrame, corpus_df: pd.DataFrame | None) -> dict:
    """Per-column shift of an incoming batch vs. the existing corpus."""
    if corpus_df is None or len(corpus_df) < 50:
        return {"status": "no-baseline", "shifts": []}

    shifts = []
    worst = 0.0
    for col in _RANGES:
        if col not in new_df.columns or col not in corpus_df.columns:
            continue
        a = pd.to_numeric(corpus_df[col], errors="coerce").dropna()
        b = pd.to_numeric(new_df[col], errors="coerce").dropna()
        if len(a) < 30 or len(b) < 10:
            continue
        psi = _psi(a.to_numpy(), b.to_numpy())
        worst = max(worst, psi)
        if psi >= config.DRIFT_PSI_WARN:
            shifts.append({
                "column": col,
                "psi": round(psi, 3),
                "corpusMean": round(float(a.mean()), 3),
                "batchMean": round(float(b.mean()), 3),
            })

    status = "ok"
    if worst >= config.DRIFT_PSI_ALERT:
        status = "alert"
    elif worst >= config.DRIFT_PSI_WARN:
        status = "warn"
    return {"status": status, "worstPsi": round(worst, 3), "shifts": shifts}


def ensure_golden_set(corpus_df: pd.DataFrame, frac: float = 0.15, cap: int = 800) -> int:
    """Freeze a trusted validation slice the first time we train. Never
    overwritten afterwards, so later (possibly poisoned) uploads can't touch
    the yardstick. Returns the golden-set size."""
    if _GOLDEN_FILE.exists():
        try:
            return len(pd.read_csv(_GOLDEN_FILE))
        except Exception:  # noqa: BLE001
            pass
    n = min(cap, max(50, int(len(corpus_df) * frac)))
    sample = corpus_df.sample(n=min(n, len(corpus_df)), random_state=42)
    try:
        _GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(_GOLDEN_FILE, index=False)
    except OSError:
        logger.warning("could not write golden set", exc_info=True)
    return len(sample)


def load_golden_set() -> pd.DataFrame | None:
    try:
        if _GOLDEN_FILE.exists():
            return pd.read_csv(_GOLDEN_FILE)
    except Exception:  # noqa: BLE001
        pass
    return None


def promotion_guard(candidate_acc: float, active_acc: float | None) -> dict:
    """Decide whether a newly trained model may become 'active'."""
    if active_acc is None:
        return {"promote": True, "reason": "first model - nothing to regress"}
    drop = active_acc - candidate_acc
    if drop > config.MODEL_PROMOTE_MAX_REGRESSION:
        return {
            "promote": False,
            "reason": (
                f"candidate accuracy {candidate_acc:.3f} is {drop:.3f} below the active "
                f"model ({active_acc:.3f}) on the golden set - exceeds the "
                f"{config.MODEL_PROMOTE_MAX_REGRESSION:.3f} regression limit; "
                "possible data poisoning - keeping the current model active"
            ),
        }
    return {"promote": True, "reason": f"golden-set accuracy {candidate_acc:.3f} (d {(-drop):+.3f})"}


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples."""
    qs = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, qs))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
