# Persistent, growing training dataset.
#
# Every training CSV that's uploaded is appended here (deduped by account_id),
# so each new upload ADDS to the corpus rather than replacing it. Models are
# always retrained on the full accumulated file — that's how "old + new data
# combine into a better model" actually works for tabular ML.

import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).resolve().parents[1] / "data" / "training"
DATASET_FILE = TRAINING_DIR / "accumulated.csv"
_KEY = "account_id"

# Columns dropped before training: identifiers, free dates, and features with
# no variance in this schema.
_DROP = {
    "account_id", "counterparty_id", "statement_start_date", "statement_end_date",
    "transaction_date", "transaction_narration", "bank_name", "mcc_code",
    "emi_coverage_ratio", "aa_data_completeness",
}
_CATEGORICAL = ["account_type", "account_status", "transaction_type", "counterparty_type", "payment_mode"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def ingest_csv(raw: bytes) -> dict:
    """Append the rows of an uploaded CSV to the accumulated dataset."""
    return ingest_frame(_normalise_columns(pd.read_csv(io.BytesIO(raw))))


def ingest_frame(new: "pd.DataFrame") -> dict:
    """Append the rows of an already-parsed (and ideally already-validated)
    DataFrame to the accumulated dataset, deduped by account_id."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    new = _normalise_columns(new.copy())
    if _KEY not in new.columns:
        new[_KEY] = [f"row_{i}" for i in range(len(new))]

    if DATASET_FILE.exists():
        existing = _normalise_columns(pd.read_csv(DATASET_FILE))
        before = len(existing)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=[_KEY], keep="last").reset_index(drop=True)
        added = len(combined) - before
    else:
        combined = new.drop_duplicates(subset=[_KEY], keep="last").reset_index(drop=True)
        added = len(combined)

    combined.to_csv(DATASET_FILE, index=False)
    logger.info("Dataset ingest: +%d rows, total %d.", added, len(combined))
    return {
        "added": int(added),
        "skippedDuplicates": int(len(new) - added),
        "total": int(len(combined)),
        "columns": int(combined.shape[1]),
    }


def load() -> pd.DataFrame | None:
    if not DATASET_FILE.exists():
        return None
    return _normalise_columns(pd.read_csv(DATASET_FILE))


def stats() -> dict:
    df = load()
    if df is None:
        return {"total": 0, "columns": 0, "banks": [], "file": None}
    return {
        "total": int(len(df)),
        "columns": int(df.shape[1]),
        "banks": sorted(df["bank_name"].dropna().unique().tolist()) if "bank_name" in df else [],
        "file": str(DATASET_FILE),
    }


def feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Returns (X, numeric_feature_names, onehot_feature_names) ready for the model."""
    keep_num = [
        c for c in df.columns
        if c not in _DROP and c not in _CATEGORICAL and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[keep_num].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    onehot_cols: list[str] = []
    for c in _CATEGORICAL:
        if c in df.columns:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c)
            onehot_cols.extend(dummies.columns.tolist())
            X = pd.concat([X, dummies.astype(float)], axis=1)

    return X, keep_num, onehot_cols


def reset() -> None:
    try:
        DATASET_FILE.unlink(missing_ok=True)
    except OSError:
        pass
