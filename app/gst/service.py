"""High-level GST helpers used outside the package (the Model Hub pipeline).

`ingest_gst_file` parses one uploaded GST file, scores every row with the model,
optionally appends the rows to the training corpus, and returns a summary in the
shape the Model Hub upload card expects.
"""
from __future__ import annotations

import logging

import pandas as pd

from app.gst import model, parser
from app.gst.schema import SCORE_TARGET

logger = logging.getLogger(__name__)


def ingest_gst_file(buf: bytes, file_name: str, *, add_to_corpus: bool = True) -> dict:
    """Returns:
        {
          "records":       int,
          "predictions":   { available, count, avgUnderwritingScore, riskCounts, ... },
          "completeness":  0-100  (how many canonical fields were present, mean),
          "corpusRows":    int | None,
          "rows":          [ parsed record dicts ],
          "warnings":      [str],
        }
    Raises ValueError if the file cannot be parsed.
    """
    rows = parser.parse_gst(buf, file_name)
    if not rows:
        raise ValueError("No GST rows found in the file.")

    warnings: list[str] = []
    corpus_rows = None
    if add_to_corpus:
        try:
            corpus_rows = model.append_to_corpus(pd.DataFrame(rows))
        except Exception as exc:  # noqa: BLE001 — never fail ingest on corpus write
            logger.warning("Could not append GST rows to corpus: %s", exc)
            warnings.append("Rows scored but not added to the training corpus.")

    predictions = model.predict_many(
        [{k: v for k, v in r.items() if k != SCORE_TARGET} for r in rows]
    )
    if not predictions.get("available"):
        warnings.append(predictions.get("message", "GST model unavailable."))

    # completeness = average share of the ~50 canonical fields actually filled
    from app.gst.schema import CANONICAL
    filled = [
        sum(1 for f in CANONICAL if str(r.get(f, "")).strip() not in ("", "nan"))
        for r in rows
    ]
    completeness = round(100 * (sum(filled) / len(filled)) / len(CANONICAL)) if filled else 0

    return {
        "records": len(rows),
        "predictions": predictions,
        "completeness": completeness,
        "corpusRows": corpus_rows,
        "rows": rows,
        "warnings": warnings,
    }
