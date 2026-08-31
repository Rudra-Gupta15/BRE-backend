"""High-level GST helpers used outside the package (the Model Hub pipeline).

`ingest_gst_file` accepts either:
  * GST return files — GSTR-1 / GSTR-3B / GSTR-2A / GSTR-2B, one file with many
    (gstin, period) rows — which are rolled up per business into the model's
    feature profile, or
  * a flat "one row per business" GST summary (the training-set shape).

It parses, scores every business, and returns a Model-Hub-friendly summary.
"""
from __future__ import annotations

import logging

import pandas as pd

from app.gst import aggregate, model, parser, present, returns
from app.gst.schema import CANONICAL, SCORE_TARGET

logger = logging.getLogger(__name__)


def _completeness(records: list[dict]) -> int:
    if not records:
        return 0
    filled = [
        sum(1 for f in CANONICAL if str(r.get(f, "")).strip() not in ("", "nan", "NA"))
        for r in records
    ]
    return round(100 * (sum(filled) / len(filled)) / len(CANONICAL))


def ingest_gst_file(buf: bytes, file_name: str, *, add_to_corpus: bool = True) -> dict:
    """Returns a summary dict:
        { mode, records, businesses, predictions, completeness, corpusRows,
          rows, profiles, returnsSeen, warnings }
    Raises ValueError if the file can't be read as GST data at all.
    """
    warnings: list[str] = []

    # ── 1. GST return files (GSTR-1/3B/2A/2B) ─────────────────────────────
    return_rows = returns.parse_returns_file(buf, file_name)
    if return_rows:
        profiles = aggregate.build_profiles(return_rows)
        profiles = [p for p in profiles if p]
        model_input = [{k: v for k, v in p.items() if k != "_meta"} for p in profiles]
        predictions = model.predict_many(model_input)
        if not predictions.get("available"):
            warnings.append(predictions.get("message", "GST model unavailable."))

        seen = {t: 0 for t in returns.RETURN_TYPES}
        for r in return_rows:
            seen[r["return_type"]] = seen.get(r["return_type"], 0) + 1

        corpus_rows = None
        if add_to_corpus and model_input:
            try:
                corpus_rows = model.append_to_corpus(pd.DataFrame(model_input))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not append aggregated GST profiles to corpus: %s", exc)
                warnings.append("Profiles scored but not added to the training corpus.")

        pred_list = predictions.get("predictions", [])
        detail = [
            present.business_view(mi, pr, meta=p.get("_meta", {}))
            for mi, pr, p in zip(model_input, pred_list, profiles)
        ][:10]

        return {
            "mode": "returns",
            "records": len(return_rows),
            "businesses": len(profiles),
            "predictions": predictions,
            "completeness": _completeness(model_input),
            "corpusRows": corpus_rows,
            "rows": return_rows[:50],
            "profiles": profiles,
            "detail": detail,
            "returnsSeen": seen,
            "warnings": warnings,
        }

    # ── 2. flat one-row-per-business summary ──────────────────────────────
    rows = parser.parse_gst(buf, file_name)
    if not rows:
        raise ValueError("Not a GST returns file and not a GST summary — "
                         "expected GSTR-1/3B/2A/2B rows or a per-business summary.")

    corpus_rows = None
    if add_to_corpus:
        try:
            corpus_rows = model.append_to_corpus(pd.DataFrame(rows))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not append GST rows to corpus: %s", exc)
            warnings.append("Rows scored but not added to the training corpus.")

    model_input = [{k: v for k, v in r.items() if k != SCORE_TARGET} for r in rows]
    predictions = model.predict_many(model_input)
    if not predictions.get("available"):
        warnings.append(predictions.get("message", "GST model unavailable."))

    pred_list = predictions.get("predictions", [])
    detail = [
        present.business_view(mi, pr)
        for mi, pr in zip(model_input, pred_list)
    ][:10]

    return {
        "mode": "summary",
        "records": len(rows),
        "businesses": len(rows),
        "predictions": predictions,
        "completeness": _completeness(rows),
        "corpusRows": corpus_rows,
        "rows": rows[:50],
        "profiles": [],
        "detail": detail,
        "returnsSeen": {},
        "warnings": warnings,
    }
