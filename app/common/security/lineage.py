"""Data lineage - provenance for every artefact that feeds a decision.

Small helpers; the actual recording happens in app/common/persistence.py, which
writes:

  statements.file_sha256 / parser_model / parser_version / parse_confidence
  inference_runs.model_version_id  (-> which exact registry artifact scored them)
  dataset_batches                  (one row per ingested training CSV)
  model_versions.trained_from_batches / artifact_sha256 / artifact_signature

so a full trace exists: file -> parser -> feature vector -> model artifact ->
training batches -> uploader.
"""
from __future__ import annotations

import hashlib

from app.common import config


def file_digest(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def parser_identity() -> dict:
    """Which parser produced an extraction - recorded on every statement."""
    return {
        "parser_model": config.STATEMENT_LLM_MODEL,
        "parser_version": config.STATEMENT_PARSER_VERSION,
        "llm_host": config.OLLAMA_HOST,
    }


def extraction_confidence(parsed: dict) -> float:
    """0-1: fraction of parsed transactions that carry all four key fields."""
    txns = parsed.get("transactions") or []
    if not txns:
        return 0.0
    complete = sum(
        1 for t in txns
        if t.get("date") and t.get("narration")
        and t.get("amount") is not None and t.get("balance") is not None
    )
    return round(complete / len(txns), 3)
