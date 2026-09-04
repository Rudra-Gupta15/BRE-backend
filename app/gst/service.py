"""High-level GST helpers used outside the package (the Model Hub pipeline).

`ingest_gst_file` accepts either:
  * GST return files — GSTR-1 / GSTR-3B / GSTR-2A / GSTR-2B, one file with many
    (gstin, period) rows — which are rolled up per business into the model's
    feature profile, or
  * a flat "one row per business" GST summary (the training-set shape).

It parses, scores every business, and returns a Model-Hub-friendly summary.

`train_for_hub` / `head_evaluations` / `evaluation_rows` / `rerun` are what
app.aa.routes.models (the generic /models/* endpoints) calls into for
sourceId == "gst_data" — GST's own training/evaluation shaping lives HERE,
not in the generic route, so the route stays a thin dispatcher.
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


# ── Model Hub training / evaluation (used by app.aa.routes.models) ─────────

def train_cards(g: dict) -> list[dict]:
    """One Model-Hub card per trained GST head (4 models)."""
    from datetime import datetime

    from app.aa.catalog import ML_ALGORITHMS

    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    algo_label = next(
        (a["label"] for a in ML_ALGORITHMS if a["value"] == g.get("algorithm")),
        "Gradient Boosting",
    )
    return [
        {
            "id": h["id"], "name": h["name"], "desc": h["desc"],
            "accuracy": h["accuracyLabel"], "algorithm": algo_label, "createdDate": created,
            "cvFolds": 3, "sampleCount": g["nSamples"], "features": h["nFeatures"],
            "realData": True, "kind": "gst",
            "metrics": h["metrics"], "metricLine": h["metricLine"], "target": h["target"],
            "modelKind": h["kind"], "version": g["version"], "classes": h.get("classes"),
        }
        for h in g.get("models", [])
    ]


def train_for_hub(algorithm: str) -> dict:
    """Train the GST model + save its pattern-fraud baseline — the full
    /models/train response body for sourceId == "gst_data" (minus
    `trainedAt`, which the generic route stamps). Raises ValueError /
    FileNotFoundError / ImportError on failure (route maps those to 400)."""
    g = model.train(algorithm)
    cards = train_cards(g)
    try:
        from app.gst import patterns
        patterns.save_training_baseline()
    except Exception:  # noqa: BLE001
        logger.warning("GST pattern-baseline save failed", exc_info=True)
    return {
        "models": cards, "algorithm": algorithm,
        "realFeatures": None, "gstFeatureSummary": g.get("featureSummary"),
        "gstRegistry": model.registry_view(),
        "txCount": g["nSamples"],
    }


def head_evaluations() -> dict:
    """{head_id: {evalMetrics, cvFolds, name}} for the active GST bundle + the
    GST Fraud/Anomaly Pattern models, or {}."""
    try:
        from app.gst.patterns import pattern_evaluations
        return {**model.head_evaluations(), **pattern_evaluations()}
    except Exception:  # noqa: BLE001 — GST is optional; never break the caller
        logger.exception("GST head evaluations unavailable")
        return {}


def evaluation_rows() -> dict:
    """Rows for the Model Evaluation summary panel — GST heads + the Fraud/
    Anomaly Pattern models, lazily backfilling per-fold CV / training a
    pattern model that hasn't run yet this process lifetime. Returns
    {"models": [...], "algorithm": str | None, "trainedAt": str | None}."""
    from app.gst.patterns import pattern_evaluations, train_pattern_models

    evals = head_evaluations()
    if not evals:
        try:
            if model.is_trained():
                evals = model.reevaluate()
        except Exception:  # noqa: BLE001
            logger.exception("GST eval backfill failed")

    try:
        pat = pattern_evaluations()
        if not pat and (evals or model.is_trained()):
            pat = train_pattern_models()
        evals = {**evals, **pat}
    except Exception:  # noqa: BLE001
        logger.exception("GST pattern evaluations unavailable")

    rows = []
    for hid, ev in evals.items():
        em = ev.get("evalMetrics", {})
        meta = em.get("metricMeta", {})
        rows.append({
            "modelId": hid,
            "name": ev.get("name", hid),
            "kind": "pattern" if hid.startswith("gst_pattern_") else "gst",
            "metricLabel": meta.get("r2Score", {}).get("name", "Score"),
            "metricValue": em.get("r2Score"),
            "precision": em.get("precision"),
            "recall": em.get("recall"),
            "f1": em.get("f1Score"),
            "folds": len(ev.get("cvFolds", [])),
        })

    ctx = {}
    if rows:
        try:
            ctx = model.eval_context()
        except Exception:  # noqa: BLE001
            logger.exception("GST eval context unavailable")
    return {"models": rows, "algorithm": ctx.get("algorithm"), "trainedAt": ctx.get("trainedAt")}


def rerun(model_id: str) -> dict | None:
    """Retrain/re-CV one GST model id — a Fraud/Anomaly Pattern model or a
    head. Returns {"evalMetrics", "cvFolds"}, or None if `model_id` doesn't
    belong to GST at all. Raises ValueError if it IS a GST id but nothing is
    trained yet (route maps that to 400)."""
    from app.gst.model import HEAD_IDS
    from app.gst.patterns import PATTERN_MODEL_IDS, train_pattern_models

    if model_id in PATTERN_MODEL_IDS:
        pat = train_pattern_models()
        entry = pat.get(model_id)
        if not entry:
            raise ValueError("Train the GST model first.")
        return {"evalMetrics": entry["evalMetrics"], "cvFolds": entry["cvFolds"]}

    if model_id in HEAD_IDS:
        try:
            evals = model.reevaluate()
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(str(exc)) from exc
        gst = evals.get(model_id)
        if not gst:
            raise ValueError("Train the GST model first (GST data source).")
        return {"evalMetrics": gst["evalMetrics"], "cvFolds": gst["cvFolds"]}

    return None
