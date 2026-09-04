"""High-level BBPS helpers used outside the package (the Model Hub pipeline).

`train_for_hub` / `head_evaluations` / `evaluation_rows` / `rerun` are what
app.aa.routes.models (the generic /models/* endpoints) calls into for
sourceId == "bbps_utility" — BBPS's own training/evaluation shaping lives
HERE, not in the generic route, so the route stays a thin dispatcher. Twin of
app.gst.service.
"""
from __future__ import annotations

import logging

from app.bbps import model

logger = logging.getLogger(__name__)


def train_cards(b: dict) -> list[dict]:
    """One Model-Hub card per trained BBPS head (4 models)."""
    from datetime import datetime

    from app.aa.catalog import ML_ALGORITHMS

    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    algo_label = next(
        (a["label"] for a in ML_ALGORITHMS if a["value"] == b.get("algorithm")),
        "Gradient Boosting",
    )
    return [
        {
            "id": h["id"], "name": h["name"], "desc": h["desc"],
            "accuracy": h["accuracyLabel"], "algorithm": algo_label, "createdDate": created,
            "cvFolds": 3, "sampleCount": b["nSamples"], "features": h["nFeatures"],
            "realData": True, "kind": "bbps",
            "metrics": h["metrics"], "metricLine": h["metricLine"], "target": h["target"],
            "modelKind": h["kind"], "version": b["version"], "classes": h.get("classes"),
        }
        for h in b.get("models", [])
    ]


def train_for_hub(algorithm: str) -> dict:
    """Train the BBPS model + save its pattern-fraud baseline — the full
    /models/train response body for sourceId == "bbps_utility" (minus
    `trainedAt`, which the generic route stamps). Raises ValueError on
    failure (route maps that to 400)."""
    b = model.train(algorithm)
    cards = train_cards(b)
    try:
        from app.bbps import patterns
        patterns.save_training_baseline()
    except Exception:  # noqa: BLE001
        logger.warning("BBPS pattern-baseline save failed", exc_info=True)
    return {
        "models": cards, "algorithm": algorithm,
        "realFeatures": None, "bbpsRegistry": model.registry_view(),
        "txCount": b["nSamples"],
    }


def head_evaluations() -> dict:
    """{head_id: {evalMetrics, cvFolds, name}} for the active BBPS bundle +
    the BBPS Fraud/Anomaly Pattern models, or {}."""
    try:
        from app.bbps.patterns import pattern_evaluations
        return {**model.head_evaluations(), **pattern_evaluations()}
    except Exception:  # noqa: BLE001 — BBPS is optional; never break the caller
        logger.exception("BBPS head evaluations unavailable")
        return {}


def evaluation_rows() -> dict:
    """Rows for the Model Evaluation summary panel — BBPS heads + the Fraud/
    Anomaly Pattern models, lazily backfilling per-fold CV / training a
    pattern model that hasn't run yet this process lifetime. Returns
    {"models": [...], "algorithm": str | None, "trainedAt": str | None}."""
    from app.bbps.patterns import pattern_evaluations, train_pattern_models

    evals = head_evaluations()
    if not evals:
        try:
            if model.is_trained():
                evals = model.reevaluate()
        except Exception:  # noqa: BLE001
            logger.exception("BBPS eval backfill failed")

    try:
        pat = pattern_evaluations()
        if not pat and (evals or model.is_trained()):
            pat = train_pattern_models()
        evals = {**evals, **pat}
    except Exception:  # noqa: BLE001
        logger.exception("BBPS pattern evaluations unavailable")

    rows = []
    for hid, ev in evals.items():
        em = ev.get("evalMetrics", {})
        meta = em.get("metricMeta", {})
        rows.append({
            "modelId": hid,
            "name": ev.get("name", hid),
            "kind": "pattern" if hid.startswith("bbps_pattern_") else "bbps",
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
            logger.exception("BBPS eval context unavailable")
    return {"models": rows, "algorithm": ctx.get("algorithm"), "trainedAt": ctx.get("trainedAt")}


def rerun(model_id: str) -> dict | None:
    """Retrain/re-CV one BBPS model id — a Fraud/Anomaly Pattern model or a
    head. Returns {"evalMetrics", "cvFolds"}, or None if `model_id` doesn't
    belong to BBPS at all. Raises ValueError if it IS a BBPS id but nothing
    is trained yet (route maps that to 400)."""
    from app.bbps.model import HEAD_IDS
    from app.bbps.patterns import PATTERN_MODEL_IDS, train_pattern_models

    if model_id in PATTERN_MODEL_IDS:
        pat = train_pattern_models()
        entry = pat.get(model_id)
        if not entry:
            raise ValueError("Train the BBPS model first.")
        return {"evalMetrics": entry["evalMetrics"], "cvFolds": entry["cvFolds"]}

    if model_id in HEAD_IDS:
        try:
            evals = model.reevaluate()
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(str(exc)) from exc
        bbps = evals.get(model_id)
        if not bbps:
            raise ValueError("Train the BBPS model first (BBPS data source).")
        return {"evalMetrics": bbps["evalMetrics"], "cvFolds": bbps["cvFolds"]}

    return None
