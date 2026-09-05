"""High-level UPI helpers used outside the package (the Model Hub pipeline).

`ingest_upi_file` parses one UPI transaction file, scores it, and appends its
real feature row to the training corpus — the Model-Hub-facing entry point
app.aa.routes.pipeline calls for sourceId == "upi_enrichment". Twin of
app.gst.service.ingest_gst_file.

`train_for_hub` / `head_evaluations` / `evaluation_rows` / `rerun` are what
app.aa.routes.models (the generic /models/* endpoints) never touches — UPI
training/evaluation happens entirely on its own /upi/* endpoints. Twin of
app.gst.service / app.bbps.service.
"""
from __future__ import annotations

import logging

from app.upi import analysis, model, parser, rules, schema

logger = logging.getLogger(__name__)


def ingest_upi_file(buf: bytes, file_name: str, *, add_to_corpus: bool = True) -> dict:
    """Parse + score one UPI transaction file. Raises ValueError if nothing
    UPI-shaped could be read from it."""
    txns = parser.parse_upi(buf, file_name)
    if not txns:
        raise ValueError("Not a UPI transaction file — expected rows with a date, amount and DEBIT/CREDIT type.")

    result = analysis.analyze_upi(txns)
    rule_result = rules.evaluate_upi_rules(result)
    fv = schema.feature_vector_from_analysis(result)
    prediction = model.predict(fv) if fv is not None else {"available": False}

    corpus_rows = None
    if add_to_corpus and fv is not None:
        try:
            corpus_rows = model.append_to_corpus(fv)
        except Exception:  # noqa: BLE001 — corpus growth must never break an upload
            logger.warning("Could not append UPI file to training corpus.", exc_info=True)

    complete = sum(
        1 for t in txns
        if t.get("date") and t.get("amount") is not None and t.get("type")
    )
    completeness = round(max(20, min(99, (complete / len(txns)) * 100))) if txns else 0

    return {
        "transactionCount": len(txns),
        "rawTransactions": txns,
        "analysis": result,
        "rules": rule_result,
        "prediction": prediction,
        "corpusRows": corpus_rows,
        "completeness": completeness,
    }


# ── Model Hub training / evaluation (used by app.aa.routes.pipeline) ───────

def train_cards(u: dict) -> list[dict]:
    """One Model-Hub card per trained UPI head (4 models)."""
    from datetime import datetime

    from app.aa.catalog import ML_ALGORITHMS

    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    algo_label = next(
        (a["label"] for a in ML_ALGORITHMS if a["value"] == u.get("algorithm")),
        "Gradient Boosting",
    )
    return [
        {
            "id": h["id"], "name": h["name"], "desc": h["desc"],
            "accuracy": h["accuracyLabel"], "algorithm": algo_label, "createdDate": created,
            "cvFolds": 3, "sampleCount": u["nSamples"], "features": h["nFeatures"],
            "realData": True, "kind": "upi",
            "metrics": h["metrics"], "metricLine": h["metricLine"], "target": h["target"],
            "modelKind": h["kind"], "version": u["version"], "classes": h.get("classes"),
        }
        for h in u.get("models", [])
    ]


def train_for_hub(algorithm: str) -> dict:
    """Train the UPI model + save its pattern-fraud baseline — the full
    /upi/train response body (minus `trainedAt`, which the route stamps).
    Raises ValueError on failure (route maps that to 400)."""
    u = model.train(algorithm)
    cards = train_cards(u)
    try:
        from app.upi import patterns
        patterns.save_training_baseline()
    except Exception:  # noqa: BLE001
        logger.warning("UPI pattern-baseline save failed", exc_info=True)
    return {
        "models": cards, "algorithm": algorithm,
        "realFeatures": None, "upiRegistry": model.registry_view(),
        "txCount": u["nSamples"],
    }


def head_evaluations() -> dict:
    """{head_id: {evalMetrics, cvFolds, name}} for the active UPI bundle +
    the UPI Fraud/Anomaly Pattern models, or {}."""
    try:
        from app.upi.patterns import pattern_evaluations
        return {**model.head_evaluations(), **pattern_evaluations()}
    except Exception:  # noqa: BLE001 — UPI is optional; never break the caller
        logger.exception("UPI head evaluations unavailable")
        return {}


def evaluation_rows() -> dict:
    """Rows for the Model Evaluation summary panel — UPI heads + the Fraud/
    Anomaly Pattern models, lazily backfilling per-fold CV / training a
    pattern model that hasn't run yet this process lifetime. Returns
    {"models": [...], "algorithm": str | None, "trainedAt": str | None}."""
    from app.upi.patterns import pattern_evaluations, train_pattern_models

    evals = head_evaluations()
    if not evals:
        try:
            if model.is_trained():
                evals = model.reevaluate()
        except Exception:  # noqa: BLE001
            logger.exception("UPI eval backfill failed")

    try:
        pat = pattern_evaluations()
        if not pat and (evals or model.is_trained()):
            pat = train_pattern_models()
        evals = {**evals, **pat}
    except Exception:  # noqa: BLE001
        logger.exception("UPI pattern evaluations unavailable")

    rows = []
    for hid, ev in evals.items():
        em = ev.get("evalMetrics", {})
        meta = em.get("metricMeta", {})
        rows.append({
            "modelId": hid,
            "name": ev.get("name", hid),
            "kind": "pattern" if hid.startswith("upi_pattern_") else "upi",
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
            logger.exception("UPI eval context unavailable")
    return {"models": rows, "algorithm": ctx.get("algorithm"), "trainedAt": ctx.get("trainedAt")}


def rerun(model_id: str) -> dict | None:
    """Retrain/re-CV one UPI model id — a Fraud/Anomaly Pattern model or a
    head. Returns {"evalMetrics", "cvFolds"}, or None if `model_id` doesn't
    belong to UPI at all. Raises ValueError if it IS a UPI id but nothing is
    trained yet (route maps that to 400)."""
    from app.upi.model import HEAD_IDS
    from app.upi.patterns import PATTERN_MODEL_IDS, train_pattern_models

    if model_id in PATTERN_MODEL_IDS:
        pat = train_pattern_models()
        entry = pat.get(model_id)
        if not entry:
            raise ValueError("Train the UPI model first.")
        return {"evalMetrics": entry["evalMetrics"], "cvFolds": entry["cvFolds"]}

    if model_id in HEAD_IDS:
        try:
            evals = model.reevaluate()
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(str(exc)) from exc
        upi = evals.get(model_id)
        if not upi:
            raise ValueError("Train the UPI model first (UPI data source).")
        return {"evalMetrics": upi["evalMetrics"], "cvFolds": upi["cvFolds"]}

    return None
