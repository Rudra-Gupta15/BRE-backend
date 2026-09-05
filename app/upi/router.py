"""FastAPI router for the UPI model — status, version registry, retrain,
Model Testing scoring, and rolling back the active version. Twin of
app.gst.router / app.bbps.router. Corpus growth from an uploaded file happens
via app.aa.routes.pipeline -> app.upi.service.ingest_upi_file (the Model Hub
"Start Process" upload flow, same as GST/BBPS)."""

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.upi import model
from app.common.state.session import session_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upi", tags=["upi"])


@router.get("/model")
async def get_model_status():
    """Is the UPI model trained, and its metrics."""
    return model.status()


@router.get("/model/registry")
async def get_model_registry():
    """Version list for the AI Setting training-history table."""
    return model.registry_view()


@router.get("/patterns")
async def get_patterns():
    """Post-training pattern recognition for the UPI model — behavioral
    patterns across the uploaded file(s). Twin of /api/models/patterns,
    /api/gst/patterns, /api/bbps/patterns; same "Detected Patterns" card."""
    from app.upi import patterns

    return patterns.compute()


@router.get("/pattern-match")
async def pattern_match():
    """Anomaly / fraud pattern match for the UPI file uploaded on the Model
    Testing page — trained pattern models + typology checks + deviation
    table + an overall verdict. Twin of /api/gst/pattern-match, /api/bbps/pattern-match."""
    from app.upi import patterns

    return patterns.test_patterns()


class TrainBody(BaseModel):
    algorithm: str = "gradient_boosting"


@router.post("/train")
async def train_model(body: TrainBody):
    """Train / retrain the 4 UPI heads + the Fraud/Anomaly Pattern models —
    the Model Hub "Start Training" button posts here directly for
    sourceId == "upi_enrichment" (no shared /models/train dispatcher). Old
    model versions are kept."""
    from datetime import datetime, timezone

    from app.upi import service

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(service.train_for_hub, body.algorithm))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result["trainedAt"] = datetime.now(timezone.utc).isoformat()
    return result


@router.get("/evaluation/summary")
async def evaluation_summary():
    """Model Evaluation panel rows for UPI — the 4 heads + the Fraud/
    Anomaly Pattern models, real cross-validated. Twin of
    /api/models/evaluation/summary, /api/gst/evaluation/summary, /api/bbps/evaluation/summary."""
    from app.upi import service

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, service.evaluation_rows)


@router.get("/evaluation")
async def get_evaluation(model_id: str):
    """Real cross-validation detail for one UPI model id (a head or a
    Fraud/Anomaly Pattern model) — the "shown below" per-fold table."""
    from app.upi import service

    ev = service.head_evaluations().get(model_id)
    trained_at = None
    if ev:
        try:
            trained_at = model.eval_context().get("trainedAt")
        except Exception:  # noqa: BLE001
            pass
    return {
        "modelId": model_id,
        "evaluation": {"evalMetrics": ev["evalMetrics"], "cvFolds": ev["cvFolds"]} if ev else None,
        "trainedAt": trained_at,
    }


@router.post("/evaluation/{model_id}/re-run")
async def rerun_evaluation(model_id: str):
    """Retrain/re-CV one UPI model id — the "Re-evaluate" button."""
    from app.upi import service

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(service.rerun, model_id))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if result is None:
        raise HTTPException(404, f"Unknown UPI model '{model_id}'.")
    return {"modelId": model_id, "evaluation": result}


class ActiveVersionBody(BaseModel):
    version: int


@router.put("/model/active")
async def set_active_version(body: ActiveVersionBody):
    """Roll the active UPI model to a specific trained version."""
    if not model.set_active_version(body.version):
        raise HTTPException(404, f"No UPI model version {body.version}.")
    return model.registry_view()


@router.post("/model/{head_id}/deploy")
async def toggle_head_deploy(head_id: str):
    """Deploy / revoke one UPI head (Risk / Reliability / Behaviour / Stability)."""
    current = model.deploy_state().get(head_id)
    if current is None:
        raise HTTPException(404, f"Unknown UPI model '{head_id}'.")
    try:
        st = model.set_deployed(head_id, not current)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return {"deployed": st}


def _uploaded_upi_transactions(scope: str = "testing") -> list[dict]:
    """UPI's real payload lives in each statement's own `upi.rawTransactions`
    side-channel, not the generic `transactions` field — session_state's
    merged_statement_for() only merges `transactions`, so it can't be used
    here (same reason app.gst.router reads statements_for() directly instead
    of the merged helper). Concatenates every uploaded file's transactions."""
    out: list[dict] = []
    for s in session_state.statements_for("upi_enrichment", scope):
        out.extend(((s or {}).get("upi") or {}).get("rawTransactions") or [])
    return out


def _score_current_statement() -> dict:
    """Score the UPI file(s) uploaded on the Model Testing page
    (scope='testing'). Shared by /score-testing, /bre-evaluate and
    /record-test."""
    from app.upi import analysis, rules, schema

    txns = _uploaded_upi_transactions("testing")
    if not txns:
        return {"available": False,
                "message": "Upload a UPI file on this page first — transactions are scored from real data."}

    upi_result = analysis.analyze_upi(txns)
    if not upi_result.get("available"):
        return {"available": False,
                "message": upi_result.get("message", "No UPI transactions found in this file.")}

    rule_result = rules.evaluate_upi_rules(upi_result)
    fv = schema.feature_vector_from_analysis(upi_result)
    prediction = model.predict(fv) if fv is not None else {"available": False}

    return {
        "available": True,
        "analysis": upi_result,
        "rules": rule_result,
        "prediction": prediction,
        "modelVersion": prediction.get("modelVersion"),
        "deployed": model.deploy_state(),
    }


@router.get("/score-testing")
async def score_testing():
    """Score the UPI file uploaded on the Model Testing page (scope='testing').
    Twin of app.gst.router.score_testing / app.bbps.router.score_testing."""
    return _score_current_statement()


@router.post("/bre-evaluate")
async def bre_evaluate():
    """Evaluate the computable upi_enrichment BRE rules against the UPI file
    uploaded on the Model Testing page — the "BRE payload" tab."""
    from app.upi import rules

    result = _score_current_statement()
    if not result.get("available"):
        return {"available": False,
                "message": result.get("message", "Upload a UPI file on this page first — BRE rules run on real data.")}

    heads = (result["prediction"] or {}).get("headScores") or {}
    return {
        "available": True,
        "payload": rules.payload(result["analysis"], heads, result["prediction"]),
        "evaluation": rules.evaluate(result["analysis"], heads),
    }


class RecordTestBody(BaseModel):
    customId: str | None = None
    fileName: str | None = None


@router.post("/record-test")
async def record_test(body: RecordTestBody):
    """Save the current UPI test to Model Testing history — ONE row per
    upload, all 4 heads folded in. Twin of app.gst.router.record_test / app.bbps.router.record_test."""
    result = _score_current_statement()
    if not result.get("available"):
        return {"recorded": False, "message": result.get("message")}
    from app.common.persistence import record_upi_test_history
    record_upi_test_history(result, "upi_enrichment", body.fileName, body.customId)
    return {"recorded": True}
