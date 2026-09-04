"""FastAPI router for the BBPS model — status, version registry, retrain,
and rolling back the active version. Twin of app.gst.router's model
endpoints. Ingestion itself happens in app.aa.routes.pipeline (BBPS
statements go through the same /pipeline/uploads flow as bank statements)."""

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.bbps import model
from app.common.state.session import session_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bbps", tags=["bbps"])


@router.get("/model")
async def get_model_status():
    """Is the BBPS model trained, and its metrics."""
    return model.status()


@router.get("/model/registry")
async def get_model_registry():
    """Version list for the AI Setting training-history table."""
    return model.registry_view()


@router.get("/patterns")
async def get_patterns():
    """Post-training pattern recognition for the BBPS model — behavioral
    patterns across the uploaded statement(s). Twin of /api/models/patterns
    and /api/gst/patterns; same "Detected Patterns" card."""
    from app.bbps import patterns

    return patterns.compute()


@router.get("/pattern-match")
async def pattern_match():
    """Anomaly / fraud pattern match for the BBPS file uploaded on the Model
    Testing page — trained pattern models + typology checks + deviation
    table + an overall verdict. Twin of /api/gst/pattern-match."""
    from app.bbps import patterns

    return patterns.test_patterns()


class TrainBody(BaseModel):
    algorithm: str = "gradient_boosting"


@router.post("/train")
async def train_model(body: TrainBody):
    """Train / retrain the 4 BBPS heads + the Fraud/Anomaly Pattern models —
    the Model Hub "Start Training" button posts here directly for
    sourceId == "bbps_utility" (no shared /models/train dispatcher). Old
    model versions are kept."""
    from datetime import datetime, timezone

    from app.bbps import service

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(service.train_for_hub, body.algorithm))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result["trainedAt"] = datetime.now(timezone.utc).isoformat()
    return result


@router.get("/evaluation/summary")
async def evaluation_summary():
    """Model Evaluation panel rows for BBPS — the 4 heads + the Fraud/
    Anomaly Pattern models, real cross-validated. Twin of
    /api/models/evaluation/summary and /api/gst/evaluation/summary."""
    from app.bbps import service

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, service.evaluation_rows)


@router.get("/evaluation")
async def get_evaluation(model_id: str):
    """Real cross-validation detail for one BBPS model id (a head or a
    Fraud/Anomaly Pattern model) — the "shown below" per-fold table."""
    from app.bbps import service

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
    """Retrain/re-CV one BBPS model id — the "Re-evaluate" button."""
    from app.bbps import service

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(service.rerun, model_id))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if result is None:
        raise HTTPException(404, f"Unknown BBPS model '{model_id}'.")
    return {"modelId": model_id, "evaluation": result}


class ActiveVersionBody(BaseModel):
    version: int


@router.put("/model/active")
async def set_active_version(body: ActiveVersionBody):
    """Roll the active BBPS model to a specific trained version."""
    if not model.set_active_version(body.version):
        raise HTTPException(404, f"No BBPS model version {body.version}.")
    return model.registry_view()


@router.post("/model/{head_id}/deploy")
async def toggle_head_deploy(head_id: str):
    """Deploy / revoke one BBPS head (Risk / Discipline / Behaviour / Stability)."""
    current = model.deploy_state().get(head_id)
    if current is None:
        raise HTTPException(404, f"Unknown BBPS model '{head_id}'.")
    try:
        st = model.set_deployed(head_id, not current)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return {"deployed": st}


def _score_current_statement() -> dict:
    """Score the BBPS file(s) uploaded on the Model Testing page
    (scope='testing'). Shared by /score-testing, /bre-evaluate and
    /record-test."""
    from app.bbps import analysis, rules, schema

    stmt = session_state.merged_statement_for("bbps_utility", "testing")
    if not (stmt and stmt.get("transactions")):
        return {"available": False,
                "message": "Upload a BBPS file on this page first — utility-bill "
                           "payments are scored from real transactions."}

    bbps_result = analysis.analyze_bbps(stmt["transactions"])
    if not bbps_result.get("available"):
        return {"available": False,
                "message": bbps_result.get("message", "No BBPS / utility bill payments found in this statement.")}

    rule_result = rules.evaluate_bbps_rules(bbps_result)
    fv = schema.feature_vector_from_analysis(bbps_result)
    prediction = model.predict(fv) if fv is not None else {"available": False}

    return {
        "available": True,
        "analysis": bbps_result,
        "rules": rule_result,
        "prediction": prediction,
        "modelVersion": prediction.get("modelVersion"),
        "deployed": model.deploy_state(),
    }


@router.get("/score-testing")
async def score_testing():
    """Score the BBPS file uploaded on the Model Testing page (scope='testing').
    Twin of app.gst.router.score_testing."""
    return _score_current_statement()


@router.post("/bre-evaluate")
async def bre_evaluate():
    """Evaluate the computable bbps_utility BRE rules against the BBPS file
    uploaded on the Model Testing page — the "BRE payload" tab. Twin of
    app.gst.router.bre_evaluate."""
    from app.bbps import rules

    result = _score_current_statement()
    if not result.get("available"):
        return {"available": False,
                "message": result.get("message", "Upload a BBPS file on this page first — BRE rules run on real data.")}

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
    """Save the current BBPS test to Model Testing history — ONE row per
    upload, all 4 heads folded in. Twin of app.gst.router.record_test."""
    result = _score_current_statement()
    if not result.get("available"):
        return {"recorded": False, "message": result.get("message")}
    from app.common.persistence import record_bbps_test_history
    record_bbps_test_history(result, "bbps_utility", body.fileName, body.customId)
    return {"recorded": True}
