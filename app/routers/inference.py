from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.data.model_catalog import MODEL_TEMPLATES
from app.services.inference_engine import (
    apply_rule_engine,
    compute_credit_score,
    compute_real_credit_score,
    compute_real_feature_vector,
    generate_analytics,
    generate_anomalies,
    generate_bre_payload,
    generate_evaluation,
    generate_transactions,
    map_real_transactions,
)
from app.state.models_state import known_model_ids, models_state
from app.state.session_state import session_state

router = APIRouter(prefix="/inference", tags=["inference"])


def _resolve_model(model_id: str) -> dict:
    from_trained = next((m for m in models_state.trained_models if m["id"] == model_id), None)
    from_template = next((m for m in MODEL_TEMPLATES if m["id"] == model_id), None)
    return from_trained or from_template or {"id": model_id, "name": "Risk Model"}


def _build_bundle(model_id: str, custom_id: str, bank_name: str, source_id: str | None) -> dict:
    """Uses the real parsed statement for this source when one was uploaded
    and actually contained transactions; otherwise falls back to the
    deterministic customId-seeded simulation."""
    model = _resolve_model(model_id)
    version = models_state.selected_version_map.get(model_id, "v3.4")

    real_statement = session_state.parsed_statements.get(source_id) if source_id else None
    has_real_data = bool(real_statement and real_statement["transactions"])

    transactions = map_real_transactions(real_statement) if has_real_data else generate_transactions(custom_id)
    anomalies = generate_anomalies(custom_id, transactions)
    evaluation = generate_evaluation(custom_id, model_id)

    real_feature_vector = compute_real_feature_vector(real_statement) if has_real_data else None
    risk_score = compute_real_credit_score(real_feature_vector) if has_real_data else compute_credit_score(custom_id)
    risk_score = apply_rule_engine(risk_score)

    # Always pass the (possibly rule-gated) risk_score through, for both real
    # and simulated data — otherwise the analytics chart / BRE payload would
    # recompute their own ungated score and disagree with the Credit Score tab.
    analytics = generate_analytics(
        custom_id, model_id, version,
        real_risk_score=risk_score,
        real_feature_vector=real_feature_vector,
    )

    bre_payload = generate_bre_payload(
        custom_id, bank_name, model["name"], version, transactions,
        real_feature_vector=real_feature_vector,
        real_risk_score=risk_score,
    )

    return {
        "model": {"id": model["id"], "name": model["name"], "version": version},
        "customId": custom_id,
        "bankName": bank_name or None,
        "dataSource": "UPLOADED_STATEMENT" if has_real_data else "SIMULATED",
        "transactions": transactions,
        "anomalies": anomalies,
        "analytics": analytics,
        "evaluation": evaluation,
        "riskScore": risk_score,
        "brePayload": bre_payload,
    }


@router.get("/deployed-models")
async def list_deployed_models():
    source = models_state.trained_models if models_state.trained_models else MODEL_TEMPLATES
    deployed = [m for m in source if models_state.deployed_status_map.get(m["id"]) == "Deployed"]
    return {
        "deployedModels": deployed,
        "selectedVersionMap": models_state.selected_version_map,
        "deployedStatusMap": models_state.deployed_status_map,
    }


class RunInferenceBody(BaseModel):
    modelId: str = "risk_model"
    customId: str = "cust_demo_medium_1"
    bankName: str = ""
    sourceId: str = ""


@router.post("/run")
async def run_inference(body: RunInferenceBody):
    if body.modelId not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{body.modelId}'.")
    if not body.customId or not body.customId.strip():
        raise HTTPException(400, "customId is required.")

    bundle = _build_bundle(body.modelId, body.customId.strip(), body.bankName.strip(), body.sourceId or None)

    session_state.inference_history.insert(0, {
        "id": bundle["customId"],
        "bank": bundle["bankName"] or "Not Specified",
        "date": datetime.now(timezone.utc).isoformat(),
        "txCount": len(bundle["transactions"]),
        "riskScore": bundle["riskScore"]["score"],
        "grade": bundle["riskScore"]["grade"],
        "status": "ANALYZED",
    })
    session_state.inference_history = session_state.inference_history[:25]

    return bundle


class ReEvaluateBody(BaseModel):
    customId: str = "cust_demo_medium_1"


@router.post("/evaluate/{model_id}")
async def re_evaluate(model_id: str, body: ReEvaluateBody):
    if model_id not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{model_id}'.")
    return generate_evaluation(f"{body.customId}:{datetime.now().timestamp()}", model_id)


@router.get("/history")
async def get_history():
    return {"history": session_state.inference_history}
