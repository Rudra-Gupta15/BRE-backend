"""/inference routes — Model Testing: score one applicant (real statement or
simulated), run the enabled BRE rules, and read/replay test history.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.aa.catalog import MODEL_TEMPLATES
from app.aa.rules import evaluate_bre_rules
from app.common.persistence import (
    get_test_history_bundle,
    list_test_history,
    log_security_event,
    record_test_history,
    save_bre_evaluation,
    save_inference_run,
)
from app.common.security import guardrails, outliers
from app.aa.scoring import (
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
from app.aa.model_state import known_model_ids, models_state
from app.common.state.session import session_state
from app.aa.settings_state import settings_state

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

    # Inference always runs against the Model Testing page's own upload, never
    # Model Hub's — the two are deliberately isolated.
    real_statement = session_state.merged_statement_for(source_id, "testing") if source_id else None
    has_real_data = bool(real_statement and real_statement["transactions"])

    stmt_summary = real_statement.get("summary", {}) if has_real_data else {}
    _first_file = (session_state.files_for(source_id, "testing")[:1] or [{}])[0] if source_id else {}
    file_name = _first_file.get("fileName")
    # Prefer what the user typed; else what was read off the statement; else the file.
    effective_bank = (bank_name or "").strip() or stmt_summary.get("bankName") or ""
    account_holder = stmt_summary.get("accountHolder")
    effective_id = custom_id
    if custom_id in ("applicant", ""):
        effective_id = account_holder or (file_name.rsplit(".", 1)[0] if file_name else custom_id)

    transactions = map_real_transactions(real_statement) if has_real_data else generate_transactions(custom_id)
    evaluation = generate_evaluation(custom_id, model_id)

    if has_real_data:
        opening_balance = (real_statement.get("summary") or {}).get("openingBalance")
        anomalies = generate_anomalies(
            custom_id, real_statement["transactions"], real=True, opening_balance=opening_balance,
        )
    else:
        anomalies = generate_anomalies(custom_id, transactions)

    real_feature_vector = compute_real_feature_vector(real_statement) if has_real_data else None

    # ── Guardrail: keep the 11 underwriting features in valid ranges ─────
    security: dict = {"guardrailStatus": "ok", "guardrailWarnings": [], "outlier": {}}
    if real_feature_vector is not None:
        real_feature_vector, gr_warn = guardrails.clamp_feature_vector(real_feature_vector)
        if gr_warn:
            security["guardrailStatus"] = "warn"
            security["guardrailWarnings"] = gr_warn
            log_security_event("guardrail", "warn", f"inference:{source_id}",
                               {"feature_warnings": gr_warn})
        # ── Outlier detection: is this vector within the scored population? ──
        security["outlier"] = outliers.score(real_feature_vector)
        if security["outlier"].get("isOutlier"):
            log_security_event("outlier", "warn", f"inference:{source_id}",
                               {"flags": security["outlier"]["flags"]})

    risk_score = (
        compute_real_credit_score(real_feature_vector, real_statement)
        if has_real_data else compute_credit_score(custom_id)
    )
    risk_score = apply_rule_engine(risk_score)
    risk_score["score"] = guardrails.bound_score(risk_score.get("score"))

    # Always pass the (possibly rule-gated) risk_score through, for both real
    # and simulated data — otherwise the analytics chart / BRE payload would
    # recompute their own ungated score and disagree with the Credit Score tab.
    analytics = generate_analytics(
        custom_id, model_id, version,
        real_risk_score=risk_score,
        real_feature_vector=real_feature_vector,
        real_statement=real_statement if has_real_data else None,
    )

    bre_payload = generate_bre_payload(
        effective_id, effective_bank, model["name"], version, transactions,
        real_feature_vector=real_feature_vector,
        real_risk_score=risk_score,
    )

    return {
        "model": {"id": model["id"], "name": model["name"], "version": version},
        "customId": custom_id,
        "bankName": effective_bank or None,
        "accountHolder": account_holder,
        "fileName": file_name,
        "statementLabel": effective_id if effective_id != "applicant" else None,
        "dataSource": "UPLOADED_STATEMENT" if has_real_data else "SIMULATED",
        "transactions": transactions,
        "anomalies": anomalies,
        "analytics": analytics,
        "evaluation": evaluation,
        "riskScore": risk_score,
        "brePayload": bre_payload,
        "security": security,
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
    customId: str = "applicant"
    bankName: str = ""
    sourceId: str = ""
    # True only for a deliberate run (file upload / Run Analysis click) — those
    # get appended to the persistent test_history. The incidental re-fires
    # (model switch, typing in the ref-id box) leave it False.
    record: bool = False


@router.post("/run")
async def run_inference(body: RunInferenceBody):
    if body.modelId not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{body.modelId}'.")
    if not body.customId or not body.customId.strip():
        raise HTTPException(400, "customId is required.")

    bundle = _build_bundle(body.modelId, body.customId.strip(), body.bankName.strip(), body.sourceId or None)
    save_inference_run(bundle, body.sourceId or None)

    if body.record:
        _first = (session_state.files_for(body.sourceId, "testing")[:1] or [{}])[0] if body.sourceId else {}
        record_test_history(bundle, body.sourceId or None, _first.get("fileName"), body.customId.strip())

    # Identifies the *person / statement* — a new applicant on the same feed is
    # a NEW entry; re-running the same one just refreshes it.
    person_key = (
        bundle.get("accountHolder")
        or bundle.get("fileName")
        or bundle.get("statementLabel")
        or f"{body.sourceId}:{body.customId}"
    )
    entry = {
        "id": bundle.get("statementLabel") or bundle.get("accountHolder") or bundle["customId"],
        "bank": bundle["bankName"] or "Not Specified",
        "personKey": person_key,
        "modelId": body.modelId,
        "date": datetime.now(timezone.utc).isoformat(),
        "txCount": len(bundle["transactions"]),
        "riskScore": bundle["riskScore"]["score"],
        "grade": bundle["riskScore"].get("gradeRaw") or bundle["riskScore"]["grade"],
        "decision": bundle["riskScore"].get("decision"),
        "status": "ANALYZED",
    }
    # One entry per person (across models) — drop any prior entry for the same
    # person so re-runs don't pile up, but keep every distinct applicant.
    session_state.inference_history = [
        h for h in session_state.inference_history if h.get("personKey") != person_key
    ]
    session_state.inference_history.insert(0, entry)
    session_state.inference_history = session_state.inference_history[:50]
    session_state.persist()

    return bundle


class ReEvaluateBody(BaseModel):
    customId: str = "applicant"


@router.post("/evaluate/{model_id}")
async def re_evaluate(model_id: str, body: ReEvaluateBody):
    if model_id not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{model_id}'.")
    return generate_evaluation(f"{body.customId}:{datetime.now().timestamp()}", model_id)


class BreRulesBody(BaseModel):
    customId: str = "applicant"
    sourceId: str = ""


@router.post("/bre-rules")
async def run_bre_rules(body: BreRulesBody):
    """Evaluates every BRE rule currently enabled on the Settings page against
    the applicant's real uploaded statement + derived feature vector + credit
    score, returning PASS / FAIL / SKIP per rule and an overall decision."""
    real_statement = session_state.merged_statement_for(body.sourceId, "testing") if body.sourceId else None
    if not (real_statement and real_statement.get("transactions")):
        return {
            "available": False,
            "message": "Upload a bank statement first — BRE rules run against real transaction data.",
        }

    fv = compute_real_feature_vector(real_statement)
    risk = apply_rule_engine(compute_real_credit_score(fv, real_statement))
    opening_balance = (real_statement.get("summary") or {}).get("openingBalance")

    result = evaluate_bre_rules(
        fv, risk, real_statement["transactions"], opening_balance, settings_state.enabled_rules,
    )
    result["available"] = True
    save_bre_evaluation(result, body.sourceId or None)
    return result


class PatternMatchBody(BaseModel):
    customId: str = "applicant"
    sourceId: str = ""


@router.post("/patterns")
async def pattern_match(body: PatternMatchBody):
    """Anomaly / fraud pattern match — runs the trained Fraud/Anomaly Pattern
    models + rule checks on this applicant, compares to the training baseline,
    and returns an overall verdict. Model Testing "Pattern Match" tab."""
    from app.aa import patterns

    return patterns.test_patterns(body.sourceId or None, body.customId.strip() or "applicant")


@router.get("/history")
async def get_history():
    """Test history — one row per tested application, newest first. Reads the
    persistent test_history table when the DB is on; falls back to the in-memory
    session log otherwise."""
    db_history = list_test_history()
    return {"history": db_history if db_history else session_state.inference_history}


@router.get("/history/{row_id}")
async def get_history_entry(row_id: int):
    """The stored analysis for one history row — reopens that application's output."""
    bundle = get_test_history_bundle(row_id)
    if bundle is None:
        raise HTTPException(404, "No stored result for that history entry.")
    return bundle
