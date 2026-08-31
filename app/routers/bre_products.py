from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.data.bre_product_rules import PRODUCT_NAMES, PRODUCT_RULE_IDS, product_catalogue
from app.data.data_source_rules import rules_for
from app.data.rule_descriptions import PRODUCT_DESCRIPTIONS, describe_rule
from app.services.bre_product_engine import evaluate_product_rules
from app.services.inference_engine import (
    apply_rule_engine,
    compute_real_credit_score,
    compute_real_feature_vector,
)
from app.state.bre_product_state import bre_product_state
from app.state.product_source_rule_state import product_source_rule_state
from app.state.session_state import session_state

router = APIRouter(prefix="/bre-products", tags=["bre-products"])


@router.get("")
async def list_products():
    """Every loan product, its curated (actually-evaluable) rule catalogue, and
    the current enabled/disabled config. `activeProduct` is the one the Model
    Testing BRE tab evaluates against."""
    return {
        "products": [
            {
                "id": pid,
                "name": PRODUCT_NAMES.get(pid, pid),
                "description": PRODUCT_DESCRIPTIONS.get(pid, ""),
                "rules": [
                    {**m, "description": describe_rule(m["label"])}
                    for m in product_catalogue(pid)
                ],
                "enabled": bre_product_state.for_product(pid),
            }
            for pid in PRODUCT_RULE_IDS
        ],
        "activeProduct": bre_product_state.active_product,
    }


@router.get("/source-usage")
async def source_usage():
    """For the Data Sources page: which loan products currently have each data
    source switched on. Product toggles are independent of each other."""
    return {
        "usage": product_source_rule_state.usage_by_source(),
        "productNames": PRODUCT_NAMES,
    }


class RulesBody(BaseModel):
    enabled: dict[str, bool] | None = None
    setAll: bool | None = None
    reset: bool | None = None


@router.put("/{product_id}/rules")
async def set_rules(product_id: str, body: RulesBody):
    if product_id not in PRODUCT_RULE_IDS:
        raise HTTPException(404, f"Unknown product '{product_id}'.")
    if body.reset:
        bre_product_state.reset_product(product_id)
    elif body.setAll is not None:
        bre_product_state.set_all(product_id, body.setAll)
    elif body.enabled is not None:
        bre_product_state.set_rules(product_id, body.enabled)
    return {"id": product_id, "enabled": bre_product_state.for_product(product_id)}


@router.get("/{product_id}/sources")
async def list_product_sources(product_id: str):
    """The 11 data sources shown inside a product, each with its per-product
    rule count."""
    if product_id not in PRODUCT_RULE_IDS:
        raise HTTPException(404, f"Unknown product '{product_id}'.")
    out = []
    for s in session_state.all_data_sources():
        sid = s["id"]
        active, total = product_source_rule_state.counts_for(product_id, sid)
        out.append({
            "id": sid,
            "title": s["title"],
            "ruleCount": total,
            "active": active,
            "sourceActive": product_source_rule_state.is_source_active(product_id, sid),
        })
    return {"productId": product_id, "productName": PRODUCT_NAMES.get(product_id, product_id), "sources": out}


class SourceActiveBody(BaseModel):
    active: bool


@router.put("/{product_id}/sources/{source_id}/active")
async def set_product_source_active(product_id: str, source_id: str, body: SourceActiveBody):
    if product_id not in PRODUCT_RULE_IDS:
        raise HTTPException(404, f"Unknown product '{product_id}'.")
    product_source_rule_state.set_source_active(product_id, source_id, body.active)
    return {
        "productId": product_id,
        "sourceId": source_id,
        "sourceActive": product_source_rule_state.is_source_active(product_id, source_id),
    }


@router.get("/{product_id}/sources/{source_id}/rules")
async def get_product_source_rules(product_id: str, source_id: str):
    if product_id not in PRODUCT_RULE_IDS:
        raise HTTPException(404, f"Unknown product '{product_id}'.")
    return {
        "productId": product_id,
        "sourceId": source_id,
        "rules": [
            {**r, "description": describe_rule(r["label"])}
            for r in rules_for(source_id)
        ],
        "enabled": product_source_rule_state.for_ps(product_id, source_id),
    }


@router.put("/{product_id}/sources/{source_id}/rules")
async def set_product_source_rules(product_id: str, source_id: str, body: RulesBody):
    if product_id not in PRODUCT_RULE_IDS:
        raise HTTPException(404, f"Unknown product '{product_id}'.")
    if body.reset:
        product_source_rule_state.reset_ps(product_id, source_id)
    elif body.setAll is not None:
        product_source_rule_state.set_all(product_id, source_id, body.setAll)
    elif body.enabled is not None:
        product_source_rule_state.set_rules(product_id, source_id, body.enabled)
    return {
        "productId": product_id,
        "sourceId": source_id,
        "enabled": product_source_rule_state.for_ps(product_id, source_id),
    }


class ActiveBody(BaseModel):
    productId: str | None = None


@router.put("/active")
async def set_active(body: ActiveBody):
    bre_product_state.set_active(body.productId)
    return {"activeProduct": bre_product_state.active_product}


class EvaluateBody(BaseModel):
    sourceId: str = ""
    productId: str | None = None   # defaults to the active product


@router.post("/evaluate")
async def evaluate(body: EvaluateBody):
    """Run a product's enabled BRE rules against the Model Testing statement."""
    product_id = body.productId or bre_product_state.active_product
    if product_id not in PRODUCT_RULE_IDS:
        return {"available": False, "message": "No loan product selected — pick one in BRE Rule Training."}

    stmt = session_state.merged_statement_for(body.sourceId, "testing") if body.sourceId else None
    if not (stmt and stmt.get("transactions")):
        return {"available": False, "message": "Upload a bank statement first — BRE rules run on real data."}

    fv = compute_real_feature_vector(stmt)
    risk = apply_rule_engine(compute_real_credit_score(fv, stmt))
    opening_balance = (stmt.get("summary") or {}).get("openingBalance")

    result = evaluate_product_rules(
        product_id, bre_product_state.for_product(product_id),
        fv, risk, stmt["transactions"], opening_balance,
    )
    result["available"] = True
    return result
