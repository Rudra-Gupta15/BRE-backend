"""/bre-products routes — per-loan-product BRE rule config (LAP/SBL, Machine,
Vehicle, MSME) and the product-level PASS/FAIL/SKIP evaluation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.aa.product_rules import PRODUCT_NAMES, PRODUCT_RULE_IDS, product_catalogue
from app.common.source_rules import DATA_SOURCE_RULES
from app.common.rule_text import PRODUCT_DESCRIPTIONS, describe_rule
from app.aa.product_engine import evaluate_product_rules
from app.aa.scoring import (
    apply_rule_engine,
    compute_real_credit_score,
    compute_real_feature_vector,
)
from app.aa.product_state import bre_product_state
from app.aa.product_source_state import product_source_rule_state
from app.common.state.rule_catalog import rule_catalog_state
from app.common.state.source_rules import data_source_rule_state
from app.common.state.session import session_state

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
            {**r, "description": r["description"] or describe_rule(r["label"])}
            for r in rule_catalog_state.rules_for(source_id)
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


class RuleEditBody(BaseModel):
    label: str | None = None
    threshold: str | None = None
    description: str | None = None


@router.put("/{product_id}/sources/{source_id}/rules/{rule_id}")
async def edit_source_rule(product_id: str, source_id: str, rule_id: str, body: RuleEditBody):
    """Rename a rule, change its threshold, or edit its explanation."""
    if product_id not in PRODUCT_RULE_IDS:
        raise HTTPException(404, f"Unknown product '{product_id}'.")
    label = body.label.strip() if body.label is not None else None
    if label == "":
        raise HTTPException(422, "Rule name can't be empty.")
    ok = rule_catalog_state.edit_rule(
        source_id, rule_id,
        label=label,
        threshold=body.threshold.strip() if body.threshold is not None else None,
        description=body.description.strip() if body.description is not None else None,
    )
    if not ok:
        raise HTTPException(404, f"Unknown rule '{rule_id}' for source '{source_id}'.")
    rule = next((r for r in rule_catalog_state.rules_for(source_id) if r["id"] == rule_id), None)
    return {"rule": {**rule, "description": rule["description"] or describe_rule(rule["label"])}}


class RuleCreateBody(BaseModel):
    label: str
    threshold: str = ""
    description: str = ""


@router.post("/{product_id}/sources/{source_id}/rules")
async def add_source_rule(product_id: str, source_id: str, body: RuleCreateBody):
    """Add a brand-new custom signal/rule to this data source's catalogue."""
    if product_id not in PRODUCT_RULE_IDS:
        raise HTTPException(404, f"Unknown product '{product_id}'.")
    if source_id not in DATA_SOURCE_RULES:
        raise HTTPException(404, f"Unknown data source '{source_id}'.")
    label = body.label.strip()
    if not label:
        raise HTTPException(422, "Rule name is required.")
    rule = rule_catalog_state.add_rule(source_id, label, body.threshold.strip(), body.description.strip())
    product_source_rule_state.register_rule(source_id, rule["id"], True)
    data_source_rule_state.register_rule(source_id, rule["id"], True)
    return {
        "productId": product_id,
        "sourceId": source_id,
        "rules": [
            {**r, "description": r["description"] or describe_rule(r["label"])}
            for r in rule_catalog_state.rules_for(source_id)
        ],
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
