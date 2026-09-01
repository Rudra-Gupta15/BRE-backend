"""/settings routes — the credit scorecard weights + decision thresholds and the
underwriting BRE rule on/off map (the "BRE Signals" settings page).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.aa.rule_catalog import UNDERWRITING_RULE_CATEGORIES
from app.aa.settings_state import build_default_rule_map, settings_state

router = APIRouter(prefix="/settings", tags=["settings"])


def _rule_id_set() -> set[str]:
    ids = set()
    for category in UNDERWRITING_RULE_CATEGORIES:
        for rule in category["rules"]:
            ids.add(rule["id"])
    return ids


@router.get("/rules")
async def get_rules():
    return {"categories": UNDERWRITING_RULE_CATEGORIES, "enabledRules": settings_state.enabled_rules}


@router.put("/rules/{rule_id}/toggle")
async def toggle_rule(rule_id: str):
    if rule_id not in _rule_id_set():
        raise HTTPException(404, f"Unknown rule '{rule_id}'.")
    settings_state.enabled_rules[rule_id] = not settings_state.enabled_rules.get(rule_id, False)
    settings_state.persist()
    return {"ruleId": rule_id, "enabled": settings_state.enabled_rules[rule_id]}


@router.get("/scoring")
async def get_scoring():
    """The credit-score scorecard weights + decision thresholds (editable)."""
    return {"scoring": settings_state.scoring.to_dict()}


@router.put("/scoring")
async def update_scoring(patch: dict):
    changed = settings_state.scoring.apply(patch)
    settings_state.persist()
    return {"scoring": settings_state.scoring.to_dict(), "changed": changed}


@router.post("/scoring/reset")
async def reset_scoring():
    from app.aa.scoring_config import ScoringConfig

    settings_state.scoring = ScoringConfig()
    settings_state.persist()
    return {"scoring": settings_state.scoring.to_dict()}


class SetAllBody(BaseModel):
    enabled: bool


@router.post("/rules/set-all")
async def set_all_rules(body: SetAllBody):
    for rule_id in _rule_id_set():
        settings_state.enabled_rules[rule_id] = body.enabled
    settings_state.persist()
    return {"enabledRules": settings_state.enabled_rules}


@router.post("/rules/reset")
async def reset_rules():
    settings_state.enabled_rules = build_default_rule_map()
    settings_state.general = {"defaultLLM": "gemma", "threshold": "60", "environment": "production"}
    settings_state.persist()
    return {"enabledRules": settings_state.enabled_rules, "general": settings_state.general}


@router.post("/save")
async def save_settings():
    settings_state.saved_at = datetime.now(timezone.utc).isoformat()
    settings_state.persist()
    return {"savedAt": settings_state.saved_at}


@router.get("/general")
async def get_general_settings():
    return {"general": settings_state.general}


class GeneralSettingsBody(BaseModel):
    defaultLLM: str | None = None
    threshold: str | None = None
    environment: str | None = None


@router.put("/general")
async def update_general_settings(body: GeneralSettingsBody):
    if body.defaultLLM:
        settings_state.general["defaultLLM"] = body.defaultLLM
    if body.threshold:
        settings_state.general["threshold"] = body.threshold
    if body.environment:
        settings_state.general["environment"] = body.environment
    settings_state.persist()
    return {"general": settings_state.general}
