"""/data-source-rules routes — the per-data-source BRE rule catalogue and its
enable/disable state (the "Data Source Signals" popup on each source card).
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.common.source_rules import rules_for
from app.common.rule_text import describe_rule
from app.common.state.source_rules import data_source_rule_state

router = APIRouter(prefix="/data-source-rules", tags=["data-source-rules"])


@router.get("/{source_id}")
async def get_rules(source_id: str):
    """The rule catalogue + enabled state for one data source. Unknown / rule-less
    sources return an empty list (the popup shows its empty state)."""
    rules = [{**r, "description": describe_rule(r["label"])} for r in rules_for(source_id)]
    return {
        "id": source_id,
        "rules": rules,
        "enabled": data_source_rule_state.for_source(source_id),
    }


class RulesBody(BaseModel):
    enabled: dict[str, bool] | None = None
    setAll: bool | None = None
    reset: bool | None = None


@router.put("/{source_id}")
async def set_rules(source_id: str, body: RulesBody):
    if body.reset:
        data_source_rule_state.reset_source(source_id)
    elif body.setAll is not None:
        data_source_rule_state.set_all(source_id, body.setAll)
    elif body.enabled is not None:
        data_source_rule_state.set_rules(source_id, body.enabled)
    return {"id": source_id, "enabled": data_source_rule_state.for_source(source_id)}
