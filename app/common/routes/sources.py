"""/data-sources routes — the 11-source catalogue, quick-start presets, the
user's current selection, and per-source publish status.
"""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.common.sources import PRESET_CONFIGS
from app.common.state.published import published_state
from app.common.state.session import session_state

router = APIRouter(prefix="/data-sources", tags=["data-sources"])


@router.get("")
async def list_data_sources():
    return {"dataSources": session_state.all_data_sources()}


@router.get("/presets")
async def list_presets():
    return {"presets": PRESET_CONFIGS}


@router.get("/selection")
async def get_selection():
    # "Published" sources survive /reset and restarts — rehydrate the live
    # session selection from them so the pipeline sees the same set.
    valid_ids = {s["id"] for s in session_state.all_data_sources()}
    ids = [i for i in published_state.published_ids if i in valid_ids]
    session_state.selected_ids = ids
    return {"selectedIds": ids}


class SelectionBody(BaseModel):
    selectedIds: list[str]


@router.put("/selection")
async def set_selection(body: SelectionBody):
    valid_ids = {s["id"] for s in session_state.all_data_sources()}
    ids = [i for i in body.selectedIds if i in valid_ids]
    session_state.selected_ids = ids
    session_state.persist()
    published_state.set_published_ids(ids)
    return {"selectedIds": ids}


@router.get("/status")
async def get_statuses():
    valid_ids = {s["id"] for s in session_state.all_data_sources()}
    return {
        "statuses": {sid: published_state.status_of(sid) for sid in valid_ids},
    }


class StatusBody(BaseModel):
    statuses: dict[str, str]


@router.put("/status")
async def set_statuses(body: StatusBody):
    valid_ids = {s["id"] for s in session_state.all_data_sources()}
    clean = {k: v for k, v in body.statuses.items() if k in valid_ids}
    published_state.set_many(clean)
    session_state.selected_ids = [
        i for i in published_state.published_ids if i in valid_ids
    ]
    return {"statuses": {sid: published_state.status_of(sid) for sid in valid_ids}}


class AddSourceBody(BaseModel):
    title: str | None = None
    desc: str | None = None
    fields: str | list[str] | None = None


@router.post("", status_code=201)
async def add_data_source(body: AddSourceBody):
    if not body.title or not body.title.strip():
        raise HTTPException(400, "Data source title is required.")

    source_id = f"custom_feed_{int(time.time() * 1000)}"
    if isinstance(body.fields, list):
        field_list = body.fields
    elif isinstance(body.fields, str) and body.fields.strip():
        field_list = [f.strip() for f in body.fields.split(",")]
    else:
        field_list = ["custom_metric_1", "custom_ratio_2"]

    desc = (body.desc or "").strip()
    new_source = {
        "id": source_id,
        "title": body.title.strip(),
        "category": "Custom Feed",
        "icon": "Sparkles",
        "badge": "Custom Integration",
        "shortDesc": desc or "Custom financial data integration feed.",
        "fullDesc": desc or "Custom data feed integrated for risk model scoring.",
        "coverage": "N/A",
        "featuresCount": len(field_list),
        "features": field_list,
        "sampleSchema": {"record_id": "REC-9910"},
    }

    session_state.custom_sources.append(new_source)
    session_state.persist()
    return {"dataSource": new_source}


@router.get("/{source_id}")
async def get_data_source(source_id: str):
    source = next((s for s in session_state.all_data_sources() if s["id"] == source_id), None)
    if not source:
        raise HTTPException(404, f"Data source '{source_id}' not found.")
    return {"dataSource": source}
