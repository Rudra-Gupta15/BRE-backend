import random

from fastapi import APIRouter
from pydantic import BaseModel

from app.state.ai_architecture_state import ai_architecture_state

router = APIRouter(prefix="/ai-architecture", tags=["ai-architecture"])

LLM_OPTIONS = [
    {"value": "gemma", "label": "Gemma 2 (vLLM Engine)"},
    {"value": "qwen", "label": "Qwen 2.5 (vLLM Engine)"},
    {"value": "llama", "label": "Llama 3.1 8B (vLLM Engine)"},
    {"value": "mistral", "label": "Mistral NeMo (vLLM Engine)"},
]


@router.get("")
async def get_ai_architecture():
    return {
        "selectedLLM": ai_architecture_state.selected_llm,
        "dataExtracted": ai_architecture_state.data_extracted,
        "cleanlinessPercent": ai_architecture_state.cleanliness_percent,
        "vllm": ai_architecture_state.vllm,
        "llmOptions": LLM_OPTIONS,
    }


class SelectedLLMBody(BaseModel):
    selectedLLM: str | None = None


@router.put("/llm")
async def set_selected_llm(body: SelectedLLMBody):
    if body.selectedLLM and any(o["value"] == body.selectedLLM for o in LLM_OPTIONS):
        ai_architecture_state.selected_llm = body.selectedLLM
    return {"selectedLLM": ai_architecture_state.selected_llm}


@router.post("/extract")
async def extract_data():
    # Simulates the LLM extraction pass; a real implementation would enqueue
    # a document-parsing job here.
    ai_architecture_state.data_extracted = True
    return {"dataExtracted": True, "engine": ai_architecture_state.selected_llm}


class CleanlinessBody(BaseModel):
    cleanlinessPercent: int | None = None


@router.put("/cleanliness")
async def set_cleanliness(body: CleanlinessBody):
    if body.cleanlinessPercent in (60, 70, 80):
        ai_architecture_state.cleanliness_percent = body.cleanlinessPercent
    return {
        "cleanlinessPercent": ai_architecture_state.cleanliness_percent,
        "llmActive": ai_architecture_state.cleanliness_percent < 60,
    }


class VllmConfigBody(BaseModel):
    enabled: bool | None = None
    endpoint: str | None = None
    modelName: str | None = None
    gpuCount: str | None = None
    gpuMemoryUtil: str | None = None


@router.put("/vllm")
async def update_vllm_config(body: VllmConfigBody):
    v = ai_architecture_state.vllm
    if body.enabled is not None:
        v["enabled"] = body.enabled
    if body.endpoint and body.endpoint.strip():
        v["endpoint"] = body.endpoint.strip()
    if body.modelName and body.modelName.strip():
        v["modelName"] = body.modelName.strip()
    if body.gpuCount is not None:
        v["gpuCount"] = body.gpuCount
    if body.gpuMemoryUtil is not None:
        v["gpuMemoryUtil"] = body.gpuMemoryUtil
    return {"vllm": v}


@router.post("/vllm/test")
async def test_vllm_connection():
    v = ai_architecture_state.vllm
    if not v["enabled"]:
        v["status"] = "disconnected"
        return {"vllm": v, "latencyMs": None}
    v["status"] = "connected"
    latency_ms = 8 + random.randint(0, 12)
    return {"vllm": v, "latencyMs": latency_ms}
