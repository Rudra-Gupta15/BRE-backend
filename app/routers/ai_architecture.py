import json
import time
import urllib.error
import urllib.request

from fastapi import APIRouter
from pydantic import BaseModel

from app import config
from app.state.ai_architecture_state import ai_architecture_state

router = APIRouter(prefix="/ai-architecture", tags=["ai-architecture"])

LLM_OPTIONS = [
    {"value": "gemma", "label": "Gemma 4 (Ollama)"},
    {"value": "qwen", "label": "Qwen 2.5 VL (Ollama)"},
    {"value": "llama", "label": "Llama 3.2 Vision (Ollama)"},
    {"value": "mistral", "label": "Mistral (Ollama)"},
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


@router.put("/vllm")
async def update_vllm_config(body: VllmConfigBody):
    v = ai_architecture_state.vllm
    if body.enabled is not None:
        v["enabled"] = body.enabled
    if body.endpoint and body.endpoint.strip():
        v["endpoint"] = body.endpoint.strip()
    if body.modelName and body.modelName.strip():
        v["modelName"] = body.modelName.strip()
    return {"vllm": v}


@router.post("/vllm/test")
async def test_vllm_connection():
    """Real connectivity check — hits Ollama's /api/tags and reports whether the
    configured model is available, with the actual round-trip latency."""
    v = ai_architecture_state.vllm
    if not v["enabled"]:
        v["status"] = "disconnected"
        return {"vllm": v, "latencyMs": None, "detail": "Engine disabled."}

    host = (v["endpoint"] or config.OLLAMA_HOST).rstrip("/")
    model = v["modelName"] or config.STATEMENT_LLM_MODEL
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=8) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        latency_ms = round((time.perf_counter() - t0) * 1000)
        names = {m.get("name", "") for m in tags.get("models", [])}
        base = model.split(":")[0]
        available = model in names or any(n.split(":")[0] == base for n in names)
        v["status"] = "connected" if available else "model-missing"
        detail = (
            f"Ollama reachable ({len(names)} models). '{model}' "
            + ("available." if available else "NOT pulled — run `ollama pull " + model + "`.")
        )
        return {"vllm": v, "latencyMs": latency_ms, "detail": detail}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        v["status"] = "disconnected"
        return {"vllm": v, "latencyMs": None, "detail": f"Cannot reach Ollama at {host} ({exc})."}
