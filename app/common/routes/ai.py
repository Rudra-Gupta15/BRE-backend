"""/ai-architecture routes — LLM / vision-model config for the statement parser.

Lists locally-installed Ollama models, tests connectivity, and lets the user
pick the vision model used for scanned-PDF extraction.
"""

import json
import time
import urllib.error
import urllib.request

from fastapi import APIRouter
from pydantic import BaseModel

from app.common import config
from app.common.state.ai_architecture import ai_architecture_state

router = APIRouter(prefix="/ai-architecture", tags=["ai-architecture"])

LLM_OPTIONS = [
    {"value": "gemma", "label": "Gemma 4 (Ollama)"},
    {"value": "qwen", "label": "Qwen 2.5 VL (Ollama)"},
    {"value": "llama", "label": "Llama 3.2 Vision (Ollama)"},
    {"value": "mistral", "label": "Mistral (Ollama)"},
]

# Model families / name fragments that indicate a vision-capable model.
_VISION_FAMILIES = {
    "clip", "mllama", "qwen2vl", "qwen2.5vl", "gemma3", "llava", "minicpmv",
    "llama4", "siglip", "pixtral", "internvl", "moondream", "phi3v",
}
_VISION_NAME_HINTS = ("vl", "vision", "llava", "bakllava", "moondream", "minicpm-v", "pixtral", "-v")


def _looks_vision(name: str, m: dict) -> bool:
    low = name.lower()
    if any(h in low for h in _VISION_NAME_HINTS):
        return True
    fams = {f.lower() for f in (m.get("details") or {}).get("families") or []}
    return bool(fams & _VISION_FAMILIES)


def _show_is_vision(host: str, name: str) -> bool:
    """Authoritative check on newer Ollama — /api/show returns `capabilities`."""
    try:
        req = urllib.request.Request(
            f"{host}/api/show",
            data=json.dumps({"model": name}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False
    if "vision" in [c.lower() for c in data.get("capabilities") or []]:
        return True
    fams = {f.lower() for f in (data.get("details") or {}).get("families") or []}
    return bool(fams & _VISION_FAMILIES)


@router.get("/local-models")
async def list_local_models():
    """Scan the machine's Ollama install and return the vision-capable models."""
    host = (ai_architecture_state.vllm.get("endpoint") or config.OLLAMA_HOST).rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=8) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"available": False, "host": host, "models": [],
                "detail": f"Cannot reach Ollama at {host} — is it running? ({exc})"}

    installed = tags.get("models", [])
    vision = []
    for m in installed:
        name = m.get("name") or m.get("model") or ""
        if not name:
            continue
        if not (_looks_vision(name, m) or _show_is_vision(host, name)):
            continue
        size = m.get("size") or 0
        details = m.get("details") or {}
        vision.append({
            "value": name,
            "label": name,
            "family": details.get("family") or "",
            "params": details.get("parameter_size"),
            "sizeGB": round(size / 1e9, 1) if size else None,
        })
    return {
        "available": True, "host": host,
        "models": vision,
        "installedCount": len(installed),
        "detail": (
            f"{len(vision)} vision model(s) of {len(installed)} installed at {host}."
            if vision else
            f"No vision models among {len(installed)} installed at {host} — "
            "try `ollama pull qwen2.5vl` or `ollama pull llama3.2-vision`."
        ),
    }


@router.get("")
async def get_ai_architecture():
    return {
        "selectedLLM": ai_architecture_state.selected_llm,
        "activeVisionModel": config.active_vision_model(),
        "dataExtracted": ai_architecture_state.data_extracted,
        "cleanlinessPercent": ai_architecture_state.cleanliness_percent,
        "vllm": ai_architecture_state.vllm,
        "llmOptions": LLM_OPTIONS,
    }


class SelectedLLMBody(BaseModel):
    selectedLLM: str | None = None


@router.put("/llm")
async def set_selected_llm(body: SelectedLLMBody):
    val = (body.selectedLLM or "").strip()
    if val:
        ai_architecture_state.selected_llm = val
        # A real Ollama model tag (has ':' or '/') → use it for PDF parsing.
        config.set_vision_model(val if (":" in val or "/" in val) else None)
    return {
        "selectedLLM": ai_architecture_state.selected_llm,
        "activeVisionModel": config.active_vision_model(),
    }


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
