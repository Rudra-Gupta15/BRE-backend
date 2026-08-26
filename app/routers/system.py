import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.state.models_state import models_state
from app.state.session_state import session_state

router = APIRouter(tags=["system"])

_start_time = time.time()


@router.get("/health")
async def health_check():
    return {"status": "ok", "uptime": time.time() - _start_time, "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/reset")
async def reset_all():
    """Mirrors the frontend's handleReset(): clears source selection,
    uploads, pipeline output, trained models, and version/deployment maps."""
    session_state.reset()
    models_state.reset()
    return {"reset": True}
