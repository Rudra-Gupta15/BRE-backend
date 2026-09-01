"""/health and /reset — liveness + the session wipe the frontend fires on every
browser reload.
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.aa.model_state import models_state
from app.common.state.session import session_state

router = APIRouter(tags=["system"])

_start_time = time.time()


@router.get("/health")
async def health_check():
    from app.common import db

    return {
        "status": "ok",
        "uptime": time.time() - _start_time,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {
            "connected": db.DB_ENABLED,
            "url": db.engine.url.render_as_string(hide_password=True) if db.engine else None,
        },
        "analyzedThisSession": len(session_state.inference_history),
    }


@router.post("/reset")
async def reset_all():
    """Mirrors the frontend's handleReset(): clears source selection,
    uploads, pipeline output, trained models, and version/deployment maps."""
    session_state.reset()
    models_state.reset()
    return {"reset": True}
