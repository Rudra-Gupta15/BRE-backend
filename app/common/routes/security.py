"""ML-security surface: overview, drift monitoring, audit log, lineage lookups."""
import asyncio

from fastapi import APIRouter, HTTPException

from app.aa import registry as model_registry
from app.common.persistence import (
    all_feature_vectors,
    recent_dataset_batches,
    recent_security_events,
    save_drift_snapshot,
    security_overview,
)
from app.common.security import drift, outliers

router = APIRouter(prefix="/security", tags=["security"])


@router.get("/overview")
async def overview():
    snap = security_overview()
    if snap is None:
        return {
            "available": False,
            "message": "PostgreSQL is off - security telemetry needs the database.",
        }
    return {"available": True, **snap}


@router.get("/drift")
async def get_drift():
    """Latest drift computation (from bre.drift_snapshots via the overview)."""
    snap = security_overview()
    return {"drift": (snap or {}).get("drift")}


@router.post("/drift/compute")
async def compute_drift():
    """Compare the most-recent scored applicants to the earliest ones."""
    rows = all_feature_vectors(limit=5000, order="asc")
    if len(rows) < 40:
        raise HTTPException(400, f"Need at least 40 scored statements to measure drift (have {len(rows)}).")

    half = max(20, len(rows) // 2)
    reference = [r["fv"] for r in rows[:half]]
    recent = [r["fv"] for r in rows[-half:]]
    ref_scores = [r["score"] for r in rows[:half] if isinstance(r["score"], (int, float))]
    rec_scores = [r["score"] for r in rows[-half:] if isinstance(r["score"], (int, float))]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: drift.compute(reference, recent, ref_scores, rec_scores)
    )
    active = model_registry.active_meta()
    snap_id = save_drift_snapshot(result, active["version"] if active else None)
    return {"snapshotId": snap_id, **result}


@router.post("/outliers/rebuild-profile")
async def rebuild_outlier_profile():
    """Recompute the outlier baseline from every stored feature vector."""
    rows = all_feature_vectors(limit=5000, order="asc")
    profile = outliers.rebuild_profile([r["fv"] for r in rows])
    if profile is None:
        raise HTTPException(400, f"Need at least 30 scored statements (have {len(rows)}).")
    return {"profile": profile}


@router.get("/events")
async def events(limit: int = 50):
    return {"events": recent_security_events(limit)}


@router.get("/batches")
async def batches(limit: int = 25):
    return {"batches": recent_dataset_batches(limit)}


@router.get("/models/integrity")
async def model_integrity():
    """Hash/signature status of every registry artifact."""
    out = []
    for e in model_registry.list_versions():
        out.append({
            "version": e["version"],
            "algorithm": e.get("algorithm"),
            "hasHash": bool(e.get("sha256")),
            "hasSignature": bool(e.get("signature")),
            "goldenAccuracy": e.get("goldenAccuracy"),
            "active": e.get("active", False),
            "promotionNote": e.get("promotionNote"),
            "trainedFromBatches": e.get("batches") or [],
        })
    return {"models": out}
