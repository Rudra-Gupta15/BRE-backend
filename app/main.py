from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGIN
from app.db import init_db
from app.routers import (
    ai_architecture,
    auth,
    dashboard,
    data_sources,
    inference,
    models,
    pipeline,
    security,
    settings,
    system,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # connects to PostgreSQL if DATABASE_URL is set; no-op otherwise
    _seed_reference_rows()
    yield


def _seed_reference_rows() -> None:
    """Make sure the deployment table + population-model registry are mirrored
    into the DB from the moment the server starts (before any training run)."""
    try:
        from app.routers.models import _persist_deployment_state
        from app.services import model_registry
        from app.services.persistence import all_feature_vectors, sync_model_versions
        from app.services.security import outliers

        model_registry.backfill_hashes()
        _persist_deployment_state()
        sync_model_versions(model_registry.list_versions())
        # Warm the outlier baseline from whatever scored history exists.
        rows = all_feature_vectors(limit=5000, order="asc")
        if rows:
            outliers.rebuild_profile([r["fv"] for r in rows])
    except Exception:  # noqa: BLE001 — never block startup on this
        import logging
        logging.getLogger(__name__).exception("reference-row seed failed")


app = FastAPI(title="SFL BRE Portal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(data_sources.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
app.include_router(ai_architecture.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(security.router, prefix="/api")
app.include_router(system.router, prefix="/api")
