"""FastAPI application entry point.

Builds the app, wires CORS, and mounts every router under /api:

    app.aa.router        -> /pipeline, /models, /inference, /bre-products, /settings
    app.gst.api_router   -> /gst/*
    app.bbps.api_router  -> /bbps/*
    app.common.routes.*  -> /auth, /reset, /health, /dashboard, /security,
                            /data-sources, /data-source-rules, /ai-architecture

On startup it connects to PostgreSQL if DATABASE_URL is set (otherwise runs
in-memory) and seeds the deployment table + population-model registry.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.aa.router import router as aa_router
from app.common.config import CORS_ORIGIN
from app.common.db import init_db
from app.common.routes import (
    ai,
    auth,
    dashboard,
    external_db,
    security,
    source_rules,
    sources,
    system,
)
from app.bbps import api_router as bbps_api_router
from app.gst import api_router as gst_api_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # connects to PostgreSQL if DATABASE_URL is set; no-op otherwise
    _seed_reference_rows()
    yield


def _seed_reference_rows() -> None:
    """Make sure the deployment table + population-model registry are mirrored
    into the DB from the moment the server starts (before any training run)."""
    try:
        from app.aa import registry as model_registry
        from app.aa.routes.models import _persist_deployment_state
        from app.common.persistence import all_feature_vectors, sync_model_versions
        from app.common.security import outliers

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
app.include_router(sources.router, prefix="/api")
app.include_router(aa_router, prefix="/api")
app.include_router(gst_api_router, prefix="/api")
app.include_router(bbps_api_router, prefix="/api")
app.include_router(source_rules.router, prefix="/api")
app.include_router(external_db.router, prefix="/api")
app.include_router(ai.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(security.router, prefix="/api")
app.include_router(system.router, prefix="/api")
