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
    settings,
    system,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # connects to PostgreSQL if DATABASE_URL is set; no-op otherwise
    yield


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
app.include_router(system.router, prefix="/api")
