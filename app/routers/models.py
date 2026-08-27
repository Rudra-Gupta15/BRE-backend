import asyncio
from datetime import datetime, timezone
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.data.model_catalog import ML_ALGORITHMS, VERSION_OPTIONS
from app.services.ml_trainer import train_models_live
from app.services.persistence import save_model_run
from app.state.models_state import known_model_ids, models_state
from app.state.session_state import session_state

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/algorithms")
async def list_algorithms():
    return {"algorithms": ML_ALGORITHMS, "versionOptions": VERSION_OPTIONS}


class TrainBody(BaseModel):
    algorithm: str = "gradient_boosting"
    datasetFile: str = "processed_features_vector.csv"


@router.post("/train")
async def train_models_handler(body: TrainBody):
    if not any(a["value"] == body.algorithm for a in ML_ALGORITHMS):
        raise HTTPException(400, f"Unknown algorithm '{body.algorithm}'.")

    # Run real sklearn training in a thread pool — it's CPU-bound and would
    # otherwise block FastAPI's async event loop for several seconds.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            train_models_live,
            body.algorithm,
            dict(session_state.parsed_statements),
        ),
    )

    trained_at = datetime.now(timezone.utc).isoformat()

    # Persist trained sklearn objects for inference
    models_state.trained_sklearn_map = result["trainedMap"]
    models_state.real_features       = result["realFeatures"]
    models_state.trained_models      = result["models"]
    models_state.evaluation_cache    = result.get("evaluations", {})

    save_model_run({**result, "datasetFile": body.datasetFile})
    models_state.last_training_run   = {
        "algorithm":   body.algorithm,
        "datasetFile": body.datasetFile,
        "trainedAt":   trained_at,
        "txCount":     result["txCount"],
        "realData":    True,
    }

    return {
        "models":       result["models"],
        "algorithm":    body.algorithm,
        "trainedAt":    trained_at,
        "realFeatures": result["realFeatures"],
        "txCount":      result["txCount"],
    }


@router.get("")
async def list_models():
    return {
        "trainedModels": models_state.trained_models,
        "selectedVersionMap": models_state.selected_version_map,
        "deployedStatusMap": models_state.deployed_status_map,
    }


class VersionBody(BaseModel):
    version: str


@router.put("/{model_id}/version")
async def set_model_version(model_id: str, body: VersionBody):
    if model_id not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{model_id}'.")
    if not any(v["value"] == body.version for v in VERSION_OPTIONS):
        raise HTTPException(400, f"Unknown version '{body.version}'.")

    models_state.selected_version_map[model_id] = body.version
    return {"selectedVersionMap": models_state.selected_version_map}


@router.post("/{model_id}/deploy")
async def toggle_deploy(model_id: str):
    if model_id not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{model_id}'.")

    current = models_state.deployed_status_map.get(model_id, "Ready")
    models_state.deployed_status_map[model_id] = "Ready" if current == "Deployed" else "Deployed"
    return {"deployedStatusMap": models_state.deployed_status_map}
