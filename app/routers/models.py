import asyncio
from datetime import datetime, timezone
from functools import partial

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.data.model_catalog import ML_ALGORITHMS, VERSION_OPTIONS
from app.services import dataset_store, model_registry
from app.services.ml_trainer import train_models_live, train_on_dataset
from app.services.persistence import (
    log_security_event,
    save_dataset_batch,
    save_model_run,
    sync_model_deployments,
    sync_model_versions,
)
from app.services.security import lineage, poisoning
from app.state.models_state import known_model_ids, models_state
from app.state.session_state import session_state

router = APIRouter(prefix="/models", tags=["models"])

_MODEL_NAMES = {
    "risk_model": "Risk Model",
    "cashflow_model": "Cashflow Model",
    "fraud_model": "Fraud Model",
    "money_balance_model": "Money Balance Model",
}


def _persist_deployment_state() -> None:
    sync_model_deployments(
        models_state.selected_version_map,
        models_state.deployed_status_map,
        _MODEL_NAMES,
    )


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
    _persist_deployment_state()

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
    _persist_deployment_state()
    return {"selectedVersionMap": models_state.selected_version_map}


@router.post("/{model_id}/deploy")
async def toggle_deploy(model_id: str):
    if model_id not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{model_id}'.")

    current = models_state.deployed_status_map.get(model_id, "Ready")
    models_state.deployed_status_map[model_id] = "Ready" if current == "Deployed" else "Deployed"
    _persist_deployment_state()
    return {"deployedStatusMap": models_state.deployed_status_map}


# ── Dataset-based training + versioned model registry ───────────────────────

@router.get("/dataset/status")
async def dataset_status():
    return {"dataset": dataset_store.stats(), "registry": model_registry.list_versions()}


@router.post("/dataset/train")
async def train_from_dataset(
    algorithm: str = "gradient_boosting",
    file: UploadFile | None = File(default=None),
):
    """Optionally ingest a training CSV, then (re)train on the FULL accumulated
    dataset and save a new model version. Old versions are kept.

    Every uploaded CSV is validated row-by-row (data-poisoning guard), profiled
    against the existing corpus, and recorded as a dataset_batch for lineage.
    A candidate model that regresses the frozen golden set is saved but NOT
    promoted to active."""
    ingest = None
    batch_id = None
    if file is not None:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Empty file.")

        # 1. Poisoning guard — validate + quarantine bad rows before ingest.
        try:
            clean_df, report = poisoning.validate_dataframe(raw)
        except ValueError as exc:
            log_security_event("poisoning", "block", f"dataset:{file.filename}", {"error": str(exc)})
            raise HTTPException(422, f"Training CSV rejected: {exc}") from exc

        # 2. Distribution shift vs. the existing corpus.
        dist = poisoning.distribution_report(clean_df, dataset_store.load())

        # 3. Ingest only the clean rows.
        ingest = dataset_store.ingest_frame(clean_df)

        # 4. Lineage: one dataset_batch row per upload.
        batch_id = save_dataset_batch({
            "fileName": file.filename,
            "sha256": lineage.file_digest(raw),
            "uploadedBy": None,
            "rowsIn": report["rowsIn"],
            "rowsAccepted": report["rowsAccepted"],
            "rowsRejected": report["rowsRejected"],
            "rowsAdded": ingest.get("added"),
            "reasons": report["reasons"],
            "distributionCheck": dist,
            "accepted": True,
        })
        sev = "warn" if (report["rowsRejected"] or dist["status"] in ("warn", "alert")) else "info"
        log_security_event("poisoning", sev, f"dataset:{file.filename}", {
            "batchId": batch_id, "report": report, "distribution": dist,
        })

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, partial(train_on_dataset, algorithm, None, [batch_id] if batch_id else None)
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    save_model_run({
        "algorithm": algorithm,
        "datasetFile": "accumulated.csv",
        "txCount": result["nSamples"],
        "models": [{"id": "risk_model_dataset", "version": result["version"], **result["metrics"]}],
        "evaluations": {"risk_model_dataset": {"evalMetrics": result["metrics"]}},
    })
    sync_model_versions(model_registry.list_versions())
    return {"ingest": ingest, "batchId": batch_id, **result}


@router.get("/evaluation")
async def get_evaluation(model_id: str = "risk_model"):
    """Real 5-fold cross-validation results for the models trained on the Model
    Hub page (computed at training time, cached on models_state)."""
    cache = models_state.evaluation_cache or {}
    return {
        "modelId": model_id,
        "evaluation": cache.get(model_id),
        "available": list(cache.keys()),
        "trainedAt": (models_state.last_training_run or {}).get("trainedAt"),
    }


@router.get("/evaluation/summary")
async def evaluation_summary():
    """One-glance accuracy of every model — the real CV headline metric for each
    session model, plus the active population (dataset) model."""
    cache = models_state.evaluation_cache or {}
    run = models_state.last_training_run or {}

    session_models = []
    for mid, name in _MODEL_NAMES.items():
        ev = cache.get(mid)
        if not ev:
            continue
        em = ev.get("evalMetrics", {})
        meta = em.get("metricMeta", {})
        session_models.append({
            "modelId": mid,
            "name": name,
            "metricLabel": meta.get("r2Score", {}).get("name", "Score"),
            "metricValue": em.get("r2Score"),
            "precision": em.get("precision"),
            "recall": em.get("recall"),
            "f1": em.get("f1Score"),
            "folds": len(ev.get("cvFolds", [])),
        })

    dataset_model = None
    active = model_registry.active_meta()
    if active:
        m = active.get("metrics", {})
        dataset_model = {
            "version": active["version"],
            "algorithm": active["algorithm"],
            "nSamples": active["nSamples"],
            "accuracy": m.get("accuracy"),
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "scoreR2": m.get("scoreR2"),
            "trainedAt": active.get("trainedAt"),
        }

    return {
        "sessionModels": session_models,
        "sessionAlgorithm": run.get("algorithm"),
        "sessionTrainedAt": run.get("trainedAt"),
        "sessionTxCount": run.get("txCount"),
        "datasetModel": dataset_model,
    }


@router.post("/evaluation/{model_id}/re-run")
async def rerun_evaluation(model_id: str):
    if model_id not in known_model_ids():
        raise HTTPException(404, f"Unknown model '{model_id}'.")
    if not models_state.trained_sklearn_map.get(model_id):
        raise HTTPException(400, "Train a model first (Model Training Process).")
    from app.services.ml_trainer import evaluate_trained

    algo = (models_state.last_training_run or {}).get("algorithm", "gradient_boosting")
    loop = asyncio.get_event_loop()
    # Rebuild the CV dataset from the stored real features and re-evaluate.
    from app.services.ml_trainer import _build_dataset, _build_labels
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    def _run():
        X, _ = _build_dataset(models_state.real_features, n_samples=600, seed=42)
        labels = _build_labels(X, np.random.default_rng(42))
        X_sc = StandardScaler().fit_transform(X)
        return evaluate_trained(algo, X_sc, labels)

    evals = await loop.run_in_executor(None, _run)
    models_state.evaluation_cache = evals
    return {"modelId": model_id, "evaluation": evals.get(model_id)}


@router.get("/registry")
async def get_registry():
    return {"versions": model_registry.list_versions(), "active": model_registry.active_meta()}


@router.put("/registry/{version}/activate")
async def activate_version(version: int):
    if not model_registry.set_active(version):
        raise HTTPException(404, f"No model version {version}.")
    sync_model_versions(model_registry.list_versions())
    return {"active": model_registry.active_meta()}
