import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.gst import model, parser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gst", tags=["gst"])


@router.get("/model")
async def get_model_status():
    """Is the GST underwriting model trained, and its metrics."""
    return model.status()


@router.post("/train")
async def train_model(file: UploadFile | None = File(default=None)):
    """Train / retrain. Optionally upload a CSV / XLSX / JSON / PDF of GST rows
    first — it is appended to the accumulating corpus, then training runs on
    everything. Old model versions are kept."""
    appended = 0
    if file is not None:
        raw = await file.read()
        try:
            records = parser.parse_gst(raw, file.filename or "upload")
        except ValueError as exc:
            raise HTTPException(400, f"Could not read the training file: {exc}")
        if not records:
            raise HTTPException(400, "No GST rows found in the uploaded file.")
        import pandas as pd
        appended = model.append_to_corpus(pd.DataFrame(records))

    try:
        result = model.train()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    result["appendedRows"] = appended
    return result


class RecordBody(BaseModel):
    record: dict


@router.post("/predict-record")
async def predict_record(body: RecordBody):
    """Predict the GST underwriting score + risk flag for one GST record."""
    return model.predict(body.record or {})


@router.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    """Parse an uploaded GST file (CSV / JSON / XLSX / PDF) and predict every row."""
    raw = await file.read()
    try:
        records = parser.parse_gst(raw, file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(400, f"Could not read the GST file: {exc}")
    if not records:
        raise HTTPException(400, "No GST rows found in the file.")
    return model.predict_many(records)
