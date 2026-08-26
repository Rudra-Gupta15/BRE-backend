from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.file_analysis import analyze_file
from app.services.pipeline_engine import run_pipeline
from app.services.statement_parser import parse_statement
from app.state.session_state import session_state

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/uploads")
async def get_uploads():
    return {"uploadedFiles": session_state.uploaded_files}


@router.post("/uploads")
async def upload_file(sourceId: str = Form(...), file: UploadFile = File(...)):
    """Accepts a real multipart file, scans its actual bytes for a
    cleanliness score (file_analysis.py), and — for CSV/TXT/PDF — parses
    real transactions out of it (statement_parser.py) so downstream
    pipeline/inference steps can use genuine data instead of mocks.

    For PDFs the cleanliness score is computed from the LLM extraction result
    (completeness of parsed rows) rather than byte entropy, which gives a
    meaningful quality signal."""
    valid_ids = {s["id"] for s in session_state.all_data_sources()}
    if sourceId not in valid_ids:
        raise HTTPException(404, f"Data source '{sourceId}' not found.")

    buf = await file.read()
    if not buf:
        raise HTTPException(400, "No file uploaded.")

    file_name = file.filename or "upload.bin"
    analysis  = analyze_file(buf, file_name)
    parsed    = await parse_statement(buf, file_name)

    # For PDFs: override byte-entropy cleanliness with LLM extraction quality.
    # - 0 transactions → 10% (the LLM couldn't read anything meaningful)
    # - Non-zero transactions: score based on how many rows have all 4 key
    #   fields (date, narration, amount, balance) — missing fields = noise.
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    if ext == "pdf" and parsed.get("transactions") is not None:
        txns = parsed["transactions"]
        if not txns:
            llm_cleanliness = 10
        else:
            complete = sum(
                1 for t in txns
                if t.get("date") and t.get("narration") and t.get("amount") and t.get("balance") is not None
            )
            llm_cleanliness = round(max(20, min(99, (complete / len(txns)) * 100)))
        analysis["cleanlinessPercent"] = llm_cleanliness

    session_state.uploaded_files[sourceId] = {
        **analysis,
        "autoFilled":        False,
        "statementSummary":  parsed["summary"],
        "transactionsParsed": len(parsed["transactions"]),
    }
    session_state.parsed_statements[sourceId] = parsed

    return {"uploadedFiles": session_state.uploaded_files, "analysis": analysis, "statement": parsed}



class AutofillBody(BaseModel):
    sourceIds: list[str] | None = None


@router.post("/uploads/autofill")
async def autofill_uploads(body: AutofillBody):
    ids = body.sourceIds if body.sourceIds else session_state.selected_ids
    sources_by_id = {s["id"]: s for s in session_state.all_data_sources()}

    for source_id in ids:
        source = sources_by_id.get(source_id)
        coverage = source.get("coverage") if source else None
        cleanliness = 70
        if isinstance(coverage, str):
            import re

            m = re.search(r"[\d.]+", coverage)
            if m:
                cleanliness = round(float(m.group()))
        session_state.uploaded_files[source_id] = {
            "fileName": f"{source_id}_data.pdf",
            "sizeBytes": None,
            "format": "pdf",
            "cleanlinessPercent": cleanliness,
            "stats": None,
            "scannedAt": datetime.now(timezone.utc).isoformat(),
            "autoFilled": True,
        }
    return {"uploadedFiles": session_state.uploaded_files}


class RunPipelineBody(BaseModel):
    selectedIds: list[str] | None = None


@router.post("/run")
async def run_pipeline_handler(body: RunPipelineBody):
    ids = body.selectedIds if body.selectedIds else session_state.selected_ids
    if not ids:
        raise HTTPException(400, "Select at least one data source before running the pipeline.")

    result = run_pipeline(ids, session_state.uploaded_files, session_state.all_data_sources(), session_state.parsed_statements)

    session_state.pipeline = {
        "status": "done",
        "currentStage": len(result["stages"]),
        "noisePercent": result["noisePercent"],
        "llmActive": result["llmActive"],
        "processedTable": result["processedTable"],
        "lastRunAt": result["completedAt"],
    }

    return {"pipeline": session_state.pipeline, "stages": result["stages"], "logs": result["logs"], "storedFile": result["storedFile"]}


@router.get("/status")
async def get_pipeline_status():
    return {"pipeline": session_state.pipeline}
