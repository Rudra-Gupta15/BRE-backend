from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.file_analysis import analyze_file
from app.services.persistence import log_security_event, save_statement
from app.services.pipeline_engine import run_pipeline
from app.services.security import guardrails, lineage
from app.services.statement_parser import parse_statement
from app.state.session_state import session_state

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/uploads")
async def get_uploads():
    return {"uploadedFiles": session_state.uploaded_files}


_MAX_FOLDER_FILES = 40


async def _ingest_one(source_id: str, file: UploadFile) -> dict:
    """Scan + parse a single uploaded file. Returns
    {meta, statement, statementId, lineage, warnings} or raises HTTPException."""
    buf = await file.read()
    file_name = file.filename or "upload.bin"
    # Folder uploads carry a relative path in the filename — keep just the leaf.
    file_name = file_name.replace("\\", "/").rsplit("/", 1)[-1]

    try:
        guardrails.validate_upload(buf, file_name)
    except guardrails.GuardrailError as exc:
        log_security_event("guardrail", "block", f"upload:{file_name}", {"error": str(exc)})
        raise HTTPException(422, f"Upload rejected ({file_name}): {exc}") from exc

    analysis = analyze_file(buf, file_name)
    try:
        parsed = await parse_statement(buf, file_name)
    except guardrails.GuardrailError as exc:
        log_security_event("guardrail", "block", f"parse:{file_name}", {"error": str(exc)})
        raise HTTPException(422, f"Statement rejected ({file_name}): {exc}") from exc

    parse_warnings: list[str] = []
    try:
        parsed, parse_warnings = guardrails.validate_llm_extraction(parsed)
    except guardrails.GuardrailError as exc:
        log_security_event("guardrail", "block", f"parse:{file_name}", {"error": str(exc)})
        raise HTTPException(422, f"Statement rejected ({file_name}): {exc}") from exc
    if parse_warnings:
        log_security_event("guardrail", "warn", f"parse:{file_name}", {"warnings": parse_warnings})

    # For PDFs: override byte-entropy cleanliness with LLM extraction quality.
    # Cleanliness = how completely the statement actually parsed, not raw byte
    # entropy. 0 transactions → the file couldn't be read as a statement.
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    if parsed.get("transactions") is not None:
        txns = parsed["transactions"]
        if not txns:
            analysis["cleanlinessPercent"] = 10
        elif ext in ("pdf", "json", "xlsx", "md"):
            complete = sum(
                1 for t in txns
                if t.get("date") and t.get("narration") and t.get("amount") and t.get("balance") is not None
            )
            analysis["cleanlinessPercent"] = round(max(20, min(99, (complete / len(txns)) * 100)))

    lin = {
        "file_sha256": lineage.file_digest(buf),
        "parse_confidence": lineage.extraction_confidence(parsed),
        "parse_warnings": parse_warnings or None,
        **lineage.parser_identity(),
    }
    statement_id = save_statement(source_id, analysis, parsed, lineage=lin)

    meta = {
        **analysis,
        "autoFilled":         False,
        "statementSummary":   parsed["summary"],
        "transactionsParsed": len(parsed["transactions"]),
        "statementId":        statement_id,
    }
    return {"meta": meta, "statement": parsed, "lineage": lin, "warnings": parse_warnings}


@router.post("/uploads")
async def upload_files(
    sourceId: str = Form(...),
    files: list[UploadFile] | None = File(None),
    file: UploadFile | None = File(None),
    scope: str = Form("hub"),
):
    """Accepts a *folder* of real files for one data source (any mix of PDF /
    CSV / TSV / TXT) via `files`, or a single `file`. Each file is byte-scanned
    for a cleanliness score and parsed into real transactions; files are kept
    separate so training runs across all of them.

    `scope`: "hub" (default) stores into Model Hub's uploads. "testing" stores
    into the Model Testing page's OWN separate slot — the two pages never share
    an upload and a Testing upload never affects the Model Hub pipeline.

    An upload replaces whatever files that source previously held in that scope."""
    scope = "testing" if scope == "testing" else "hub"
    valid_ids = {s["id"] for s in session_state.all_data_sources()}
    if sourceId not in valid_ids:
        raise HTTPException(404, f"Data source '{sourceId}' not found.")

    incoming = list(files or [])
    if file is not None:
        incoming.append(file)
    incoming = [f for f in incoming if (f.filename or "").strip()]
    if not incoming:
        raise HTTPException(400, "No files in the selected folder.")
    if len(incoming) > _MAX_FOLDER_FILES:
        raise HTTPException(422, f"Too many files ({len(incoming)}); max {_MAX_FOLDER_FILES} per folder.")

    metas: list[dict] = []
    statements: list[dict] = []
    lineages: list[dict] = []
    warnings: list[str] = []
    skipped: list[str] = []
    for f in incoming:
        leaf = (f.filename or "file").replace("\\", "/").rsplit("/", 1)[-1]
        try:
            res = await _ingest_one(sourceId, f)
        except HTTPException as exc:
            # One bad file (unsupported type, corrupt bytes) must not sink the
            # whole folder — record it and move on.
            ext = (leaf.rsplit(".", 1)[-1] if "." in leaf else "").lower()
            skipped.append(f"{leaf}: {exc.detail}")
            metas.append({"fileName": leaf, "format": ext, "sizeBytes": None,
                          "cleanlinessPercent": 0, "autoFilled": False,
                          "transactionsParsed": 0, "error": str(exc.detail)})
            statements.append({"transactions": [], "summary": {}, "error": str(exc.detail)})
            continue
        metas.append(res["meta"])
        statements.append(res["statement"])
        lineages.append(res["lineage"])
        warnings.extend(res["warnings"] or [])

    if all(m.get("error") for m in metas):
        raise HTTPException(422, "No file in the folder could be processed. " + " | ".join(skipped))

    session_state._upload_store(scope)[sourceId] = metas
    session_state._statement_store(scope)[sourceId] = statements
    session_state.persist()

    return {
        "uploadedFiles": session_state._upload_store(scope),
        "files": metas,
        "statements": statements,
        "statement": session_state.merged_statement_for(sourceId, scope),  # back-compat (single)
        "lineage": lineages,
        "guardrailWarnings": warnings,
        "skipped": skipped,
        "scope": scope,
    }



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
        session_state.uploaded_files[source_id] = [{
            "fileName": f"{source_id}_data.pdf",
            "sizeBytes": None,
            "format": "pdf",
            "cleanlinessPercent": cleanliness,
            "stats": None,
            "scannedAt": datetime.now(timezone.utc).isoformat(),
            "autoFilled": True,
        }]
    session_state.persist()
    return {"uploadedFiles": session_state.uploaded_files}


class RunPipelineBody(BaseModel):
    selectedIds: list[str] | None = None


@router.post("/run")
async def run_pipeline_handler(body: RunPipelineBody):
    ids = body.selectedIds if body.selectedIds else session_state.selected_ids
    if not ids:
        raise HTTPException(400, "Select at least one data source before running the pipeline.")

    # The pipeline engine reasons about one statement per source — collapse each
    # source's folder of files into a single merged statement + representative
    # upload meta (average cleanliness across the folder).
    merged_parsed: dict = {}
    merged_uploads: dict = {}
    for sid in ids:
        m = session_state.merged_statement_for(sid)
        if m is not None:
            merged_parsed[sid] = m
        file_metas = session_state.files_for(sid)
        if file_metas:
            scores = [f["cleanlinessPercent"] for f in file_metas
                      if isinstance(f.get("cleanlinessPercent"), (int, float))]
            merged_uploads[sid] = {
                "cleanlinessPercent": round(sum(scores) / len(scores)) if scores else None,
                "autoFilled": all(f.get("autoFilled") for f in file_metas),
                "fileCount": len(file_metas),
            }

    result = run_pipeline(ids, merged_uploads, session_state.all_data_sources(), merged_parsed)

    session_state.pipeline = {
        "status": "done",
        "currentStage": len(result["stages"]),
        "noisePercent": result["noisePercent"],
        "llmActive": result["llmActive"],
        "processedTable": result["processedTable"],
        "lastRunAt": result["completedAt"],
    }
    session_state.persist()

    return {
        "pipeline": session_state.pipeline,
        "stages": result["stages"],
        "storedFile": result["storedFile"],
        "selectedFeatures": result["selectedFeatures"],
        "normalizeTable": result["normalizeTable"],
        "engineeredTable": result["engineeredTable"],
        "selectionTable": result["selectionTable"],
    }


@router.get("/status")
async def get_pipeline_status():
    return {"pipeline": session_state.pipeline}
