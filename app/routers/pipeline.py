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


def _ingest_gst(source_id: str, buf: bytes, file_name: str, analysis: dict) -> dict:
    """One GST file → parsed by the GST parser, scored by the GST model, and
    appended to the GST training corpus."""
    from app.gst import service as gst_service

    try:
        res = gst_service.ingest_gst_file(buf, file_name)
    except ValueError as exc:
        raise HTTPException(422, f"GST file rejected ({file_name}): {exc}") from exc

    preds = res["predictions"]
    analysis["cleanlinessPercent"] = res["completeness"]

    gst_summary = {
        "mode": res["mode"],                       # "returns" | "summary"
        "records": res["records"],                 # rows in the file
        "businesses": res["businesses"],           # distinct GSTINs scored
        "returnsSeen": res.get("returnsSeen", {}),  # {GSTR1: n, GSTR3B: n, ...}
        "avgUnderwritingScore": preds.get("avgUnderwritingScore"),
        "riskCounts": preds.get("riskCounts", {}),
        "modelVersion": preds.get("modelVersion"),
        "completeness": res["completeness"],
        "corpusRows": res["corpusRows"],
        "modelAvailable": bool(preds.get("available")),
    }
    parsed = {
        "transactions": [],
        "summary": {
            "transactionCount": 0,
            "gstRecords": res["records"],
            "gstBusinesses": res["businesses"],
            "kind": "gst",
        },
        # Store the summary + a small sample only — a large folder must not
        # bloat the session cache / DB row.
        "gst": {**gst_summary,
                "sample": preds.get("predictions", [])[:20],
                "detail": res.get("detail", [])[:10],
                "profiles": [p.get("_meta") for p in res.get("profiles", [])][:20]},
    }

    lin = {
        "file_sha256": lineage.file_digest(buf),
        "parse_confidence": (res["completeness"] / 100.0),
        "parse_warnings": res["warnings"] or None,
        **lineage.parser_identity(),
    }
    statement_id = save_statement(source_id, analysis, parsed, lineage=lin)

    meta = {
        **analysis,
        "autoFilled": False,
        "statementSummary": parsed["summary"],
        "transactionsParsed": 0,
        "gst": gst_summary,
        "statementId": statement_id,
    }
    return {"meta": meta, "statement": parsed, "lineage": lin, "warnings": res["warnings"]}


async def _ingest_gst_folder(source_id: str, incoming: list[UploadFile]):
    """A whole folder of GST files. Return files (GSTR-1/3B/2A/2B) are MERGED and
    rolled into one profile per GSTIN; a flat per-business summary file is handled
    on its own. Returns (metas, statements, lineages, warnings, skipped) or None
    if no file here is GST data (fall back to the normal per-file path)."""
    from app.gst import aggregate as gst_agg
    from app.gst import model as gst_model
    from app.gst import present as gst_present
    from app.gst import returns as gst_returns
    from app.gst import service as gst_service

    bufs = []
    for f in incoming:
        leaf = (f.filename or "file").replace("\\", "/").rsplit("/", 1)[-1]
        bufs.append((leaf, await f.read()))

    return_rows: list[dict] = []
    return_files: list[tuple[str, int]] = []
    summary_files: list[tuple[str, bytes]] = []
    skipped: list[str] = []

    for leaf, buf in bufs:
        try:
            guardrails.validate_upload(buf, leaf)
        except guardrails.GuardrailError as exc:
            skipped.append(f"{leaf}: {exc}")
            continue
        rows = gst_returns.parse_returns_file(buf, leaf)
        if rows:
            return_rows.extend(rows)
            return_files.append((leaf, len(rows)))
        else:
            summary_files.append((leaf, buf))

    if not return_rows and not summary_files:
        return None  # nothing GST here

    metas, statements, lineages, warnings = [], [], [], []

    # ── merged return set → one profile per business ──────────────────────
    if return_rows:
        profiles = [p for p in gst_agg.build_profiles(return_rows) if p]
        model_input = [{k: v for k, v in p.items() if k != "_meta"} for p in profiles]
        preds = gst_model.predict_many(model_input)
        corpus_rows = None
        try:
            import pandas as pd
            corpus_rows = gst_model.append_to_corpus(pd.DataFrame(model_input))
        except Exception:  # noqa: BLE001
            warnings.append("Profiles scored but not added to the training corpus.")
        seen = {t: 0 for t in gst_returns.RETURN_TYPES}
        for r in return_rows:
            seen[r["return_type"]] += 1

        gst_block = {
            "mode": "returns", "records": len(return_rows), "businesses": len(profiles),
            "returnsSeen": seen, "avgUnderwritingScore": preds.get("avgUnderwritingScore"),
            "riskCounts": preds.get("riskCounts", {}), "modelVersion": preds.get("modelVersion"),
            "completeness": 100, "corpusRows": corpus_rows,
            "modelAvailable": bool(preds.get("available")),
            "files": [{"fileName": n, "returnRows": c} for n, c in return_files],
        }
        detail = [
            gst_present.business_view(mi, pr, meta=p.get("_meta", {}))
            for mi, pr, p in zip(model_input, preds.get("predictions", []), profiles)
        ][:10]
        parsed = {
            "transactions": [],
            "summary": {"transactionCount": 0, "gstRecords": len(return_rows),
                        "gstBusinesses": len(profiles), "kind": "gst"},
            "gst": {**gst_block, "sample": preds.get("predictions", [])[:20],
                    "detail": detail,
                    "profiles": [p.get("_meta") for p in profiles][:20]},
        }
        lin = {"file_sha256": lineage.file_digest(b"".join(b for _, b in bufs)),
               "parse_confidence": 1.0 if preds.get("available") else 0.5,
               "parse_warnings": warnings or None, **lineage.parser_identity()}
        sid = save_statement(source_id, {"cleanlinessPercent": 100}, parsed, lineage=lin)
        # one combined "file" row representing the whole return set
        label = " + ".join(t for t, c in seen.items() if c) or "GST returns"
        metas.append({
            "fileName": f"{label}  ({len(return_files)} file{'s' if len(return_files) != 1 else ''})",
            "format": "gst-returns", "sizeBytes": sum(len(b) for _, b in bufs),
            "cleanlinessPercent": 100, "autoFilled": False, "transactionsParsed": 0,
            "gst": gst_block, "statementId": sid,
            "statementSummary": parsed["summary"],
        })
        statements.append(parsed)
        lineages.append(lin)

    # ── any flat per-business summary files, handled individually ─────────
    for leaf, buf in summary_files:
        analysis = analyze_file(buf, leaf)
        try:
            res = gst_service.ingest_gst_file(buf, leaf)
        except ValueError as exc:
            skipped.append(f"{leaf}: {exc}")
            metas.append({"fileName": leaf, "format": leaf.rsplit('.', 1)[-1].lower(),
                          "cleanlinessPercent": 0, "autoFilled": False,
                          "transactionsParsed": 0, "error": str(exc)})
            statements.append({"transactions": [], "summary": {}, "error": str(exc)})
            continue
        preds = res["predictions"]
        analysis["cleanlinessPercent"] = res["completeness"]
        gst_block = {
            "mode": res["mode"], "records": res["records"], "businesses": res["businesses"],
            "returnsSeen": res.get("returnsSeen", {}),
            "avgUnderwritingScore": preds.get("avgUnderwritingScore"),
            "riskCounts": preds.get("riskCounts", {}), "modelVersion": preds.get("modelVersion"),
            "completeness": res["completeness"], "corpusRows": res["corpusRows"],
            "modelAvailable": bool(preds.get("available")),
        }
        parsed = {"transactions": [],
                  "summary": {"transactionCount": 0, "gstRecords": res["records"], "kind": "gst"},
                  "gst": {**gst_block, "sample": preds.get("predictions", [])[:20],
                          "detail": res.get("detail", [])[:10]}}
        lin = {"file_sha256": lineage.file_digest(buf), "parse_confidence": res["completeness"] / 100.0,
               "parse_warnings": res["warnings"] or None, **lineage.parser_identity()}
        sid = save_statement(source_id, analysis, parsed, lineage=lin)
        metas.append({**analysis, "autoFilled": False, "transactionsParsed": 0,
                      "gst": gst_block, "statementId": sid, "statementSummary": parsed["summary"]})
        statements.append(parsed)
        lineages.append(lin)
        warnings.extend(res["warnings"] or [])

    return metas, statements, lineages, warnings, skipped


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

    # GST Transaction Data → the GST subsystem (parser + model), not the bank
    # statement pipeline. These files carry GST return fields, not transactions.
    if source_id == "gst_data":
        return _ingest_gst(source_id, buf, file_name, analysis)

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

    # GST folder: return files (GSTR-1/3B/2A/2B) must be MERGED across the whole
    # folder before the per-business roll-up — not processed one file at a time.
    if sourceId == "gst_data":
        gst_folder = await _ingest_gst_folder(sourceId, incoming)
        if gst_folder is not None:
            metas, statements, lineages, warnings, skipped = gst_folder
            session_state._upload_store(scope)[sourceId] = metas
            session_state._statement_store(scope)[sourceId] = statements
            session_state.persist()
            return {
                "uploadedFiles": session_state._upload_store(scope),
                "files": metas, "statements": statements,
                "statement": session_state.merged_statement_for(sourceId, scope),
                "lineage": lineages, "guardrailWarnings": warnings,
                "skipped": skipped, "scope": scope,
            }

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


def _gst_pipeline_result() -> dict:
    """Stand-alone pipeline for the GST source: parse → roll up per business →
    score → rank the GST model's own features. No bank-statement stages."""
    from datetime import datetime, timezone

    from app.gst import model as gst_model

    gst_stmts = [s for s in session_state.statements_for("gst_data") if s and s.get("gst")]
    if not gst_stmts:
        raise HTTPException(400, "Upload GST return / summary files first.")

    blocks = [s["gst"] for s in gst_stmts]
    scores = [b["avgUnderwritingScore"] for b in blocks if b.get("avgUnderwritingScore") is not None]
    risk: dict = {}
    seen: dict = {}
    for b in blocks:
        for k, v in (b.get("riskCounts") or {}).items():
            risk[k] = risk.get(k, 0) + v
        for k, v in (b.get("returnsSeen") or {}).items():
            seen[k] = seen.get(k, 0) + v
    businesses = sum(b.get("businesses") or b.get("records") or 0 for b in blocks)
    avg = round(sum(scores) / len(scores), 2) if scores else None
    mode = blocks[0].get("mode")

    ranking = gst_model.feature_ranking()
    selection_table = [
        {"rank": i + 1, "feature": r["feature"], "variance": r["variance"],
         "selected": r["selected"], "kind": "gst"}
        for i, r in enumerate(ranking)
    ]
    seen_str = " + ".join(k for k, v in seen.items() if v) or (
        "GST summary" if mode == "summary" else "GST returns")
    processed_row = {
        "id": "gst_data", "kind": "gst",
        "adb": f"{businesses} business{'' if businesses == 1 else 'es'}",
        "gstDelta": f"score {avg}" if avg is not None else "—",
        "upiVelocity": seen_str,
        "cersai": " / ".join(f"{k} {v}" for k, v in risk.items()) or "—",
        "normScore": f"{avg / 100:.3f}" if avg is not None else "—",
        "status": "GST scored",
    }
    stages = [
        {"id": 1, "name": "1. Parse GST Files", "desc": "GSTR-1 / 3B / 2A / 2B or summary rows", "durationMs": 0,
         "detail": f"{sum(seen.values()) or businesses} row(s) across {len(gst_stmts)} file(s)."},
        {"id": 2, "name": "2. Normalise Fields", "desc": "Coerce numbers, strip currency/%",
         "durationMs": 0, "detail": "All GST fields normalised."},
        {"id": 3, "name": "3. Roll Up Per Business", "desc": "12-24 months → one profile per GSTIN",
         "durationMs": 0, "detail": f"{businesses} business profile(s)."},
        {"id": 4, "name": "4. Score", "desc": "GST Underwriting Model — score + risk flag",
         "durationMs": 0, "detail": f"avg score {avg}; " + (", ".join(f"{k} {v}" for k, v in risk.items()) or "no flags")},
        {"id": 5, "name": "5. Rank Features", "desc": "GST model inputs by variance",
         "durationMs": 0, "detail": f"{len(ranking)} GST feature(s) ranked."},
    ]
    return {
        "stages": stages, "noisePercent": 0, "llmActive": False,
        "processedTable": [processed_row], "storedFile": "gst_feature_profiles.csv",
        "selectedFeatures": [r["feature"] for r in ranking if r["selected"]],
        "normalizeTable": [], "engineeredTable": [], "selectionTable": selection_table,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/run")
async def run_pipeline_handler(body: RunPipelineBody):
    ids = body.selectedIds if body.selectedIds else session_state.selected_ids
    if not ids:
        raise HTTPException(400, "Select at least one data source before running the pipeline.")

    # GST is its own pipeline — never mixed with the bank-statement stages.
    if set(ids) == {"gst_data"}:
        result = _gst_pipeline_result()
    else:
        ids = [i for i in ids if i != "gst_data"] or ids
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
