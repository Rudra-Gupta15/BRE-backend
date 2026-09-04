"""/pipeline routes — upload files for a data source and run the Model Hub
pipeline. Bank statements parse + score here; sourceId=="gst_data" delegates
to app.gst.service / app.gst.pipeline. sourceId=="bbps_utility" parses as a
normal statement here too, then hands the transactions to app.bbps.analyze_bbps.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.common.file_scan import analyze_file
from app.common.persistence import log_security_event, save_statement
from app.aa.pipeline import run_pipeline
from app.common.security import guardrails, lineage
from app.aa.parser import parse_statement
from app import bbps
from app.common.state.session import session_state

logger = logging.getLogger(__name__)

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
        # analysis["cleanlinessPercent"] stays the real byte/cell-level scan from
        # analyze_file() above — it used to get overwritten with res["completeness"]
        # (coverage against the full 53-field GSTR-return schema), which made a
        # perfectly well-formed flat summary file — this one only carries 9 of
        # those fields by design — look like it was almost entirely dirty. That
        # coverage number is real and useful, just not "cleanliness"; it's kept
        # under its own name in gst_block["completeness"] below.
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

    # BBPS Utility Payment History arrives as the same statement formats —
    # BBPS payments are line items inside a real bank statement, not a
    # separate feed — so mine the utility-bill subset out of what just parsed,
    # evaluate the 16 BBPS rules against it, and score it with the trained
    # BBPS model. A real statement's derived feature row also gets appended
    # to the model's training corpus (real features; the label is still the
    # documented weak-supervision formula — see app.bbps.model's docstring).
    bbps_block = None
    if source_id == "bbps_utility":
        bbps_result = bbps.analyze_bbps(parsed["transactions"])
        bbps_block = {**bbps_result, "rules": bbps.rules.evaluate_bbps_rules(bbps_result)}
        fv = bbps.schema.feature_vector_from_analysis(bbps_result)
        if fv is not None:
            bbps_block["model"] = bbps.model.predict(fv)
            try:
                bbps.model.append_to_corpus(fv)
            except Exception:  # noqa: BLE001 — corpus growth must never break an upload
                logger.warning("Could not append BBPS statement to training corpus.", exc_info=True)
        parsed["bbps"] = bbps_block

    meta = {
        **analysis,
        "autoFilled":         False,
        "statementSummary":   parsed["summary"],
        "transactionsParsed": len(parsed["transactions"]),
        "statementId":        statement_id,
        **({"bbps": bbps_block} if bbps_block is not None else {}),
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
    """Thin adapter over app.gst.pipeline.run_gst_pipeline (translates its
    NoGstData into the 400 the frontend expects)."""
    from app.gst.pipeline import NoGstData, run_gst_pipeline

    try:
        return run_gst_pipeline()
    except NoGstData as exc:
        raise HTTPException(400, str(exc)) from exc


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
                scored = [f for f in file_metas if isinstance(f.get("cleanlinessPercent"), (int, float))]
                # Worst file in the folder, not the average — a single dirty
                # file must still trip the noise threshold for this source,
                # not get diluted away by the clean files next to it. Named,
                # not just counted, so the noise card can point at exactly
                # which file needs attention.
                worst = min(scored, key=lambda f: f["cleanlinessPercent"]) if scored else None
                merged_uploads[sid] = {
                    "cleanlinessPercent": worst["cleanlinessPercent"] if worst else None,
                    "worstFileName": worst.get("fileName") if worst else None,
                    "autoFilled": all(f.get("autoFilled") for f in file_metas),
                    "fileCount": len(file_metas),
                }
        result = run_pipeline(ids, merged_uploads, session_state.all_data_sources(), merged_parsed)

    session_state.pipeline = {
        "status": "done",
        "currentStage": len(result["stages"]),
        "noisePercent": result["noisePercent"],
        "llmActive": result["llmActive"],
        "noiseBySource": result.get("noiseBySource", {}),
        "cleaningBySource": result.get("cleaningBySource", {}),
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
