import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.gst import aggregate, model, returns, rules, service
from app.common.state.session import session_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gst", tags=["gst"])


@router.get("/model")
async def get_model_status():
    """Is the GST underwriting model trained, and its metrics."""
    return model.status()


@router.get("/model/registry")
async def get_model_registry():
    """Version list + per-head deploy state for the Model Hub deployment table."""
    return model.registry_view()


@router.get("/patterns")
async def get_patterns():
    """Post-training pattern recognition for the GST model — behavioral patterns
    across the uploaded business(es) + per-head feature weights. Twin of
    /api/models/patterns; same "Detected Patterns" card."""
    from app.gst import patterns

    return patterns.compute()


@router.get("/pattern-match")
async def pattern_match():
    """Anomaly / fraud pattern match for the GST file uploaded on the Model
    Testing page — trained pattern models + rule checks + deviation table + an
    overall verdict. Twin of POST /api/inference/patterns."""
    from app.gst import patterns

    return patterns.test_patterns()


class ActiveVersionBody(BaseModel):
    version: int


@router.put("/model/active")
async def set_active_version(body: ActiveVersionBody):
    """Roll the active GST bundle to a specific trained version."""
    if not model.set_active_version(body.version):
        raise HTTPException(404, f"No GST model version {body.version}.")
    return model.registry_view()


@router.get("/score-testing")
async def score_testing():
    """Score the GST file uploaded on the Model Testing page (scope='testing').
    Returns one consolidated result across every business in the file plus each
    GST head's output."""
    stmts = [s for s in session_state.statements_for("gst_data", "testing") if s and s.get("gst")]
    if not stmts:
        return {"available": False,
                "message": "Upload a GST file on this page first (GSTR-1/3B/2A/2B or a per-business summary)."}

    samples: list[dict] = []
    detail: list[dict] = []
    businesses = 0
    modes: set[str] = set()
    returns_seen: dict = {}
    for s in stmts:
        g = s["gst"]
        businesses += g.get("businesses") or g.get("records") or 0
        modes.add(g.get("mode") or "summary")
        for k, v in (g.get("returnsSeen") or {}).items():
            returns_seen[k] = returns_seen.get(k, 0) + (v or 0)
        samples.extend(g.get("sample") or [])
        detail.extend(g.get("detail") or [])

    ok = [p for p in samples if p.get("available")]
    if not ok:
        return {"available": False, "message": "The GST model could not score this file."}

    avg = round(sum(p["underwritingScore"] for p in ok) / len(ok), 2)
    from collections import Counter

    head_summary = model._aggregate_head_scores(ok)
    for hid, meta in model.active_heads_meta().items():
        if hid in head_summary:
            head_summary[hid].update({k: v for k, v in meta.items() if v is not None})

    return {
        "available": True,
        "mode": "returns" if "returns" in modes else "summary",
        "businesses": businesses or len(ok),
        "scored": len(ok),
        "returnsSeen": {k: v for k, v in returns_seen.items() if v},
        "avgUnderwritingScore": avg,
        "riskCounts": dict(Counter(p["riskFlag"] for p in ok)),
        "modelVersion": ok[0].get("modelVersion"),
        "headSummary": head_summary,
        "deployed": model.deploy_state(),
        "detail": detail[:20],
        "predictions": ok[:50],
    }


@router.post("/bre-evaluate")
async def bre_evaluate():
    """Evaluate the computable gst_data BRE rules against the GST file uploaded on
    the Model Testing page — the "BRE payload" tab."""
    stmts = [s for s in session_state.statements_for("gst_data", "testing") if s and s.get("gst")]
    detail: list[dict] = []
    for s in stmts:
        detail.extend(s["gst"].get("detail") or [])
    if not detail:
        return {"available": False,
                "message": "Upload a GST file on this page first — BRE rules run on real data."}

    primary = detail[0]
    profile = primary.get("profile") or {}
    pred = primary.get("prediction") or {}
    heads = pred.get("headScores") or {}
    return {
        "available": True,
        "payload": rules.payload(profile, heads, pred),
        "evaluation": rules.evaluate(profile, heads),
        "businessCount": len(detail),
    }


class RecordTestBody(BaseModel):
    customId: str | None = None
    fileName: str | None = None


@router.post("/record-test")
async def record_test(body: RecordTestBody):
    """Save the current GST test to Model Testing history — ONE row per upload,
    all 4 heads folded in. Called once on a deliberate upload / re-run."""
    result = await score_testing()
    if not result.get("available"):
        return {"recorded": False, "message": result.get("message")}
    from app.common.persistence import record_gst_test_history
    record_gst_test_history(result, "gst_data", body.fileName, body.customId)
    return {"recorded": True}


@router.post("/model/{head_id}/deploy")
async def toggle_head_deploy(head_id: str):
    """Deploy / revoke one GST head (Underwriting Score / Risk Flag / Loan
    Eligibility / Filing Compliance)."""
    current = model.deploy_state().get(head_id)
    if current is None:
        raise HTTPException(404, f"Unknown GST model '{head_id}'.")
    try:
        st = model.set_deployed(head_id, not current)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return {"deployed": st}


@router.post("/train")
async def train_model(file: UploadFile | None = File(default=None)):
    """Train / retrain. Optionally upload a file of GST rows first (a flat
    per-business summary, or GSTR-1/3B/2A/2B returns which are rolled into
    profiles) — appended to the corpus, then training runs on everything.
    Old model versions are kept."""
    appended = 0
    if file is not None:
        raw = await file.read()
        try:
            res = service.ingest_gst_file(raw, file.filename or "upload", add_to_corpus=True)
        except ValueError as exc:
            raise HTTPException(400, f"Could not read the training file: {exc}")
        appended = res.get("corpusRows") or res.get("businesses") or res.get("records") or 0

    try:
        result = model.train()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    result["appendedRows"] = appended
    try:
        from app.gst import patterns
        patterns.save_training_baseline()
        patterns.train_pattern_models()
    except Exception:  # noqa: BLE001
        logger.warning("GST pattern-model training failed", exc_info=True)
    return result


class RecordBody(BaseModel):
    record: dict


@router.post("/predict-record")
async def predict_record(body: RecordBody):
    """Predict the GST underwriting score + risk flag for one GST profile."""
    return model.predict(body.record or {})


@router.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    """One GST file → predictions. Accepts GSTR-1/3B/2A/2B returns (rolled up
    per business) or a flat per-business summary."""
    raw = await file.read()
    try:
        res = service.ingest_gst_file(raw, file.filename or "upload", add_to_corpus=False)
    except ValueError as exc:
        raise HTTPException(400, f"Could not read the GST file: {exc}")
    return {
        "mode": res["mode"],
        "records": res["records"],
        "businesses": res["businesses"],
        "returnsSeen": res.get("returnsSeen", {}),
        **res["predictions"],
        "profiles": [
            {**{k: v for k, v in p.items() if k != "_meta"}, "_meta": p.get("_meta")}
            for p in res.get("profiles", [])
        ][:100],
    }


@router.post("/returns")
async def predict_returns(files: list[UploadFile] = File(...)):
    """Upload a FOLDER of GST return files (GSTR-1, GSTR-3B, GSTR-2A, GSTR-2B —
    12-24 months per business). They are merged, rolled up into one profile per
    GSTIN, scored, and added to the training corpus."""
    all_rows: list[dict] = []
    per_file = []
    for f in files:
        raw = await f.read()
        rows = returns.parse_returns_file(raw, f.filename or "upload")
        per_file.append({"file": f.filename, "returnRows": len(rows)})
        all_rows.extend(rows)

    if not all_rows:
        raise HTTPException(400, "No GST return rows found. Expected GSTR-1 / "
                                 "GSTR-3B / GSTR-2A / GSTR-2B files.")

    profiles = [p for p in aggregate.build_profiles(all_rows) if p]
    model_input = [{k: v for k, v in p.items() if k != "_meta"} for p in profiles]

    corpus_rows = None
    if model_input:
        try:
            import pandas as pd
            corpus_rows = model.append_to_corpus(pd.DataFrame(model_input))
        except Exception:  # noqa: BLE001
            logger.warning("Could not append return profiles to corpus.", exc_info=True)

    preds = model.predict_many(model_input)
    seen = {t: 0 for t in returns.RETURN_TYPES}
    for r in all_rows:
        seen[r["return_type"]] += 1

    return {
        "files": per_file,
        "returnRows": len(all_rows),
        "returnsSeen": seen,
        "businesses": len(profiles),
        "corpusRows": corpus_rows,
        **preds,
        "profiles": [
            {**{k: v for k, v in p.items() if k != "_meta"}, "_meta": p["_meta"]}
            for p in profiles
        ],
    }
