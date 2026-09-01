"""One-off: push everything currently in .session_cache.json into PostgreSQL.

- The one statement still fully in `parsed_statements` is saved in full
  (statement + transactions + inference run).
- Every other person in `inference_history` is backfilled as an applicant +
  a lightweight inference run (score / grade / decision / txn count / date).
  Their transaction-level data was overwritten in the session cache long ago
  and cannot be recovered.

Idempotent: re-running it will not create duplicates.

    ./venv/Scripts/python.exe scripts/backfill_from_cache.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
import re
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

import app.common.db as db
from app.common.db import init_db
from app.common.db_models import Applicant, InferenceRun
from app.common.persistence import save_inference_run, save_statement

CACHE = Path(__file__).resolve().parents[1].joinpath(".session_cache.json")


def _clean_name(raw: str) -> str:
    s = re.sub(r"\.(pdf|csv|txt|xlsx?)$", "", raw or "", flags=re.I)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)          # drop " (1)"
    s = re.sub(r"^stmt[_-]?\d*[_-]?", "", s, flags=re.I)  # drop "stmt_08_"
    s = s.replace("_", " ").strip()
    return s.title() if s else raw


def main() -> None:
    init_db()
    if not db.DB_ENABLED:
        raise SystemExit("PostgreSQL is not connected — check Backend/.env and use the venv Python.")

    cache = json.loads(CACHE.read_text("utf-8"))
    parsed = cache.get("parsed_statements") or {}
    uploads = cache.get("uploaded_files") or {}
    history = cache.get("inference_history") or []

    # ── 1. Full statement(s) still in the cache ────────────────────────────
    saved_full = 0
    for source_id, ps in parsed.items():
        if not (ps and ps.get("transactions")):
            continue
        analysis = dict(uploads.get(source_id) or {})
        analysis.setdefault("fileName", f"{source_id}.pdf")
        stmt_id = save_statement(source_id, analysis, ps)
        if stmt_id:
            saved_full += 1
            # build a proper bundle so the inference run links to this statement
            from app.aa.routes.inference import _build_bundle
            summary = ps.get("summary") or {}
            cid = summary.get("accountHolder") or _clean_name(analysis["fileName"])
            bundle = _build_bundle("risk_model", cid, summary.get("bankName") or "", source_id)
            run_id = save_inference_run(bundle, source_id)
            print(f"  [full] {analysis['fileName']}: statement #{stmt_id}, "
                  f"{len(ps['transactions'])} txns, inference run #{run_id}")

    # ── 2. Historical summaries (transaction data already lost) ────────────
    session = db.SessionLocal()
    saved_hist = skipped = 0
    try:
        for e in history:
            pk = e.get("personKey") or ""
            if pk in (":applicant", "account_aggregator:applicant") or e.get("id") == "applicant":
                skipped += 1
                continue

            name = _clean_name(e.get("id") or pk)
            bank = e.get("bank") if e.get("bank") not in (None, "Not Specified", "Not Specified") else None

            applicant = session.scalar(select(Applicant).where(Applicant.name == name))
            if applicant is None:
                applicant = Applicant(name=name, bank_name=bank)
                session.add(applicant)
                session.flush()
            elif bank and not applicant.bank_name:
                applicant.bank_name = bank

            model_id = e.get("modelId", "risk_model")
            # dedupe: one lightweight row per (applicant, model) with no statement
            existing = session.scalar(
                select(InferenceRun).where(
                    InferenceRun.applicant_id == applicant.id,
                    InferenceRun.model_id == model_id,
                    InferenceRun.statement_id.is_(None),
                )
            )
            if existing is not None:
                skipped += 1
                continue

            try:
                created = datetime.fromisoformat(e["date"])
            except Exception:
                created = None

            session.add(InferenceRun(
                applicant_id=applicant.id,
                statement_id=None,
                model_id=model_id,
                model_version=None,
                data_source="UPLOADED_STATEMENT",
                credit_score=e.get("riskScore"),
                risk_grade=e.get("grade"),
                decision=e.get("decision"),
                transaction_count=e.get("txCount") or 0,
                anomaly_count=0,
                created_at=created,
            ))
            saved_hist += 1
            print(f"  [hist] {name}: score {e.get('riskScore')} {e.get('grade')} "
                  f"({e.get('txCount')} txns, {e.get('date', '')[:10]})")
        session.commit()
    finally:
        session.close()

    # ── Report ────────────────────────────────────────────────────────────
    with db.engine.connect() as c:
        from sqlalchemy import text
        counts = {t: c.execute(text(f"select count(*) from bre.{t}")).scalar()
                  for t in ("applicants", "statements", "transactions",
                            "inference_runs", "bre_evaluations", "model_runs")}
    print(f"\nBackfilled {saved_full} full statement(s) + {saved_hist} historical "
          f"summaries ({skipped} skipped as duplicates/simulated).")
    print("bre.* row counts:", counts)


if __name__ == "__main__":
    main()
