# Thin persistence layer between the routers and PostgreSQL.
#
# Every function is a safe no-op when the DB is disabled or errors — the API
# keeps working off the in-memory state either way. The DB is the durable,
# queryable record (browse it in pgAdmin).

import logging

from sqlalchemy import select

from app.db import DB_ENABLED, get_session
from app.models_db import (
    Applicant,
    BreEvaluation,
    InferenceRun,
    ModelRun,
    Statement,
    Transaction,
)

logger = logging.getLogger(__name__)


def _f(x):
    """Coerce to float or None (Numeric columns reject stray strings/NaN)."""
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def save_statement(source_id: str, analysis: dict, parsed: dict) -> int | None:
    """Upsert the applicant and insert the statement + its transactions.
    Returns the new statement id (or None when the DB is off)."""
    if not DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        summary = parsed.get("summary", {}) or {}
        txns = parsed.get("transactions", []) or []
        holder = summary.get("accountHolder")
        bank = summary.get("bankName")

        applicant = None
        if holder:
            applicant = session.scalar(select(Applicant).where(Applicant.name == holder))
        if applicant is None:
            applicant = Applicant(name=holder, bank_name=bank)
            session.add(applicant)
            session.flush()
        elif bank and not applicant.bank_name:
            applicant.bank_name = bank

        stmt = Statement(
            applicant_id=applicant.id,
            source_id=source_id,
            file_name=analysis.get("fileName"),
            file_format=analysis.get("format"),
            size_bytes=analysis.get("sizeBytes"),
            cleanliness_percent=analysis.get("cleanlinessPercent"),
            statement_months=_f(summary.get("statementMonths")),
            opening_balance=_f(summary.get("openingBalance")),
            closing_balance=_f(summary.get("closingBalance")),
            min_balance=_f(summary.get("minBalance")),
            max_balance=_f(summary.get("maxBalance")),
            total_credit=_f(summary.get("totalCredit")),
            total_debit=_f(summary.get("totalDebit")),
            transaction_count=len(txns),
            summary=summary,
        )
        session.add(stmt)
        session.flush()

        session.add_all([
            Transaction(
                statement_id=stmt.id, seq=i,
                txn_date=t.get("date"),
                narration=(t.get("narration") or "")[:2000],
                txn_type=t.get("type"),
                amount=_f(t.get("amount")),
                balance=_f(t.get("balance")),
            )
            for i, t in enumerate(txns)
        ])
        session.commit()
        logger.info("Saved statement #%d (%d transactions) for applicant #%d.", stmt.id, len(txns), applicant.id)
        return stmt.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_statement failed")
        return None
    finally:
        session.close()


def _latest_statement(session, source_id: str) -> Statement | None:
    return session.scalars(
        select(Statement).where(Statement.source_id == source_id).order_by(Statement.id.desc()).limit(1)
    ).first()


def save_inference_run(bundle: dict, source_id: str | None) -> int | None:
    """Record one analysis run (Model Testing page). Returns inference_run id."""
    if not DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        stmt = _latest_statement(session, source_id) if source_id else None
        applicant_id = stmt.applicant_id if stmt else None

        # If simulated (no upload) but the user typed a ref id / bank, still log an applicant.
        if applicant_id is None and (bundle.get("statementLabel") or bundle.get("bankName")):
            a = Applicant(
                ref_id=bundle.get("brePayload", {}).get("statement_id"),
                name=bundle.get("accountHolder") or bundle.get("statementLabel"),
                bank_name=bundle.get("bankName"),
            )
            session.add(a)
            session.flush()
            applicant_id = a.id
        elif stmt is not None:
            a = session.get(Applicant, applicant_id)
            typed = bundle.get("brePayload", {}).get("statement_id")
            if a is not None and typed and typed != "applicant" and not a.ref_id:
                a.ref_id = typed

        risk = bundle.get("riskScore", {}) or {}
        bp = bundle.get("brePayload", {}) or {}
        run = InferenceRun(
            applicant_id=applicant_id,
            statement_id=stmt.id if stmt else None,
            model_id=bundle.get("model", {}).get("id", "risk_model"),
            model_version=bundle.get("model", {}).get("version"),
            data_source=bundle.get("dataSource", "SIMULATED"),
            credit_score=risk.get("score"),
            probability_of_default=_f(bp.get("model_metadata", {}).get("probability_of_default")),
            risk_grade=risk.get("grade"),
            decision=risk.get("decision"),
            feature_vector=bp.get("feature_vector"),
            bre_payload=bp,
        )
        session.add(run)
        session.commit()
        return run.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_inference_run failed")
        return None
    finally:
        session.close()


def save_bre_evaluation(result: dict, source_id: str | None) -> int | None:
    """Attach a BRE rule evaluation to the applicant's most recent inference run."""
    if not DB_ENABLED or result.get("available") is False:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        stmt = _latest_statement(session, source_id) if source_id else None
        run = None
        if stmt is not None:
            run = session.scalars(
                select(InferenceRun)
                .where(InferenceRun.statement_id == stmt.id)
                .order_by(InferenceRun.id.desc())
                .limit(1)
            ).first()
        if run is None:
            logger.info("save_bre_evaluation: no inference run to attach to — skipped.")
            return None

        # Replace any prior evaluation for this run.
        old = session.scalars(select(BreEvaluation).where(BreEvaluation.inference_run_id == run.id)).all()
        for o in old:
            session.delete(o)

        ev = BreEvaluation(
            inference_run_id=run.id,
            decision=result.get("decision"),
            applicant_profile=result.get("applicantProfile"),
            credit_score=result.get("creditScore"),
            passed=result.get("passed", 0),
            failed=result.get("failed", 0),
            skipped=result.get("skipped", 0),
            enabled_count=result.get("enabledCount", 0),
            results=result.get("results"),
            serious_flags=result.get("seriousFlags"),
        )
        session.add(ev)
        session.commit()
        return ev.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_bre_evaluation failed")
        return None
    finally:
        session.close()


def save_model_run(result: dict) -> int | None:
    if not DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        run = ModelRun(
            algorithm=result.get("algorithm", "gradient_boosting"),
            dataset_file=result.get("datasetFile"),
            tx_count=result.get("txCount"),
            real_features=result.get("realFeatures"),
            models=result.get("models"),
            evaluations=result.get("evaluations"),
        )
        session.add(run)
        session.commit()
        return run.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_model_run failed")
        return None
    finally:
        session.close()
