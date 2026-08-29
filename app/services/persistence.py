# Thin persistence layer between the routers and PostgreSQL.
#
# Every function is a safe no-op when the DB is disabled or errors — the API
# keeps working off the in-memory state either way. The DB is the durable,
# queryable record (browse it in pgAdmin).

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select


def _now() -> datetime:
    return datetime.now(timezone.utc)

from app import db
from app.db import get_session
from app.models_db import (
    AnalyticsMonth,
    Anomaly,
    Applicant,
    BreEvaluation,
    DatasetBatch,
    DriftSnapshot,
    InferenceRun,
    ModelDeployment,
    ModelRun,
    ModelVersion,
    SecurityEvent,
    Statement,
    TestHistory,
    TrainedModel,
    Transaction,
)

logger = logging.getLogger(__name__)


def _f(x):
    """Coerce to float or None (Numeric columns reject stray strings/NaN)."""
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _num(x):
    """Parse a number out of '308.30', '72.5%', '₹1,23,456' → float / None."""
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return None
    cleaned = x.replace("₹", "").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def save_statement(source_id: str, analysis: dict, parsed: dict,
                   lineage: dict | None = None) -> int | None:
    """Upsert the applicant and insert the statement + its transactions.
    `lineage` carries file_sha256 / parser_model / parser_version /
    parse_confidence / parse_warnings for the data-lineage trail.
    Returns the new statement id (or None when the DB is off)."""
    if not db.DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    lin = lineage or {}
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
            file_sha256=lin.get("file_sha256"),
            parser_model=lin.get("parser_model"),
            parser_version=lin.get("parser_version"),
            parse_confidence=_f(lin.get("parse_confidence")),
            parse_warnings=lin.get("parse_warnings"),
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
    if not db.DB_ENABLED:
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
        model_id = bundle.get("model", {}).get("id", "risk_model")
        sec = bundle.get("security", {}) or {}
        outl = sec.get("outlier", {}) or {}

        # which exact registry artifact scored this applicant
        mv_id = None
        src = risk.get("modelSource") or ""
        if src.startswith("dataset-v"):
            try:
                mv_id = session.scalar(
                    select(ModelVersion.id).where(ModelVersion.version == int(src.split("v")[-1]))
                )
            except (ValueError, TypeError):
                mv_id = None

        fields = dict(
            applicant_id=applicant_id,
            statement_id=stmt.id if stmt else None,
            model_id=model_id,
            model_version=bundle.get("model", {}).get("version"),
            data_source=bundle.get("dataSource", "SIMULATED"),
            credit_score=risk.get("score"),
            probability_of_default=_f(bp.get("model_metadata", {}).get("probability_of_default")),
            # the true 3-tier band (LOW/MEDIUM/HIGH) — `decision` carries the
            # gated APPROVED/REJECTED outcome
            risk_grade=risk.get("gradeRaw") or risk.get("grade"),
            decision=risk.get("decision"),
            anomaly_count=len(bundle.get("anomalies") or []),
            transaction_count=len(bundle.get("transactions") or []),
            scorecard_score=risk.get("scorecardScore"),
            model_score=risk.get("modelScore"),
            model_source=risk.get("modelSource"),
            ml_blended=risk.get("mlBlended"),
            feature_vector=bp.get("feature_vector"),
            bre_payload=bp,
            model_version_id=mv_id,
            outlier_score=_f(outl.get("outlierScore")),
            is_outlier=outl.get("isOutlier"),
            outlier_flags=outl.get("flags"),
            guardrail_status=sec.get("guardrailStatus"),
            guardrail_warnings=sec.get("guardrailWarnings"),
        )

        # The Model Testing page re-fires /inference/run on every input change,
        # so keep ONE current row per (statement, model) — update it in place
        # instead of piling up duplicates. (Simulated runs with no statement
        # still just append.)
        existing = None
        if stmt is not None:
            existing = session.scalars(
                select(InferenceRun).where(
                    InferenceRun.statement_id == stmt.id, InferenceRun.model_id == model_id,
                ).limit(1)
            ).first()

        if existing is not None:
            for k, v in fields.items():
                setattr(existing, k, v)
            existing.created_at = _now()
            run = existing
        else:
            run = InferenceRun(**fields)
            session.add(run)
        session.flush()

        _replace_anomalies(session, run.id, bundle.get("anomalies") or [])
        _replace_analytics(session, run.id, bundle.get("analytics") or {})

        session.commit()
        return run.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_inference_run failed")
        return None
    finally:
        session.close()


def _applicant_key(bundle: dict, custom_id: str | None, file_name: str | None, source_id: str | None) -> str:
    """Stable per-APPLICATION key — a re-test of the same application updates its
    one history row rather than adding a new one."""
    typed = (custom_id or "").strip()
    if typed and typed.lower() not in ("applicant", ""):
        return f"ref:{typed.lower()}"
    holder = (bundle.get("accountHolder") or bundle.get("statementLabel") or "").strip()
    if holder:
        return f"name:{holder.lower()}"
    if file_name:
        return f"file:{file_name.rsplit('.', 1)[0].strip().lower()}"
    return f"src:{source_id or 'unknown'}"


def record_test_history(bundle: dict, source_id: str | None, file_name: str | None,
                        custom_id: str | None = None) -> None:
    """Upsert the persistent Model Testing history — ONE row per (application,
    model). Re-testing the same application with the same model updates that row;
    a different model on the same application adds a new row.
    Called only on deliberate runs (upload / Run Analysis)."""
    if not db.DB_ENABLED:
        return
    session = get_session()
    if session is None:
        return
    try:
        risk = bundle.get("riskScore", {}) or {}
        model = bundle.get("model", {}) or {}
        key = _applicant_key(bundle, custom_id, file_name, source_id)
        fields = dict(
            ref_id=(bundle.get("brePayload", {}) or {}).get("statement_id"),
            applicant_name=bundle.get("accountHolder") or bundle.get("statementLabel"),
            bank_name=bundle.get("bankName"),
            model_id=model.get("id"),
            model_name=model.get("name"),
            model_version=model.get("version"),
            data_source=bundle.get("dataSource"),
            credit_score=risk.get("score"),
            risk_grade=risk.get("gradeRaw") or risk.get("grade"),
            decision=risk.get("decision"),
            transaction_count=len(bundle.get("transactions") or []),
            source_id=source_id,
            custom_id=custom_id,
            file_name=file_name,
            result_bundle=bundle,
            tested_at=_now(),
        )
        existing = session.scalar(select(TestHistory).where(
            TestHistory.applicant_key == key,
            TestHistory.model_id == model.get("id"),
        ))
        if existing is not None:
            for k, v in fields.items():
                setattr(existing, k, v)
        else:
            session.add(TestHistory(applicant_key=key, first_tested_at=_now(), **fields))
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("record_test_history failed")
    finally:
        session.close()


def list_test_history(limit: int = 1000) -> list[dict]:
    """One row per tested application, newest first. Empty when the DB is off."""
    if not db.DB_ENABLED:
        return []
    session = get_session()
    if session is None:
        return []
    try:
        rows = session.scalars(
            select(TestHistory)
            .where(TestHistory.applicant_key.is_not(None))
            .order_by(TestHistory.tested_at.desc())
            .limit(limit)
        ).all()
        out = []
        for r in rows:
            label = r.ref_id or r.applicant_name
            if not label and r.file_name:
                label = r.file_name.rsplit(".", 1)[0]
            out.append({
                "rowId": r.id,
                "id": label or "—",
                "bank": r.bank_name or "—",
                "modelId": r.model_id,
                "model": r.model_name or r.model_id or "—",
                "version": r.model_version,
                "dataSource": r.data_source,
                "riskScore": r.credit_score,
                "grade": r.risk_grade,
                "decision": r.decision,
                "txCount": r.transaction_count,
                "date": r.tested_at.isoformat() if r.tested_at else None,
                "status": "ANALYZED",
            })
        return out
    except Exception:  # noqa: BLE001
        logger.exception("list_test_history failed")
        return []
    finally:
        session.close()


def get_test_history_bundle(row_id: int) -> dict | None:
    """The full stored analysis for one history row — to reopen its output."""
    if not db.DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        r = session.get(TestHistory, row_id)
        return r.result_bundle if r else None
    except Exception:  # noqa: BLE001
        logger.exception("get_test_history_bundle failed")
        return None
    finally:
        session.close()


def _replace_anomalies(session, run_id: int, anomalies: list[dict]) -> None:
    """Rewrite the anomaly rows for one inference run (the Anomalies tab)."""
    for old in session.scalars(select(Anomaly).where(Anomaly.inference_run_id == run_id)).all():
        session.delete(old)
    session.add_all([
        Anomaly(
            inference_run_id=run_id,
            txn_date=a.get("date") or None,
            narration=(a.get("narration") or "")[:2000],
            amount=_num(a.get("amount")),
            score_percent=_num(a.get("score")),
            level=a.get("level"),
            reasons=a.get("reasons"),
        )
        for a in anomalies
    ])


def _replace_analytics(session, run_id: int, analytics: dict) -> None:
    """Rewrite the monthly analytics rows for one inference run (Analytics tab).
    Only the risk-model chart carries the raw inflow/outflow/ADB numbers."""
    for old in session.scalars(
        select(AnalyticsMonth).where(AnalyticsMonth.inference_run_id == run_id)
    ).all():
        session.delete(old)

    chart = analytics.get("chart") or []
    rows = []
    for i, pt in enumerate(chart):
        rows.append(AnalyticsMonth(
            inference_run_id=run_id,
            month=str(pt.get("month") or f"Period {i + 1}")[:24],
            seq=i,
            inflow=_num(pt.get("Inflow")),
            outflow=_num(pt.get("Outflow")),
            net_cashflow=_num(pt.get("NetCashflow")),
            adb_score=_num(pt.get("ADB") if pt.get("ADB") is not None else pt.get("ADBScore")),
            min_balance=_num(pt.get("MinBal")),
            pd_risk_percent=_num(pt.get("PDRiskPct")),
            credit_score=int(_num(pt.get("CreditScore"))) if _num(pt.get("CreditScore")) is not None else None,
        ))
    session.add_all(rows)


def save_bre_evaluation(result: dict, source_id: str | None) -> int | None:
    """Attach a BRE rule evaluation to the applicant's most recent inference run."""
    if not db.DB_ENABLED or result.get("available") is False:
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
            # No persisted statement yet (uploaded before the DB was on): attach
            # to the most recent real inference run instead.
            run = session.scalars(
                select(InferenceRun)
                .where(InferenceRun.data_source == "UPLOADED_STATEMENT")
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


def dashboard_snapshot() -> dict | None:
    """Real KPIs / charts / recent list aggregated from the DB. None when the
    DB is off (caller then uses the in-memory/baseline numbers)."""
    if not db.DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        real = InferenceRun.data_source == "UPLOADED_STATEMENT"

        # One representative run per applicant/statement (the risk_model view), so
        # re-running the same person doesn't inflate the counts. Works even when
        # the run isn't linked to a `statements` row (statement uploaded before
        # the DB was enabled) — falls back to the applicant, then the run id.
        entity = func.coalesce(
            InferenceRun.statement_id,
            0 - InferenceRun.applicant_id,
            0 - 1_000_000 - InferenceRun.id,
        )
        per_entity = (
            select(func.max(InferenceRun.id))
            .where(real, InferenceRun.model_id == "risk_model")
            .group_by(entity)
            .scalar_subquery()
        )
        one = InferenceRun.id.in_(per_entity)

        analyzed = session.scalar(
            select(func.count()).select_from(select(1).where(one).subquery())
        ) or 0
        processed = session.scalar(
            select(func.coalesce(func.sum(InferenceRun.transaction_count), 0)).where(one)
        ) or 0
        avg_score = session.scalar(select(func.avg(InferenceRun.credit_score)).where(one))
        anomalies = session.scalar(select(func.coalesce(func.sum(InferenceRun.anomaly_count), 0)).where(one)) or 0
        pending = session.scalar(
            select(func.count(func.distinct(InferenceRun.statement_id)))
            .select_from(BreEvaluation)
            .join(InferenceRun, BreEvaluation.inference_run_id == InferenceRun.id)
            .where(BreEvaluation.decision.in_(("CONDITIONAL APPROVAL", "APPROVED WITH NOTES")))
        ) or 0

        by_grade = dict(
            session.execute(
                select(InferenceRun.risk_grade, func.count(InferenceRun.id)).where(one).group_by(InferenceRun.risk_grade)
            ).all()
        )
        by_decision = dict(
            session.execute(
                select(InferenceRun.decision, func.count(InferenceRun.id)).where(one).group_by(InferenceRun.decision)
            ).all()
        )

        recent_rows = session.execute(
            select(InferenceRun, Applicant, Statement)
            .join(Applicant, InferenceRun.applicant_id == Applicant.id, isouter=True)
            .join(Statement, InferenceRun.statement_id == Statement.id, isouter=True)
            .where(one)
            .order_by(InferenceRun.id.desc())
            .limit(8)
        ).all()
        recent = [
            {
                "id": (a.ref_id if a and a.ref_id else a.name if a and a.name
                       else f"{s.file_name}" if s and s.file_name else f"Statement #{s.id}" if s else f"RUN-{r.id}"),
                "bank": (a.bank_name if a and a.bank_name else s.file_name if s else "—"),
                "date": r.created_at.isoformat() if r.created_at else None,
                "txCount": r.transaction_count,
                "riskScore": r.credit_score,
                "grade": r.risk_grade,
                "status": "ANALYZED",
            }
            for r, a, s in recent_rows
        ]

        return {
            "analyzed": int(analyzed),
            "processed": int(processed),
            "avgScore": round(float(avg_score), 1) if avg_score is not None else None,
            "anomalies": int(anomalies),
            "pending": int(pending),
            "byRiskGrade": {
                "LOW": by_grade.get("LOW", 0),
                "MEDIUM": by_grade.get("MEDIUM", 0),
                "HIGH": by_grade.get("HIGH", 0),
            },
            "byDecision": by_decision,
            "recent": recent,
        }
    except Exception:  # noqa: BLE001
        logger.exception("dashboard_snapshot failed")
        return None
    finally:
        session.close()


def save_model_run(result: dict) -> int | None:
    if not db.DB_ENABLED:
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
        session.flush()

        # Break the per-model accuracy blob into queryable rows — the 4 model
        # cards + the Model Evaluation table on the Model Hub page.
        evals = result.get("evaluations") or {}
        for m in result.get("models") or []:
            ev = (evals.get(m["id"]) or {}).get("evalMetrics", {}) if isinstance(evals, dict) else {}
            session.add(TrainedModel(
                model_run_id=run.id,
                model_id=m.get("id"),
                name=m.get("name") or m.get("id"),
                description=m.get("desc"),
                algorithm=m.get("algorithm") or run.algorithm,
                accuracy=_num(m.get("accuracy")),
                precision=_num(ev.get("precision")),
                recall=_num(ev.get("recall")),
                f1=_num(ev.get("f1Score")),
                brier_score=_num(ev.get("mse")),
                error_rate=_num(ev.get("mae")),
                cv_folds=m.get("cvFolds"),
                sample_count=m.get("sampleCount") or m.get("nSamples"),
                is_live=True,
                metric_meta=ev.get("metricMeta"),
                cv_detail=(evals.get(m["id"]) or {}).get("cvFolds") if isinstance(evals, dict) else None,
            ))
        session.commit()
        return run.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_model_run failed")
        return None
    finally:
        session.close()


def sync_model_deployments(version_map: dict, status_map: dict, name_map: dict) -> None:
    """Upsert the current version + deploy state for every model — the
    Model Version & Deployment Management Table. One row per model_id."""
    if not db.DB_ENABLED:
        return
    session = get_session()
    if session is None:
        return
    try:
        for model_id, name in name_map.items():
            row = session.scalar(select(ModelDeployment).where(ModelDeployment.model_id == model_id))
            if row is None:
                row = ModelDeployment(model_id=model_id, model_name=name)
                session.add(row)
            row.model_name = name
            row.selected_version = version_map.get(model_id)
            row.status = status_map.get(model_id, "Ready")
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("sync_model_deployments failed")
    finally:
        session.close()


def sync_model_versions(registry: list[dict]) -> None:
    """Mirror the on-disk population-model registry into the DB — the
    'Population Model vN' row and the version history."""
    if not db.DB_ENABLED:
        return
    session = get_session()
    if session is None:
        return
    try:
        for e in registry or []:
            v = e.get("version")
            if v is None:
                continue
            row = session.scalar(select(ModelVersion).where(ModelVersion.version == v))
            if row is None:
                row = ModelVersion(version=v)
                session.add(row)
            m = e.get("metrics", {}) or {}
            row.algorithm = e.get("algorithm")
            row.artifact_path = e.get("path")
            row.n_samples = e.get("nSamples")
            row.accuracy = _f(m.get("accuracy"))
            row.f1 = _f(m.get("f1"))
            row.precision = _f(m.get("precision"))
            row.recall = _f(m.get("recall"))
            row.score_r2 = _f(m.get("scoreR2"))
            row.is_active = bool(e.get("active"))
            row.lineage = e.get("lineage")
            row.artifact_sha256 = e.get("sha256")
            row.artifact_signature = e.get("signature")
            row.trained_from_batches = e.get("batches")
            row.golden_accuracy = _f(e.get("goldenAccuracy"))
            row.promotion_note = e.get("promotionNote")
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("sync_model_versions failed")
    finally:
        session.close()


# ── ML-security records ────────────────────────────────────────────────────
def log_security_event(event_type: str, severity: str, source: str | None,
                       detail: dict | None = None) -> None:
    """Append-only audit line for a security check. Never raises."""
    if not db.DB_ENABLED:
        return
    session = get_session()
    if session is None:
        return
    try:
        session.add(SecurityEvent(
            event_type=event_type, severity=severity, source=source, detail=detail,
        ))
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("log_security_event failed")
    finally:
        session.close()


def save_dataset_batch(meta: dict) -> int | None:
    if not db.DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        row = DatasetBatch(
            file_name=meta.get("fileName"),
            file_sha256=meta.get("sha256"),
            uploaded_by=meta.get("uploadedBy"),
            rows_in=meta.get("rowsIn"),
            rows_accepted=meta.get("rowsAccepted"),
            rows_rejected=meta.get("rowsRejected"),
            rows_added=meta.get("rowsAdded"),
            rejection_reasons=meta.get("reasons"),
            distribution_check=meta.get("distributionCheck"),
            accepted=meta.get("accepted", True),
        )
        session.add(row)
        session.commit()
        return row.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_dataset_batch failed")
        return None
    finally:
        session.close()


def save_drift_snapshot(result: dict, model_version: int | None = None) -> int | None:
    if not db.DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        mv_id = None
        if model_version is not None:
            mv_id = session.scalar(select(ModelVersion.id).where(ModelVersion.version == model_version))
        row = DriftSnapshot(
            model_version_id=mv_id,
            status=result.get("status", "unknown"),
            overall_psi=_f(result.get("overallPsi")),
            reference_n=result.get("referenceN"),
            recent_n=result.get("recentN"),
            feature_psi=result.get("features"),
            prediction_drift=result.get("prediction"),
        )
        session.add(row)
        session.commit()
        return row.id
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("save_drift_snapshot failed")
        return None
    finally:
        session.close()


def all_feature_vectors(limit: int = 5000, order: str = "asc") -> list[dict]:
    """Every stored inference-run feature vector (oldest-first by default) —
    feeds the outlier profile and the drift reference/recent windows."""
    if not db.DB_ENABLED:
        return []
    session = get_session()
    if session is None:
        return []
    try:
        col = InferenceRun.id.asc() if order == "asc" else InferenceRun.id.desc()
        rows = session.execute(
            select(InferenceRun.feature_vector, InferenceRun.credit_score)
            .where(InferenceRun.feature_vector.is_not(None))
            .order_by(col).limit(limit)
        ).all()
        return [{"fv": r[0], "score": r[1]} for r in rows if isinstance(r[0], dict)]
    except Exception:  # noqa: BLE001
        logger.exception("all_feature_vectors failed")
        return []
    finally:
        session.close()


def recent_security_events(limit: int = 50) -> list[dict]:
    if not db.DB_ENABLED:
        return []
    session = get_session()
    if session is None:
        return []
    try:
        rows = session.scalars(
            select(SecurityEvent).order_by(SecurityEvent.id.desc()).limit(limit)
        ).all()
        return [{
            "id": r.id, "type": r.event_type, "severity": r.severity,
            "source": r.source, "detail": r.detail,
            "at": r.created_at.isoformat() if r.created_at else None,
        } for r in rows]
    except Exception:  # noqa: BLE001
        logger.exception("recent_security_events failed")
        return []
    finally:
        session.close()


def recent_dataset_batches(limit: int = 25) -> list[dict]:
    if not db.DB_ENABLED:
        return []
    session = get_session()
    if session is None:
        return []
    try:
        rows = session.scalars(
            select(DatasetBatch).order_by(DatasetBatch.id.desc()).limit(limit)
        ).all()
        return [{
            "id": r.id, "fileName": r.file_name, "sha256": (r.file_sha256 or "")[:16],
            "rowsIn": r.rows_in, "rowsAccepted": r.rows_accepted, "rowsRejected": r.rows_rejected,
            "rowsAdded": r.rows_added, "reasons": r.rejection_reasons,
            "distribution": r.distribution_check, "accepted": r.accepted,
            "at": r.ingested_at.isoformat() if r.ingested_at else None,
        } for r in rows]
    except Exception:  # noqa: BLE001
        logger.exception("recent_dataset_batches failed")
        return []
    finally:
        session.close()


def security_overview() -> dict | None:
    """Counts + latest status for the Security page."""
    if not db.DB_ENABLED:
        return None
    session = get_session()
    if session is None:
        return None
    try:
        from sqlalchemy import desc

        events = dict(session.execute(
            select(SecurityEvent.severity, func.count(SecurityEvent.id)).group_by(SecurityEvent.severity)
        ).all())
        outliers = session.scalar(
            select(func.count(InferenceRun.id)).where(InferenceRun.is_outlier.is_(True))
        ) or 0
        scored = session.scalar(select(func.count(InferenceRun.id))) or 0
        latest_drift = session.scalars(
            select(DriftSnapshot).order_by(desc(DriftSnapshot.id)).limit(1)
        ).first()
        batches = session.scalar(select(func.count(DatasetBatch.id))) or 0
        rejected = session.scalar(
            select(func.coalesce(func.sum(DatasetBatch.rows_rejected), 0))
        ) or 0
        unverified = session.scalar(
            select(func.count(ModelVersion.id)).where(ModelVersion.artifact_sha256.is_(None))
        ) or 0
        return {
            "events": {"info": events.get("info", 0), "warn": events.get("warn", 0), "block": events.get("block", 0)},
            "outlierRuns": int(outliers),
            "scoredRuns": int(scored),
            "drift": {
                "status": latest_drift.status if latest_drift else "not-computed",
                "overallPsi": latest_drift.overall_psi if latest_drift else None,
                "computedAt": latest_drift.computed_at.isoformat() if latest_drift and latest_drift.computed_at else None,
            },
            "datasetBatches": int(batches),
            "rowsRejectedAllTime": int(rejected),
            "unverifiedModels": int(unverified),
        }
    except Exception:  # noqa: BLE001
        logger.exception("security_overview failed")
        return None
    finally:
        session.close()
