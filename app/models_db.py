# Relational schema for the BRE portal.
#
#   applicants ─┬─< statements ─┬─< transactions
#               │               └─< inference_runs ─┬─ bre_evaluations
#               │                                   ├─< anomalies
#               │                                   └─< analytics_months
#               └─< inference_runs
#   model_runs ──< trained_models   (one training run → its 4 session models)
#   model_deployments   (one row per model_id — current version + deploy state)
#   model_versions      (the population/dataset model registry — versioned)

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Applicant(Base):
    __tablename__ = "applicants"

    id: Mapped[int] = mapped_column(primary_key=True)
    ref_id: Mapped[str | None] = mapped_column(String(120), index=True)  # user-typed Application ID
    name: Mapped[str | None] = mapped_column(String(120))               # account holder
    bank_name: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    statements: Mapped[list["Statement"]] = relationship(back_populates="applicant", cascade="all, delete-orphan")
    inference_runs: Mapped[list["InferenceRun"]] = relationship(back_populates="applicant", cascade="all, delete-orphan")


class Statement(Base):
    __tablename__ = "statements"

    id: Mapped[int] = mapped_column(primary_key=True)
    applicant_id: Mapped[int] = mapped_column(ForeignKey("applicants.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[str] = mapped_column(String(64), index=True)       # e.g. "account_aggregator"
    file_name: Mapped[str | None] = mapped_column(String(255))
    file_format: Mapped[str | None] = mapped_column(String(16))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    cleanliness_percent: Mapped[int | None] = mapped_column(Integer)

    # ── data lineage — where this extraction came from ──────────────────
    file_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    parser_model: Mapped[str | None] = mapped_column(String(80))
    parser_version: Mapped[str | None] = mapped_column(String(40))
    parse_confidence: Mapped[float | None] = mapped_column(Float)   # 0-1
    parse_warnings: Mapped[list | None] = mapped_column(JSON)       # guardrail notes

    statement_months: Mapped[float | None] = mapped_column(Float)
    opening_balance: Mapped[float | None] = mapped_column(Numeric(16, 2))
    closing_balance: Mapped[float | None] = mapped_column(Numeric(16, 2))
    min_balance: Mapped[float | None] = mapped_column(Numeric(16, 2))
    max_balance: Mapped[float | None] = mapped_column(Numeric(16, 2))
    total_credit: Mapped[float | None] = mapped_column(Numeric(16, 2))
    total_debit: Mapped[float | None] = mapped_column(Numeric(16, 2))
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)

    summary: Mapped[dict | None] = mapped_column(JSON)                   # full parsed summary
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    applicant: Mapped["Applicant"] = relationship(back_populates="statements")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="statement", cascade="all, delete-orphan")
    inference_runs: Mapped[list["InferenceRun"]] = relationship(back_populates="statement")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    statement_id: Mapped[int] = mapped_column(ForeignKey("statements.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)                           # order within the statement
    txn_date: Mapped[str | None] = mapped_column(String(32))           # kept as printed on the statement
    narration: Mapped[str | None] = mapped_column(Text)
    txn_type: Mapped[str | None] = mapped_column(String(8))            # DEBIT / CREDIT
    amount: Mapped[float | None] = mapped_column(Numeric(16, 2))
    balance: Mapped[float | None] = mapped_column(Numeric(16, 2))

    statement: Mapped["Statement"] = relationship(back_populates="transactions")


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    applicant_id: Mapped[int | None] = mapped_column(ForeignKey("applicants.id", ondelete="SET NULL"), index=True)
    statement_id: Mapped[int | None] = mapped_column(ForeignKey("statements.id", ondelete="SET NULL"), index=True)

    model_id: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(16))
    data_source: Mapped[str] = mapped_column(String(32))               # UPLOADED_STATEMENT / SIMULATED

    credit_score: Mapped[int | None] = mapped_column(Integer)
    probability_of_default: Mapped[float | None] = mapped_column(Float)
    risk_grade: Mapped[str | None] = mapped_column(String(16))
    decision: Mapped[str | None] = mapped_column(String(32))
    anomaly_count: Mapped[int] = mapped_column(Integer, default=0)
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)

    # Credit Score tab breakdown — the scorecard vs. ML-model split
    scorecard_score: Mapped[int | None] = mapped_column(Integer)
    model_score: Mapped[int | None] = mapped_column(Integer)
    model_source: Mapped[str | None] = mapped_column(String(32))   # registry vN / session / none
    ml_blended: Mapped[bool | None] = mapped_column(Boolean)

    # ── ML-security signals ────────────────────────────────────────────
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), index=True
    )
    outlier_score: Mapped[float | None] = mapped_column(Float)     # 0-100
    is_outlier: Mapped[bool | None] = mapped_column(Boolean)
    outlier_flags: Mapped[list | None] = mapped_column(JSON)
    guardrail_status: Mapped[str | None] = mapped_column(String(16))  # ok / warn / blocked
    guardrail_warnings: Mapped[list | None] = mapped_column(JSON)

    feature_vector: Mapped[dict | None] = mapped_column(JSON)
    bre_payload: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    applicant: Mapped["Applicant"] = relationship(back_populates="inference_runs")
    statement: Mapped["Statement"] = relationship(back_populates="inference_runs")
    bre_evaluation: Mapped["BreEvaluation"] = relationship(
        back_populates="inference_run", cascade="all, delete-orphan", uselist=False,
    )
    anomalies: Mapped[list["Anomaly"]] = relationship(
        back_populates="inference_run", cascade="all, delete-orphan",
    )
    analytics_months: Mapped[list["AnalyticsMonth"]] = relationship(
        back_populates="inference_run", cascade="all, delete-orphan",
    )


class BreEvaluation(Base):
    __tablename__ = "bre_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    inference_run_id: Mapped[int] = mapped_column(ForeignKey("inference_runs.id", ondelete="CASCADE"), index=True)

    decision: Mapped[str | None] = mapped_column(String(32))
    applicant_profile: Mapped[str | None] = mapped_column(String(32))
    credit_score: Mapped[int | None] = mapped_column(Integer)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    enabled_count: Mapped[int] = mapped_column(Integer, default=0)

    results: Mapped[list | None] = mapped_column(JSON)                  # per-rule PASS/FAIL/SKIP
    serious_flags: Mapped[list | None] = mapped_column(JSON)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    inference_run: Mapped["InferenceRun"] = relationship(back_populates="bre_evaluation")


class Anomaly(Base):
    """One flagged transaction — the Anomalies tab on the Model Testing page."""
    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(primary_key=True)
    inference_run_id: Mapped[int] = mapped_column(ForeignKey("inference_runs.id", ondelete="CASCADE"), index=True)

    txn_date: Mapped[str | None] = mapped_column(String(32))
    narration: Mapped[str | None] = mapped_column(Text)
    amount: Mapped[float | None] = mapped_column(Numeric(16, 2))
    score_percent: Mapped[float | None] = mapped_column(Float)          # 0-100 anomaly score
    level: Mapped[str | None] = mapped_column(String(16))              # HIGH / MEDIUM
    reasons: Mapped[str | None] = mapped_column(Text)                  # "; "-joined

    inference_run: Mapped["InferenceRun"] = relationship(back_populates="anomalies")


class AnalyticsMonth(Base):
    """One calendar-month bucket — the Analytics tab on the Model Testing page."""
    __tablename__ = "analytics_months"

    id: Mapped[int] = mapped_column(primary_key=True)
    inference_run_id: Mapped[int] = mapped_column(ForeignKey("inference_runs.id", ondelete="CASCADE"), index=True)

    month: Mapped[str] = mapped_column(String(24))                     # "Feb 2026" etc.
    seq: Mapped[int] = mapped_column(Integer, default=0)               # order within the statement
    inflow: Mapped[float | None] = mapped_column(Numeric(16, 2))
    outflow: Mapped[float | None] = mapped_column(Numeric(16, 2))
    net_cashflow: Mapped[float | None] = mapped_column(Numeric(16, 2))
    adb_score: Mapped[float | None] = mapped_column(Numeric(16, 2))    # average daily balance
    min_balance: Mapped[float | None] = mapped_column(Numeric(16, 2))
    pd_risk_percent: Mapped[float | None] = mapped_column(Float)
    credit_score: Mapped[int | None] = mapped_column(Integer)

    inference_run: Mapped["InferenceRun"] = relationship(back_populates="analytics_months")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    dataset_file: Mapped[str | None] = mapped_column(String(255))
    tx_count: Mapped[int | None] = mapped_column(Integer)
    real_features: Mapped[dict | None] = mapped_column(JSON)
    models: Mapped[list | None] = mapped_column(JSON)                   # per-model accuracy metadata
    evaluations: Mapped[dict | None] = mapped_column(JSON)              # real CV results
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    trained_models: Mapped[list["TrainedModel"]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan",
    )


class TrainedModel(Base):
    """One session model from a training run — the 4 model cards + the
    Model Evaluation table on the Model Hub page."""
    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)

    model_id: Mapped[str] = mapped_column(String(64), index=True)      # risk_model, cashflow_model, …
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str | None] = mapped_column(String(255))
    algorithm: Mapped[str | None] = mapped_column(String(64))

    accuracy: Mapped[float | None] = mapped_column(Float)             # headline metric, 0-100
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1: Mapped[float | None] = mapped_column(Float)
    brier_score: Mapped[float | None] = mapped_column(Float)
    error_rate: Mapped[float | None] = mapped_column(Float)

    cv_folds: Mapped[int | None] = mapped_column(Integer)
    sample_count: Mapped[int | None] = mapped_column(Integer)
    is_live: Mapped[bool] = mapped_column(Boolean, default=True)
    metric_meta: Mapped[dict | None] = mapped_column(JSON)            # per-metric display labels
    cv_detail: Mapped[list | None] = mapped_column(JSON)             # per-fold rows
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    model_run: Mapped["ModelRun"] = relationship(back_populates="trained_models")


class ModelDeployment(Base):
    """Current version + deploy state for one model — the Model Version &
    Deployment Management Table. Exactly one row per model_id."""
    __tablename__ = "model_deployments"
    __table_args__ = (UniqueConstraint("model_id", name="uq_model_deployment_model_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(80))
    selected_version: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="Ready")  # Deployed / Ready
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class ModelVersion(Base):
    """The population/dataset model registry — the 'Population Model vN' row and
    the version history on the AI Architecture page. One row per trained version."""
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("version", name="uq_model_version_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(Integer, index=True)
    algorithm: Mapped[str | None] = mapped_column(String(64))
    artifact_path: Mapped[str | None] = mapped_column(String(255))
    n_samples: Mapped[int | None] = mapped_column(Integer)

    accuracy: Mapped[float | None] = mapped_column(Float)
    f1: Mapped[float | None] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    score_r2: Mapped[float | None] = mapped_column(Float)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    lineage: Mapped[list | None] = mapped_column(JSON)               # prior versions this builds on
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    # ── ML-security: artifact integrity + training-data provenance ──────
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    artifact_signature: Mapped[str | None] = mapped_column(String(64))
    trained_from_batches: Mapped[list | None] = mapped_column(JSON)   # dataset_batches ids
    golden_accuracy: Mapped[float | None] = mapped_column(Float)      # acc on the frozen golden set
    promotion_note: Mapped[str | None] = mapped_column(Text)          # why it was / wasn't promoted


class DatasetBatch(Base):
    """One ingested training CSV — the data-lineage + poisoning audit record."""
    __tablename__ = "dataset_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_name: Mapped[str | None] = mapped_column(String(255))
    file_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    uploaded_by: Mapped[str | None] = mapped_column(String(120))

    rows_in: Mapped[int | None] = mapped_column(Integer)
    rows_accepted: Mapped[int | None] = mapped_column(Integer)
    rows_rejected: Mapped[int | None] = mapped_column(Integer)
    rows_added: Mapped[int | None] = mapped_column(Integer)          # net new after dedupe
    rejection_reasons: Mapped[dict | None] = mapped_column(JSON)
    distribution_check: Mapped[dict | None] = mapped_column(JSON)    # PSI vs. existing corpus
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class DriftSnapshot(Base):
    """One concept-drift computation — recent scored applicants vs. a reference
    window. Surfaced on the Security page; an 'alert' is the retrain trigger."""
    __tablename__ = "drift_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(24))                  # stable / warn / alert
    overall_psi: Mapped[float | None] = mapped_column(Float)
    reference_n: Mapped[int | None] = mapped_column(Integer)
    recent_n: Mapped[int | None] = mapped_column(Integer)
    feature_psi: Mapped[list | None] = mapped_column(JSON)
    prediction_drift: Mapped[dict | None] = mapped_column(JSON)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class SecurityEvent(Base):
    """Append-only audit log for every ML-security check that fired."""
    __tablename__ = "security_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)  # guardrail / outlier / poisoning / drift / integrity
    severity: Mapped[str] = mapped_column(String(16))               # info / warn / block
    source: Mapped[str | None] = mapped_column(String(120))         # endpoint / file / model
    detail: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
