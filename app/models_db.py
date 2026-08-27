# Relational schema for the BRE portal.
#
#   applicants ─┬─< statements ─┬─< transactions
#               │               └─< inference_runs ─── bre_evaluations
#               └─< inference_runs
#   model_runs  (standalone — one row per Model Hub training run)

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
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

    feature_vector: Mapped[dict | None] = mapped_column(JSON)
    bre_payload: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    applicant: Mapped["Applicant"] = relationship(back_populates="inference_runs")
    statement: Mapped["Statement"] = relationship(back_populates="inference_runs")
    bre_evaluation: Mapped["BreEvaluation"] = relationship(
        back_populates="inference_run", cascade="all, delete-orphan", uselist=False,
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
