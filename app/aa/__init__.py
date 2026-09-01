"""Account Aggregator (AA) domain — bank-statement underwriting.

Everything that turns an uploaded bank statement into a credit decision: the
multi-format parser, the feature-vector + credit-score engine, the per-statement
and per-loan-product BRE rule engines, the 4 session ML models + their registry,
the Model Hub pipeline, and the AA settings/deployment state. Mirrors app/gst/.
"""
from app.aa.router import router  # noqa: F401 — FastAPI router for main.py
