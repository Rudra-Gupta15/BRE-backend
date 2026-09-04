"""BBPS Utility Payment History subsystem — self-contained.

    app/bbps/
      analysis.py   — analyze_bbps(transactions) -> real per-applicant
                       utility-payment signals (punctuality, missed
                       payments, per-type averages)
      schema.py     — feature/target names + feature_vector_from_analysis()
      model.py      — train / predict / status: 4 real heads (Utility Payment
                       Risk Model, Payment Discipline Score, Bill Payment
                       Behaviour Model, Utility Expense Stability Model),
                       weak-supervision labels — see its module docstring
      rules.py      — evaluate_bbps_rules() -> the 16 catalogue rules'
                       real PASS/FAIL/SKIP verdicts
      patterns.py   — Fraud & Anomaly Pattern detection (twin of
                       app.aa.patterns / app.gst.patterns) — 2 trained
                       classifiers + corpus-wide typology scan
      service.py    — Model Hub training/evaluation shaping (twin of
                       app.gst.service) — what app.aa.routes.models calls
                       into for sourceId == "bbps_utility"
      router.py     — FastAPI router, mounted at /api/bbps

No dedicated parser: BBPS payments are line items *inside* a real bank
statement (not a separate feed), so ingestion reuses app.aa.parser's existing
PDF/CSV/JSON/XLSX/MD statement parsing — app.aa.routes.pipeline calls into
this package when source_id=="bbps_utility".
"""

from app.bbps import model, patterns, rules, schema, service
from app.bbps.analysis import analyze_bbps, extract_bbps_transactions
from app.bbps.router import router as api_router

__all__ = ["analyze_bbps", "extract_bbps_transactions", "model", "patterns", "rules", "schema",
           "service", "api_router"]
