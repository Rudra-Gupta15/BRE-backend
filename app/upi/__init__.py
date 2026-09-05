"""UPI Transaction Data Enrichment subsystem — self-contained.

    app/upi/
      parser.py     — parse_upi(bytes, filename) -> list[real transaction dict]
                       (a DEDICATED parser, unlike BBPS — UPI's structured
                       fields (VPA, MCC, mode) don't exist in a generic bank
                       statement narration, so this can't reuse app.aa.parser)
      schema.py      — canonical field names / feature names / targets
      analysis.py    — analyze_upi(transactions) -> real per-applicant
                        signals (P2P/P2M mix, reliability, network diversity,
                        MCC risk exposure, weekend/weekday spend split)
      model.py       — train / predict / status: 4 real heads (UPI Transaction
                        Risk Model, Payment Reliability Score, Spend Behaviour
                        Model, Network Stability Score), weak-supervision
                        labels — see its module docstring
      rules.py       — evaluate_upi_rules() -> the 16 catalogue rules' real
                        PASS/FAIL/SKIP verdicts + evaluate()/payload() for the
                        Model Testing "BRE payload" tab
      patterns.py    — Fraud & Anomaly Pattern detection (twin of
                        app.gst.patterns / app.bbps.patterns) — 2 trained
                        classifiers + corpus-wide typology scan
      service.py     — ingest_upi_file() (Model Hub upload path) + Model Hub
                        training/evaluation shaping (twin of app.gst.service)
      router.py      — FastAPI router, mounted at /api/upi

One uploaded file = one PERSON's UPI transaction history (P2P transfers +
P2M/QR merchant payments mixed together), not a merchant's own QR-collection
settlement ledger — see app.aa.routes.pipeline for where ingestion hooks in
for source_id == "upi_enrichment".
"""

from app.upi import analysis, model, parser, patterns, rules, schema, service
from app.upi.router import router as api_router

__all__ = ["analysis", "model", "parser", "patterns", "rules", "schema", "service", "api_router"]
