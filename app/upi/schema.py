"""UPI Transaction Data Enrichment — canonical field names, model targets, and
the feature-vector builder. Twin of app.gst.schema / app.bbps.schema.

One uploaded file = one PERSON's UPI transaction history (not a merchant's QR
collection ledger — see app/upi/__init__.py's docstring for why).
"""

from __future__ import annotations

# ── raw transaction record fields (what parser.py extracts per row) ────────
CANONICAL = {
    "transaction_id", "date", "time", "type", "amount",
    "payer_vpa", "payer_name", "payee_vpa", "payee_name",
    "mode", "mcc", "status", "remarks",
}

# ── four prediction targets — weak-supervision, not real ground truth (see
# model.py's _compute_targets docstring for the documented formulas). ──────
RISK_TARGET = "upi_transaction_risk_flag"          # LOW / MEDIUM / HIGH classification
RELIABILITY_TARGET = "upi_payment_reliability_score"  # 0-100 regression — success-rate only
BEHAVIOUR_TARGET = "upi_spend_behaviour"            # REGULAR / IRREGULAR classification
STABILITY_TARGET = "upi_network_stability_score"    # 0-100 regression — payee/payer diversity + tenure

RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]
BEHAVIOUR_ORDER = ["IRREGULAR", "REGULAR"]

# High-risk Merchant Category Codes — gambling, quasi-cash/money-transfer
# agents, pawn shops. Real MCC assignments (ISO 18245), not invented.
HIGH_RISK_MCC = {"7995", "6051", "5933", "7800", "6211"}

FEATURES = [
    "total_transactions",
    "span_months",
    "p2p_ratio",
    "p2m_ratio",
    "success_ratio",
    "failed_ratio",
    "unique_payees",
    "unique_payers",
    "recurring_payee_count",
    "daily_avg_transactions",
    "weekend_spend_pct",
    "avg_ticket_size",
    "high_risk_mcc_spend_pct",
    "p2p_debit_velocity",
    "p2p_credit_velocity",
]


def feature_vector_from_analysis(result: dict) -> dict | None:
    """analyze_upi()'s output -> the flat numeric feature row the model
    trains/predicts on. None when the file had no usable UPI transactions."""
    if not result or not result.get("available"):
        return None
    return {
        "total_transactions":       result.get("totalTransactions", 0),
        "span_months":               result.get("spanMonths", 1),
        "p2p_ratio":                 result.get("p2pRatio", 0.0),
        "p2m_ratio":                 result.get("p2mRatio", 0.0),
        "success_ratio":             result.get("successRatio", 1.0),
        "failed_ratio":              result.get("failedRatio", 0.0),
        "unique_payees":             result.get("uniquePayees", 0),
        "unique_payers":             result.get("uniquePayers", 0),
        "recurring_payee_count":     result.get("recurringPayeeCount", 0),
        "daily_avg_transactions":    result.get("dailyAvgTransactions", 0.0),
        "weekend_spend_pct":         result.get("weekendSpendPct", 0.0),
        "avg_ticket_size":           result.get("avgTicketSize", 0.0),
        "high_risk_mcc_spend_pct":   result.get("highRiskMccSpendPct", 0.0),
        "p2p_debit_velocity":        result.get("p2pDebitVelocity", 0.0),
        "p2p_credit_velocity":       result.get("p2pCreditVelocity", 0.0),
    }
