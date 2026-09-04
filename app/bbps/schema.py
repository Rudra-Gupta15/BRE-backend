"""BBPS model feature/target names — single source of truth shared by
model.py and the feature-vector builder in analysis.py's caller."""

# Four prediction targets — weak-supervision, not real ground truth (see
# model.py's _compute_targets docstring for the documented formulas). Each
# target weights the same real features differently, so each head learns a
# genuinely distinct signal instead of four copies of the same score.
RISK_TARGET = "bbps_risk_flag"                          # LOW / MEDIUM / HIGH classification
DISCIPLINE_TARGET = "payment_discipline_score"           # 0-100 regression — punctuality only
BEHAVIOUR_TARGET = "bill_payment_behaviour"               # REGULAR / IRREGULAR classification
STABILITY_TARGET = "utility_expense_stability_score"      # 0-100 regression — diversity + tenure

RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]
BEHAVIOUR_ORDER = ["IRREGULAR", "REGULAR"]

FEATURES = [
    "utility_accounts",
    "payments_count",
    "span_months",
    "on_time_payment_ratio",
    "missed_payment_count",
    "average_bill_amount",
    "has_electricity",
    "has_water",
    "has_gas",
    "has_broadband",
    "has_mobile_dth",
    "recurring_type_count",
]


def feature_vector_from_analysis(result: dict) -> dict | None:
    """analyze_bbps()'s output -> the flat numeric feature row the model
    trains/predicts on. None when the statement had no BBPS activity at all
    (nothing to score)."""
    if not result or not result.get("available"):
        return None
    by_type = {r["utilityType"]: r for r in result.get("byType", [])}
    return {
        "utility_accounts": result.get("utilityAccounts", 0),
        "payments_count": result.get("paymentsLast12m", 0),
        "span_months": result.get("spanMonths", 1),
        "on_time_payment_ratio": result.get("onTimePaymentRatio", 0.0),
        "missed_payment_count": result.get("missedPaymentCount", 0),
        "average_bill_amount": result.get("averageBillAmount", 0.0),
        "has_electricity": 1 if "ELECTRICITY" in by_type else 0,
        "has_water": 1 if "WATER" in by_type else 0,
        "has_gas": 1 if "GAS" in by_type else 0,
        "has_broadband": 1 if "BROADBAND" in by_type else 0,
        "has_mobile_dth": 1 if "MOBILE_DTH" in by_type else 0,
        "recurring_type_count": sum(1 for r in by_type.values() if r.get("recurring")),
    }
