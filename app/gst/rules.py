"""GST BRE rule evaluation for the Model Testing "BRE payload" tab.

Evaluates the computable subset of the `gst_data` rule catalogue
(data_source_rules.py) against one scored GST business — the profile record the
model saw plus the 4 head predictions. Same result shape as
`/bre-products/evaluate`.
"""
from __future__ import annotations

from app.common.source_rules import DATA_SOURCE_RULES
from app.aa.product_state import bre_product_state
from app.aa.product_source_state import product_source_rule_state


def _f(v, d: float = 0.0) -> float:
    try:
        f = float(str(v).replace(",", "").replace("₹", "").replace("%", "").strip())
        return f if f == f else d  # str(nan) parses back to nan without raising — catch it here
    except (TypeError, ValueError):
        return d


def _P(detail: str):
    return "PASS", detail


def _F(detail: str):
    return "FAIL", detail


def _S(detail: str):
    return "SKIP", detail


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


# label -> (serious?, fn(profile, heads) -> (status, detail))
_RULES: dict[str, tuple[bool, object]] = {
    "GSTIN Validation Rule": (False, lambda p, h: (
        _P(f"GSTIN {p.get('gstin') or p.get('customer_id')} present and well-formed.")
        if (p.get("gstin") or p.get("customer_id")) else _F("No GSTIN on the record."))),
    "GST Registration Status Rule": (True, lambda p, h: (
        _P("Registration is ACTIVE.") if str(p.get("gst_status", "")).upper() == "ACTIVE"
        else _F(f"Registration status is {p.get('gst_status') or 'UNKNOWN'}."))),
    "GST Registration Vintage Rule": (False, lambda p, h: (
        _P(f"GST vintage {_f(p.get('business_vintage_years')):.1f} yrs (≥ 1).")
        if _f(p.get("business_vintage_years")) >= 1
        else _F(f"GST vintage {_f(p.get('business_vintage_years')):.1f} yrs (< 1)."))),
    "GST Return Filing Regularity Rule": (True, lambda p, h: (
        _P(f"Filing regularity {_f(p.get('filing_regularity_percentage')):.0f}% (≥ 80%).")
        if _f(p.get("filing_regularity_percentage")) >= 80
        else _F(f"Filing regularity {_f(p.get('filing_regularity_percentage')):.0f}% (< 80%)."))),
    "GST Filing Delay Rule": (False, lambda p, h: (
        _P(f"Late returns: {int(_f(p.get('late_return_count')))} (≤ 3).")
        if _f(p.get("late_return_count")) <= 3
        else _F(f"Late returns: {int(_f(p.get('late_return_count')))} (> 3)."))),
    "Missed Return Rule": (True, lambda p, h: (
        _P("No missed GST returns.") if _f(p.get("missed_return_count")) == 0
        else _F(f"{int(_f(p.get('missed_return_count')))} missed GST return(s)."))),
    "Late Return Rule": (False, lambda p, h: (
        _P(f"{int(_f(p.get('late_return_count')))} late return(s) (≤ 2).")
        if _f(p.get("late_return_count")) <= 2
        else _F(f"{int(_f(p.get('late_return_count')))} late return(s) (> 2)."))),
    "GSTR-1 vs GSTR-3B Sales Matching Rule": (False, lambda p, h: (
        _S("GSTR-1 vs 3B mismatch not available (summary file).")
        if p.get("gstr1_vs_gstr3b_mismatch_pct") is None
        else _P(f"GSTR-1 vs 3B mismatch {_f(p.get('gstr1_vs_gstr3b_mismatch_pct')):.1f}% (≤ 10%).")
        if _f(p.get("gstr1_vs_gstr3b_mismatch_pct")) <= 10
        else _F(f"GSTR-1 vs 3B mismatch {_f(p.get('gstr1_vs_gstr3b_mismatch_pct')):.1f}% (> 10%)."))),
    "GST Turnover Validation Rule": (False, lambda p, h: (
        _P(f"Annualised GST turnover {_f(p.get('annualised_gst_turnover')):,.0f}.")
        if _f(p.get("annualised_gst_turnover")) > 0 else _F("Annualised GST turnover is zero."))),
    "Monthly Turnover Minimum Rule": (False, lambda p, h: (
        _P(f"Monthly turnover ₹{_f(p.get('monthly_turnover')):,.0f} (≥ ₹50k).")
        if _f(p.get("monthly_turnover")) >= 50_000
        else _F(f"Monthly turnover ₹{_f(p.get('monthly_turnover')):,.0f} (< ₹50k)."))),
    "Quarterly Turnover Minimum Rule": (False, lambda p, h: (
        _P(f"Quarterly turnover ₹{_f(p.get('quarterly_turnover')):,.0f} (≥ ₹1.5L).")
        if _f(p.get("quarterly_turnover")) >= 150_000
        else _F(f"Quarterly turnover ₹{_f(p.get('quarterly_turnover')):,.0f} (< ₹1.5L)."))),
    "Year-on-Year Turnover Growth Rule": (False, lambda p, h: (
        _P(f"YoY turnover growth {_f(p.get('turnover_growth_yoy')):.1f}% (≥ -20%).")
        if _f(p.get("turnover_growth_yoy")) >= -20
        else _F(f"YoY turnover growth {_f(p.get('turnover_growth_yoy')):.1f}% (< -20%)."))),
    "Turnover Decline Rule": (True, lambda p, h: (
        _P(f"Turnover decline {_f(p.get('turnover_decline_percentage')):.1f}% (≤ 25%).")
        if _f(p.get("turnover_decline_percentage")) <= 25
        else _F(f"Turnover decline {_f(p.get('turnover_decline_percentage')):.1f}% (> 25%)."))),
    "Consecutive Declining Quarters Rule": (False, lambda p, h: (
        _P(f"{int(_f(p.get('consecutive_declining_quarters')))} declining quarter(s) (≤ 2).")
        if _f(p.get("consecutive_declining_quarters")) <= 2
        else _F(f"{int(_f(p.get('consecutive_declining_quarters')))} declining quarter(s) (> 2)."))),
    "ITC Claim Ratio Rule": (False, lambda p, h: (
        _P(f"ITC claimed ÷ available = {_ratio(_f(p.get('itc_claimed_amount')), _f(p.get('itc_available_amount'))):.2f} (≤ 1.05).")
        if _ratio(_f(p.get("itc_claimed_amount")), _f(p.get("itc_available_amount"))) <= 1.05
        else _F(f"ITC claimed ÷ available = {_ratio(_f(p.get('itc_claimed_amount')), _f(p.get('itc_available_amount'))):.2f} (> 1.05)."))),
    "ITC-to-Turnover Ratio Rule": (False, lambda p, h: (
        _P(f"Net ITC ÷ monthly turnover = {_ratio(_f(p.get('net_itc_amount')), _f(p.get('monthly_turnover'))):.2f} (≤ 0.6).")
        if _ratio(_f(p.get("net_itc_amount")), _f(p.get("monthly_turnover"))) <= 0.6
        else _F(f"Net ITC ÷ monthly turnover = {_ratio(_f(p.get('net_itc_amount')), _f(p.get('monthly_turnover'))):.2f} (> 0.6)."))),
    "Top Buyer Concentration Rule": (False, lambda p, h: (
        _P(f"Top buyer {_f(p.get('top_buyer_sales_percentage')):.0f}% of sales (≤ 50%).")
        if _f(p.get("top_buyer_sales_percentage")) <= 50
        else _F(f"Top buyer {_f(p.get('top_buyer_sales_percentage')):.0f}% of sales (> 50%)."))),
    "Buyer Concentration Level Rule": (False, lambda p, h: (
        _F(f"Buyer concentration {p.get('buyer_concentration_level')}.")
        if str(p.get("buyer_concentration_level", "")).upper() == "HIGH"
        else _P(f"Buyer concentration {p.get('buyer_concentration_level') or 'LOW'}."))),
    "GST Turnover-to-Loan Ratio Rule": (False, lambda p, h: (
        _S("No proposed loan on the profile.")
        if _f(p.get("gst_turnover_to_loan_ratio")) == 0
        else _P(f"GST turnover ÷ loan = {_f(p.get('gst_turnover_to_loan_ratio')):.2f} (≥ 1.0).")
        if _f(p.get("gst_turnover_to_loan_ratio")) >= 1.0
        else _F(f"GST turnover ÷ loan = {_f(p.get('gst_turnover_to_loan_ratio')):.2f} (< 1.0)."))),
    "Maximum Loan Eligibility by GST Turnover Rule": (False, lambda p, h: (
        _P(f"Max GST-rule loan ≈ ₹{_f((h.get('gst_loan_eligibility_model') or {}).get('value')):,.0f}.")
        if _f((h.get("gst_loan_eligibility_model") or {}).get("value")) > 0
        else _F("GST-rule loan eligibility is zero."))),
    "GST Data Completeness Rule": (False, lambda p, h: (
        _P(f"Data completeness {_f(p.get('gst_data_completeness_score')) * 100:.0f}% (≥ 70%).")
        if _f(p.get("gst_data_completeness_score")) >= 0.7
        else _F(f"Data completeness {_f(p.get('gst_data_completeness_score')) * 100:.0f}% (< 70%)."))),
    "GST Risk Flag Rule": (True, lambda p, h: (
        _F("GST model flags this profile HIGH risk.")
        if str((h.get("gst_risk_flag_model") or {}).get("label", "")).upper() == "HIGH"
        else _P(f"GST risk flag: {(h.get('gst_risk_flag_model') or {}).get('label', 'LOW')}."))),
    "GST Underwriting Score Rule": (True, lambda p, h: (
        _P(f"GST underwriting score {_f((h.get('gst_underwriting_score_model') or {}).get('value')):.0f} (≥ 55).")
        if _f((h.get("gst_underwriting_score_model") or {}).get("value")) >= 55
        else _F(f"GST underwriting score {_f((h.get('gst_underwriting_score_model') or {}).get('value')):.0f} (< 55)."))),
    "GST Filing Behaviour Risk Rule": (False, lambda p, h: (
        _P(f"Predicted on-time filing {_f((h.get('gst_filing_compliance_model') or {}).get('value')):.0f}% (≥ 75%).")
        if _f((h.get("gst_filing_compliance_model") or {}).get("value")) >= 75
        else _F(f"Predicted on-time filing {_f((h.get('gst_filing_compliance_model') or {}).get('value')):.0f}% (< 75%)."))),
    "GST-Based Auto-Rejection Rule": (True, lambda p, h: (
        _F("Auto-reject: score < 45 or HIGH risk.")
        if (_f((h.get("gst_underwriting_score_model") or {}).get("value")) < 45
            or str((h.get("gst_risk_flag_model") or {}).get("label", "")).upper() == "HIGH")
        else _P("No auto-rejection trigger."))),
    "GST-Based Auto-Approval Rule": (False, lambda p, h: (
        _P("Auto-approve: score ≥ 75, LOW risk, no missed returns.")
        if (_f((h.get("gst_underwriting_score_model") or {}).get("value")) >= 75
            and str((h.get("gst_risk_flag_model") or {}).get("label", "")).upper() == "LOW"
            and _f(p.get("missed_return_count")) == 0)
        else _S("Auto-approval conditions not all met — manual/rule review."))),
}


def _label_to_id() -> dict[str, str]:
    return {r["label"]: r["id"] for r in DATA_SOURCE_RULES.get("gst_data", [])}


def evaluate(profile: dict, heads: dict, *, product_id: str | None = None) -> dict:
    """Run the computable gst_data rules against one scored business."""
    product_id = product_id or bre_product_state.active_product
    enabled_map = product_source_rule_state.for_ps(product_id, "gst_data") if product_id else {}
    lid = _label_to_id()

    results: list[dict] = []
    passed = failed = skipped = 0
    serious_flags: list[str] = []

    for label, rid in lid.items():
        if enabled_map and enabled_map.get(rid) is False:
            continue  # rule switched off for this product
        spec = _RULES.get(label)
        if not spec:
            results.append({"id": rid, "label": label, "status": "SKIP", "serious": False,
                            "detail": "No evaluator — needs the full GST rule engine."})
            skipped += 1
            continue
        serious, fn = spec
        try:
            status, detail = fn(profile, heads)
        except Exception as exc:  # noqa: BLE001
            status, detail = "SKIP", f"Could not evaluate ({exc})."
        results.append({"id": rid, "label": label, "status": status,
                        "serious": serious, "detail": detail})
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
            if serious:
                serious_flags.append(label)
        else:
            skipped += 1

    score = _f((heads.get("gst_underwriting_score_model") or {}).get("value"))
    if serious_flags:
        decision = "REJECTED"
    elif failed > 0 or score < 55:
        decision = "CONDITIONAL APPROVAL"
    elif score >= 75:
        decision = "APPROVED"
    else:
        decision = "APPROVED WITH NOTES"

    return {
        "available": True,
        "decision": decision,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "enabledCount": passed + failed + skipped,
        "seriousFlags": serious_flags,
        "creditScore": round(score),
        "gateThreshold": 55,
        "applicantProfile": "GST-only",
        "productName": None,
        "results": results,
    }


def payload(profile: dict, heads: dict, prediction: dict) -> dict:
    """The JSON blob shown in the BRE Output Payload panel."""
    keys = [
        "gstin", "customer_id", "gst_status", "filing_frequency",
        "annualised_gst_turnover", "monthly_turnover", "quarterly_turnover",
        "total_taxable_turnover", "turnover_growth_yoy", "turnover_growth_qoq",
        "turnover_decline_percentage", "consecutive_declining_quarters",
        "filing_regularity_percentage", "missed_return_count", "late_return_count",
        "gstr1_vs_gstr3b_mismatch_pct", "net_itc_amount", "itc_claimed_amount",
        "itc_available_amount", "top_buyer_sales_percentage", "unique_buyer_count",
        "buyer_concentration_level", "business_vintage_years",
        "gst_turnover_to_loan_ratio", "gst_data_completeness_score",
    ]
    return {
        "source": "gst_data",
        "gstin": profile.get("gstin") or profile.get("customer_id"),
        "profile": {k: profile[k] for k in keys if k in profile},
        "modelOutputs": {
            "underwritingScore": prediction.get("underwritingScore"),
            "riskFlag": prediction.get("riskFlag"),
            "riskProbability": prediction.get("riskProbability"),
            "loanEligibility": (heads.get("gst_loan_eligibility_model") or {}).get("value"),
            "filingCompliancePct": (heads.get("gst_filing_compliance_model") or {}).get("value"),
            "modelVersion": prediction.get("modelVersion"),
        },
        "topFactors": prediction.get("topFactors", []),
    }
