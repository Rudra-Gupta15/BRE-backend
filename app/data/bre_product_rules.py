# Per-loan-product BRE rule catalogue + evaluators.
#
# EVERY rule from every product is in the catalogue. Rules that can be computed
# from what we have — the uploaded bank statement, the derived feature vector
# and the credit score / PD / grade — have a real evaluator and are ENABLED by
# default. Rules that need an external feed we don't ingest (property valuation,
# GST-portal filing data, bureau DPD, dealer / OEM / RC KYC, machine invoices…)
# are still listed but are DISABLED by default and evaluate to SKIP with the
# reason, so an underwriter can see the full rule set and turn them on later.
#
# Evaluable rule: fn(c, fv, risk) -> ("PASS" | "FAIL" | "SKIP", detail_str)
#   c   = bre_engine._build_context(...)   (~50 derived quantities)
#   fv  = feature vector
#   risk = { score, pd, grade, decision, ... }

from app.services.bre_engine import _F, _need, _P, _ratio, _rupees
from app.state.settings_state import settings_state


def _GATE() -> int:
    return settings_state.scoring.gate_threshold


def _pd(risk: dict) -> float:
    try:
        v = float(risk.get("pd"))
    except (TypeError, ValueError):
        return 0.0
    return v * 100 if v <= 1 else v


def _monthly_surplus(c: dict) -> float:
    return (c["total_credit"] - c["total_debit"]) / max(c["n_months"], 1)


# ── Evaluable rules (real logic, enabled by default) ───────────────────────
# {id: {label, fn, serious?, business_only?, gate?}}

RULES: dict[str, dict] = {
    "min_income": {"label": "Minimum Monthly Income Rule", "fn": lambda c, fv, r: (
        _P(f"Avg monthly inflow {_rupees(fv['avg_monthly_inflow'])} (≥ ₹15,000).")
        if fv["avg_monthly_inflow"] >= 15000
        else _F(f"Avg monthly inflow {_rupees(fv['avg_monthly_inflow'])} (< ₹15,000)."))},
    "min_monthly_turnover": {"label": "Minimum Monthly Turnover Rule", "fn": lambda c, fv, r: (
        _P(f"Monthly credit turnover {_rupees(fv['avg_monthly_inflow'])} (≥ ₹50,000).")
        if fv["avg_monthly_inflow"] >= 50000
        else _F(f"Monthly credit turnover {_rupees(fv['avg_monthly_inflow'])} (< ₹50,000)."))},
    "min_annual_turnover": {"label": "Minimum Annual Turnover Rule", "fn": lambda c, fv, r: (
        _P(f"Annualised turnover ≈ {_rupees(fv['avg_monthly_inflow'] * 12)} (≥ ₹10L).")
        if fv["avg_monthly_inflow"] * 12 >= 1_000_000
        else _F(f"Annualised turnover ≈ {_rupees(fv['avg_monthly_inflow'] * 12)} (< ₹10L)."))},
    "income_stability": {"label": "Income Stability Rule", "fn": lambda c, fv, r: (
        _P(f"Income stability {fv['income_stability'] * 100:.0f}% (≥ 60%).") if fv["income_stability"] >= 0.60
        else _F(f"Income stability {fv['income_stability'] * 100:.0f}% (< 60%)."))},
    "income_validation": {"label": "Bank Statement Income Validation Rule", "fn": lambda c, fv, r: (
        _P(f"{c['salary_in_count']} recurring salary credit(s) identified.") if c["salary_in_count"] >= 1
        else _P(f"Regular inflow {_rupees(fv['avg_monthly_inflow'])}/mo (business, no salary tag)."))},
    "bank_credit_volume": {"label": "Bank Credit Volume Rule", "fn": lambda c, fv, r:
        _P(f"Total credits {_rupees(c['total_credit'])} over {c['n_months']} month(s).")},
    "bank_debit_volume": {"label": "Bank Debit Volume Rule", "fn": lambda c, fv, r:
        _P(f"Total debits {_rupees(c['total_debit'])} over {c['n_months']} month(s).")},

    "net_cashflow": {"label": "Monthly Net Cash Flow Rule", "serious": True, "fn": lambda c, fv, r: (
        _P(f"Net surplus {_rupees(c['total_credit'] - c['total_debit'])} over the statement.")
        if c["total_credit"] - c["total_debit"] > 0
        else _F(f"Net deficit {_rupees(c['total_debit'] - c['total_credit'])} over the statement."))},
    "cashflow_stability": {"label": "Cash-Flow Stability Rule", "fn": lambda c, fv, r: (
        _need(2) if len(c["monthly_inflows"]) < 2
        else _P(f"Monthly inflow CV {c['inflow_cv']:.2f} (< 0.30).") if c["inflow_cv"] < 0.30
        else _F(f"Monthly inflow CV {c['inflow_cv']:.2f} (≥ 0.30)."))},
    "monthly_cashflow": {"label": "Monthly Cash-Flow Rule", "fn": lambda c, fv, r: (
        _P(f"Positive monthly surplus ≈ {_rupees(_monthly_surplus(c))}.") if _monthly_surplus(c) > 0
        else _F(f"Monthly cash flow negative ≈ {_rupees(_monthly_surplus(c))}."))},
    "credit_debit_ratio": {"label": "Credit-Debit Ratio Rule", "fn": lambda c, fv, r: (
        _P(f"Credits ÷ debits = {_ratio(c['total_credit'], c['total_debit']):.2f} (> 1).")
        if _ratio(c["total_credit"], c["total_debit"]) > 1.0
        else _F(f"Credits ÷ debits = {_ratio(c['total_credit'], c['total_debit']):.2f} (≤ 1)."))},
    "avg_balance": {"label": "Average Bank Balance Rule", "fn": lambda c, fv, r: (
        _P(f"Average balance {_rupees(c['avg_bal'])} (≥ ₹10,000).") if c["avg_bal"] >= 10000
        else _F(f"Average balance {_rupees(c['avg_bal'])} (< ₹10,000)."))},
    "min_balance": {"label": "Minimum Bank Balance Rule", "fn": lambda c, fv, r: (
        _P(f"Lowest balance {_rupees(c['min_bal'])} (≥ ₹5,000).") if c["min_bal"] >= 5000
        else _F(f"Lowest balance {_rupees(c['min_bal'])} (< ₹5,000)."))},
    "negative_balance": {"label": "Negative Balance Rule", "serious": True, "fn": lambda c, fv, r: (
        _F(f"Negative balance at {c['neg_points']} point(s).") if c["neg_points"] > 0
        else _P("No negative-balance episodes."))},
    "overdraft": {"label": "Overdraft Rule", "serious": True, "fn": lambda c, fv, r: (
        _F(f"Balance went negative at {c['neg_points']} point(s).") if c["neg_points"] > 0
        else _P("Balance never went negative."))},
    "expense_ratio": {"label": "Expense-to-Income Ratio Rule", "fn": lambda c, fv, r: (
        _P(f"Expenses {c['expense_ratio'] * 100:.0f}% of inflow (≤ 85%).") if c["expense_ratio"] <= 0.85
        else _F(f"Expenses {c['expense_ratio'] * 100:.0f}% of inflow (> 85%)."))},

    "existing_emi": {"label": "Existing EMI Obligation Rule", "fn": lambda c, fv, r:
        _P(f"{c['emi_txn_count']} EMI/NACH debit(s) ≈ {_rupees(c['monthly_emi'])}/mo.")},
    "existing_debt_burden": {"label": "Existing Debt Burden Rule", "serious": True, "fn": lambda c, fv, r: (
        _F(f"Recurring payments to {c['emi_lenders']} lenders (possible loan stacking).") if c["emi_lenders"] >= 3
        else _P(f"{c['emi_lenders']} distinct lender(s) — no stacking."))},
    "existing_loan_count": {"label": "Existing Loan Count Rule", "fn": lambda c, fv, r: (
        _F(f"{c['emi_lenders']} active loan repayments detected (> 2).") if c["emi_lenders"] > 2
        else _P(f"{c['emi_lenders']} active loan repayment(s)."))},
    "emi_burden": {"label": "Existing EMI Burden Rule", "serious": True, "fn": lambda c, fv, r: (
        _P(f"EMI burden {c['emi_burden'] * 100:.0f}% of inflow (≤ 40%).") if c["emi_burden"] <= 0.40
        else _F(f"EMI burden {c['emi_burden'] * 100:.0f}% of inflow (> 40%)."))},
    "emi_to_income": {"label": "EMI-to-Income Ratio Rule", "serious": True, "fn": lambda c, fv, r: (
        _P(f"EMI-to-income {c['emi_burden'] * 100:.0f}% (≤ 40%).") if c["emi_burden"] <= 0.40
        else _F(f"EMI-to-income {c['emi_burden'] * 100:.0f}% (> 40%)."))},
    "emi_to_credit": {"label": "EMI-to-Credit Ratio Rule", "fn": lambda c, fv, r: (
        _P(f"EMI ÷ monthly credits {_ratio(c['monthly_emi'], fv['avg_monthly_inflow']) * 100:.0f}% (≤ 35%).")
        if _ratio(c["monthly_emi"], fv["avg_monthly_inflow"]) <= 0.35
        else _F(f"EMI ÷ monthly credits {_ratio(c['monthly_emi'], fv['avg_monthly_inflow']) * 100:.0f}% (> 35%)."))},
    "foir": {"label": "FOIR Rule", "serious": True, "fn": lambda c, fv, r: (
        _P(f"FOIR {fv['foir_ratio'] * 100:.0f}% (≤ 50%).") if fv["foir_ratio"] <= 0.50
        else _F(f"FOIR {fv['foir_ratio'] * 100:.0f}% (> 50%)."))},
    "dscr": {"label": "DSCR Rule", "serious": True, "fn": lambda c, fv, r: (
        _P(f"DSCR {fv['dscr_ratio']:.2f}x (≥ 1.25).") if fv["dscr_ratio"] >= 1.25
        else _F(f"DSCR {fv['dscr_ratio']:.2f}x (< 1.25)."))},
    "proposed_emi_affordability": {"label": "Proposed EMI Affordability Rule", "fn": lambda c, fv, r: (
        _P(f"Monthly surplus {_rupees(_monthly_surplus(c))} available for a new EMI.") if _monthly_surplus(c) > 0
        else _F(f"No monthly surplus ({_rupees(_monthly_surplus(c))}) to service a new EMI."))},
    "proposed_emi_coverage": {"label": "Proposed EMI Coverage Rule", "fn": lambda c, fv, r: (
        _P(f"Surplus ÷ current EMI = {_ratio(_monthly_surplus(c), c['monthly_emi'] or 1):.1f}x head-room.")
        if _monthly_surplus(c) > 0 else _F("Insufficient surplus to add another obligation."))},

    "bounce_count": {"label": "Bounce Count Rule", "serious": True, "fn": lambda c, fv, r: (
        _F(f"{c['bounce_count']} returned / failed debit(s).") if c["bounce_count"] > 0
        else _P("No cheque / NACH / ECS returns."))},
    "cash_deposit": {"label": "Cash Deposit Rule", "fn": lambda c, fv, r: (
        _P(f"Cash deposits {c['cash_deposit_ratio'] * 100:.0f}% of credits (≤ 40%).") if c["cash_deposit_ratio"] <= 0.40
        else _F(f"Cash deposits {c['cash_deposit_ratio'] * 100:.0f}% of credits (> 40%)."))},
    "cash_withdrawal": {"label": "Cash Withdrawal Rule", "fn": lambda c, fv, r: (
        _P(f"Cash withdrawals {fv['cash_withdrawal_ratio'] * 100:.0f}% of debits (≤ 20%).") if fv["cash_withdrawal_ratio"] <= 0.20
        else _F(f"Cash withdrawals {fv['cash_withdrawal_ratio'] * 100:.0f}% of debits (> 20%)."))},
    "upi_payment_behaviour": {"label": "UPI Payment Behaviour Rule", "fn": lambda c, fv, r: (
        _P(f"{c['upi_credit_count']} UPI credit(s) ≈ {_rupees(c['upi_credit_amt'])}.") if c["upi_credit_count"] > 0
        else ("SKIP", "No UPI activity in the statement."))},
    "upi_business_txn": {"label": "UPI Business Transaction Rule", "business_only": True, "fn": lambda c, fv, r: (
        _P(f"{c['upi_credit_count']} UPI business receipt(s) ≈ {_rupees(c['upi_credit_amt'])}.")
        if c["upi_credit_count"] >= max(1, c["n_months"] - 1)
        else _F(f"Only {c['upi_credit_count']} UPI receipt(s) across {c['n_months']} month(s).") if c["upi_credit_count"] > 0
        else ("SKIP", "No UPI business receipts detected."))},
    "bbps_payment_behaviour": {"label": "BBPS Payment Behaviour Rule", "fn": lambda c, fv, r: (
        _P(f"{c.get('util_count', 0)} recurring utility / BBPS bill payment(s).") if c.get("util_count", 0) > 0
        else ("SKIP", "No BBPS / utility bill payments detected."))},
    "dormancy": {"label": "Account Dormancy Rule", "fn": lambda c, fv, r: (
        _F("A month with ≤ 1 transaction (dormant period).") if c["dormant"] and c["n_months"] >= 2
        else _P("No dormant months."))},
    "data_completeness": {"label": "Data Completeness Rule", "fn": lambda c, fv, r: (
        _P(f"{c['n_months']} month(s) of statement data (≥ 3).") if c["n_months"] >= 3
        else _F(f"Only {c['n_months']} month(s) of data (< 3)."))},

    "business_eligibility": {"label": "Business Eligibility Rule", "business_only": True, "fn": lambda c, fv, r: (
        _P("Regular business inflow with repeat counterparties.") if c["repeat_payees"] >= 1 and fv["avg_monthly_inflow"] > 0
        else _F("No recurring business receipts identified."))},
    "business_vintage": {"label": "Business / Account Vintage Rule", "fn": lambda c, fv, r: (
        _P(f"Account age ≈ {fv['account_age_days'] / 365:.1f} yr (≥ 3 yr).")
        if fv.get("account_age_days") and fv["account_age_days"] >= 3 * 365
        else _F(f"Account age ≈ {(fv.get('account_age_days') or 0) / 365:.1f} yr (< 3 yr).") if fv.get("account_age_days")
        else ("SKIP", "Account-opening date not on the statement."))},
    "revenue_stability": {"label": "Business Revenue Stability Rule", "business_only": True, "fn": lambda c, fv, r: (
        _need(2) if len(c["monthly_inflows"]) < 2
        else _P(f"Low revenue volatility (CV {c['inflow_cv']:.2f}).") if c["inflow_cv"] < 0.30
        else _F(f"Revenue volatility CV {c['inflow_cv']:.2f} (≥ 0.30)."))},
    "revenue_growth": {"label": "Revenue Growth Rule", "business_only": True, "fn": lambda c, fv, r: (
        _need(4) if c["growth"] is None
        else _P(f"Revenue trend +{c['growth'] * 100:.0f}% (2nd half vs 1st).") if c["growth"] >= 0
        else _F(f"Revenue trend {c['growth'] * 100:.0f}% (declining)."))},
    "counterparty_concentration": {"label": "Counterparty Concentration Rule", "business_only": True, "fn": lambda c, fv, r: (
        _F(f"Top payee = {c['top1_share'] * 100:.0f}% of credits (> 50%).") if c["top1_share"] > 0.50
        else _P(f"Top payee = {c['top1_share'] * 100:.0f}% of credits (≤ 50%)."))},
    "counterparty_diversity": {"label": "Business Counterparty Diversity Rule", "business_only": True, "fn": lambda c, fv, r: (
        _P(f"{c['unique_payees']} distinct payers, top 5 = {c['top5_share'] * 100:.0f}% of credits.")
        if c["top5_share"] < 0.60 else _F(f"Top 5 payers = {c['top5_share'] * 100:.0f}% of credits (≥ 60%)."))},
    "supplier_concentration": {"label": "Supplier Concentration Rule", "business_only": True, "fn": lambda c, fv, r: (
        _F(f"Top supplier = {c['top_supplier_share'] * 100:.0f}% of debits (> 50%).") if c["top_supplier_share"] > 0.50
        else _P(f"Top supplier = {c['top_supplier_share'] * 100:.0f}% of debits (≤ 50%)."))},
    "gst_payment_detected": {"label": "GST Payment (in-statement) Rule", "business_only": True, "fn": lambda c, fv, r: (
        _P(f"{c['gst_count']} GST payment(s) ≈ {_rupees(c['monthly_gst'])}/mo.") if c["gst_count"] > 0
        else ("SKIP", "No GST payments visible in the statement."))},

    "credit_score": {"label": "Credit Score Rule", "serious": True, "gate": True, "fn": lambda c, fv, r: (
        _P(f"Credit score {r['score']} > {_GATE()} (gate).") if r["score"] > _GATE()
        else _F(f"Credit score {r['score']} ≤ {_GATE()} — hard cutoff."))},
    "risk_score_threshold": {"label": "Risk Score Threshold Rule", "fn": lambda c, fv, r: (
        _P(f"Score {r['score']} ≥ 600 (grade {r['grade']}).") if r["score"] >= 600
        else _F(f"Score {r['score']} < 600 (grade {r['grade']})."))},
    "pd_threshold": {"label": "PD Threshold Rule", "serious": True, "fn": lambda c, fv, r: (
        _P(f"Probability of default {_pd(r):.1f}% (≤ 10%).") if _pd(r) <= 10
        else _F(f"Probability of default {_pd(r):.1f}% (> 10%)."))},
    "income_to_loan": {"label": "Income-to-Loan Ratio Rule", "fn": lambda c, fv, r:
        _P(f"Annual inflow ≈ {_rupees(fv['avg_monthly_inflow'] * 12)} — indicative loan cap on surplus applies.")},
    "loan_to_turnover": {"label": "Loan-to-Turnover Rule", "fn": lambda c, fv, r:
        _P(f"Annual turnover ≈ {_rupees(fv['avg_monthly_inflow'] * 12)} — 25% cap ≈ {_rupees(fv['avg_monthly_inflow'] * 3)}.")},
    "max_loan_eligibility": {"label": "Maximum Loan Eligibility Rule", "fn": lambda c, fv, r:
        _P(f"On surplus {_rupees(_monthly_surplus(c))}/mo, indicative eligible EMI ≈ {_rupees(max(0.0, _monthly_surplus(c)) * 0.6)}/mo.")},
    "txn_anomaly": {"label": "Transaction Anomaly Rule", "serious": True, "fn": lambda c, fv, r: (
        _F(f"Same-day in/out ratio {c['same_day_ratio'] * 100:.0f}% or {c['rapid_dup']} rapid duplicate(s).")
        if c["same_day_ratio"] > 0.30 or c["rapid_dup"] >= 2 else _P("No transaction anomaly pattern."))},
    "fraud_composite": {"label": "Fraud / Suspicious Activity Rule", "serious": True, "fn": lambda c, fv, r: (
        _F("Suspicious pattern: " + ", ".join(x for x in [
            f"{c['same_day_inout']} same-day in/out" if c["same_day_inout"] > 0 else "",
            f"{c['round_credits']} round ₹50k credit(s)" if c["round_credits"] >= 3 else "",
            f"{c['rapid_dup']} rapid duplicate(s)" if c["rapid_dup"] >= 2 else "",
            "circular transfers" if c["same_day_ratio"] > 0.30 else "",
        ] if x) + ".")
        if (c["same_day_inout"] > 0 or c["round_credits"] >= 3 or c["rapid_dup"] >= 2 or c["same_day_ratio"] > 0.30)
        else _P("No fraud / round-tripping / circular-transfer pattern."))},

    # Outcome rules — fn is None; the engine fills these from the aggregate decision.
    "manual_review": {"label": "Manual Review Rule", "fn": None},
    "auto_approval": {"label": "Auto-Approval Rule", "fn": None},
    "auto_rejection": {"label": "Auto-Rejection Rule", "fn": None},
}


# ── Non-computable rules — listed, but OFF by default & evaluate to SKIP ────
_EXT_BUREAU = "Needs a credit-bureau pull (DPD / default history) — not in the bank statement."
_EXT_GST = "Needs GST-portal data (filing / GSTR / turnover) — not in the bank statement."
_EXT_PROP = "Needs the property/collateral file (valuation, title, encumbrance) — not in the bank statement."
_EXT_ASSET = "Needs the asset invoice / valuation / dealer KYC — not in the bank statement."
_EXT_KYC = "Needs KYC / demographic data (age, identity, address) — not in the bank statement."
_EXT_DECL = "Needs the applicant's declared figures to reconcile against — not provided."

EXTERNAL: dict[str, dict] = {
    "applicant_eligibility":      {"label": "Applicant Eligibility Rule", "reason": _EXT_KYC},
    "employment_business_vintage":{"label": "Employment/Business Vintage Rule", "reason": _EXT_KYC},
    "dpd_history":                {"label": "DPD History Rule", "reason": _EXT_BUREAU},
    "recent_default":             {"label": "Recent Default Rule", "reason": _EXT_BUREAU},

    "gst_registration_status":    {"label": "GST Registration Status Rule", "reason": _EXT_GST},
    "gst_registration":           {"label": "GST Registration Rule", "reason": _EXT_GST},
    "gst_turnover":               {"label": "GST Turnover Rule", "reason": _EXT_GST},
    "min_gst_turnover":           {"label": "Minimum GST Turnover Rule", "reason": _EXT_GST},
    "gst_filing_regularity":      {"label": "GST Filing Regularity Rule", "reason": _EXT_GST},
    "gst_filing_delay":           {"label": "GST Filing Delay Rule", "reason": _EXT_GST},
    "gst_return_consistency":     {"label": "GST Return Consistency Rule", "reason": _EXT_GST},
    "gstr1_vs_gstr3b":            {"label": "GSTR-1 vs GSTR-3B Matching Rule", "reason": _EXT_GST},
    "gst_turnover_growth":        {"label": "GST Turnover Growth Rule", "reason": _EXT_GST},
    "gst_risk_flag":              {"label": "GST Risk Flag Rule", "reason": _EXT_GST},
    "gst_bank_turnover_matching": {"label": "GST-Bank Turnover Matching Rule", "reason": _EXT_GST},
    "gst_to_bank_turnover_matching": {"label": "GST-to-Bank Turnover Matching Rule", "reason": _EXT_GST},
    "loan_to_gst_turnover":       {"label": "Loan-to-GST-Turnover Rule", "reason": _EXT_GST},
    "turnover_decline":           {"label": "Turnover Decline Rule", "reason": _EXT_GST},
    "consecutive_declining_quarter": {"label": "Consecutive Declining Quarter Rule", "reason": _EXT_GST},
    "b2b_sales_pct":              {"label": "B2B Sales Percentage Rule", "reason": _EXT_GST},
    "b2c_sales_pct":              {"label": "B2C Sales Percentage Rule", "reason": _EXT_GST},
    "buyer_concentration":        {"label": "Buyer Concentration Rule", "reason": _EXT_GST},
    "top_buyer_sales_pct":        {"label": "Top Buyer Sales Percentage Rule", "reason": _EXT_GST},
    "export_sales":               {"label": "Export Sales Rule", "reason": _EXT_GST},

    "ltv":                        {"label": "Loan-to-Value (LTV) Rule", "reason": _EXT_PROP},
    "vehicle_loan_to_value":      {"label": "Loan-to-Value Rule", "reason": _EXT_ASSET},
    "property_valuation":         {"label": "Property Valuation Rule", "reason": _EXT_PROP},
    "property_type":              {"label": "Property Type Eligibility Rule", "reason": _EXT_PROP},
    "property_ownership":         {"label": "Property Ownership Rule", "reason": _EXT_PROP},
    "property_encumbrance":       {"label": "Property Encumbrance Rule", "reason": _EXT_PROP},
    "property_location":          {"label": "Property Location Rule", "reason": _EXT_PROP},
    "loan_to_business_turnover":  {"label": "Loan-to-Business-Turnover Rule", "reason": _EXT_DECL},
    "income_bank_statement_matching": {"label": "Income-Bank Statement Matching Rule", "reason": _EXT_DECL},

    "machine_cost_validation":    {"label": "Machine Cost Validation Rule", "reason": _EXT_ASSET},
    "machine_invoice_validation": {"label": "Machine Invoice Validation Rule", "reason": _EXT_ASSET},
    "machine_valuation":          {"label": "Machine Valuation Rule", "reason": _EXT_ASSET},
    "machine_type_eligibility":   {"label": "Machine Type Eligibility Rule", "reason": _EXT_ASSET},
    "new_used_machine":           {"label": "New/Used Machine Eligibility Rule", "reason": _EXT_ASSET},
    "loan_to_machine_value":      {"label": "Loan-to-Machine-Value Rule", "reason": _EXT_ASSET},
    "margin_money":               {"label": "Margin Money Rule", "reason": _EXT_ASSET},
    "down_payment":               {"label": "Down Payment Rule", "reason": _EXT_ASSET},
    "asset_age":                  {"label": "Asset Age Rule", "reason": _EXT_ASSET},
    "dealer_supplier_validation": {"label": "Dealer/Supplier Validation Rule", "reason": _EXT_ASSET},
    "supplier_verification":      {"label": "Supplier Concentration/Verification Rule", "reason": _EXT_ASSET},

    "vehicle_price_to_income":    {"label": "Vehicle Price-to-Income Rule", "reason": _EXT_ASSET},
    "vehicle_valuation":          {"label": "Vehicle Valuation Rule", "reason": _EXT_ASSET},
    "vehicle_type_eligibility":   {"label": "Vehicle Type Eligibility Rule", "reason": _EXT_ASSET},
    "new_used_vehicle":           {"label": "New/Used Vehicle Rule", "reason": _EXT_ASSET},
    "vehicle_age":                {"label": "Vehicle Age Rule", "reason": _EXT_ASSET},
    "vehicle_down_payment":       {"label": "Down Payment Rule", "reason": _EXT_ASSET},
    "vehicle_margin_money":       {"label": "Margin Money Rule", "reason": _EXT_ASSET},
    "dealer_validation":          {"label": "Dealer Validation Rule", "reason": _EXT_ASSET},
    "oem_validation":             {"label": "OEM Validation Rule", "reason": _EXT_ASSET},
    "insurance_validation":       {"label": "Insurance Validation Rule", "reason": _EXT_ASSET},
    "rc_validation":              {"label": "Registration/RC Validation Rule", "reason": _EXT_ASSET},
    "vehicle_cost_validation":    {"label": "Vehicle Cost Validation Rule", "reason": _EXT_ASSET},
}


# ── Full per-product rule lists (every rule, in the order given) ───────────
# (id, ...) — id resolves in RULES (evaluable, default ON) or EXTERNAL (SKIP, default OFF)

PRODUCT_RULE_IDS: dict[str, list[str]] = {
    "lap_sbl": [
        "applicant_eligibility", "min_income", "business_vintage",
        "gst_registration_status", "gst_turnover", "gst_filing_regularity", "gst_turnover_growth", "gstr1_vs_gstr3b",
        "bank_credit_volume", "bank_debit_volume", "net_cashflow", "cashflow_stability", "credit_debit_ratio",
        "avg_balance", "min_balance", "bounce_count", "overdraft",
        "emi_burden", "emi_to_income", "dscr", "proposed_emi_affordability", "existing_debt_burden",
        "credit_score", "dpd_history",
        "ltv", "property_valuation", "property_type", "property_ownership", "property_encumbrance", "property_location",
        "loan_to_gst_turnover", "loan_to_business_turnover", "gst_bank_turnover_matching", "income_bank_statement_matching",
        "upi_business_txn", "counterparty_concentration", "txn_anomaly", "fraud_composite",
        "risk_score_threshold", "pd_threshold", "max_loan_eligibility",
        "manual_review", "auto_approval", "auto_rejection",
    ],
    "machine": [
        "applicant_eligibility", "business_vintage", "min_income", "min_gst_turnover",
        "gst_filing_regularity", "gst_turnover_growth",
        "bank_credit_volume", "net_cashflow", "cashflow_stability", "credit_debit_ratio",
        "avg_balance", "min_balance", "bounce_count", "overdraft",
        "existing_debt_burden", "existing_emi", "emi_to_income", "dscr", "proposed_emi_affordability",
        "credit_score", "dpd_history",
        "machine_cost_validation", "machine_invoice_validation", "machine_valuation", "machine_type_eligibility",
        "new_used_machine", "loan_to_machine_value", "margin_money", "down_payment", "asset_age",
        "dealer_supplier_validation", "supplier_verification",
        "gst_bank_turnover_matching", "income_bank_statement_matching",
        "revenue_stability", "txn_anomaly", "fraud_composite",
        "risk_score_threshold", "pd_threshold", "max_loan_eligibility",
        "manual_review", "auto_approval", "auto_rejection",
    ],
    "vehicle": [
        "applicant_eligibility", "min_income", "employment_business_vintage", "income_stability", "foir",
        "emi_to_income", "emi_burden", "proposed_emi_affordability",
        "credit_score", "dpd_history", "recent_default", "existing_loan_count",
        "income_validation", "monthly_cashflow", "cashflow_stability", "avg_balance", "negative_balance",
        "bounce_count", "overdraft", "upi_payment_behaviour", "bbps_payment_behaviour",
        "vehicle_price_to_income", "vehicle_valuation", "vehicle_type_eligibility", "new_used_vehicle", "vehicle_age",
        "vehicle_loan_to_value", "vehicle_down_payment", "vehicle_margin_money",
        "dealer_validation", "oem_validation", "insurance_validation", "rc_validation", "vehicle_cost_validation",
        "income_to_loan", "risk_score_threshold", "pd_threshold", "txn_anomaly", "fraud_composite",
        "max_loan_eligibility", "manual_review", "auto_approval", "auto_rejection",
    ],
    "msme": [
        "business_eligibility", "business_vintage", "min_annual_turnover", "min_monthly_turnover",
        "gst_registration", "gst_filing_regularity", "gst_filing_delay", "gst_return_consistency",
        "gstr1_vs_gstr3b", "gst_turnover_growth", "turnover_decline", "consecutive_declining_quarter",
        "b2b_sales_pct", "b2c_sales_pct", "buyer_concentration", "top_buyer_sales_pct", "export_sales",
        "bank_credit_volume", "bank_debit_volume", "net_cashflow", "cashflow_stability", "credit_debit_ratio",
        "avg_balance", "min_balance", "cash_deposit", "cash_withdrawal", "bounce_count", "overdraft",
        "existing_debt_burden", "existing_emi", "emi_to_credit", "dscr", "proposed_emi_coverage",
        "credit_score", "dpd_history",
        "upi_business_txn", "counterparty_diversity", "gst_to_bank_turnover_matching", "income_bank_statement_matching",
        "revenue_stability", "txn_anomaly", "gst_risk_flag",
        "risk_score_threshold", "pd_threshold", "loan_to_turnover", "max_loan_eligibility", "data_completeness",
        "fraud_composite", "manual_review", "auto_approval", "auto_rejection",
    ],
}

PRODUCT_NAMES = {
    "lap_sbl": "LAP / SBL",
    "machine": "Machine Loan",
    "vehicle": "Vehicle Loan",
    "msme": "MSME Loan",
}


def rule_meta(rid: str) -> dict | None:
    if rid in RULES:
        m = RULES[rid]
        return {"id": rid, "label": m["label"], "serious": bool(m.get("serious")),
                "computable": True, "default": True}
    if rid in EXTERNAL:
        return {"id": rid, "label": EXTERNAL[rid]["label"], "serious": False,
                "computable": False, "default": False}
    return None


def product_catalogue(product_id: str) -> list[dict]:
    return [m for rid in PRODUCT_RULE_IDS.get(product_id, []) if (m := rule_meta(rid))]
