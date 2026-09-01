"""AA per-statement BRE engine — PASS / FAIL / SKIP for every enabled rule.

`evaluate_bre_rules(...)` runs the rules toggled on the Settings page against a
real applicant's bank statement + derived feature vector + credit score and
returns a verdict per rule plus an overall decision. `build_context()` derives
the ~50 quantities the rule evaluators need and is reused by
app.aa.product_engine.

  PASS = rule evaluated favourably (no concern)
  FAIL = the rule's risk condition triggered (a concern)
  SKIP = enabled but not computable from one statement (needs an external feed —
         GST turnover, bureau — or 6-12 months of history)

The GST equivalent is app.gst.rules.
"""

import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

from app.aa.rule_catalog import (
    CREDIT_SCORE_GATE_RULE_ID,
    UNDERWRITING_RULE_CATEGORIES,
)
from app.aa.settings_state import settings_state
from app.aa.scoring import (
    BOUNCE_RE,
    CASH_RE,
    _narr_sig,
    _parse_tx_date,
    _running_balances,
)

_EMI_RE = re.compile(r"\b(emi|loan|nach|standing\s*instr|instal?ment|repay(?:ment)?|\bsi\b|auto\s*debit)\b", re.I)
_GST_RE = re.compile(r"\bgst\b|goods\s+and\s+services\s+tax|gstin|gstpmt", re.I)
_TAX_RE = re.compile(r"\b(income\s*tax|tds|advance\s*tax|self\s*assessment\s*tax|itns|tax\s*challan)\b", re.I)
_SAL_RE = re.compile(r"\bsalary\b|payroll|\bsal\s*cr\b|sal[-/]cr|wages", re.I)
_RENT_RE = re.compile(r"\brent\b|lease\s*(?:rent|payment)", re.I)
_UTIL_RE = re.compile(r"\b(electricity|water\s*bill|gas\s*bill|broadband|mobile\s*bill|dth|utility|bbps|recharge)\b", re.I)
_UPI_RE = re.compile(r"\bupi\b|@ok|@ybl|@paytm|@axl|@apl", re.I)
_REVERSAL_RE = re.compile(r"\breversal\b|charge\s*back|chargeback|txn\s+reversed|payment\s+reversed", re.I)


def _ratio(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _rupees(x: float) -> str:
    return f"₹{round(x):,}"


# ─────────────────────────────────────────────────────────────────────────────
# Context: every quantity the rules need, derived once from the statement.
# ─────────────────────────────────────────────────────────────────────────────

def build_context(fv: dict, risk: dict, transactions: list[dict], opening_balance: float | None) -> dict:
    txns = [t for t in transactions if isinstance(t.get("amount"), (int, float)) and t["amount"] > 0]
    credits = [t for t in txns if t["type"] == "CREDIT"]
    debits = [t for t in txns if t["type"] == "DEBIT"]
    total_credit = sum(t["amount"] for t in credits)
    total_debit = sum(t["amount"] for t in debits)

    months = float(fv.get("statement_months") or 1.0)

    # Per calendar-month buckets
    mc: dict = defaultdict(float)
    md: dict = defaultdict(float)
    mcount: dict = defaultdict(int)
    month_end_bal: dict = {}
    running = _running_balances(transactions, opening_balance)
    for t, bal in zip(transactions, running):
        if not (isinstance(t.get("amount"), (int, float)) and t["amount"] > 0):
            continue
        d = _parse_tx_date(t.get("date"))
        if not d:
            continue
        k = (d.year, d.month)
        (mc if t["type"] == "CREDIT" else md)[k] += t["amount"]
        mcount[k] += 1
        if bal is not None:
            month_end_bal[k] = bal
    month_keys = sorted(set(list(mc) + list(md) + list(mcount)))
    monthly_inflows = [mc.get(k, 0.0) for k in month_keys]
    monthly_outflows = [md.get(k, 0.0) for k in month_keys]
    monthly_counts = [mcount.get(k, 0) for k in month_keys]
    n_months = len(month_keys) or 1

    inflow_mean = (sum(monthly_inflows) / len(monthly_inflows)) if monthly_inflows else 0.0
    inflow_cv = (statistics.pstdev(monthly_inflows) / inflow_mean) if len(monthly_inflows) >= 2 and inflow_mean else 0.0

    bals = [b for b in running if b is not None]
    neg_points = sum(1 for b in bals if b < 0)
    low_bal_points = sum(1 for b in bals if 0 <= b < 2000)
    avg_bal = (sum(bals) / len(bals)) if bals else float(fv.get("avg_balance") or 0)
    min_bal = min(bals) if bals else float(fv.get("minimum_balance") or 0)

    emi_txns = [t for t in debits if _EMI_RE.search(t["narration"])]
    monthly_emi = sum(t["amount"] for t in emi_txns) / months
    emi_burden = _ratio(monthly_emi, fv.get("avg_monthly_inflow") or 0)
    emi_lenders = {_narr_sig(t["narration"]) for t in emi_txns if _narr_sig(t["narration"])}
    emi_months = len({(_parse_tx_date(t["date"]).year, _parse_tx_date(t["date"]).month)
                      for t in emi_txns if _parse_tx_date(t.get("date"))})

    gst_txns = [t for t in debits if _GST_RE.search(t["narration"])]
    tax_txns = [t for t in debits if _TAX_RE.search(t["narration"])]
    rent_txns = [t for t in debits if _RENT_RE.search(t["narration"])]
    util_txns = [t for t in debits if _UTIL_RE.search(t["narration"])]
    salary_in = [t for t in credits if _SAL_RE.search(t["narration"])]
    upi_credits = [t for t in credits if _UPI_RE.search(t["narration"])]
    cash_deposits = [t for t in credits if CASH_RE.search(t["narration"])]
    cash_withdrawals = [t for t in debits if CASH_RE.search(t["narration"])]
    reversals = [t for t in txns if _REVERSAL_RE.search(t["narration"])]
    bounces = [t for t in txns if BOUNCE_RE.search(t["narration"])]

    monthly_gst = sum(t["amount"] for t in gst_txns) / months
    monthly_rent = sum(t["amount"] for t in rent_txns) / months
    fixed_burden = (monthly_emi + monthly_rent
                    + sum(t["amount"] for t in util_txns) / months)

    largest_credit = max((t["amount"] for t in credits), default=0.0)
    largest_credit_pct = _ratio(largest_credit, fv.get("avg_monthly_inflow") or 0)

    # Counterparty (credit) concentration
    cp_amt: dict = defaultdict(float)
    cp_cnt: Counter = Counter()
    for t in credits:
        s = _narr_sig(t["narration"]) or t["narration"][:16]
        cp_amt[s] += t["amount"]
        cp_cnt[s] += 1
    cp_sorted = sorted(cp_amt.values(), reverse=True)
    top1_share = _ratio(cp_sorted[0], total_credit) if cp_sorted else 0.0
    top3_share = _ratio(sum(cp_sorted[:3]), total_credit) if cp_sorted else 0.0
    top5_share = _ratio(sum(cp_sorted[:5]), total_credit) if cp_sorted else 0.0
    unique_payees = len(cp_amt)
    repeat_payees = sum(1 for v in cp_cnt.values() if v >= 2)

    # Supplier (debit) concentration
    sup_amt: dict = defaultdict(float)
    for t in debits:
        sup_amt[_narr_sig(t["narration"]) or t["narration"][:16]] += t["amount"]
    sup_sorted = sorted(sup_amt.values(), reverse=True)
    top_supplier_share = _ratio(sup_sorted[0], total_debit) if sup_sorted else 0.0

    # Same-day large in→out (possible round-tripping)
    dated = sorted(
        ((_parse_tx_date(t["date"]), t) for t in txns if _parse_tx_date(t.get("date"))),
        key=lambda x: x[0],
    )
    big = 0.5 * (fv.get("avg_monthly_inflow") or 0)
    same_day_inout = 0
    for i, (d, t) in enumerate(dated):
        if t["type"] != "CREDIT" or t["amount"] < big or big <= 0:
            continue
        for d2, t2 in dated[i + 1:]:
            if (d2 - d).days > 2:
                break
            if t2["type"] == "DEBIT" and t2["amount"] >= 0.8 * t["amount"]:
                same_day_inout += 1
                break
    same_day_ratio = _ratio(same_day_inout, len(credits))

    # Rapid duplicates = same counterparty + same amount within 2 days (the
    # actual fraud pattern — a monthly ₹649 subscription is NOT this).
    rapid_dup = 0
    for i, (d, t) in enumerate(dated):
        s = _narr_sig(t["narration"])
        for d2, t2 in dated[i + 1:]:
            if (d2 - d).days > 2:
                break
            if _narr_sig(t2["narration"]) == s and abs(t2["amount"] - t["amount"]) < 1.0:
                rapid_dup += 1
                break
    round_credits = sum(1 for t in credits if t["amount"] >= 50000 and t["amount"] % 50000 == 0)

    # First-half vs second-half monthly inflow (growth / decline proxy)
    growth = None
    if len(monthly_inflows) >= 4:
        h = len(monthly_inflows) // 2
        a = sum(monthly_inflows[:h]) / h
        b = sum(monthly_inflows[h:]) / (len(monthly_inflows) - h)
        growth = _ratio(b - a, a) if a else 0.0

    # Month-end balance trend
    end_bals = [month_end_bal[k] for k in month_keys if k in month_end_bal]
    balance_declining = len(end_bals) >= 3 and all(
        end_bals[i] >= end_bals[i + 1] for i in range(len(end_bals) - 1)
    )

    zero_inflow_months = sum(1 for v in monthly_inflows if v <= 1.0)
    deficit_months = sum(1 for a, b in zip(monthly_inflows, monthly_outflows) if b > a)

    expense_ratio = _ratio(total_debit, total_credit)
    tx_per_month = len(txns) / months
    dormant = min(monthly_counts) <= 1 if monthly_counts else False

    # Applicant profile: a recurring salary credit that dominates the inflow
    # means the business-customer / supplier / GST rules don't apply.
    salary_total = sum(t["amount"] for t in salary_in)
    is_salaried = (
        len(salary_in) >= max(2, n_months - 1)
        or (len(salary_in) >= 1 and _ratio(salary_total, total_credit) >= 0.55)
    )

    return {
        "months": months, "n_months": n_months,
        "total_credit": total_credit, "total_debit": total_debit,
        "monthly_inflows": monthly_inflows, "monthly_outflows": monthly_outflows,
        "monthly_counts": monthly_counts,
        "inflow_cv": inflow_cv,
        "avg_bal": avg_bal, "min_bal": min_bal,
        "neg_points": neg_points, "low_bal_points": low_bal_points,
        "monthly_emi": monthly_emi, "emi_burden": emi_burden,
        "emi_txn_count": len(emi_txns), "emi_lenders": len(emi_lenders), "emi_months": emi_months,
        "gst_count": len(gst_txns), "monthly_gst": monthly_gst,
        "tax_count": len(tax_txns),
        "rent_count": len(rent_txns), "monthly_rent": monthly_rent,
        "util_count": len(util_txns),
        "fixed_burden": fixed_burden, "fixed_burden_ratio": _ratio(fixed_burden, fv.get("avg_monthly_inflow") or 0),
        "salary_in_count": len(salary_in),
        "upi_credit_count": len(upi_credits), "upi_credit_amt": sum(t["amount"] for t in upi_credits),
        "cash_deposit_amt": sum(t["amount"] for t in cash_deposits), "cash_deposit_ratio": _ratio(sum(t["amount"] for t in cash_deposits), total_credit),
        "cash_wd_count": len(cash_withdrawals),
        "reversal_count": len(reversals), "bounce_count": len(bounces),
        "largest_credit": largest_credit, "largest_credit_pct": largest_credit_pct,
        "top1_share": top1_share, "top3_share": top3_share, "top5_share": top5_share,
        "top_supplier_share": top_supplier_share,
        "unique_payees": unique_payees, "repeat_payees": repeat_payees,
        "same_day_inout": same_day_inout, "same_day_ratio": same_day_ratio,
        "rapid_dup": rapid_dup, "round_credits": round_credits,
        "growth": growth, "balance_declining": balance_declining,
        "zero_inflow_months": zero_inflow_months, "deficit_months": deficit_months,
        "expense_ratio": expense_ratio, "tx_per_month": tx_per_month, "dormant": dormant,
        "is_salaried": is_salaried,
    }


# Rules that only make sense for a business/SME applicant — skipped when the
# statement is a salaried individual's (one dominant recurring salary credit).
_BUSINESS_ONLY = {
    "rev_monthly_inflow", "rev_growth", "rev_decline", "rev_active_months",
    "rev_large_credit_dep", "rev_cust_conc", "rev_cust_div", "rev_recurring_receipts",
    "exp_ratio", "exp_high_ratio", "exp_fixed_burden", "exp_rent_burden", "exp_unusual",
    "cf_coverage",
    "gst_payment_detected", "gst_consistency", "tax_regularity",
    "sal_consistency",
    "cash_deposit_ratio", "cash_high_dep",
    "upi_receipts", "upi_consistency",
    "acct_completeness", "acct_missing_months",
    "fraud_round_credits", "fraud_temp_inflation",
    "stab_revenue", "stab_growing_rev", "stab_declining_rev", "stab_unpredictable",
    "eligibility_strong_cashflow", "eligibility_high_risk_flag",
    "eligibility_stable_business", "eligibility_concentration_risk",
}

# FAILs here escalate the decision even when the credit-score gate passes.
_SERIOUS_FAIL = {
    "cf_neg_bal_days", "emi_bounce", "bank_cheque_bounce", "bank_nach_failure",
    "bank_ecs_return", "emi_high_burden", "cf_deficit", "exp_high_ratio",
    "fraud_same_day_in_out", "fraud_circular", "eligibility_high_risk_flag",
    "eligibility_transaction_risk", "cf_deterioration",
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-rule evaluators.  fn(c, fv, risk) -> ("PASS"|"FAIL"|"SKIP", detail)
# ─────────────────────────────────────────────────────────────────────────────

def _P(detail):
    return ("PASS", detail)


def _F(detail):
    return ("FAIL", detail)


def _need(months_needed):
    return ("SKIP", f"Needs ≥{months_needed} months of history.")


_EVALUATORS = {
    # ── Business Revenue ────────────────────────────────────────────────────
    "rev_monthly_inflow": lambda c, fv, r: _P(f"Avg monthly inflow {_rupees(fv['avg_monthly_inflow'])}."),
    "rev_consistency": lambda c, fv, r: (
        _P(f"Monthly inflow CV {c['inflow_cv']:.2f} (< 0.30).") if c["inflow_cv"] < 0.30 and len(c["monthly_inflows"]) >= 2
        else _F(f"Monthly inflow CV {c['inflow_cv']:.2f} (≥ 0.30).") if len(c["monthly_inflows"]) >= 2
        else _need(2)
    ),
    "rev_volatility": lambda c, fv, r: (
        _F(f"Monthly inflow CV {c['inflow_cv']:.2f} (> 0.50).") if len(c["monthly_inflows"]) >= 2 and c["inflow_cv"] > 0.50
        else _P(f"Monthly inflow CV {c['inflow_cv']:.2f} (≤ 0.50).") if len(c["monthly_inflows"]) >= 2
        else _need(2)
    ),
    "rev_active_months": lambda c, fv, r: (
        _need(10) if c["n_months"] < 10
        else _P(f"Inflow in {c['n_months'] - c['zero_inflow_months']}/{c['n_months']} months.")
        if c["zero_inflow_months"] == 0 else _F(f"{c['zero_inflow_months']} month(s) with no inflow.")
    ),
    "rev_zero_months": lambda c, fv, r: (
        _F(f"{c['zero_inflow_months']} month(s) with no inflow.") if c["zero_inflow_months"] >= 2
        else _P("No 2-month inflow gap.")
    ),
    "rev_large_credit_dep": lambda c, fv, r: (
        _F(f"Largest single credit is {c['largest_credit_pct'] * 100:.0f}% of monthly inflow ({_rupees(c['largest_credit'])}).")
        if c["largest_credit_pct"] > 0.30
        else _P(f"Largest single credit {c['largest_credit_pct'] * 100:.0f}% of monthly inflow.")
    ),
    "rev_growth": lambda c, fv, r: (
        _need(4) if c["growth"] is None
        else _P(f"Inflow trend +{c['growth'] * 100:.0f}% (2nd half vs 1st).") if c["growth"] >= 0
        else _F(f"Inflow trend {c['growth'] * 100:.0f}% (2nd half vs 1st).")
    ),
    "rev_decline": lambda c, fv, r: (
        _need(4) if c["growth"] is None
        else _F(f"Inflow down {abs(c['growth']) * 100:.0f}% (> 20%).") if c["growth"] < -0.20
        else _P(f"Inflow trend {c['growth'] * 100:+.0f}% (no >20% decline).")
    ),
    "rev_cust_conc": lambda c, fv, r: (
        _F(f"Top 3 payees = {c['top3_share'] * 100:.0f}% of credits (> 50%).") if c["top3_share"] > 0.50
        else _P(f"Top 3 payees = {c['top3_share'] * 100:.0f}% of credits.")
    ),
    "rev_cust_div": lambda c, fv, r: (
        _P(f"Top 5 payees = {c['top5_share'] * 100:.0f}% of credits (< 50%).") if c["top5_share"] < 0.50
        else _F(f"Top 5 payees = {c['top5_share'] * 100:.0f}% of credits (≥ 50%).")
    ),
    "rev_recurring_receipts": lambda c, fv, r: (
        _P(f"{c['repeat_payees']} recurring payee(s).") if c["repeat_payees"] >= 1
        else _F("No repeat credit counterparties found.")
    ),

    # ── Cash Flow ──────────────────────────────────────────────────────────
    "cf_avg_bal": lambda c, fv, r: _P(f"Average balance {_rupees(c['avg_bal'])}."),
    "cf_min_bal": lambda c, fv, r: (
        _F(f"Lowest balance {_rupees(c['min_bal'])}.") if c["min_bal"] < 5000
        else _P(f"Lowest balance {_rupees(c['min_bal'])} (≥ ₹5,000).")
    ),
    "cf_end_month_bal": lambda c, fv, r: (
        _F("Month-end balance declining every month.") if c["balance_declining"]
        else _P("Month-end balance not in continuous decline.")
    ),
    "cf_neg_bal_days": lambda c, fv, r: (
        _F(f"Balance went negative at {c['neg_points']} point(s).") if c["neg_points"] > 0
        else _P("Balance never went negative.")
    ),
    "cf_low_bal_days": lambda c, fv, r: (
        _F(f"Balance under ₹2,000 at {c['low_bal_points']} point(s).") if c["low_bal_points"] > 3
        else _P(f"Balance rarely under ₹2,000 ({c['low_bal_points']} point(s)).")
    ),
    "cf_coverage": lambda c, fv, r: (
        _P(f"Inflow ÷ fixed obligations = {_ratio(fv['avg_monthly_inflow'], c['fixed_burden']):.1f}x.")
        if c["fixed_burden"] > 0 and _ratio(fv["avg_monthly_inflow"], c["fixed_burden"]) >= 1.5
        else _F(f"Inflow ÷ fixed obligations = {_ratio(fv['avg_monthly_inflow'], c['fixed_burden']):.1f}x (< 1.5).")
        if c["fixed_burden"] > 0
        else _P("No fixed obligations detected.")
    ),
    "cf_inflow_outflow_ratio": lambda c, fv, r: (
        _P(f"Credits ÷ debits = {_ratio(c['total_credit'], c['total_debit']):.2f} (> 1).")
        if _ratio(c["total_credit"], c["total_debit"]) > 1.0
        else _F(f"Credits ÷ debits = {_ratio(c['total_credit'], c['total_debit']):.2f} (≤ 1).")
    ),
    "cf_surplus": lambda c, fv, r: (
        _P(f"Net surplus {_rupees(c['total_credit'] - c['total_debit'])} over the statement.")
        if c["total_credit"] - c["total_debit"] > 0
        else _F(f"Net deficit {_rupees(c['total_debit'] - c['total_credit'])} over the statement.")
    ),
    "cf_deficit": lambda c, fv, r: (
        _F(f"Outflow exceeded inflow in {c['deficit_months']}/{c['n_months']} months.")
        if c["n_months"] >= 2 and c["deficit_months"] > c["n_months"] / 2
        else _P(f"Outflow exceeded inflow in {c['deficit_months']}/{c['n_months']} months.")
    ),
    "cf_deterioration": lambda c, fv, r: (
        _F("Balance continuously declining.") if c["balance_declining"]
        else _P("Balance not continuously declining.")
    ),

    # ── Expense ────────────────────────────────────────────────────────────
    "exp_ratio": lambda c, fv, r: _P(f"Expense ratio {c['expense_ratio'] * 100:.0f}% of inflow."),
    "exp_high_ratio": lambda c, fv, r: (
        _F(f"Expenses {c['expense_ratio'] * 100:.0f}% of inflow (> 80%).") if c["expense_ratio"] > 0.80
        else _P(f"Expenses {c['expense_ratio'] * 100:.0f}% of inflow (≤ 80%).")
    ),
    "exp_fixed_burden": lambda c, fv, r: _P(
        f"Fixed obligations ≈ {_rupees(c['fixed_burden'])}/mo ({c['fixed_burden_ratio'] * 100:.0f}% of inflow)."
    ),
    "exp_rent_burden": lambda c, fv, r: (
        _need("rent transactions") if c["rent_count"] == 0
        else _F(f"Rent {_ratio(c['monthly_rent'], fv['avg_monthly_inflow']) * 100:.0f}% of inflow (> 40%).")
        if _ratio(c["monthly_rent"], fv["avg_monthly_inflow"]) > 0.40
        else _P(f"Rent {_ratio(c['monthly_rent'], fv['avg_monthly_inflow']) * 100:.0f}% of inflow.")
    ),
    "exp_unusual": lambda c, fv, r: (
        _F(f"{c['round_credits']} large round-number credit(s) / abnormal debits present.")
        if c["round_credits"] >= 2 else _P("No abnormal large debits flagged.")
    ),

    # ── EMI / Existing Debt ────────────────────────────────────────────────
    "emi_existing_count": lambda c, fv, r: _P(f"{c['emi_txn_count']} EMI/NACH debit(s) detected."),
    "emi_total_burden": lambda c, fv, r: _P(f"EMI burden {c['emi_burden'] * 100:.0f}% of monthly inflow ({_rupees(c['monthly_emi'])})."),
    "emi_high_burden": lambda c, fv, r: (
        _F(f"EMI burden {c['emi_burden'] * 100:.0f}% (> 40%).") if c["emi_burden"] > 0.40
        else _P(f"EMI burden {c['emi_burden'] * 100:.0f}% (≤ 40%).")
    ),
    "emi_moderate_burden": lambda c, fv, r: (
        _F(f"EMI burden {c['emi_burden'] * 100:.0f}% (in 25-40% band).") if 0.25 <= c["emi_burden"] <= 0.40
        else _P(f"EMI burden {c['emi_burden'] * 100:.0f}% (outside 25-40%).")
    ),
    "emi_low_burden": lambda c, fv, r: (
        _P(f"EMI burden {c['emi_burden'] * 100:.0f}% (< 25%).") if c["emi_burden"] < 0.25
        else _F(f"EMI burden {c['emi_burden'] * 100:.0f}% (≥ 25%).")
    ),
    "emi_bounce": lambda c, fv, r: (
        _F(f"{c['bounce_count']} returned/failed debit(s).") if c["bounce_count"] > 0
        else _P("No failed/returned EMI.")
    ),
    "emi_multiple_lenders": lambda c, fv, r: (
        _F(f"Recurring payments to {c['emi_lenders']} distinct lenders (loan stacking).") if c["emi_lenders"] >= 3
        else _P(f"{c['emi_lenders']} distinct lender(s).")
    ),
    "emi_repayment_consistency": lambda c, fv, r: (
        _need("EMI transactions") if c["emi_txn_count"] == 0
        else _P(f"EMI paid in {c['emi_months']}/{c['n_months']} months, no bounces.")
        if c["bounce_count"] == 0 and c["emi_months"] >= max(1, c["n_months"] - 1)
        else _F(f"EMI irregular ({c['emi_months']}/{c['n_months']} months) or bounced.")
    ),

    # ── Banking Behaviour ──────────────────────────────────────────────────
    "bank_cheque_bounce": lambda c, fv, r: (
        _F(f"{c['bounce_count']} return/bounce entr(y/ies).") if c["bounce_count"] > 0 else _P("No cheque returns.")
    ),
    "bank_nach_failure": lambda c, fv, r: (
        _F(f"{c['bounce_count']} NACH/auto-debit failure(s).") if c["bounce_count"] > 0 else _P("No NACH failures.")
    ),
    "bank_ecs_return": lambda c, fv, r: (
        _F(f"{c['bounce_count']} ECS return(s).") if c["bounce_count"] > 0 else _P("No ECS returns.")
    ),
    "bank_reversals": lambda c, fv, r: (
        _F(f"{c['reversal_count']} reversal/refund entr(y/ies).") if c["reversal_count"] > 3
        else _P(f"{c['reversal_count']} reversal entr(y/ies).")
    ),
    "bank_dormant": lambda c, fv, r: (
        _F("A month with ≤1 transaction (dormant period).") if c["dormant"] and c["n_months"] >= 2
        else _P("No dormant months.")
    ),
    "bank_frequency": lambda c, fv, r: _P(f"{c['tx_per_month']:.0f} transactions / month."),
    "bank_active_usage": lambda c, fv, r: (
        _P("Regular credits and debits every month.")
        if c["n_months"] >= 1 and all(x > 0 for x in c["monthly_inflows"]) and all(x > 0 for x in c["monthly_outflows"])
        else _F("Some months missing credits or debits.")
    ),

    # ── GST / Tax ──────────────────────────────────────────────────────────
    "gst_payment_detected": lambda c, fv, r: (
        _P(f"{c['gst_count']} GST payment(s) ≈ {_rupees(c['monthly_gst'])}/mo.") if c["gst_count"] > 0
        else ("SKIP", "No GST payments in the statement.")
    ),
    "gst_consistency": lambda c, fv, r: (
        _P(f"GST paid across {c['gst_count']} entries.") if c["gst_count"] >= max(2, c["n_months"] - 1)
        else _F(f"GST paid only {c['gst_count']} time(s) in {c['n_months']} months.") if c["gst_count"] > 0
        else ("SKIP", "No GST payments to assess.")
    ),
    "tax_regularity": lambda c, fv, r: (
        _P(f"{c['tax_count']} tax payment(s) detected.") if c["tax_count"] > 0
        else ("SKIP", "No direct-tax payments in the statement.")
    ),

    # ── Salary / Employee (business paying salaries) ───────────────────────
    "sal_consistency": lambda c, fv, r: ("SKIP", "No outgoing payroll pattern detected."),

    # ── Cash Deposit / Withdrawal ─────────────────────────────────────────
    "cash_deposit_ratio": lambda c, fv, r: _P(f"Cash deposits {c['cash_deposit_ratio'] * 100:.0f}% of credits."),
    "cash_high_dep": lambda c, fv, r: (
        _F(f"Cash deposits {c['cash_deposit_ratio'] * 100:.0f}% of credits (> 40%).") if c["cash_deposit_ratio"] > 0.40
        else _P(f"Cash deposits {c['cash_deposit_ratio'] * 100:.0f}% of credits (≤ 40%).")
    ),
    "cash_withdrawal_ratio": lambda c, fv, r: (
        _F(f"Cash withdrawals {fv['cash_withdrawal_ratio'] * 100:.0f}% of debits (> 20%).") if fv["cash_withdrawal_ratio"] > 0.20
        else _P(f"Cash withdrawals {fv['cash_withdrawal_ratio'] * 100:.0f}% of debits.")
    ),
    "cash_freq_withdrawal": lambda c, fv, r: (
        _F(f"{c['cash_wd_count']} cash withdrawals ({c['cash_wd_count'] / c['months']:.1f}/mo).")
        if c["cash_wd_count"] / c["months"] > 6 else _P(f"{c['cash_wd_count']} cash withdrawal(s).")
    ),

    # ── UPI / Digital ─────────────────────────────────────────────────────
    "upi_receipts": lambda c, fv, r: _P(f"{c['upi_credit_count']} UPI credits ≈ {_rupees(c['upi_credit_amt'])}."),
    "upi_consistency": lambda c, fv, r: (
        _P(f"{c['upi_credit_count']} UPI credits across the period.") if c["upi_credit_count"] >= c["n_months"]
        else _F(f"Only {c['upi_credit_count']} UPI credits in {c['n_months']} months.")
    ),

    # ── Account Quality ──────────────────────────────────────────────────
    "acct_completeness": lambda c, fv, r: (
        _P(f"{c['n_months']} months available (≥ 6).") if c["n_months"] >= 6
        else _F(f"Only {c['n_months']} month(s) available (< 6).")
    ),
    "acct_missing_months": lambda c, fv, r: (
        _F(f"{c['zero_inflow_months']} month(s) with almost no activity.") if c["zero_inflow_months"] > 0
        else _P("No missing periods.")
    ),

    # ── Fraud / Anomaly ─────────────────────────────────────────────────
    "fraud_round_credits": lambda c, fv, r: (
        _F(f"{c['round_credits']} round ₹50k/₹1L credit(s).") if c["round_credits"] >= 3
        else _P(f"{c['round_credits']} round-number credit(s).")
    ),
    "fraud_same_day_in_out": lambda c, fv, r: (
        _F(f"{c['same_day_inout']} large credit(s) reversed out within 2 days.") if c["same_day_inout"] > 0
        else _P("No same-day in/out pattern.")
    ),
    "fraud_temp_inflation": lambda c, fv, r: (
        _F(f"Largest credit ({_rupees(c['largest_credit'])}) is {c['largest_credit_pct'] * 100:.0f}% of monthly inflow.")
        if c["largest_credit_pct"] > 2.0 else _P("No obvious balance-inflation credit.")
    ),
    "fraud_reversal_pattern": lambda c, fv, r: (
        _F(f"{c['reversal_count']} reversal entries.") if c["reversal_count"] > 3
        else _P(f"{c['reversal_count']} reversal entr(y/ies).")
    ),
    "fraud_duplicates": lambda c, fv, r: (
        _F(f"{c['rapid_dup']} same-payee same-amount transaction(s) within 2 days.") if c["rapid_dup"] >= 2
        else _P(f"{c['rapid_dup']} rapid duplicate(s).")
    ),
    "fraud_circular": lambda c, fv, r: (
        _F(f"{c['same_day_inout']} possible circular transfer(s).") if c["same_day_inout"] > 0
        else _P("No circular-transfer pattern.")
    ),
    "fraud_abnormal_spike": lambda c, fv, r: (
        _F("A month's activity is far above the others.")
        if len(c["monthly_counts"]) >= 3 and max(c["monthly_counts"]) > 3 * (sum(c["monthly_counts"]) / len(c["monthly_counts"]))
        else _P("No abnormal activity spike.")
    ),

    # ── Business Stability ──────────────────────────────────────────────
    "stab_revenue": lambda c, fv, r: (
        _need(2) if len(c["monthly_inflows"]) < 2
        else _P(f"Low monthly volatility (CV {c['inflow_cv']:.2f}).") if c["inflow_cv"] < 0.30
        else _F(f"Monthly volatility CV {c['inflow_cv']:.2f} (≥ 0.30).")
    ),
    "stab_growing_rev": lambda c, fv, r: (
        _need(6) if c["growth"] is None or c["n_months"] < 6
        else _P(f"Positive revenue trend (+{c['growth'] * 100:.0f}%).") if c["growth"] > 0.05
        else _F(f"Revenue trend {c['growth'] * 100:+.0f}% (not growing).")
    ),
    "stab_declining_rev": lambda c, fv, r: (
        _need(6) if c["growth"] is None or c["n_months"] < 6
        else _F(f"Declining revenue trend ({c['growth'] * 100:.0f}%).") if c["growth"] < -0.05
        else _P(f"Revenue trend {c['growth'] * 100:+.0f}% (not declining).")
    ),
    "stab_unpredictable": lambda c, fv, r: (
        _F(f"High unexplained volatility (CV {c['inflow_cv']:.2f} > 0.50).")
        if len(c["monthly_inflows"]) >= 2 and c["inflow_cv"] > 0.50
        else _P(f"Volatility within range (CV {c['inflow_cv']:.2f}).") if len(c["monthly_inflows"]) >= 2
        else _need(2)
    ),

    # ── Loan Eligibility & Decisioning ─────────────────────────────────
    CREDIT_SCORE_GATE_RULE_ID: lambda c, fv, r: (
        _P(f"Credit score {r['score']} > {settings_state.scoring.gate_threshold}.")
        if r["score"] > settings_state.scoring.gate_threshold
        else _F(f"Credit score {r['score']} ≤ {settings_state.scoring.gate_threshold} — hard cutoff.")
    ),
    "eligibility_strong_cashflow": lambda c, fv, r: (
        _P("Consistent revenue, positive surplus, EMI ≤ 40%, no bounces, healthy balance.")
        if (c["inflow_cv"] < 0.30 and c["total_credit"] > c["total_debit"] and c["emi_burden"] <= 0.40
            and c["bounce_count"] == 0 and c["avg_bal"] > 10000)
        else _F("One or more strong-cashflow conditions not met "
                f"(CV {c['inflow_cv']:.2f}, surplus {'yes' if c['total_credit'] > c['total_debit'] else 'no'}, "
                f"EMI {c['emi_burden'] * 100:.0f}%, bounces {c['bounce_count']}).")
    ),
    "eligibility_high_risk_flag": lambda c, fv, r: (
        _F("Revenue declining >20%, negative surplus AND EMI burden >40%.")
        if (c["growth"] is not None and c["growth"] < -0.20 and c["total_credit"] < c["total_debit"] and c["emi_burden"] > 0.40)
        else _P("High-risk composite not triggered.")
    ),
    "eligibility_stable_business": lambda c, fv, r: (
        _need(10) if c["n_months"] < 10
        else _P("Inflow consistency ≥ 70%, ≥10 active months, no bounces, no negative balance.")
        if (c["inflow_cv"] <= 0.30 and c["zero_inflow_months"] == 0 and c["bounce_count"] == 0 and c["neg_points"] == 0)
        else _F("One or more stable-business conditions not met.")
    ),
    "eligibility_concentration_risk": lambda c, fv, r: (
        _F(f"Top payee {c['top1_share'] * 100:.0f}% / top supplier {c['top_supplier_share'] * 100:.0f}% (> 50%).")
        if c["top1_share"] > 0.50 or c["top_supplier_share"] > 0.50
        else _P(f"Top payee {c['top1_share'] * 100:.0f}%, top supplier {c['top_supplier_share'] * 100:.0f}%.")
    ),
    "eligibility_transaction_risk": lambda c, fv, r: (
        _F(f"Same-day in/out ratio {c['same_day_ratio'] * 100:.0f}% (> 30%) or circular transfers.")
        if c["same_day_ratio"] > 0.30 or c["same_day_inout"] > 2
        else _P(f"Same-day in/out ratio {c['same_day_ratio'] * 100:.0f}%.")
    ),
}


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bre_rules(
    fv: dict, risk: dict, transactions: list[dict],
    opening_balance: float | None, enabled_rules: dict[str, bool],
) -> dict:
    c = build_context(fv, risk, transactions, opening_balance)

    results = []
    passed = failed = skipped = 0
    for cat in UNDERWRITING_RULE_CATEGORIES:
        for rule in cat["rules"]:
            rid = rule["id"]
            if not enabled_rules.get(rid):
                continue
            if c["is_salaried"] and rid in _BUSINESS_ONLY:
                status, detail = "SKIP", "Not applicable — salaried applicant (business/SME rule)."
            else:
                fn = _EVALUATORS.get(rid)
                if fn is None:
                    status, detail = "SKIP", "Not computable from a single bank statement — needs an external feed or longer history."
                else:
                    try:
                        status, detail = fn(c, fv, risk)
                    except Exception as exc:  # noqa: BLE001
                        status, detail = "SKIP", f"Could not evaluate ({exc})."
            results.append({
                "id": rid,
                "name": rule["name"],
                "category": cat["title"],
                "condition": rule["condition"],
                "signal": rule["signal"],
                "status": status,
                "detail": detail,
            })
            passed += status == "PASS"
            failed += status == "FAIL"
            skipped += status == "SKIP"

    gate = next((x for x in results if x["id"] == CREDIT_SCORE_GATE_RULE_ID), None)
    gate_enabled = gate is not None
    gate_passed = gate is None or gate["status"] == "PASS"

    # Collapse the 4 bounce/return rules (they detect the same event) into one
    # signal for the decision.
    _BOUNCE_FAMILY = {"emi_bounce", "bank_cheque_bounce", "bank_nach_failure", "bank_ecs_return"}
    serious_ids = {x["id"] for x in results if x["status"] == "FAIL" and x["id"] in _SERIOUS_FAIL}
    distinct_serious = len(serious_ids - _BOUNCE_FAMILY) + (1 if serious_ids & _BOUNCE_FAMILY else 0)
    serious_fails = [x for x in results if x["status"] == "FAIL" and x["id"] in _SERIOUS_FAIL]

    # The Credit Score Gate is the sole hard decision rule (see underwriting_rules.py).
    # Serious advisory fails downgrade an approval; they never override the gate.
    if gate_enabled and not gate_passed:
        decision = "REJECTED"
    elif distinct_serious >= 2:
        decision = "REJECTED"
    elif distinct_serious == 1:
        decision = "CONDITIONAL APPROVAL"
    elif failed > 0:
        decision = "APPROVED WITH NOTES"
    else:
        decision = "APPROVED"

    return {
        "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "creditScore": risk["score"],
        "riskGrade": risk["grade"],
        "gateThreshold": settings_state.scoring.gate_threshold,
        "gateEnabled": gate_enabled,
        "applicantProfile": "SALARIED" if c["is_salaried"] else "BUSINESS / SELF-EMPLOYED",
        "enabledCount": sum(1 for v in enabled_rules.values() if v),
        "evaluatedCount": passed + failed,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "seriousFlags": [x["name"] for x in serious_fails],
        "decision": decision,
        "results": results,
    }
