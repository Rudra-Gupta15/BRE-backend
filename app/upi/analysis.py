"""UPI Transaction Data Enrichment — real signals mined out of one person's
UPI transaction history. No ML, no guessing here — counts/ratios/averages
computed directly from the dates, amounts and counterparties that are
actually there. Twin of app.bbps.analysis.
"""

from __future__ import annotations

from collections import defaultdict

from app.common.dates import parse_tx_date as _parse_tx_date
from app.upi.schema import HIGH_RISK_MCC

_MCC_LABELS = {
    "5411": "Groceries", "5812": "Restaurants", "5541": "Fuel", "4900": "Utilities",
    "5399": "E-commerce", "5311": "Department Stores", "5732": "Electronics",
    "5912": "Pharmacy", "4111": "Transport", "5941": "Sporting Goods",
    "7995": "Gambling", "6051": "Money Transfer / Quasi-Cash", "5933": "Pawn Shop",
    "7800": "Government Lottery", "6211": "Securities / Brokerage",
}


def analyze_upi(transactions: list[dict]) -> dict:
    """Real, computed UPI signals from one person's transaction log."""
    txns = [t for t in (transactions or []) if t.get("date") and t.get("amount") is not None]
    if not txns:
        return {
            "available": False,
            "message": "No UPI transactions found in this file.",
        }

    dated = [(t, _parse_tx_date(t.get("date"))) for t in txns]
    dated = [(t, d) for t, d in dated if d]
    if not dated:
        return {
            "available": False,
            "message": "UPI transactions found, but none had a readable date.",
        }

    months = {(d.year, d.month) for _, d in dated}
    span_months = len(months) or 1
    days_span = max(1, (max(d for _, d in dated) - min(d for _, d in dated)).days + 1)

    total = len(dated)
    p2p = [(t, d) for t, d in dated if t.get("mode") == "P2P"]
    p2m = [(t, d) for t, d in dated if t.get("mode") == "P2M"]
    succeeded = [t for t, _ in dated if t.get("status") == "SUCCESS"]
    failed = [t for t, _ in dated if t.get("status") == "FAILED"]

    debits = [(t, d) for t, d in dated if t.get("type") == "DEBIT"]
    credits = [(t, d) for t, d in dated if t.get("type") == "CREDIT"]

    payees = {t["payee_vpa"] for t, _ in debits if t.get("payee_vpa")}
    payers = {t["payer_vpa"] for t, _ in credits if t.get("payer_vpa")}

    # A payee paid in >= 2 distinct months is a recurring relationship
    # (subscription, regular merchant, informal recurring lending, ...).
    payee_months: dict[str, set] = defaultdict(set)
    for t, d in debits:
        vpa = t.get("payee_vpa")
        if vpa:
            payee_months[vpa].add((d.year, d.month))
    recurring_payees = [vpa for vpa, ms in payee_months.items() if len(ms) >= 2]

    per_day: dict = defaultdict(int)
    for _, d in dated:
        per_day[d] += 1
    peak_day_count = max(per_day.values(), default=0)

    weekend_amt = sum(t["amount"] for t, d in p2m if t.get("type") == "DEBIT" and d.weekday() >= 5)
    weekday_amt = sum(t["amount"] for t, d in p2m if t.get("type") == "DEBIT" and d.weekday() < 5)
    p2m_debit_total = weekend_amt + weekday_amt

    high_risk_amt = sum(t["amount"] for t, _ in p2m
                        if t.get("type") == "DEBIT" and (t.get("mcc") or "") in HIGH_RISK_MCC)

    p2p_debit_amt = sum(t["amount"] for t, _ in p2p if t.get("type") == "DEBIT")
    p2p_credit_amt = sum(t["amount"] for t, _ in p2p if t.get("type") == "CREDIT")

    by_mcc: dict[str, list] = defaultdict(list)
    for t, _ in p2m:
        by_mcc[t.get("mcc") or "OTHER"].append(t)
    mcc_rows = []
    for mcc, rows in by_mcc.items():
        total_amt = sum(t["amount"] for t in rows)
        mcc_rows.append({
            "mcc": mcc, "label": _MCC_LABELS.get(mcc, "Other"),
            "paymentCount": len(rows), "totalAmount": round(total_amt, 2),
            "highRisk": mcc in HIGH_RISK_MCC,
        })
    mcc_rows.sort(key=lambda r: r["totalAmount"], reverse=True)

    all_amounts = [t["amount"] for t, _ in dated]

    return {
        "available": True,
        "spanMonths": span_months,
        "totalTransactions": total,
        "p2pCount": len(p2p),
        "p2mCount": len(p2m),
        "p2pRatio": round(len(p2p) / total, 4),
        "p2mRatio": round(len(p2m) / total, 4),
        "successRatio": round(len(succeeded) / total, 4),
        "failedRatio": round(len(failed) / total, 4),
        "failedCount": len(failed),
        "uniquePayees": len(payees),
        "uniquePayers": len(payers),
        "recurringPayeeCount": len(recurring_payees),
        "dailyAvgTransactions": round(total / days_span, 3),
        "peakDayTransactionCount": peak_day_count,
        "weekendSpendPct": round((weekend_amt / p2m_debit_total * 100) if p2m_debit_total else 0.0, 1),
        "avgTicketSize": round(sum(all_amounts) / total, 2),
        "highRiskMccSpendPct": round((high_risk_amt / p2m_debit_total * 100) if p2m_debit_total else 0.0, 1),
        "p2pDebitVelocity": round(p2p_debit_amt / span_months, 2),
        "p2pCreditVelocity": round(p2p_credit_amt / span_months, 2),
        "byMcc": mcc_rows,
    }
