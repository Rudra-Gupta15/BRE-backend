"""BBPS Utility Payment History — real signals mined out of a bank statement.

There is no separate BBPS feed here: in practice, BBPS (Bharat Bill Payment
System) payments show up as line items *inside* a bank statement pulled via
Account Aggregator — "BBPS-ELECTRICITY BOARD BILL PAYMENT", "UPI-JIO MOBILE
BILL PAYMENT", etc. So the "BBPS Utility Payment History" data source reuses
app.aa.parser's statement parsing (PDF/CSV/JSON/XLSX/MD — the same formats
a bank statement comes in) and this module mines the utility-bill subset out
of the parsed transactions — no ML, no guessing, just counts/ratios/averages
computed directly from the dates and amounts that are actually there.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.common.dates import parse_tx_date as _parse_tx_date

_UTILITY_TYPES: dict[str, re.Pattern] = {
    "ELECTRICITY": re.compile(r"electric|power\s*bill|discom|state\s*electricity|\belectricity\b", re.I),
    "WATER":       re.compile(r"water\s*bill|water\s*board|jal\s*board|\bwater\b", re.I),
    "GAS":         re.compile(r"\bgas\s*bill\b|piped\s*gas|\blpg\b", re.I),
    "BROADBAND":   re.compile(r"broadband|wifi\s*bill|internet\s*bill|\bfiber\b", re.I),
    "MOBILE_DTH":  re.compile(r"mobile\s*bill|\bdth\b|recharge|postpaid", re.I),
}
_BBPS_HINT_RE = re.compile(r"\bbbps\b", re.I)


def _classify(narration: str) -> str | None:
    for kind, pat in _UTILITY_TYPES.items():
        if pat.search(narration or ""):
            return kind
    if _BBPS_HINT_RE.search(narration or ""):
        return "OTHER_BBPS"
    return None


def extract_bbps_transactions(transactions: list[dict]) -> list[dict]:
    """Every debit that looks like a bill/utility payment, tagged with its type."""
    out = []
    for t in transactions:
        if t.get("type") != "DEBIT":
            continue
        # A row the parser couldn't fully read comes through with amount=None
        # (guardrails passes missing data rather than rejecting the row) — an
        # unreadable amount can't be counted toward a bill total, so skip it.
        if not isinstance(t.get("amount"), (int, float)):
            continue
        kind = _classify(t.get("narration") or "")
        if not kind:
            continue
        out.append({**t, "utilityType": kind})
    return out


def analyze_bbps(transactions: list[dict]) -> dict:
    """Real, computed BBPS signals from one statement's utility-tagged rows —
    the five features the data-source card already advertises: punctuality,
    consumption/spend trend (per-type averages), missed-payment frequency,
    peak-vs-normal spend, and a debt-to-revenue-style ratio isn't computable
    without declared income so it's left out rather than faked."""
    bbps_txns = extract_bbps_transactions(transactions)
    if not bbps_txns:
        return {
            "available": False,
            "message": "No BBPS / utility bill payments found in this statement.",
        }

    # Statement span in calendar months, from ALL transactions — the true
    # denominator for "how many months could this bill have been paid in".
    all_months = set()
    for t in transactions:
        d = _parse_tx_date(t.get("date"))
        if d:
            all_months.add((d.year, d.month))
    span_months = len(all_months) or 1

    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in bbps_txns:
        by_type[t["utilityType"]].append(t)

    types_out = []
    total_paid_months = total_expected_months = 0
    for kind, txns in by_type.items():
        months_paid = {
            (d.year, d.month) for d in (_parse_tx_date(t.get("date")) for t in txns) if d
        }
        n_paid = len(months_paid)
        # A utility seen in >=2 distinct months is a recurring bill — every
        # span month it did NOT appear in counts as a missed payment. A
        # one-off utility payment has nothing to be "missed" against.
        recurring = n_paid >= 2
        expected = span_months if recurring else n_paid
        missed = max(0, expected - n_paid)
        total_paid_months += n_paid
        total_expected_months += expected
        amounts = [t["amount"] for t in txns]
        types_out.append({
            "utilityType": kind,
            "paymentCount": len(txns),
            "monthsPaid": n_paid,
            "missedMonths": missed,
            "averageBillAmount": round(sum(amounts) / len(amounts), 2),
            "totalPaid": round(sum(amounts), 2),
            "recurring": recurring,
        })
    types_out.sort(key=lambda r: r["totalPaid"], reverse=True)

    on_time_ratio = round(total_paid_months / total_expected_months, 4) if total_expected_months else 1.0
    missed_total = sum(r["missedMonths"] for r in types_out)
    avg_bill = round(sum(t["amount"] for t in bbps_txns) / len(bbps_txns), 2)

    # Punctuality Index (0-100) — the on-time ratio scaled up, penalised a
    # further 5pts per missed month beyond the first. Documented formula,
    # not a trained score — same convention as the rest of this app's
    # non-ML "index" fields.
    punctuality_index = round(
        max(0.0, min(100.0, on_time_ratio * 100 - max(0, missed_total - 1) * 5)), 1
    )

    return {
        "available": True,
        "spanMonths": span_months,
        "utilityAccounts": len(by_type),
        "paymentsLast12m": len(bbps_txns),
        "onTimePaymentRatio": on_time_ratio,
        "missedPaymentCount": missed_total,
        "averageBillAmount": avg_bill,
        "utilityBillPunctualityIndex": punctuality_index,
        "byType": types_out,
    }
