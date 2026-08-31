"""Roll a business's GST return rows (GSTR-1 / 3B / 2A / 2B across many months)
into the 53-field profile that gst_model.predict consumes."""
from __future__ import annotations

import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)


def _f(v, default=0.0) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _pdate(v) -> date | None:
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _period_key(p: str) -> int:
    y, m = p.split("-")
    return int(y) * 12 + int(m)


def _growth(series: list[float], lag: int) -> float:
    """Percent change of the latest value vs `lag` periods earlier."""
    if len(series) <= lag or series[-1 - lag] <= 0:
        return 0.0
    return round((series[-1] - series[-1 - lag]) / series[-1 - lag] * 100, 4)


def _concentration_band(top_pct: float) -> str:
    if top_pct >= 45:
        return "HIGH"
    if top_pct >= 25:
        return "MEDIUM"
    return "LOW"


def build_profile(rows: list[dict], *, proposed_loan_amount: float | None = None) -> dict:
    """`rows` = every normalised return row for ONE gstin (mixed types / periods).
    Returns a profile dict with the model's field names + a `_meta` block."""
    if not rows:
        return {}
    gstin = rows[0]["gstin"]

    by_period: dict[str, dict[str, dict]] = {}
    meta_bits: dict[str, str] = {}
    for r in rows:
        by_period.setdefault(r["period"], {})[r["return_type"]] = r
        for k in ("legal_name", "trade_name", "gst_status", "constitution",
                  "filing_frequency", "registration_date"):
            if r.get(k) and k not in meta_bits:
                meta_bits[k] = str(r[k])

    periods = sorted(by_period, key=_period_key)
    g1 = [by_period[p].get("GSTR1") for p in periods]
    g3 = [by_period[p].get("GSTR3B") for p in periods]
    g2a = [by_period[p].get("GSTR2A") for p in periods]
    g2b = [by_period[p].get("GSTR2B") for p in periods]

    # ── turnover series (prefer 3B outward, fall back to GSTR-1 taxable) ────
    turnover = []
    monthly_series = []
    for p in periods:
        b3, b1 = by_period[p].get("GSTR3B"), by_period[p].get("GSTR1")
        v = _f(b3.get("outward_taxable_value")) if b3 else 0.0
        if v <= 0 and b1:
            v = _f(b1.get("taxable_value")) or _f(b1.get("gross_total_value"))
        turnover.append(v)
        status = str((b3 or {}).get("return_status", "")).upper()
        fd, dd = _pdate((b3 or {}).get("filing_date")), _pdate((b3 or {}).get("due_date"))
        on_time = None
        if fd and dd:
            on_time = (fd - dd).days <= 0
        elif status:
            on_time = status in ("FILED", "ONTIME")
        monthly_series.append({
            "period": p,
            "turnover": round(v, 2),
            "filed": bool(b3 or b1),
            "onTime": on_time,
        })
    turnover = [t for t in turnover if t > 0] or [0.0]

    monthly_turnover = turnover[-1]
    quarterly_turnover = sum(turnover[-3:])
    annualised = sum(turnover[-12:]) if len(turnover) >= 12 else (
        sum(turnover) / len(turnover) * 12)

    yoy = _growth(turnover, 12)
    qoq = _growth(turnover, 3)
    mom = _growth(turnover, 1)
    decline_pct = abs(min(0.0, yoy))
    declining_q = 0
    for i in range(len(turnover) - 1, 0, -1):
        if turnover[i] < turnover[i - 1]:
            declining_q += 1
        else:
            break
    declining_q = min(4, declining_q)

    # ── filing behaviour (from GSTR-3B) ───────────────────────────────────
    filed_3b = [b for b in g3 if b]
    delays, late, on_time = [], 0, 0
    for b in filed_3b:
        fd, dd = _pdate(b.get("filing_date")), _pdate(b.get("due_date"))
        status = str(b.get("return_status", "")).upper()
        if fd and dd:
            d = max(0, (fd - dd).days)
            delays.append(d)
            late += d > 0
            on_time += d == 0
        elif status:
            late += status == "LATE"
            on_time += status in ("FILED", "ONTIME")
    n_expected = len(periods)
    missed = max(0, n_expected - len(filed_3b))
    filing_regularity = round(100 * on_time / max(1, len(filed_3b)), 2) if filed_3b else 50.0
    avg_delay = round(sum(delays) / len(delays), 1) if delays else 0.0

    # ── GSTR-1 vs GSTR-3B mismatch ───────────────────────────────────────
    mism = []
    for p in periods:
        b1, b3 = by_period[p].get("GSTR1"), by_period[p].get("GSTR3B")
        if b1 and b3:
            a = _f(b1.get("taxable_value")) or _f(b1.get("gross_total_value"))
            b = _f(b3.get("outward_taxable_value"))
            if b > 0:
                mism.append(abs(a - b) / b * 100)
    gstr1_vs_gstr3b_mismatch_pct = round(sum(mism) / len(mism), 2) if mism else 0.0

    # ── sales mix (mean over GSTR-1) ─────────────────────────────────────
    def _mean_g1(fn):
        vals = [fn(b) for b in g1 if b]
        return sum(vals) / len(vals) if vals else 0.0

    b2b_amt = _mean_g1(lambda b: _f(b.get("b2b_value")))
    b2c_amt = _mean_g1(lambda b: _f(b.get("b2c_value")))
    mix_total = b2b_amt + b2c_amt or monthly_turnover or 1.0
    b2b_pct = round(100 * b2b_amt / mix_total, 2) if mix_total else 60.0
    b2c_pct = round(100 - b2b_pct, 2)
    export_amt = _mean_g1(lambda b: _f(b.get("export_value")))
    sez_amt = _mean_g1(lambda b: _f(b.get("sez_value")))

    latest_g1 = next((b for b in reversed(g1) if b), None)
    unique_buyers = _f(latest_g1.get("unique_buyer_count")) if latest_g1 else 0.0
    unique_b2b = _f(latest_g1.get("b2b_invoice_count")) if latest_g1 else 0.0
    top_buyer_val = _f(latest_g1.get("top_buyer_value")) if latest_g1 else 0.0
    top_buyer_pct = round(100 * top_buyer_val / monthly_turnover, 2) if monthly_turnover else 20.0
    top_buyer_pct = min(95.0, max(0.0, top_buyer_pct))

    # ── tax + ITC (latest GSTR-3B, cross-checked with 2B/2A) ──────────────
    latest_3b = next((b for b in reversed(g3) if b), None)
    latest_2b = next((b for b in reversed(g2b) if b), None)
    latest_2a = next((b for b in reversed(g2a) if b), None)

    if latest_3b:
        igst = _f(latest_3b.get("outward_igst"))
        cgst = _f(latest_3b.get("outward_cgst"))
        sgst = _f(latest_3b.get("outward_sgst"))
        cess = _f(latest_3b.get("outward_cess"))
        net_tax_liability = _f(latest_3b.get("tax_payable")) or (igst + cgst + sgst + cess)
        itc_claimed = _f(latest_3b.get("itc_total")) or (
            _f(latest_3b.get("itc_igst")) + _f(latest_3b.get("itc_cgst"))
            + _f(latest_3b.get("itc_sgst")) + _f(latest_3b.get("itc_cess")))
        itc_reversed = _f(latest_3b.get("itc_reversed"))
        net_itc = _f(latest_3b.get("net_itc")) or max(0.0, itc_claimed - itc_reversed)
    else:
        igst = cgst = sgst = cess = net_tax_liability = 0.0
        itc_claimed = itc_reversed = net_itc = 0.0

    itc_available = (
        _f(latest_2b.get("itc_available_total")) if latest_2b else
        _f(latest_2a.get("total_taxable_value")) if latest_2a else
        itc_claimed)
    suppliers = _f((latest_2b or latest_2a or {}).get("supplier_count"))

    # ── vintage / status ─────────────────────────────────────────────────
    reg = _pdate(meta_bits.get("registration_date"))
    if reg:
        vintage = round((date.today() - reg).days / 365.25, 2)
    else:
        first = _pdate(f"{periods[0]}-01")
        vintage = round((date.today() - first).days / 365.25, 2) if first else 3.0
    status = (meta_bits.get("gst_status") or "ACTIVE").upper()

    # ── loan sizing ──────────────────────────────────────────────────────
    ratio = round(annualised / proposed_loan_amount, 2) if proposed_loan_amount else 2.0
    if not proposed_loan_amount:
        proposed_loan_amount = round(annualised / ratio, 2)
    max_loan = round(annualised * 0.40, 2)

    profile = {
        "customer_id": gstin,   # dedup key for the training corpus
        "gstin": gstin,
        "legal_name": meta_bits.get("legal_name", ""),
        "trade_name": meta_bits.get("trade_name", ""),
        "gst_status": status,
        "return_type": "GSTR3B",
        "return_status": "LATE" if late and not on_time else "FILED",
        "filing_frequency": (meta_bits.get("filing_frequency") or "MONTHLY").upper(),
        "filing_delay_days": avg_delay,
        "gstr1_sales_value": round(_mean_g1(lambda b: _f(b.get("taxable_value")) or _f(b.get("gross_total_value"))), 2),
        "gstr3b_taxable_outward_supply": round(monthly_turnover, 2),
        "gstr3b_net_tax_liability": round(net_tax_liability, 2),
        "total_taxable_turnover": round(monthly_turnover, 2),
        "b2b_sales_amount": round(b2b_amt, 2),
        "b2c_sales_amount": round(b2c_amt, 2),
        "b2b_sales_percentage": b2b_pct,
        "b2c_sales_percentage": b2c_pct,
        "export_sales_amount": round(export_amt, 2),
        "sez_sales_amount": round(sez_amt, 2),
        "reverse_charge_sales_amount": 0.0,
        "igst_amount": round(igst, 2),
        "cgst_amount": round(cgst, 2),
        "sgst_amount": round(sgst, 2),
        "cess_amount": round(cess, 2),
        "itc_available_amount": round(itc_available, 2),
        "itc_claimed_amount": round(itc_claimed, 2),
        "itc_reversed_amount": round(itc_reversed, 2),
        "net_itc_amount": round(net_itc, 2),
        "unique_buyer_count": int(unique_buyers),
        "unique_b2b_buyer_count": int(unique_b2b),
        "top_buyer_sales_percentage": top_buyer_pct,
        "buyer_concentration_level": _concentration_band(top_buyer_pct),
        "monthly_turnover": round(monthly_turnover, 2),
        "quarterly_turnover": round(quarterly_turnover, 2),
        "turnover_growth_mom": mom,
        "turnover_growth_qoq": qoq,
        "turnover_growth_yoy": yoy,
        "turnover_decline_percentage": round(decline_pct, 2),
        "consecutive_declining_quarters": declining_q,
        "filing_regularity_percentage": filing_regularity,
        "missed_return_count": int(missed),
        "late_return_count": int(late),
        "gstr1_vs_gstr3b_mismatch_pct": gstr1_vs_gstr3b_mismatch_pct,
        "business_vintage_years": vintage,
        "annualised_gst_turnover": round(annualised, 2),
        "proposed_loan_amount": round(proposed_loan_amount, 2),
        "gst_turnover_to_loan_ratio": ratio,
        "maximum_loan_by_gst_rule": max_loan,
        "gst_data_completeness_score": 1.0,
        "_meta": {
            "gstin": gstin,
            "periodsCovered": len(periods),
            "periodRange": [periods[0], periods[-1]] if periods else None,
            "returnsSeen": {
                "GSTR1": sum(1 for b in g1 if b),
                "GSTR3B": sum(1 for b in g3 if b),
                "GSTR2A": sum(1 for b in g2a if b),
                "GSTR2B": sum(1 for b in g2b if b),
            },
            "gstr1_vs_gstr3b_mismatch_pct": gstr1_vs_gstr3b_mismatch_pct,
            "suppliers": int(suppliers),
            "monthlySeries": monthly_series,
        },
    }
    return profile


def build_profiles(rows: list[dict]) -> list[dict]:
    """Group mixed return rows by gstin and build one profile each."""
    by_gstin: dict[str, list[dict]] = {}
    for r in rows:
        by_gstin.setdefault(r["gstin"], []).append(r)
    return [build_profile(rs) for rs in by_gstin.values()]
