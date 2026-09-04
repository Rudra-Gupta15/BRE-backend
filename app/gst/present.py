"""Turn a scored GST business into a display object for the Model Testing page.

Mirrors the bank-statement result: a headline metric, a 12-point turnover chart
series, and a table of the key GST underwriting metrics.
"""
from __future__ import annotations


def _f(v, default: float = 0.0) -> float:
    try:
        f = float(str(v).replace(",", "").replace("₹", "").replace("%", "").strip())
        return f if f == f else default  # str(nan) parses back to nan without raising — catch it here
    except (TypeError, ValueError):
        return default


# metric_key -> (label, kind)  ·  kind: money | pct | num | text
_METRIC_FIELDS = [
    ("annualised_gst_turnover", "Annualised Turnover", "money"),
    ("monthly_turnover", "Monthly Turnover", "money"),
    ("quarterly_turnover", "Quarterly Turnover", "money"),
    ("total_taxable_turnover", "Taxable Turnover (latest)", "money"),
    ("turnover_growth_yoy", "Turnover Growth YoY", "pct"),
    ("turnover_growth_qoq", "Turnover Growth QoQ", "pct"),
    ("filing_regularity_percentage", "Filing Regularity", "pct"),
    ("missed_return_count", "Missed Returns", "num"),
    ("late_return_count", "Late Returns", "num"),
    ("net_itc_amount", "Net ITC", "money"),
    ("itc_claimed_amount", "ITC Claimed", "money"),
    ("gstr3b_net_tax_liability", "Net Tax Liability", "money"),
    ("top_buyer_sales_percentage", "Top Buyer Share", "pct"),
    ("unique_buyer_count", "Unique Buyers", "num"),
    ("buyer_concentration_level", "Buyer Concentration", "text"),
    ("business_vintage_years", "Business Vintage (yrs)", "num"),
    ("gst_status", "GST Status", "text"),
]


def curate_metrics(rec: dict) -> list[dict]:
    out = []
    for key, label, kind in _METRIC_FIELDS:
        if key not in rec or rec[key] in (None, ""):
            continue
        raw = rec[key]
        if kind == "text":
            out.append({"label": label, "value": str(raw).upper(), "kind": kind})
        else:
            out.append({"label": label, "value": _f(raw), "kind": kind})
    return out


def synth_series(rec: dict, n: int = 12) -> list[dict]:
    """A plausible n-month turnover series for a flat summary record: start from
    monthly_turnover and walk backwards by the reported MoM growth."""
    base = _f(rec.get("monthly_turnover")) or _f(rec.get("total_taxable_turnover")) \
        or (_f(rec.get("annualised_gst_turnover")) / 12)
    if base <= 0:
        return []
    g = _f(rec.get("turnover_growth_mom")) / 100.0
    g = max(-0.4, min(0.4, g))
    vals = [base]
    for _ in range(n - 1):
        prev = vals[0] / (1 + g) if (1 + g) else vals[0]
        vals.insert(0, max(0.0, round(prev, 2)))
    return [{"period": f"M{i + 1}", "turnover": round(v, 2), "filed": True, "onTime": None}
            for i, v in enumerate(vals)]


def business_view(rec: dict, prediction: dict, meta: dict | None = None) -> dict:
    """One scored business, ready for the GST result panel."""
    meta = meta or {}
    series = meta.get("monthlySeries") or synth_series(rec)
    return {
        "gstin": meta.get("gstin") or rec.get("gstin") or rec.get("customer_id"),
        "profile": {k: v for k, v in rec.items() if k != "_meta"},
        "prediction": {
            "underwritingScore": prediction.get("underwritingScore"),
            "riskFlag": prediction.get("riskFlag"),
            "riskProbability": prediction.get("riskProbability"),
            "headScores": prediction.get("headScores", {}),
            "topFactors": prediction.get("topFactors", []),
            "modelVersion": prediction.get("modelVersion"),
        },
        "metrics": curate_metrics(rec),
        "series": series,
        "periodsCovered": meta.get("periodsCovered"),
        "periodRange": meta.get("periodRange"),
        "returnsSeen": meta.get("returnsSeen", {}),
    }
