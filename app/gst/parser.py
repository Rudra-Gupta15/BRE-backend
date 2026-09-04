"""GST file parser — turns an uploaded GST file into a list of record dicts
keyed by the canonical column names (schema.CANONICAL). Handles
CSV / TSV / TXT, JSON, XLSX, and PDF (both one-cell-per-line and wide-column
table layouts)."""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from typing import Callable

from app.gst.schema import CANONICAL

logger = logging.getLogger(__name__)


def _norm(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


# Human labels (as printed on a GST return-summary PDF / form) → canonical field.
_LABEL_MAP = {
    "customer_id": "customer_id", "gstin": "gstin", "gst_number": "gstin",
    "legal_name": "legal_name", "trade_name": "trade_name",
    "gst_status": "gst_status", "status": "gst_status",
    "registration_date": "gst_registration_date",
    "vintage_yrs": "business_vintage_years", "vintage": "business_vintage_years",
    "return_period": "gst_return_period", "return_type": "return_type",
    "filing_date": "return_filing_date", "return_status": "return_status",
    "filing_frequency": "filing_frequency",
    "filing_delay_days": "filing_delay_days",
    "gstr_1_sales_value": "gstr1_sales_value",
    "gstr_3b_taxable_supply": "gstr3b_taxable_outward_supply",
    "total_taxable_turnover": "total_taxable_turnover",
    "b2b_sales": "b2b_sales_amount", "b2c_sales": "b2c_sales_amount",
    "b2b": "b2b_sales_percentage", "b2c": "b2c_sales_percentage",
    "export_sales": "export_sales_amount", "sez_sales": "sez_sales_amount",
    "reverse_charge_sales": "reverse_charge_sales_amount",
    "net_tax_liability": "gstr3b_net_tax_liability",
    "igst": "igst_amount", "cgst": "cgst_amount", "sgst": "sgst_amount",
    "cess": "cess_amount",
    "itc_available": "itc_available_amount", "itc_claimed": "itc_claimed_amount",
    "itc_reversed": "itc_reversed_amount", "net_itc": "net_itc_amount",
    "unique_buyers": "unique_buyer_count",
    "unique_b2b_buyers": "unique_b2b_buyer_count",
    "top_buyer": "top_buyer_sales_percentage",
    "buyer_concentration": "buyer_concentration_level",
    "monthly_turnover": "monthly_turnover",
    "quarterly_turnover": "quarterly_turnover",
    "annualised_turnover": "annualised_gst_turnover",
    "mom_growth": "turnover_growth_mom", "qoq_growth": "turnover_growth_qoq",
    "yoy_growth": "turnover_growth_yoy", "decline": "turnover_decline_percentage",
    "declining_quarters": "consecutive_declining_quarters",
    "filing_regularity": "filing_regularity_percentage",
    "missed_returns": "missed_return_count", "late_returns": "late_return_count",
    "proposed_loan": "proposed_loan_amount",
    "turnover_loan_ratio": "gst_turnover_to_loan_ratio",
    "max_loan_gst_rule": "maximum_loan_by_gst_rule",
    "data_completeness": "gst_data_completeness_score",
    "risk_flag": "gst_risk_flag", "underwriting_score": "gst_underwriting_score",
}

_NUM_RE = re.compile(r"-?[\d,]+\.?\d*")


def _val(field: str, raw: str):
    s = str(raw).strip().replace("₹", "").replace(",", "").replace("%", "").strip()
    text_fields = {"customer_id", "gstin", "legal_name", "trade_name", "gst_status",
                   "gst_registration_date", "gst_return_period", "return_type",
                   "return_filing_date", "return_status", "filing_frequency",
                   "buyer_concentration_level", "gst_risk_flag"}
    if field in text_fields:
        return raw.strip()
    m = _NUM_RE.search(s)
    return m.group(0) if m else s


def _row(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        nk = _norm(k)
        if nk:
            out[nk] = v
    return out


def _from_records(records: list) -> list[dict]:
    return [r for r in (_row(x) for x in records if isinstance(x, dict)) if r]


def _sniff_delim(sample: str) -> str:
    return max(("\t", ",", ";", "|"), key=lambda d: sample.count(d))


def _parse_delimited(text: str) -> list[dict]:
    delim = _sniff_delim(text[:4096])
    try:
        rows = list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delim))
    except csv.Error:
        return []
    # markdown table separator row ("---|---|---") sneaks in as a data row — drop it
    rows = [r for r in rows if not all(set(str(v).strip()) <= set("-: ") for v in r.values() if v)]
    return _from_records(rows)


def _parse_json(raw: str) -> list[dict]:
    obj = json.loads(raw)
    if isinstance(obj, list):
        return _from_records(obj)
    if isinstance(obj, dict):
        for key in ("records", "data", "rows", "gst", "items"):
            if isinstance(obj.get(key), list):
                return _from_records(obj[key])
        return _from_records([obj])
    return []


def _parse_xlsx(buf: bytes) -> list[dict]:
    import pandas as pd
    df = pd.read_excel(io.BytesIO(buf), dtype=str).fillna("")
    df.columns = [_norm(c) for c in df.columns]
    return [r for r in df.to_dict("records") if any(v not in ("", None) for v in r.values())]


def _parse_kv_pdf_or_text(lines: list[str]) -> list[dict]:
    """Single-business "label\\nvalue" summary (a GST return-summary form).
    Also works on a plain-text form. Returns [] if too few fields match."""
    rec: dict = {}
    for i, ln in enumerate(lines[:-1]):
        field = _LABEL_MAP.get(_norm(ln))
        if not field or field in rec:
            continue
        # value is the next non-empty, non-label line
        for nxt in lines[i + 1:i + 3]:
            if _norm(nxt) in _LABEL_MAP:
                break
            v = _val(field, nxt)
            if str(v).strip():
                rec[field] = v
                break
    return [rec] if len(rec) >= 8 else []


def _parse_pdf(buf: bytes) -> list[dict]:
    import pymupdf as fitz
    doc = fitz.open(stream=buf, filetype="pdf")
    lines = []
    for page in doc:
        lines.extend(ln.strip() for ln in page.get_text("text").splitlines() if ln.strip())
    doc.close()

    kv = _parse_kv_pdf_or_text(lines)
    if kv:
        return kv
    return _parse_pdf_table(lines)


def _parse_pdf_table(lines: list[str]) -> list[dict]:
    # wide-column layout
    for i, ln in enumerate(lines):
        toks = [_norm(t) for t in re.split(r"\s{2,}|\t|\|", ln) if t.strip()]
        if len(set(toks) & CANONICAL) >= 4:
            header, recs = toks, []
            for row in lines[i + 1:]:
                cells = [c.strip() for c in re.split(r"\s{2,}|\t|\|", row) if c.strip()]
                if len(cells) >= max(3, len(header) - 4):
                    recs.append(_row(dict(zip(header, cells))))
            if recs:
                return recs
    # one-token-per-line layout
    header, start_of_data = [], len(lines)
    for i, ln in enumerate(lines):
        if _norm(ln) in CANONICAL:
            header.append(_norm(ln))
        elif header:
            start_of_data = i
            break
    if len(header) < 4:
        return []
    n, hset, values = len(header), set(header), []
    for ln in lines[start_of_data:]:
        if _norm(ln) in hset or ln.lower().startswith("gst transaction data"):
            continue
        values.append(ln)
    return [_row(dict(zip(header, values[k:k + n]))) for k in range(0, len(values) - n + 1, n)] if len(values) >= n else []


# Alternate-schema flat summaries (a simplified per-customer export, not a
# full GSTR-return dump) use different field names and coarser granularity
# than CANONICAL. Left un-mapped, a row like {"annual_turnover": 8339975.91,
# "unique_buyers": 92, ...} carries none of the model's actual feature names,
# so predict() silently falls back to the TRAINING-SET MEAN for almost every
# input — every business ends up with nearly the same score regardless of how
# different its real numbers are. Each alternate field is derived into every
# canonical field it can honestly stand in for (never overwriting a canonical
# value the row already has for real), so genuine per-row variation actually
# reaches the model.
_ALT_DERIVATIONS: list[tuple[str, str, Callable[[float], float]]] = [
    ("annual_turnover", "annualised_gst_turnover", lambda v: v),
    ("annual_turnover", "total_taxable_turnover", lambda v: v / 12.0),
    ("annual_turnover", "monthly_turnover", lambda v: v / 12.0),
    ("annual_turnover", "quarterly_turnover", lambda v: v / 4.0),
    ("turnover_growth_pct", "turnover_growth_yoy", lambda v: v),
    ("turnover_growth_pct", "turnover_decline_percentage", lambda v: max(0.0, -v)),
    ("unique_buyers", "unique_buyer_count", lambda v: v),
    ("return_delay_count", "late_return_count", lambda v: v),
    ("filing_months_last_12", "filing_regularity_percentage", lambda v: min(100.0, v / 12.0 * 100)),
    ("filing_months_last_12", "missed_return_count", lambda v: max(0.0, 12 - v)),
]


def _apply_alt_derivations(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        for alt, canon, transform in _ALT_DERIVATIONS:
            if canon in r and str(r[canon]).strip() not in ("", "nan", "NA"):
                continue  # a real canonical value is already present — don't override it
            raw = r.get(alt)
            if raw in (None, "") or str(raw).strip() in ("", "nan", "NA"):
                continue
            try:
                v = float(str(raw).replace(",", "").replace("%", "").strip())
            except (TypeError, ValueError):
                continue
            r[canon] = round(transform(v), 4)
        out.append(r)
    return out


def parse_gst(buf: bytes, file_name: str) -> list[dict]:
    """Never raises — returns [] when nothing parses."""
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    try:
        if ext in ("csv", "tsv", "txt"):
            rows = _parse_delimited(buf.decode("utf-8", "ignore"))
            if not rows or not any(set(r) & CANONICAL for r in rows):
                rows = _parse_kv_pdf_or_text([ln.strip() for ln in buf.decode("utf-8", "ignore").splitlines() if ln.strip()])
            return _apply_alt_derivations(rows)
        if ext == "json":
            return _apply_alt_derivations(_parse_json(buf.decode("utf-8", "ignore")))
        if ext in ("xlsx", "xls"):
            return _apply_alt_derivations(_parse_xlsx(buf))
        if ext in ("pdf",):
            return _apply_alt_derivations(_parse_pdf(buf))
        if ext in ("md", "html", "htm"):
            rows = _parse_kv_pdf_or_text([ln.strip() for ln in re.sub(r"[|*#>`\-]{1,}", " ", buf.decode("utf-8", "ignore")).splitlines() if ln.strip()]) \
                or _parse_delimited(buf.decode("utf-8", "ignore"))
            return _apply_alt_derivations(rows)
        head = buf[:64].lstrip()
        if head[:1] in (b"[", b"{"):
            return _apply_alt_derivations(_parse_json(buf.decode("utf-8", "ignore")))
        return _apply_alt_derivations(_parse_delimited(buf.decode("utf-8", "ignore")))
    except Exception as exc:  # noqa: BLE001 — a bad file must never 500 the upload
        logger.warning("GST parse failed for %s: %s", file_name, exc)
        return []
