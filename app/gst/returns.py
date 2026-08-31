"""Parsers for the four GST return types used in credit underwriting:

    GSTR-1   outward supplies (sales)          — monthly / quarterly
    GSTR-3B  summary return (turnover + ITC)   — monthly
    GSTR-2A  auto-drafted inward supplies      — live view
    GSTR-2B  static ITC statement              — monthly

One uploaded file holds many rows: one per (gstin, tax period). Each row is
normalised to a flat dict keyed `gstin`, `period` (YYYY-MM), `return_type`, plus
the fields below. `aggregate.build_profile` then rolls a business's rows into the
53-field profile the model consumes.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re

logger = logging.getLogger(__name__)

RETURN_TYPES = ("GSTR1", "GSTR3B", "GSTR2A", "GSTR2B")

# Fields recognised per return type (a row may carry any subset).
GSTR1_FIELDS = {
    "filing_date", "due_date", "gross_total_value", "taxable_value",
    "b2b_value", "b2c_value", "export_value", "sez_value", "nil_rated_value",
    "cdnr_value", "total_igst", "total_cgst", "total_sgst", "total_cess",
    "b2b_invoice_count", "unique_buyer_count", "top_buyer_value",
}
GSTR3B_FIELDS = {
    "filing_date", "due_date", "outward_taxable_value", "outward_igst",
    "outward_cgst", "outward_sgst", "outward_cess", "zero_rated_value",
    "exempt_nil_value", "itc_igst", "itc_cgst", "itc_sgst", "itc_cess",
    "itc_total", "itc_reversed", "net_itc", "tax_payable", "tax_paid_cash",
    "tax_paid_credit",
}
GSTR2A_FIELDS = {
    "supplier_count", "total_invoice_value", "total_taxable_value",
    "itc_igst", "itc_cgst", "itc_sgst", "itc_cess", "top_supplier_value",
}
GSTR2B_FIELDS = {
    "itc_available_igst", "itc_available_cgst", "itc_available_sgst",
    "itc_available_cess", "itc_available_total", "itc_not_available_total",
    "supplier_count",
}
_FIELD_MAP = {
    "GSTR1": GSTR1_FIELDS, "GSTR3B": GSTR3B_FIELDS,
    "GSTR2A": GSTR2A_FIELDS, "GSTR2B": GSTR2B_FIELDS,
}


def _norm(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


def _clean_type(v) -> str | None:
    s = re.sub(r"[^A-Z0-9]", "", str(v).upper())
    for rt in RETURN_TYPES:
        if s == rt or s == rt.replace("GSTR", "GSTR-"):
            return rt
    if s in ("GSTR2", "2A"):
        return "GSTR2A"
    return None


def _period(v) -> str | None:
    s = str(v).strip()
    m = re.match(r"(\d{4})[-/ ]?(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2})[-/ ]?(\d{4})", s)   # MM-YYYY
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    return None


def _detect_type(row: dict) -> str | None:
    if row.get("return_type"):
        t = _clean_type(row["return_type"])
        if t:
            return t
    keys = set(row)
    best, score = None, 0
    for rt, fields in _FIELD_MAP.items():
        hit = len(keys & fields)
        if hit > score:
            best, score = rt, hit
    return best if score >= 2 else None


def _normalise_row(raw: dict) -> dict | None:
    row = {_norm(k): v for k, v in raw.items() if _norm(k)}
    gstin = row.get("gstin") or row.get("gst_number") or row.get("gstn")
    period = _period(row.get("period") or row.get("tax_period") or row.get("return_period") or "")
    if not gstin or not period:
        return None
    rtype = _detect_type(row)
    if not rtype:
        return None
    out = {"gstin": str(gstin).strip().upper(), "period": period, "return_type": rtype}
    for f in _FIELD_MAP[rtype]:
        if f in row and str(row[f]).strip() not in ("", "nan", "NA", "None"):
            out[f] = row[f]
    # keep a few common extras if present
    for extra in ("legal_name", "trade_name", "gst_status", "registration_date",
                  "constitution", "filing_frequency", "return_status"):
        if row.get(extra) not in (None, ""):
            out[extra] = row[extra]
    return out


# ── format readers ────────────────────────────────────────────────────────
def _rows_from_csv(text: str) -> list[dict]:
    if "\x00" in text[:2000] or sum(c < " " and c not in "\r\n\t" for c in text[:2000]) > 40:
        return []   # looks binary, not a CSV
    sample = text[:4096]
    delim = max(("\t", ",", ";", "|"), key=lambda d: sample.count(d))
    try:
        return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delim))
    except csv.Error:
        return []


def _rows_from_json(raw: str) -> list[dict]:
    obj = json.loads(raw)
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("returns", "records", "data", "rows", "filings", "items"):
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
        return [obj]
    return []


def _rows_from_xlsx(buf: bytes) -> list[dict]:
    import pandas as pd
    df = pd.read_excel(io.BytesIO(buf), dtype=str).fillna("")
    return df.to_dict("records")


def parse_returns_file(buf: bytes, file_name: str) -> list[dict]:
    """One file → list of normalised return rows. Empty list if the file holds
    no recognisable GST returns (never raises)."""
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    try:
        if ext in ("csv", "tsv", "txt"):
            raw_rows = _rows_from_csv(buf.decode("utf-8", "ignore"))
        elif ext == "json":
            raw_rows = _rows_from_json(buf.decode("utf-8", "ignore"))
        elif ext in ("xlsx", "xls"):
            raw_rows = _rows_from_xlsx(buf)
        elif ext in ("pdf", "html", "htm", "md"):
            raw_rows = []   # not a return-table format — handled by parser.parse_gst
        else:
            head = buf[:64].lstrip()
            raw_rows = (_rows_from_json(buf.decode("utf-8", "ignore"))
                        if head[:1] in (b"[", b"{")
                        else _rows_from_csv(buf.decode("utf-8", "ignore")))
    except Exception as exc:  # noqa: BLE001 — a bad file must never 500 the upload
        logger.warning("GST returns parse failed for %s: %s", file_name, exc)
        return []
    try:
        return [r for r in (_normalise_row(x) for x in raw_rows) if r]
    except Exception:  # noqa: BLE001
        logger.warning("GST returns normalise failed for %s", file_name, exc_info=True)
        return []


def looks_like_returns(buf: bytes, file_name: str) -> bool:
    """Cheap sniff: does this file contain GST return rows (vs the flat
    one-row-per-business summary, or something else entirely)?"""
    try:
        return len(parse_returns_file(buf[:20000], file_name)) > 0
    except Exception:  # noqa: BLE001
        return False
