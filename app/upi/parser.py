"""UPI transaction log parser — turns an uploaded file into a list of real
transaction dicts keyed by app.upi.schema.CANONICAL. Handles CSV / TSV / TXT,
JSON, XLSX, and PDF (a digitally-exported UPI statement — selectable text,
not a scan, so no vision LLM is needed here, unlike app.aa.parser).

Twin of app.gst.parser, but a transaction LIST rather than one profile row.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re

from app.upi.schema import CANONICAL

logger = logging.getLogger(__name__)


def _norm(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


# Human column headers (as printed on a UPI app's exported statement / a PSP
# report) → canonical field.
_LABEL_MAP = {
    "transaction_id": "transaction_id", "txn_id": "transaction_id", "rrn": "transaction_id",
    "upi_ref_no": "transaction_id", "reference_no": "transaction_id", "utr": "transaction_id",
    "date": "date", "txn_date": "date", "transaction_date": "date",
    "time": "time", "txn_time": "time", "transaction_time": "time",
    "type": "type", "txn_type": "type", "transaction_type": "type", "debit_credit": "type",
    "amount": "amount", "txn_amount": "amount", "transaction_amount": "amount", "amount_inr": "amount",
    "payer_vpa": "payer_vpa", "from_vpa": "payer_vpa", "sender_vpa": "payer_vpa",
    "payer_name": "payer_name", "from_name": "payer_name", "sender_name": "payer_name",
    "payee_vpa": "payee_vpa", "to_vpa": "payee_vpa", "receiver_vpa": "payee_vpa",
    "payee_name": "payee_name", "to_name": "payee_name", "receiver_name": "payee_name", "merchant_name": "payee_name",
    "mode": "mode", "txn_mode": "mode", "payment_mode": "mode",
    "mcc": "mcc", "merchant_category_code": "mcc", "category_code": "mcc",
    "status": "status", "txn_status": "status", "transaction_status": "status",
    "remarks": "remarks", "note": "remarks", "narration": "remarks", "description": "remarks",
}

_NUM_RE = re.compile(r"-?[\d,]+\.?\d*")


def _row(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        field = _LABEL_MAP.get(_norm(k))
        if field:
            out[field] = v
    return out


def _clean_amount(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).replace("₹", "").replace(",", "").strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return abs(float(m.group(0)))
    except ValueError:
        return None


def _clean_row(r: dict) -> dict | None:
    """Coerce a raw parsed row into a well-typed transaction dict, or None if
    it's missing the fields a transaction needs (date + amount + type)."""
    if not r.get("date") or r.get("amount") is None or not r.get("type"):
        return None
    amt = _clean_amount(r.get("amount"))
    if amt is None:
        return None
    ttype = str(r.get("type", "")).strip().upper()
    if ttype not in ("DEBIT", "CREDIT"):
        # tolerate "DR"/"CR" or "sent"/"received"
        ttype = "DEBIT" if ttype.startswith(("D", "S")) else "CREDIT" if ttype.startswith(("C", "R")) else None
    if ttype is None:
        return None
    mode = str(r.get("mode", "")).strip().upper()
    if mode not in ("P2P", "P2M"):
        mode = "P2M" if r.get("mcc") else "P2P"
    status = str(r.get("status", "")).strip().upper() or "SUCCESS"
    if status not in ("SUCCESS", "FAILED", "PENDING", "REVERSED"):
        status = "SUCCESS"
    return {
        "transaction_id": str(r.get("transaction_id") or "").strip() or None,
        "date": str(r.get("date")).strip(),
        "time": str(r.get("time") or "").strip() or None,
        "type": ttype,
        "amount": round(amt, 2),
        "payer_vpa": str(r.get("payer_vpa") or "").strip() or None,
        "payer_name": str(r.get("payer_name") or "").strip() or None,
        "payee_vpa": str(r.get("payee_vpa") or "").strip() or None,
        "payee_name": str(r.get("payee_name") or "").strip() or None,
        "mode": mode,
        "mcc": str(r.get("mcc")).strip() if r.get("mcc") not in (None, "") else None,
        "status": status,
        "remarks": str(r.get("remarks") or "").strip() or None,
    }


def _from_records(records: list) -> list[dict]:
    out = []
    for x in records:
        if not isinstance(x, dict):
            continue
        c = _clean_row(_row(x))
        if c:
            out.append(c)
    return out


def _sniff_delim(sample: str) -> str:
    return max(("\t", ",", ";", "|"), key=lambda d: sample.count(d))


def _parse_delimited(text: str) -> list[dict]:
    delim = _sniff_delim(text[:4096])
    try:
        rows = list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delim))
    except csv.Error:
        return []
    rows = [r for r in rows if not all(set(str(v).strip()) <= set("-: ") for v in r.values() if v)]
    return _from_records(rows)


def _parse_json(raw: str) -> list[dict]:
    obj = json.loads(raw)
    if isinstance(obj, list):
        return _from_records(obj)
    if isinstance(obj, dict):
        for key in ("transactions", "records", "data", "rows", "upi", "items"):
            if isinstance(obj.get(key), list):
                return _from_records(obj[key])
        return _from_records([obj])
    return []


def _parse_xlsx(buf: bytes) -> list[dict]:
    import pandas as pd

    # A real export often has a title/caption row above the actual header
    # ("UPI Transaction History — Name (vpa) | Profile: ...") — pandas reads
    # row 0 as the header regardless, so if none of THOSE columns map to
    # anything we know, search the next few rows for the real header row.
    raw = pd.read_excel(io.BytesIO(buf), dtype=str, header=None).fillna("")
    header_row = 0
    for i in range(min(5, len(raw))):
        norms = {_norm(v) for v in raw.iloc[i].tolist()}
        if norms & set(_LABEL_MAP):
            header_row = i
            break
    df = pd.read_excel(io.BytesIO(buf), dtype=str, header=header_row).fillna("")
    df.columns = [_norm(c) for c in df.columns]
    return _from_records(df.to_dict("records"))


def _parse_pdf_table(lines: list[str]) -> list[dict]:
    canon_norms = {_norm(c) for c in _LABEL_MAP}
    for i, ln in enumerate(lines):
        toks = [_norm(t) for t in re.split(r"\s{2,}|\t|\|", ln) if t.strip()]
        if len(set(toks) & canon_norms) >= 3:
            header, recs = toks, []
            for row in lines[i + 1:]:
                cells = [c.strip() for c in re.split(r"\s{2,}|\t|\|", row) if c.strip()]
                if len(cells) >= max(3, len(header) - 3):
                    recs.append(dict(zip(header, cells)))
            if recs:
                return _from_records(recs)
    return []


_DATE_TOKEN_RE = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")


def _parse_pdf_positional(buf: bytes) -> list[dict]:
    """Reconstructs a real-world statement layout pymupdf's plain-text mode
    can't: a "Date / Time" and "Counterparty / Details" cell each stacking
    several lines (name, VPA, mode+MCC, remarks), and separate Debit/Credit
    columns whose meaning (which one holds the amount) only plain x/y
    position — not the linear text order — can tell apart.

    Groups words into visual lines (pymupdf's own block/line index), finds
    the header row's column x-positions, then walks the body bucketing each
    line into its column by x — a new "Date / Time" cell whose first line is
    an actual date starts a new transaction; every other line accumulates
    into the transaction currently being built."""
    import pymupdf as fitz

    doc = fitz.open(stream=buf, filetype="pdf")

    # ── 1. Extract every page's words as visual lines, tagged with a global
    # (page_index, y0) sort key — a multi-page statement usually prints the
    # column header ONLY on page 1, so header detection and body walking
    # can't both be scoped to a single page. ──────────────────────────────
    all_lines: list[dict] = []
    for pi, page in enumerate(doc):
        words = page.get_text("words")
        if not words:
            continue
        raw_lines: dict[tuple, dict] = {}
        for x0, y0, x1, y1, text, block, line, _wno in words:
            key = (block, line)
            L = raw_lines.setdefault(key, {"x0": x0, "y0": y0, "words": []})
            L["x0"] = min(L["x0"], x0)
            L["y0"] = min(L["y0"], y0)
            L["words"].append((x0, text))
        page_lines = []
        for L in raw_lines.values():
            L["words"].sort(key=lambda t: t[0])
            page_lines.append({"x0": L["x0"], "y0": L["y0"], "page": pi,
                               "text": " ".join(t[1] for t in L["words"])})
        page_lines.sort(key=lambda l: (round(l["y0"]), l["x0"]))
        all_lines.extend(page_lines)
    doc.close()
    if not all_lines:
        return []

    # ── 2. Find the header row + its column x-positions on WHICHEVER page
    # has it (normally page 1) — used for every page's body. ──────────────
    header_map = {"date": "datetime", "counterparty": "counterparty", "debit": "debit",
                  "credit": "credit", "status": "status", "rrn": "rrn"}
    header_candidates = [l for l in all_lines if any(l["text"].strip().lower().startswith(k) for k in header_map)]
    if not header_candidates:
        return []
    header_page = min(l["page"] for l in header_candidates)
    header_candidates = [l for l in header_candidates if l["page"] == header_page]
    header_y = min(l["y0"] for l in header_candidates)
    header_row = [l for l in header_candidates if abs(l["y0"] - header_y) < 3]
    cols: dict[str, float] = {}
    for l in header_row:
        t = l["text"].strip().lower()
        for prefix, col in header_map.items():
            if t.startswith(prefix) and col not in cols:
                cols[col] = l["x0"]
    if len(cols) < 5:
        return []
    # Boundaries at the MIDPOINT between adjacent header x-positions, not a
    # fixed offset from each header's own x0 — a value can start well left
    # (or right) of its own header label's x0 (e.g. a 12-digit RRN under a
    # 3-letter "RRN" label), so a fixed-offset approach put values in the
    # wrong column whenever that gap exceeded the offset.
    col_order = sorted(cols.items(), key=lambda kv: kv[1])
    col_bounds = []
    for i, (name, cx) in enumerate(col_order):
        lo = -1e9 if i == 0 else (col_order[i - 1][1] + cx) / 2
        col_bounds.append((name, lo))

    def which_col(x: float) -> str | None:
        best = None
        for name, lo in col_bounds:
            if x >= lo:
                best = name
        return best

    # ── 3. Walk every line on every page (skipping the header page's own
    # header row) as one continuous body, in (page, y) order. ─────────────
    rows: list[dict] = []
    current: dict | None = None
    for l in all_lines:
        if l["page"] == header_page and l["y0"] <= header_y + 3:
            continue
        col = which_col(l["x0"])
        if col is None:
            continue
        txt = l["text"].strip()
        if col == "datetime" and _DATE_TOKEN_RE.match(txt):
            if current:
                rows.append(current)
            current = {"datetime": [txt], "counterparty": [], "debit": None,
                      "credit": None, "status": None, "rrn": None}
            continue
        if current is None:
            continue
        if col == "datetime":
            current["datetime"].append(txt)
        elif col == "counterparty":
            current["counterparty"].append(txt)
        else:
            current[col] = txt
    if current:
        rows.append(current)

    out: list[dict] = []
    for r in rows:
        date = r["datetime"][0] if r["datetime"] else None
        time = next((t for t in r["datetime"][1:] if re.match(r"^\d{1,2}:\d{2}", t)), None)
        cp = r["counterparty"]
        name = cp[0] if cp else None
        vpa = next((c for c in cp[1:] if "@" in c), None)
        mode_line = next((c for c in cp if c.upper().startswith(("P2P", "P2M"))), None)
        mode, mcc = None, None
        if mode_line:
            parts = mode_line.split()
            mode = parts[0].upper()
            mcc = parts[1] if len(parts) > 1 else None
        used = {name, vpa, mode_line}
        remarks = " ".join(c for c in cp if c not in used) or None

        ttype = "DEBIT" if r.get("debit") else "CREDIT" if r.get("credit") else None
        amount = r.get("debit") or r.get("credit")
        if not (date and ttype and amount):
            continue
        row = {
            "transaction_id": r.get("rrn"), "date": date, "time": time,
            "type": ttype, "amount": amount, "mode": mode, "mcc": mcc,
            "status": r.get("status"), "remarks": remarks,
        }
        if ttype == "DEBIT":
            row["payee_name"], row["payee_vpa"] = name, vpa
        else:
            row["payer_name"], row["payer_vpa"] = name, vpa
        out.append(row)
    return _from_records(out)


def _parse_pdf(buf: bytes) -> list[dict]:
    import pymupdf as fitz

    positional = _parse_pdf_positional(buf)
    if positional:
        return positional

    doc = fitz.open(stream=buf, filetype="pdf")
    lines = []
    for page in doc:
        lines.extend(ln.strip() for ln in page.get_text("text").splitlines() if ln.strip())
    doc.close()
    return _parse_pdf_table(lines)


def parse_upi(buf: bytes, file_name: str) -> list[dict]:
    """Never raises — returns [] when nothing parses."""
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    try:
        if ext in ("csv", "tsv", "txt"):
            return _parse_delimited(buf.decode("utf-8", "ignore"))
        if ext == "json":
            return _parse_json(buf.decode("utf-8", "ignore"))
        if ext in ("xlsx", "xls"):
            return _parse_xlsx(buf)
        if ext == "pdf":
            return _parse_pdf(buf)
        if ext in ("md", "html", "htm"):
            text = re.sub(r"[|*#>`]{1,}", " ", buf.decode("utf-8", "ignore"))
            rows = _parse_pdf_table([ln.strip() for ln in text.splitlines() if ln.strip()])
            return rows or _parse_delimited(buf.decode("utf-8", "ignore"))
        head = buf[:64].lstrip()
        if head[:1] in (b"[", b"{"):
            return _parse_json(buf.decode("utf-8", "ignore"))
        return _parse_delimited(buf.decode("utf-8", "ignore"))
    except Exception as exc:  # noqa: BLE001 — a bad file must never 500 the upload
        logger.warning("UPI parse failed for %s: %s", file_name, exc)
        return []
