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

from app.gst.schema import CANONICAL

logger = logging.getLogger(__name__)


def _norm(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


def _row(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        nk = _norm(k)
        if nk:
            out[nk] = v
    return out


def _from_records(records: list) -> list[dict]:
    return [r for r in (_row(x) for x in records if isinstance(x, dict)) if r]


def _parse_delimited(text: str) -> list[dict]:
    sample = text[:4096]
    delim = "\t" if sample.count("\t") > sample.count(",") else ","
    if sample.count(";") > sample.count(delim):
        delim = ";"
    return _from_records(list(csv.DictReader(io.StringIO(text), delimiter=delim)))


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


def _parse_pdf(buf: bytes) -> list[dict]:
    import pymupdf as fitz
    doc = fitz.open(stream=buf, filetype="pdf")
    lines = []
    for page in doc:
        lines.extend(ln.strip() for ln in page.get_text("text").splitlines() if ln.strip())
    doc.close()

    # wide-column layout: a line already holds several known columns
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

    # one-token-per-line layout: a run of column names, then values in order
    header = []
    start_of_data = len(lines)
    for i, ln in enumerate(lines):
        if _norm(ln) in CANONICAL:
            header.append(_norm(ln))
        elif header:
            start_of_data = i
            break
    if len(header) < 4:
        raise ValueError("Could not locate a GST table header in the PDF — "
                         "upload the CSV / JSON / XLSX version instead.")

    n, hset, values = len(header), set(header), []
    for ln in lines[start_of_data:]:
        if _norm(ln) in hset or ln.lower().startswith("gst transaction data") or ln.startswith("Page "):
            continue
        values.append(ln)
    return [_row(dict(zip(header, values[k:k + n]))) for k in range(0, len(values) - n + 1, n)]


def parse_gst(buf: bytes, file_name: str) -> list[dict]:
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
        head = buf[:64].lstrip()
        if head[:1] in (b"[", b"{"):
            return _parse_json(buf.decode("utf-8", "ignore"))
        return _parse_delimited(buf.decode("utf-8", "ignore"))
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("GST parse failed for %s: %s", file_name, exc)
        raise
