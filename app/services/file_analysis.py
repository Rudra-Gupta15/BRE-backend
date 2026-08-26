# Real content analysis of an uploaded file — no lookup tables. CSV/text
# files are scanned for actual missing/empty cells; JSON is parsed and
# checked for null/empty values; anything else (PDF, XLSX, images, ...)
# gets a byte-level Shannon entropy + size read, since we can't parse those
# formats here but can still measure the real bytes that were uploaded.

import json
import math
from collections import Counter
from datetime import datetime, timezone

UNKNOWN_FORMAT_BASELINE = 70


def _shannon_entropy(buf: bytes) -> float:
    if not buf:
        return 0.0
    counts = Counter(buf)
    length = len(buf)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _analyze_delimited_text(text: str, delimiter: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    total_cells = 0
    empty_cells = 0
    for line in lines:
        cells = line.split(delimiter)
        total_cells += len(cells)
        empty_cells += sum(1 for c in cells if c.strip() == "")
    missing_ratio = (empty_cells / total_cells) if total_cells else 1.0
    cleanliness = round(max(2, min(99, (1 - missing_ratio) * 100)))
    return {
        "cleanlinessPercent": cleanliness,
        "stats": {"rows": len(lines), "totalCells": total_cells, "emptyCells": empty_cells, "missingRatio": round(missing_ratio, 3)},
    }


def _analyze_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"cleanlinessPercent": 20, "stats": {"parseError": True}}

    records = data if isinstance(data, list) else [data]
    total_fields = 0
    empty_fields = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        for value in record.values():
            total_fields += 1
            if value is None or value == "":
                empty_fields += 1
    missing_ratio = (empty_fields / total_fields) if total_fields else 0.0
    cleanliness = round(max(5, min(99, (1 - missing_ratio) * 100)))
    return {
        "cleanlinessPercent": cleanliness,
        "stats": {"records": len(records), "totalFields": total_fields, "emptyFields": empty_fields, "missingRatio": round(missing_ratio, 3)},
    }


def _analyze_binary(buf: bytes) -> dict:
    sample = buf[:65536]
    entropy = _shannon_entropy(sample)
    size_factor = 0.5 if len(buf) < 2000 else 0.85 if len(buf) < 20000 else 1.0
    cleanliness = round(max(8, min(97, (entropy / 8) * 100 * size_factor)))
    return {"cleanlinessPercent": cleanliness, "stats": {"entropyBitsPerByte": round(entropy, 2), "sampledBytes": len(sample)}}


def analyze_file(buf: bytes, file_name: str) -> dict:
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()

    if ext in ("csv", "tsv"):
        result = _analyze_delimited_text(buf.decode("utf-8", errors="replace"), "\t" if ext == "tsv" else ",")
    elif ext == "txt":
        result = _analyze_delimited_text(buf.decode("utf-8", errors="replace"), ",")
    elif ext == "json":
        result = _analyze_json(buf.decode("utf-8", errors="replace"))
    else:
        result = _analyze_binary(buf)

    return {
        "fileName": file_name,
        "sizeBytes": len(buf),
        "format": ext or "unknown",
        "cleanlinessPercent": result["cleanlinessPercent"],
        "stats": result["stats"],
        "scannedAt": datetime.now(timezone.utc).isoformat(),
    }
