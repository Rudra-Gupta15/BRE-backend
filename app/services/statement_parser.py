# statement_parser.py
#
# Real bank statement parser.
#
# PDF flow (LLM vision):
#   1. PDF pages → PNG images via pymupdf (fitz)
#   2. Each page image → Qwen2.5VL (vision LLM) via local Ollama API
#   3. LLM returns structured JSON: list of transactions with date, narration,
#      type (DEBIT/CREDIT), amount, balance
#   4. Results merged across pages → real summary computed from actual numbers
#
# CSV/TSV/TXT flow (unchanged):
#   Column-based heuristic parser — works fine for structured text files.

import base64
import json
import logging
import re
import urllib.error
import urllib.request

import pymupdf as fitz  # PyMuPDF (1.28+ uses pymupdf instead of fitz)

logger = logging.getLogger(__name__)

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_HOST    = "http://localhost:11434"
OLLAMA_MODEL   = "qwen2.5vl:3b"   # 3b is ~3x faster than 7b; still reads tables well
OLLAMA_TIMEOUT = 90               # seconds per page — 3b is much quicker

# ── Page render resolution ────────────────────────────────────────────────────
# 120 DPI: fast to render & encode, still legible for 3b vision model.
# Only used when text extraction fails (scanned/image PDFs).
PAGE_DPI = 120

# Minimum transactions text-extraction must find before we skip the LLM.
# If text extraction gets fewer than this, we assume it's a scanned PDF
# and fall back to LLM vision.
MIN_TEXT_TRANSACTIONS = 3

# ── LLM prompts ───────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a bank statement data extraction assistant. "
    "Your ONLY job is to extract transaction rows from the bank statement image. "
    "Return a valid JSON object — nothing else, no explanation, no markdown fences."
)

_USER_PROMPT = (
    "Extract every transaction visible in this bank statement image.\n\n"
    "Return ONLY a JSON object with this exact structure:\n"
    "{\n"
    "  \"opening_balance\": <number or null>,\n"
    "  \"closing_balance\": <number or null>,\n"
    "  \"transactions\": [\n"
    "    {\n"
    "      \"date\": \"DD/MM/YYYY or as shown\",\n"
    "      \"narration\": \"description of the transaction\",\n"
    "      \"type\": \"DEBIT\" or \"CREDIT\",\n"
    "      \"amount\": <positive number>,\n"
    "      \"balance\": <running balance number or null>\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- If the amount column is split into separate debit/credit columns, "
    "use whichever has a value and set type accordingly.\n"
    "- Ignore header rows, footer rows, totals rows, and page numbers.\n"
    "- If a field is not visible, use null.\n"
    "- Return ONLY the JSON. No extra text."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_result() -> dict:
    return {
        "transactions": [],
        "summary": {
            "transactionCount": 0,
            "totalDebit":       0.0,
            "totalCredit":      0.0,
            "openingBalance":   None,
            "closingBalance":   None,
            "minBalance":       None,
            "maxBalance":       None,
        },
    }


def _summarize_llm(transactions: list[dict], opening: float | None, closing: float | None) -> dict:
    total_debit  = round(sum(t["amount"] for t in transactions if t.get("type") == "DEBIT"),  2)
    total_credit = round(sum(t["amount"] for t in transactions if t.get("type") == "CREDIT"), 2)
    balances     = [t["balance"] for t in transactions if isinstance(t.get("balance"), (int, float))]
    return {
        "transactions": transactions,
        "summary": {
            "transactionCount": len(transactions),
            "totalDebit":       total_debit,
            "totalCredit":      total_credit,
            "openingBalance":   opening if opening is not None else (balances[0]  if balances else None),
            "closingBalance":   closing if closing is not None else (balances[-1] if balances else None),
            "minBalance":       min(balances) if balances else None,
            "maxBalance":       max(balances) if balances else None,
        },
    }


# ── Ollama vision call ────────────────────────────────────────────────────────

def _call_ollama_vision(image_b64: str) -> dict | None:
    """
    Sends one PNG page (base64) to Qwen2.5VL via Ollama /api/chat.
    Returns parsed page dict: { opening_balance, closing_balance, transactions }
    or None on failure / non-JSON response.
    """
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_PROMPT, "images": [image_b64]},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("Ollama request failed: %s", exc)
        return None

    try:
        content = json.loads(body)["message"]["content"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Cannot parse Ollama envelope: %s", exc)
        return None

    # Strip accidental markdown fences the model sometimes adds
    content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content.strip())

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON: %s\n%.300s", exc, content)
        return None


# ── PDF parser (Hybrid: text-first → LLM vision fallback) ────────────────────

def _try_text_extraction(doc) -> dict | None:
    """
    Fast path: extract raw text from each page using pymupdf and run it
    through the existing heuristic parser. Returns a result dict if at least
    MIN_TEXT_TRANSACTIONS transactions were found, otherwise None so the
    caller knows to fall back to LLM vision.
    """
    pages_text = []
    for page in doc:
        text = page.get_text("text")   # plain text, preserves layout order
        if text:
            pages_text.append(text)
    full_text = "\n".join(pages_text)
    if not full_text.strip():
        return None   # PDF has no extractable text → must be scanned

    result = _parse_statement_text(full_text)
    if len(result["transactions"]) >= MIN_TEXT_TRANSACTIONS:
        logger.info(
            "Text extraction succeeded: %d transactions found — skipping LLM.",
            len(result["transactions"]),
        )
        return result

    logger.info(
        "Text extraction found only %d transactions (< %d) — will try LLM vision.",
        len(result["transactions"]), MIN_TEXT_TRANSACTIONS,
    )
    return None


def _parse_pdf_statement(buf: bytes) -> dict:
    """
    Hybrid PDF parser:
      1. Try fast text extraction (pymupdf → heuristic parser). Instant.
         Covers ~90% of real bank statement PDFs (digital/selectable text).
      2. If text extraction yields < MIN_TEXT_TRANSACTIONS, fall back to
         Qwen2.5VL 3b via Ollama — handles scanned/image-only PDFs.
    """
    doc = fitz.open(stream=buf, filetype="pdf")

    # ── Fast path: text extraction ───────────────────────────────────────────
    text_result = _try_text_extraction(doc)
    if text_result is not None:
        doc.close()
        return text_result

    # ── Slow path: LLM vision (scanned PDFs only) ────────────────────────────
    logger.info("Falling back to LLM vision scan (Qwen2.5VL %s)…", OLLAMA_MODEL)
    mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)

    all_transactions: list[dict] = []
    opening_balance: float | None = None
    closing_balance: float | None = None

    for page_num, page in enumerate(doc):
        logger.info("LLM scanning PDF page %d / %d …", page_num + 1, len(doc))

        # Render to PNG in memory, encode base64 for Ollama
        pix       = page.get_pixmap(matrix=mat, alpha=False)
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")

        page_data = _call_ollama_vision(image_b64)
        if not page_data:
            logger.warning("Page %d: LLM returned no data, skipping.", page_num + 1)
            continue

        # Capture opening/closing balances (first page that has them wins)
        if opening_balance is None and page_data.get("opening_balance") is not None:
            try:
                opening_balance = float(page_data["opening_balance"])
            except (TypeError, ValueError):
                pass

        if page_data.get("closing_balance") is not None:
            try:
                closing_balance = float(page_data["closing_balance"])
            except (TypeError, ValueError):
                pass

        # Normalize and collect transactions from this page
        for tx in page_data.get("transactions") or []:
            try:
                amount = float(tx.get("amount") or 0)
                if amount <= 0:
                    continue

                tx_type = str(tx.get("type") or "CREDIT").upper()
                if tx_type not in ("DEBIT", "CREDIT"):
                    tx_type = "CREDIT"

                try:
                    balance = float(tx["balance"]) if tx.get("balance") is not None else None
                except (TypeError, ValueError):
                    balance = None

                all_transactions.append({
                    "date":      str(tx.get("date") or ""),
                    "narration": str(tx.get("narration") or "Transaction")[:200],
                    "type":      tx_type,
                    "amount":    round(amount, 2),
                    "balance":   balance,
                })
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping malformed tx row: %s", exc)

    doc.close()

    if not all_transactions:
        logger.warning("LLM extracted 0 transactions — check PDF and Ollama model.")
        result = _empty_result()
        result["error"] = "Could not extract transactions from this PDF (tried text + LLM vision)."
        return result

    return _summarize_llm(all_transactions, opening_balance, closing_balance)


# ── CSV / TSV / TXT parser (unchanged) ───────────────────────────────────────

DATE_RE   = re.compile(r"^(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b")
AMOUNT_RE = re.compile(r"-?\d+(?:,\d{2,3})*\.\d{2}")
FOOTER_RE = re.compile(
    r"\bopening\s+balance\b|\bclosing\s+balance\b|\bclosing\s*:|\bstatement\s+(summary|period)\b"
    r"|\btotal\s+(debit|credit)\b|\bpage\s+\d+\s+of\s+\d+\b|\bgenerated\s+on\b|\btransactions\s*:\s*\d+\b",
    re.IGNORECASE,
)


def _extract_amounts(s: str) -> list[float]:
    return [float(m.replace(",", "")) for m in AMOUNT_RE.findall(s)]


def _find_labeled_amount(text: str, labels: list[str]):
    for label in labels:
        label_pattern = r"\s+".join(re.escape(w) for w in label.split())
        m = re.search(rf"{label_pattern}\s*[:\-]?\s*([\d,]+\.\d{{2}})", text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _summarize(transactions: list[dict], raw_text: str) -> dict:
    total_debit  = round(sum(t["amount"] for t in transactions if t["type"] == "DEBIT"), 2)
    total_credit = round(sum(t["amount"] for t in transactions if t["type"] == "CREDIT"), 2)

    labeled_opening = _find_labeled_amount(raw_text, ["opening balance"])
    labeled_closing = _find_labeled_amount(raw_text, ["closing balance", "closing"])

    balances        = [t["balance"] for t in transactions if isinstance(t.get("balance"), (int, float))]
    opening_balance = labeled_opening if labeled_opening is not None else (balances[0]  if balances else None)
    closing_balance = labeled_closing if labeled_closing is not None else (balances[-1] if balances else None)

    return {
        "transactions": transactions,
        "summary": {
            "transactionCount": len(transactions),
            "totalDebit":       total_debit,
            "totalCredit":      total_credit,
            "openingBalance":   opening_balance,
            "closingBalance":   closing_balance,
            "minBalance":       min(balances) if balances else None,
            "maxBalance":       max(balances) if balances else None,
        },
    }


def _parse_csv_statement(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return _empty_result()

    delimiter = "\t" if "\t" in lines[0] else ","
    header    = [h.strip().lower() for h in lines[0].split(delimiter)]

    def find_col(keywords: list[str]) -> int:
        for i, h in enumerate(header):
            if any(k in h for k in keywords):
                return i
        return -1

    date_idx      = find_col(["date"])
    narration_idx = find_col(["narration", "description", "particulars", "details"])
    debit_idx     = find_col(["debit", "withdrawal"])
    credit_idx    = find_col(["credit", "deposit"])
    balance_idx   = find_col(["balance"])
    amount_idx    = find_col(["amount"])

    def parse_num(cells: list[str], idx: int):
        if idx < 0 or idx >= len(cells) or not cells[idx]:
            return None
        try:
            return float(cells[idx].replace(",", ""))
        except ValueError:
            return None

    transactions = []
    for line in lines[1:]:
        cells  = [c.strip() for c in line.split(delimiter)]
        if len(cells) < 2:
            continue

        debit   = parse_num(cells, debit_idx)
        credit  = parse_num(cells, credit_idx)
        amount  = parse_num(cells, amount_idx)
        balance = parse_num(cells, balance_idx)

        tx_type = None
        value   = None
        if debit is not None and debit > 0:
            tx_type, value = "DEBIT", debit
        elif credit is not None and credit > 0:
            tx_type, value = "CREDIT", credit
        elif amount is not None:
            tx_type, value = ("DEBIT" if amount < 0 else "CREDIT"), abs(amount)
        if value is None:
            continue

        transactions.append({
            "date":      cells[date_idx]      if date_idx      >= 0 else None,
            "narration": cells[narration_idx] if narration_idx >= 0 else line[:120],
            "type":      tx_type,
            "amount":    value,
            "balance":   balance,
        })

    return _summarize(transactions, text)


DATE_ANY_RE = re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b")
REF_NO_RE   = re.compile(r"\b\d{10,}\b")


def _parse_statement_text(text: str) -> dict:
    """Heuristic line-by-line parser for plain-text bank statements."""
    blocks:  list[str] = []
    current: str | None = None

    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line.strip())
        if not line:
            continue

        if DATE_RE.match(line):
            if current is None:
                current = line
            elif len(_extract_amounts(current)) >= 2:
                blocks.append(current)
                current = line
            else:
                current += " " + line
        elif FOOTER_RE.search(line):
            if current:
                blocks.append(current)
            current = None
        elif current is not None:
            current += " " + line

    if current:
        blocks.append(current)

    transactions = []
    prev_balance = _find_labeled_amount(text, ["opening balance"])
    for block in blocks:
        date_match = DATE_RE.match(block)
        amounts    = _extract_amounts(block)
        if len(amounts) < 2:
            continue

        balance   = amounts[-1]
        amount    = amounts[-2]
        narration = REF_NO_RE.sub("", AMOUNT_RE.sub("", DATE_ANY_RE.sub("", block)))
        narration = re.sub(r"-{2,}", "-", narration)
        narration = re.sub(r"\s+", " ", narration).strip(" -")[:120] or "Transaction"

        tx_type      = "DEBIT" if (prev_balance is not None and balance < prev_balance) else "CREDIT"
        prev_balance = balance

        transactions.append({
            "date":      date_match.group(1) if date_match else None,
            "narration": narration,
            "type":      tx_type,
            "amount":    amount,
            "balance":   balance,
        })

    return _summarize(transactions, text)


# ── Public entry point ────────────────────────────────────────────────────────

async def parse_statement(buf: bytes, file_name: str) -> dict:
    """
    Dispatches to the correct parser based on file extension.
    PDF  → LLM vision via Qwen2.5VL (Ollama) — reads the page as an image.
    CSV/TSV/TXT → column/heuristic text parser.
    """
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    try:
        if ext in ("csv", "tsv", "txt"):
            return _parse_csv_statement(buf.decode("utf-8", errors="replace"))
        if ext == "pdf":
            return _parse_pdf_statement(buf)
    except Exception as err:  # noqa: BLE001
        logger.exception("Statement parse failed for %s", file_name)
        result = _empty_result()
        result["error"] = str(err)
        return result
    return _empty_result()
