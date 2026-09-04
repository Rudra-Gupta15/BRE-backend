"""AA bank-statement parser — an uploaded file becomes real transactions.

`parse_statement(bytes, filename)` dispatches on extension and returns
`{transactions: [...], summary: {...}}` (the shape the pipeline + inference
engine consume). Called by the /pipeline/uploads route.

PDF flow:
  1. Fast path — selectable-text extraction (pymupdf) → heuristic parser.
     Covers digital bank-statement PDFs instantly.
  2. Scanned/image PDFs — each page rendered to PNG and sent to a vision LLM
     (Gemma 4 via Ollama Cloud by default: fast, no local GPU) which returns
     structured JSON transactions. Results merged → real summary.

CSV/TSV/TXT flow:
  Column-based heuristic parser — works fine for structured text files.
"""

import asyncio
import base64
import json
import logging
import re
import urllib.error
import urllib.request

import pymupdf as fitz  # PyMuPDF (1.28+ uses pymupdf instead of fitz)

from app.common import config

logger = logging.getLogger(__name__)

# ── Ollama vision-LLM config (set these in Backend/.env) ────────────────────
# gemma4:31b-cloud runs on Ollama Cloud — accurate table reading, ~3-5s/page,
# and zero load on the local machine. Set STATEMENT_LLM_MODEL=qwen2.5vl:3b in
# .env for a fully-local model.
OLLAMA_HOST    = config.OLLAMA_HOST
OLLAMA_MODEL   = config.STATEMENT_LLM_MODEL
OLLAMA_TIMEOUT = config.STATEMENT_LLM_TIMEOUT      # seconds per page

# Page render resolution — only used for scanned/image PDFs that reach the LLM.
PAGE_DPI = config.STATEMENT_LLM_DPI

# Minimum transactions text-extraction must find before we skip the LLM.
# Fewer than this → assume a scanned PDF and fall back to vision.
MIN_TEXT_TRANSACTIONS = 3

# ── LLM prompts ───────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a bank statement data extraction assistant. "
    "Your ONLY job is to extract transaction rows from the bank statement image. "
    "Return a valid JSON object — nothing else, no explanation, no markdown fences."
)

_USER_PROMPT = (
    "Extract the account details and every transaction visible in this bank statement image.\n\n"
    "Return ONLY a JSON object with this exact structure:\n"
    "{\n"
    "  \"bank_name\": \"<name of the bank, e.g. 'HDFC Bank', or null>\",\n"
    "  \"account_holder\": \"<full name of the account holder, or null>\",\n"
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
    "- bank_name / account_holder usually appear only in the header of the first page.\n"
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
            "bankName":         None,
            "accountHolder":    None,
        },
    }


def _clean_name(value) -> str | None:
    if not value or not isinstance(value, str):
        return None
    s = re.sub(r"\s+", " ", value.strip()).strip(" .:-")
    return s[:80] or None


def _summarize_llm(
    transactions: list[dict], opening: float | None, closing: float | None,
    bank_name: str | None = None, account_holder: str | None = None,
) -> dict:
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
            "bankName":         _clean_name(bank_name),
            "accountHolder":    _clean_name(account_holder),
        },
    }


# ── Ollama vision call ────────────────────────────────────────────────────────

def _call_ollama_vision(image_b64: str) -> dict | None:
    """
    Sends one PNG page (base64) to the vision LLM (OLLAMA_MODEL) via Ollama
    /api/chat. Returns parsed page dict:
    { opening_balance, closing_balance, transactions } or None on failure /
    non-JSON response.
    """
    payload = json.dumps({
        "model":  config.active_vision_model(),
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


# ── PDF parser (text-first → vision-LLM fallback) ───────────────────────────

def _try_text_extraction(doc) -> dict | None:
    """
    Fast path: extract raw text from each page using pymupdf and run it
    through the existing heuristic parser. Returns a result dict if at least
    MIN_TEXT_TRANSACTIONS transactions were found, otherwise None so the
    caller knows to fall back to the vision LLM.
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
        "Text extraction found only %d transactions (< %d) — will try the vision LLM.",
        len(result["transactions"]), MIN_TEXT_TRANSACTIONS,
    )
    return None


def _llm_vision_extraction(doc) -> dict:
    """Scanned/image-PDF path: send each rendered page to the vision LLM
    (Gemma 4 via Ollama Cloud by default) and merge the JSON it returns."""
    logger.info("Scanning PDF with vision LLM (%s)…", config.active_vision_model())
    mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)

    all_transactions: list[dict] = []
    opening_balance: float | None = None
    closing_balance: float | None = None
    bank_name: str | None = None
    account_holder: str | None = None

    for page_num, page in enumerate(doc):
        logger.info("LLM scanning PDF page %d / %d …", page_num + 1, len(doc))
        pix       = page.get_pixmap(matrix=mat, alpha=False)
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")

        page_data = _call_ollama_vision(image_b64)
        if not page_data:
            logger.warning("Page %d: LLM returned no data, skipping.", page_num + 1)
            continue

        if not bank_name and page_data.get("bank_name"):
            bank_name = str(page_data["bank_name"])
        if not account_holder and page_data.get("account_holder"):
            account_holder = str(page_data["account_holder"])

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

        for tx in page_data.get("transactions") or []:
            try:
                try:
                    amount = float(tx.get("amount")) if tx.get("amount") not in (None, "") else None
                except (TypeError, ValueError):
                    amount = None
                if amount is not None and amount <= 0:
                    amount = None
                try:
                    balance = float(tx["balance"]) if tx.get("balance") is not None else None
                except (TypeError, ValueError):
                    balance = None
                date_val = str(tx.get("date") or "")
                narr_val = str(tx.get("narration") or "")[:200]

                if amount is None:
                    # The vision model didn't return a usable amount for this
                    # row — pass it through (if it at least has a date or
                    # narration) instead of silently dropping it, so the
                    # pipeline's AI recovery pass gets a real shot at it.
                    if not (date_val or narr_val):
                        continue
                    all_transactions.append({
                        "date": date_val, "narration": narr_val or "Transaction",
                        "type": None, "amount": None, "balance": balance,
                    })
                    continue

                tx_type = str(tx.get("type") or "CREDIT").upper()
                if tx_type not in ("DEBIT", "CREDIT"):
                    tx_type = "CREDIT"
                all_transactions.append({
                    "date":      date_val,
                    "narration": narr_val or "Transaction",
                    "type":      tx_type,
                    "amount":    round(amount, 2),
                    "balance":   balance,
                })
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping malformed tx row: %s", exc)

    if not all_transactions:
        return _empty_result()
    return _summarize_llm(all_transactions, opening_balance, closing_balance, bank_name, account_holder)


def _parse_pdf_statement(buf: bytes) -> dict:
    """
    PDF parser:
      1. selectable-text extraction (pymupdf → heuristic parser) — instant,
         covers digital bank-statement PDFs.
      2. If that yields < MIN_TEXT_TRANSACTIONS, render each page and send it
         to the vision LLM (STATEMENT_LLM_MODEL, default gemma4:31b-cloud).
    """
    doc = fitz.open(stream=buf, filetype="pdf")
    try:
        # Guardrail: page-count cap + encryption check before we render anything.
        from app.common.security import guardrails
        guardrails.validate_pdf(doc)

        text_result = _try_text_extraction(doc)
        if text_result is not None:
            return text_result

        llm_result = _llm_vision_extraction(doc)
        if llm_result["transactions"]:
            return llm_result

        result = _empty_result()
        result["error"] = "Could not extract transactions from this PDF (tried text + vision AI)."
        return result
    finally:
        doc.close()


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


_BANK_RE = re.compile(
    r"\b([A-Z][A-Za-z&.]+(?:\s+[A-Z][A-Za-z&.]+){0,3}\s+bank)\b"
    r"|\b(HDFC|ICICI|SBI|AXIS|KOTAK|YES|IDFC|INDUSIND|PNB|BOB|CANARA|UNION|RBL|AU|BANDHAN|FEDERAL|IDBI)\b",
    re.IGNORECASE,
)
_HOLDER_RE = re.compile(
    r"(?:account\s*holder|customer\s*name|name\s*of\s*(?:account\s*holder|customer)|a/c\s*holder|holder\s*name)"
    r"\s*[:\-]?\s*([A-Za-z][A-Za-z .]{2,60})",
    re.IGNORECASE,
)


def _extract_account_meta(raw_text: str) -> tuple[str | None, str | None]:
    head = raw_text[:1500]  # header info is near the top
    bank = None
    m = _BANK_RE.search(head)
    if m:
        bank = _clean_name(m.group(1) or m.group(2))
        if bank and len(bank) <= 6 and "bank" not in bank.lower():
            bank = f"{bank.upper()} Bank"
    holder = None
    mh = _HOLDER_RE.search(head)
    if mh:
        holder = _clean_name(mh.group(1))
    return bank, holder


def _summarize(transactions: list[dict], raw_text: str) -> dict:
    total_debit  = round(sum(t["amount"] for t in transactions if t["type"] == "DEBIT"), 2)
    total_credit = round(sum(t["amount"] for t in transactions if t["type"] == "CREDIT"), 2)

    labeled_opening = _find_labeled_amount(raw_text, ["opening balance"])
    labeled_closing = _find_labeled_amount(raw_text, ["closing balance", "closing"])

    balances        = [t["balance"] for t in transactions if isinstance(t.get("balance"), (int, float))]
    opening_balance = labeled_opening if labeled_opening is not None else (balances[0]  if balances else None)
    closing_balance = labeled_closing if labeled_closing is not None else (balances[-1] if balances else None)
    bank_name, account_holder = _extract_account_meta(raw_text)

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
            "bankName":         bank_name,
            "accountHolder":    account_holder,
        },
    }


def _parse_csv_statement(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return _empty_result()

    # Skip any preamble rows (bank name, account holder, period …) and find the
    # real column-header row: the first line mentioning "date" plus an amount col.
    header_i = 0
    for i, line in enumerate(lines[:15]):
        low = line.lower()
        if "date" in low and any(k in low for k in ("debit", "credit", "withdrawal", "deposit", "amount", "balance")):
            header_i = i
            break

    hdr = lines[header_i]
    if "\t" in hdr:
        delimiter = "\t"
    elif "|" in hdr:
        delimiter = "|"
    elif hdr.count(";") > hdr.count(","):
        delimiter = ";"
    else:
        delimiter = ","
    header = [h.strip().lower() for h in hdr.split(delimiter)]

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
    # A single "Type" / "Dr/Cr" / "Indicator" column paired with one Amount column.
    type_idx      = find_col(["dr/cr", "cr/dr", "drcr", "type", "indicator", "txn type", "transaction type"])

    def parse_num(cells: list[str], idx: int):
        if idx < 0 or idx >= len(cells) or not cells[idx]:
            return None
        try:
            return float(cells[idx].replace(",", ""))
        except ValueError:
            return None

    transactions = []
    for line in lines[header_i + 1:]:
        cells  = [c.strip().strip("|").strip() for c in line.split(delimiter)]
        if len(cells) < 2 or re.fullmatch(r"[\s:|\-]+", line):
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
            if 0 <= type_idx < len(cells) and cells[type_idx]:
                t = cells[type_idx].strip().upper()
                is_debit = t.startswith(("D", "W")) or t in ("DR", "DEBIT", "WITHDRAWAL")
                tx_type, value = ("DEBIT" if is_debit else "CREDIT"), abs(amount)
            else:
                tx_type, value = ("DEBIT" if amount < 0 else "CREDIT"), abs(amount)

        date_val = cells[date_idx] if 0 <= date_idx < len(cells) else None
        narr_val = cells[narration_idx] if 0 <= narration_idx < len(cells) else None
        if value is None:
            # No debit/credit/amount column gave us a usable number. Don't
            # silently drop it here — a row that at least looks like a real
            # transaction (has a date or narration) is passed through with
            # amount=None so the pipeline's AI recovery pass gets a real shot
            # at reconstructing it from the balance delta or the narration
            # text; only genuinely blank/junk lines are skipped outright.
            if not (date_val or narr_val):
                continue
            transactions.append({
                "date": date_val, "narration": narr_val or line[:120],
                "type": None, "amount": None, "balance": balance,
            })
            continue

        transactions.append({
            "date":      date_val,
            "narration": narr_val if narr_val else line[:120],
            "type":      tx_type,
            "amount":    value,
            "balance":   balance,
        })

    # Delimited parse found nothing — fall back to the free-text heuristic.
    if not transactions:
        return _parse_statement_text(text)
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


# ── JSON / Markdown / XLSX statements ─────────────────────────────────────────

def _num(x):
    try:
        return float(str(x).replace(",", "").replace("₹", "").replace("Rs.", "").strip())
    except (TypeError, ValueError):
        return None


def _dict_pick(d: dict, wants: list[str]):
    for k, v in d.items():
        kl = re.sub(r"[^a-z]", "", k.lower())
        if any(w in kl for w in wants):
            return v
    return None


def _first_list_of_dicts(obj):
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _first_list_of_dicts(v)
            if found:
                return found
    return None


def _parse_json_statement(raw: str) -> dict:
    """Flexible JSON parser — a bare array of transaction objects, a
    {transactions: [...]} wrapper, or an Account-Aggregator-style nested payload.
    Field names are matched fuzzily (date/valueDate/txnDate, amount/txnAmount,
    narration/description/remarks, type/drCr, balance/currentBalance …)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        r = _empty_result()
        r["error"] = f"invalid JSON ({exc})"
        return r

    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("transactions", "txns", "entries", "data", "statement", "rows"):
            for k in data:
                if k.lower() == key and isinstance(data[k], list):
                    rows = data[k]
                    break
            if rows:
                break
        if rows is None:
            rows = _first_list_of_dicts(data)

    if not rows:
        return _empty_result()

    txns: list[dict] = []
    for it in rows:
        if not isinstance(it, dict):
            continue
        date = _dict_pick(it, ["valuedate", "txndate", "transactiondate", "date", "postingdate"])
        narr = _dict_pick(it, ["narration", "description", "remarks", "particulars", "details", "reference"])
        bal  = _num(_dict_pick(it, ["currentbalance", "runningbalance", "closingbalance", "balance"]))
        amt  = _num(_dict_pick(it, ["amount", "txnamount", "transactionamount", "value"]))
        debit  = _num(_dict_pick(it, ["debit", "withdrawal", "withdrawalamt"]))
        credit = _num(_dict_pick(it, ["credit", "deposit", "depositamt"]))
        typ    = str(_dict_pick(it, ["type", "drcr", "crdr", "transactiontype", "indicator"]) or "").upper()

        if debit and debit > 0:
            tx_type, value = "DEBIT", debit
        elif credit and credit > 0:
            tx_type, value = "CREDIT", credit
        elif amt is not None:
            is_debit = amt < 0 or typ.startswith(("D", "W")) or "DEBIT" in typ or "DR" == typ
            tx_type, value = ("DEBIT" if is_debit else "CREDIT"), abs(amt)
        else:
            continue

        txns.append({
            "date":      str(date) if date not in (None, "") else None,
            "narration": (str(narr)[:120] if narr else "Transaction"),
            "type":      tx_type,
            "amount":    value,
            "balance":   bal,
        })
    return _summarize(txns, raw)


def _parse_markdown_statement(text: str) -> dict:
    """Pull the pipe-table out of a Markdown statement and reuse the delimited
    parser; fall back to the free-text heuristic if there's no usable table."""
    rows = [l.strip().strip("|").strip() for l in text.splitlines() if l.count("|") >= 2]
    rows = [r for r in rows if not re.fullmatch(r"[\s:|\-]+", r)]
    if len(rows) >= 2:
        res = _parse_csv_statement("\n".join(r.replace("|", "\t") for r in rows))
        if res["transactions"]:
            return res
    return _parse_statement_text(text)


def _parse_xlsx_statement(buf: bytes) -> dict:
    """First sheet of an .xlsx workbook → CSV → the delimited parser."""
    import io

    import pandas as pd
    try:
        df = pd.read_excel(io.BytesIO(buf), dtype=str).fillna("")
    except Exception as exc:  # noqa: BLE001 — openpyxl / format errors
        r = _empty_result()
        r["error"] = f"could not read XLSX ({exc})"
        return r
    return _parse_csv_statement(df.to_csv(index=False))


# ── Public entry point ────────────────────────────────────────────────────────

async def parse_statement(buf: bytes, file_name: str) -> dict:
    """
    Dispatches to the correct parser based on file extension.
    PDF        → selectable-text extraction, then vision LLM for scanned PDFs.
    CSV/TSV/TXT → column parser (tab/comma/semicolon/pipe), text-heuristic fallback.
    JSON       → flexible field-matching parser (array / wrapper / AA payload).
    MD         → the pipe-table, else the text heuristic.
    XLSX       → first sheet via pandas → the column parser.
    """
    from app.common.security.guardrails import GuardrailError

    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    try:
        if ext in ("csv", "tsv", "txt"):
            return _parse_csv_statement(buf.decode("utf-8", errors="replace"))
        if ext == "json":
            return _parse_json_statement(buf.decode("utf-8", errors="replace"))
        if ext == "md":
            return _parse_markdown_statement(buf.decode("utf-8", errors="replace"))
        if ext == "xlsx":
            return await asyncio.to_thread(_parse_xlsx_statement, buf)
        if ext == "pdf":
            # _parse_pdf_statement does blocking PDF rendering and a blocking
            # urllib call to Ollama. Running it inline would freeze the whole
            # asyncio event loop — and with it every other endpoint in the app
            # — for the entire scan. Push it to a worker thread so the rest of
            # the API stays responsive.
            return await asyncio.to_thread(_parse_pdf_statement, buf)
    except GuardrailError:
        raise  # let the router turn this into a 422
    except Exception as err:  # noqa: BLE001
        logger.exception("Statement parse failed for %s", file_name)
        result = _empty_result()
        result["error"] = str(err)
        return result
    return _empty_result()

