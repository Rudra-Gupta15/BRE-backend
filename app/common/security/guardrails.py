"""Guardrails - hard bounds and schema checks on every untrusted boundary.

  validate_upload()        - file size / type / basic sniffing
  validate_pdf()           - page-count cap, encryption check
  validate_llm_extraction()- schema + sanity of the vision-LLM JSON, and a
                             deterministic cross-check of its reported totals
  clamp_feature_vector()   - keep the 11 underwriting features in valid ranges
  bound_score()            - final credit score stays in [300, 900]

A real violation raises GuardrailError (-> HTTP 400 / 422). Soft issues are
returned as a list of warning strings so the caller can log/annotate.
"""
from __future__ import annotations

import logging

from app.common import config

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"
_FEATURE_BOUNDS = {
    "account_age_days":       (0.0, 20_000.0),
    "avg_monthly_inflow":     (0.0, 5.0e8),
    "avg_monthly_debit":      (0.0, 5.0e8),
    "nach_bounce_count_90d":  (0.0, 200.0),
    "dscr_ratio":             (0.0, 50.0),
    "cash_withdrawal_ratio":  (0.0, 1.0),
    "balance_volatility":     (0.0, 20.0),
    "transaction_volatility": (0.0, 20.0),
    "minimum_balance":        (-5.0e8, 5.0e8),
    "foir_ratio":             (0.0, 1000.0),
    "income_stability":       (0.0, 1.0),
}


class GuardrailError(ValueError):
    """A hard input/output-validation failure."""


# -- uploads ----------------------------------------------------------------
def validate_upload(buf: bytes, file_name: str) -> str:
    """Returns the lower-cased extension. Raises GuardrailError on a violation."""
    if not buf:
        raise GuardrailError("empty file")

    if config.MAX_UPLOAD_BYTES and len(buf) > config.MAX_UPLOAD_BYTES:
        raise GuardrailError(
            f"file is {len(buf) / 1e6:.1f} MB - limit is {config.MAX_UPLOAD_BYTES / 1e6:.0f} MB"
        )

    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    if config.ALLOWED_UPLOAD_EXT and ext not in config.ALLOWED_UPLOAD_EXT:
        raise GuardrailError(
            f"'.{ext}' not allowed - accepted: {', '.join(config.ALLOWED_UPLOAD_EXT)}"
        )

    # light content sniff - the bytes should look like what the name claims
    if ext == "pdf" and not buf.lstrip()[:1024].startswith(_PDF_MAGIC):
        raise GuardrailError("file is named .pdf but is not a PDF")
    if ext in ("csv", "tsv", "txt", "json", "md"):
        sample = buf[:4096]
        if b"\x00" in sample:
            raise GuardrailError(f"file is named .{ext} but contains binary data")
    if ext == "xlsx" and not buf[:2] == b"PK":
        raise GuardrailError("file is named .xlsx but is not a valid workbook")
    return ext


def validate_pdf(fitz_doc) -> list[str]:
    """Page-count cap + encryption check on an opened PyMuPDF document."""
    warnings: list[str] = []
    if getattr(fitz_doc, "needs_pass", False) or getattr(fitz_doc, "is_encrypted", False):
        raise GuardrailError("PDF is password-protected / encrypted")
    if config.MAX_PDF_PAGES and fitz_doc.page_count > config.MAX_PDF_PAGES:
        raise GuardrailError(
            f"PDF has {fitz_doc.page_count} pages - limit is {config.MAX_PDF_PAGES}"
        )
    if fitz_doc.page_count == 0:
        raise GuardrailError("PDF has no pages")
    return warnings


# -- LLM extraction ---------------------------------------------------------
def validate_llm_extraction(parsed: dict) -> tuple[dict, list[str]]:
    """Schema + sanity check on a parsed statement (from the vision LLM or the
    text parser). Sanitises in place; returns (parsed, warnings). A structurally
    broken result raises GuardrailError."""
    warnings: list[str] = []
    if not isinstance(parsed, dict):
        raise GuardrailError("parser returned a non-object result")

    txns = parsed.get("transactions")
    if txns is None or not isinstance(txns, list):
        raise GuardrailError("parser result has no transactions list")

    if config.MAX_TRANSACTIONS and len(txns) > config.MAX_TRANSACTIONS:
        raise GuardrailError(
            f"{len(txns)} transactions parsed - limit is {config.MAX_TRANSACTIONS} "
            "(possible malformed or adversarial statement)"
        )

    clean: list[dict] = []
    credit_sum = debit_sum = 0.0
    for t in txns:
        if not isinstance(t, dict):
            continue
        amt = _to_float(t.get("amount"))
        if amt is None or amt < 0 or amt > 1.0e9:
            warnings.append(f"dropped transaction with implausible amount: {t.get('amount')!r}")
            continue
        narr = str(t.get("narration") or "")[:2000]
        # strip control chars that could be prompt-injection / log-forging
        narr = "".join(ch for ch in narr if ch >= " " or ch == "\t")
        ttype = str(t.get("type") or "").upper()
        if ttype not in ("DEBIT", "CREDIT"):
            ttype = "DEBIT" if amt and amt < 0 else "CREDIT"
        row = {**t, "narration": narr, "amount": abs(amt), "type": ttype}
        clean.append(row)
        if ttype == "CREDIT":
            credit_sum += abs(amt)
        else:
            debit_sum += abs(amt)

    parsed["transactions"] = clean
    summary = parsed.get("summary") or {}

    # Deterministic cross-check: don't trust the AI's own totals.
    for key, computed in (("totalCredit", credit_sum), ("totalDebit", debit_sum)):
        reported = _to_float(summary.get(key))
        if reported is not None and computed > 0:
            drift = abs(reported - computed) / computed
            if drift > 0.02:
                warnings.append(
                    f"{key}: AI said {reported:,.0f}, transactions sum to {computed:,.0f} "
                    f"({drift * 100:.0f}% off) - using the computed value"
                )
        summary[key] = round(computed, 2)

    ob, cb = _to_float(summary.get("openingBalance")), _to_float(summary.get("closingBalance"))
    if ob is not None and cb is not None:
        expected_cb = ob + credit_sum - debit_sum
        if abs(expected_cb - cb) > max(1.0, 0.02 * abs(cb or 1)):
            warnings.append(
                f"closing balance {cb:,.0f} != opening {ob:,.0f} + credits - debits "
                f"({expected_cb:,.0f}) - statement may be incomplete or altered"
            )

    parsed["summary"] = summary
    return parsed, warnings


# -- feature vector / score ------------------------------------------------
def clamp_feature_vector(fv: dict) -> tuple[dict, list[str]]:
    """Force every underwriting feature into a valid range. Out-of-range values
    are a signal something upstream went wrong - clamp and record."""
    warnings: list[str] = []
    out = dict(fv)
    for name, (lo, hi) in _FEATURE_BOUNDS.items():
        if name not in out:
            warnings.append(f"feature '{name}' missing - defaulted")
            out[name] = lo if name != "income_stability" else 0.5
            continue
        v = _to_float(out[name])
        if v is None:
            warnings.append(f"feature '{name}' not numeric ({out[name]!r}) - defaulted")
            out[name] = lo
        elif v < lo or v > hi:
            warnings.append(f"feature '{name}' = {v:g} out of [{lo:g}, {hi:g}] - clamped")
            out[name] = max(lo, min(hi, v))
    return out, warnings


def bound_score(score: float) -> int:
    try:
        return int(max(300, min(900, round(float(score)))))
    except (TypeError, ValueError):
        return 300


def _to_float(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", "").replace("Rs ", "").replace("%", "").strip())
        except ValueError:
            return None
    return None
