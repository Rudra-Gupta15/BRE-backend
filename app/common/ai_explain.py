"""Plain-English explanations for the Signals Decision tab, via the local Ollama LLM.

`explain_rule(...)` returns one short plain-English paragraph for a single BRE
rule result — why it passed, why it failed, or why it could not be evaluated —
written for a non-technical credit officer. Best-effort: if Ollama is unreachable
a deterministic fallback sentence (built from the curated rule text) is returned
instead, so the UI always has something to show.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from app.common import config
from app.common.rule_text import describe_rule

logger = logging.getLogger(__name__)

_TIMEOUT = 30


def _fallback(label: str, status: str, detail: str, serious: bool) -> str:
    """No-LLM explanation — still readable, built from the curated rule text."""
    what = describe_rule(label)
    if status == "SKIP":
        return (f"{what} It could not be checked automatically here — it needs the "
                f"full rule engine or data that wasn't part of this upload"
                f"{f' ({detail})' if detail else ''}. A reviewer would confirm it by hand.")
    if status == "FAIL":
        tail = " This is a key rule, so it weighs heavily on the decision." if serious else ""
        return f"{what} The applicant did not meet it{f' — {detail}' if detail else ''}.{tail}"
    return f"{what} The applicant met this requirement{f' — {detail}' if detail else ''}."


def explain_rule(label: str, status: str, detail: str = "",
                 serious: bool = False, product: str = "", decision: str = "") -> str:
    """One plain-English paragraph explaining this rule's result. Falls back to a
    deterministic sentence when Ollama is unreachable."""
    label = (label or "").strip()
    status = (status or "").strip().upper()
    detail = (detail or "").strip()
    fallback = _fallback(label, status, detail, serious)

    verb = {"PASS": "passed", "FAIL": "failed", "SKIP": "could not be evaluated"}.get(status, "was checked")
    ask = {
        "PASS": "Explain in 2-3 plain sentences what this rule checks and why the applicant passed it, "
                "and what that means for the loan.",
        "FAIL": "Explain in 2-3 plain sentences what this rule checks, why the applicant failed it, "
                "and how much that should worry the lender.",
        "SKIP": "Explain in 2-3 plain sentences what this rule would have checked and why it could not "
                "be run automatically here (it needs the full rule engine or data that wasn't supplied).",
    }.get(status, "Explain in 2-3 plain sentences what this rule checks and what the result means.")

    prompt = (
        f"A non-technical loan credit officer is reading an automated underwriting check. "
        f"No jargon, no restating the numbers verbatim, no bullet points — just clear prose.\n\n"
        f"Rule: {label}\n"
        f"What it checks: {describe_rule(label)}\n"
        f"Result: {verb}\n"
        f"System note: {detail or '(none)'}\n"
        f"{'Loan product: ' + product if product else ''}\n"
        f"{'Overall decision so far: ' + decision if decision else ''}\n\n"
        f"{ask}"
    )
    payload = json.dumps({
        "model": config.active_vision_model(),
        "stream": False,
        "options": {"temperature": 0.2},
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            text = (json.loads(resp.read().decode("utf-8")).get("message") or {}).get("content", "")
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        logger.warning("Rule AI explain unavailable (%s) — using fallback", exc)
        return fallback

    text = (text or "").strip()
    return text or fallback
