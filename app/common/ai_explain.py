"""Plain-English one-liners for the Pattern Match tab, via the local Ollama LLM.

`explain(kind, result)` returns {"fraud": str|None, "anomaly": str|None} — one
short sentence for each of the two boxes shown to a non-technical credit
officer. Best-effort: if Ollama is unreachable both are None and the UI falls
back to a canned sentence.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

from app.common import config

logger = logging.getLogger(__name__)

_TIMEOUT = 30


def _facts(result: dict) -> str:
    lines: list[str] = []
    for _mid, s in (result.get("modelScores") or {}).items():
        pct = int(round((s.get("probability") or 0) * 100))
        lines.append(f"- {s.get('name')}: {pct}% ({'FLAGGED' if s.get('flag') else 'not flagged'})")
    hits = [f"{t['name']} ({t['verdict']})" for t in result.get("typologies", [])
            if t.get("verdict") in ("match", "elevated")]
    lines.append("- Fraud signatures triggered: " + (", ".join(hits) if hits else "none"))
    dev = [f"{r['label']} {r['value']} vs baseline {r['baseline']} ({r['band']})"
           for r in (result.get("comparison") or {}).get("perMetric", [])
           if r.get("band") and r["band"] != "within"][:4]
    lines.append("- Numbers outside the normal range: " + (", ".join(dev) if dev else "none"))
    lines.append(f"- Overall verdict: {result.get('verdict', 'n/a')}")
    return "\n".join(lines)


def explain(kind: str, result: dict) -> dict:
    empty = {"fraud": None, "anomaly": None}
    if not result or not result.get("available"):
        return empty

    subject = "GST business" if kind == "gst" else "loan applicant's bank statements"
    prompt = (
        f"A non-technical bank credit officer is reviewing an automated check on a "
        f"{subject}. Write TWO short plain-English sentences — no jargon, no numbers.\n\n"
        f"Reply in EXACTLY this format:\n"
        f"FRAUD: <one sentence: is there any sign of deliberate fraud, and the gist why>\n"
        f"UNUSUAL: <one sentence: is the account activity normal or odd, and the gist why>\n\n"
        f"Check results:\n{_facts(result)}"
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
        logger.warning("AI explain unavailable (%s)", exc)
        return empty

    fraud = re.search(r"FRAUD:\s*(.+)", text)
    unusual = re.search(r"UNUSUAL:\s*(.+)", text)
    return {
        "fraud": fraud.group(1).strip() if fraud else None,
        "anomaly": unusual.group(1).strip() if unusual else None,
    }
