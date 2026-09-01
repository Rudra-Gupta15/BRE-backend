"""GST Model Hub pipeline — the stand-alone "process data" run for the GST source.

Parse the uploaded GST files -> roll up 12-24 months into one profile per GSTIN
-> score with the GST model -> rank the model's own features by variance. No
bank-statement stages. Consumed by the `/pipeline/run` router when the only
selected source is `gst_data`; the shape mirrors `app.aa.pipeline.run_pipeline`
so the Model Hub UI renders both the same way.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.common.state.session import session_state
from app.gst import model as gst_model


class NoGstData(ValueError):
    """Raised when `/pipeline/run` is asked for GST but nothing was uploaded."""


def run_gst_pipeline() -> dict:
    gst_stmts = [s for s in session_state.statements_for("gst_data") if s and s.get("gst")]
    if not gst_stmts:
        raise NoGstData("Upload GST return / summary files first.")

    blocks = [s["gst"] for s in gst_stmts]
    scores = [b["avgUnderwritingScore"] for b in blocks if b.get("avgUnderwritingScore") is not None]
    risk: dict = {}
    seen: dict = {}
    for b in blocks:
        for k, v in (b.get("riskCounts") or {}).items():
            risk[k] = risk.get(k, 0) + v
        for k, v in (b.get("returnsSeen") or {}).items():
            seen[k] = seen.get(k, 0) + v
    businesses = sum(b.get("businesses") or b.get("records") or 0 for b in blocks)
    avg = round(sum(scores) / len(scores), 2) if scores else None
    mode = blocks[0].get("mode")

    ranking = gst_model.feature_ranking()
    selection_table = [
        {"rank": i + 1, "feature": r["feature"], "variance": r["variance"],
         "selected": r["selected"], "kind": "gst"}
        for i, r in enumerate(ranking)
    ]
    seen_str = " + ".join(k for k, v in seen.items() if v) or (
        "GST summary" if mode == "summary" else "GST returns")
    processed_row = {
        "id": "gst_data", "kind": "gst",
        "adb": f"{businesses} business{'' if businesses == 1 else 'es'}",
        "gstDelta": f"score {avg}" if avg is not None else "—",
        "upiVelocity": seen_str,
        "cersai": " / ".join(f"{k} {v}" for k, v in risk.items()) or "—",
        "normScore": f"{avg / 100:.3f}" if avg is not None else "—",
        "status": "GST scored",
    }
    stages = [
        {"id": 1, "name": "1. Parse GST Files", "desc": "GSTR-1 / 3B / 2A / 2B or summary rows", "durationMs": 0,
         "detail": f"{sum(seen.values()) or businesses} row(s) across {len(gst_stmts)} file(s)."},
        {"id": 2, "name": "2. Normalise Fields", "desc": "Coerce numbers, strip currency/%",
         "durationMs": 0, "detail": "All GST fields normalised."},
        {"id": 3, "name": "3. Roll Up Per Business", "desc": "12-24 months → one profile per GSTIN",
         "durationMs": 0, "detail": f"{businesses} business profile(s)."},
        {"id": 4, "name": "4. Score", "desc": "GST Underwriting Model — score + risk flag",
         "durationMs": 0, "detail": f"avg score {avg}; " + (", ".join(f"{k} {v}" for k, v in risk.items()) or "no flags")},
        {"id": 5, "name": "5. Rank Features", "desc": "GST model inputs by variance",
         "durationMs": 0, "detail": f"{len(ranking)} GST feature(s) ranked."},
    ]
    return {
        "stages": stages, "noisePercent": 0, "llmActive": False,
        "processedTable": [processed_row], "storedFile": "gst_feature_profiles.csv",
        "selectedFeatures": [r["feature"] for r in ranking if r["selected"]],
        "normalizeTable": [], "engineeredTable": [], "selectionTable": selection_table,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
