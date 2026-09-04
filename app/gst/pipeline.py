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

    # Real noise/cleanliness from the actual uploaded file(s) — this used to be
    # hardcoded to 0/False regardless of what was uploaded, so the "AI Noise
    # Inspection" box always claimed a perfectly clean file even when it wasn't.
    # Worst file wins, not the average — one dirty file in a folder of clean
    # ones must still trip the threshold, not get diluted away.
    file_metas = session_state.files_for("gst_data")
    scored = [f for f in file_metas if isinstance(f.get("cleanlinessPercent"), (int, float))]
    worst = min(scored, key=lambda f: f["cleanlinessPercent"]) if scored else None
    cleanliness = worst["cleanlinessPercent"] if worst else 100
    noise_percent = min(95, max(1, round(100 - cleanliness)))
    llm_active = noise_percent > 40
    noise_by_source = {
        "gst_data": {
            "label": "GST Transaction Data", "cleanlinessPercent": round(cleanliness),
            "worstFileName": worst.get("fileName") if worst else None,
            "noisePercent": noise_percent, "llmActive": llm_active, "fileCount": len(file_metas),
        }
    }

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
    avg_completeness = round(sum((b.get("completeness") or 0) for b in blocks) / len(blocks)) if blocks else 0
    normalise_detail = (
        f"{cleanliness}% file cleanliness (byte-level scan of the upload) — "
        + (f"noise {noise_percent}% exceeds the 40% threshold, AI cleaning pass activated."
           if llm_active else
           f"noise {noise_percent}% within threshold, ingested directly — no cleaning needed.")
        + f" GSTR-field coverage {avg_completeness}% (how much of the full 53-field return schema this "
          "source supplies — a schema-coverage measure, not a cleanliness score)."
    )
    stages = [
        {"id": 1, "name": "1. Parse GST Files", "desc": "GSTR-1 / 3B / 2A / 2B or summary rows", "durationMs": 0,
         "detail": f"{sum(seen.values()) or businesses} row(s) across {len(gst_stmts)} file(s)."},
        {"id": 2, "name": "2. Normalise Fields", "desc": "Coerce numbers, strip currency/%",
         "durationMs": 0, "detail": normalise_detail},
        {"id": 3, "name": "3. Roll Up Per Business", "desc": "12-24 months → one profile per GSTIN",
         "durationMs": 0, "detail": f"{businesses} business profile(s)."},
        {"id": 4, "name": "4. Score", "desc": "GST Underwriting Model — score + risk flag",
         "durationMs": 0, "detail": f"avg score {avg}; " + (", ".join(f"{k} {v}" for k, v in risk.items()) or "no flags")},
        {"id": 5, "name": "5. Rank Features", "desc": "GST model inputs by variance",
         "durationMs": 0, "detail": f"{len(ranking)} GST feature(s) ranked."},
    ]
    return {
        "stages": stages, "noisePercent": noise_percent, "llmActive": llm_active,
        "noiseBySource": noise_by_source,
        "processedTable": [processed_row], "storedFile": "gst_feature_profiles.csv",
        "selectedFeatures": [r["feature"] for r in ranking if r["selected"]],
        "normalizeTable": [], "engineeredTable": [], "selectionTable": selection_table,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
