import re
import time
from datetime import datetime, timezone

from app.data.model_catalog import ML_ALGORITHMS, MOCK_TRAINING_LOGS, MODEL_TEMPLATES, PIPELINE_STAGES
from app.services.rng import create_rng, rand_int, rand_range

UNKNOWN_SOURCE_COVERAGE = 70  # custom/unrecognized feeds: no known quality metric


def _parse_coverage_percent(source: dict | None) -> float:
    coverage = source.get("coverage") if source else None
    if isinstance(coverage, str):
        m = re.search(r"[\d.]+", coverage)
        if m:
            return float(m.group())
    return UNKNOWN_SOURCE_COVERAGE


def compute_noise(selected_ids: list[str], uploaded_files: dict, sources: list[dict] | None = None) -> int:
    """Computes a "data noise" percentage from the real cleanliness of each
    selected source's uploaded file (see file_analysis.py, which scans the
    actual uploaded bytes). A source with no uploaded file contributes 0 (no
    data to be clean); one that was auto-filled (no real bytes to scan)
    falls back to that source's documented coverage metric. Cleanliness =
    100 - noise; the pipeline treats cleanliness <= 60% (noise > 40%) as
    needing the LLM cleaning stage, matching the platform's documented
    threshold."""
    if not selected_ids:
        return 60

    sources = sources or []
    source_map = {s["id"]: s for s in sources}

    cleanliness_scores = []
    for source_id in selected_ids:
        upload = uploaded_files.get(source_id)
        if not upload:
            cleanliness_scores.append(0)
        elif isinstance(upload.get("cleanlinessPercent"), (int, float)):
            cleanliness_scores.append(upload["cleanlinessPercent"])
        else:
            cleanliness_scores.append(_parse_coverage_percent(source_map.get(source_id)))

    avg_cleanliness = sum(cleanliness_scores) / len(cleanliness_scores)
    noise = round(100 - avg_cleanliness)
    return min(95, max(1, noise))


def _real_processed_row(source_id: str, upload: dict, statement: dict) -> dict:
    """Builds a processed-table row from a real parsed statement instead of
    random numbers — genuine average balance, credit/debit ratio, and
    transaction velocity for this specific source's uploaded data."""
    transactions = statement["transactions"]
    summary = statement["summary"]
    balances = [t["balance"] for t in transactions if isinstance(t.get("balance"), (int, float))]
    adb = (sum(balances) / len(balances)) if balances else (summary.get("minBalance") or 0)

    total_debit = summary.get("totalDebit") or 0
    total_credit = summary.get("totalCredit") or 0
    credit_debit_ratio = (total_credit / total_debit) if total_debit > 0 else 9.99

    # Statement span isn't reliably parseable across date formats, so this
    # approximates velocity over a typical ~30-day statement period.
    velocity_per_day = round(len(transactions) / 30, 1)

    return {
        "id": source_id,
        "adb": f"₹{round(adb):,}",
        "gstDelta": f"{min(9.99, credit_debit_ratio):.2f}",
        "upiVelocity": f"{velocity_per_day} / day",
        "cersai": "Clean (0)",
        "normScore": f"{(upload.get('cleanlinessPercent') or 0) / 100:.3f}",
        "status": "Normalized",
    }


def _simulated_processed_row(rng, source_id: str) -> dict:
    return {
        "id": source_id,
        "adb": f"₹{rand_int(rng, 1200000, 3400000):,}",
        "gstDelta": f"{rand_range(rng, 0.95, 1.06):.2f}",
        "upiVelocity": f"{rand_int(rng, 250, 550)} / day",
        "cersai": "Clean (0)",
        "normScore": f"{rand_range(rng, 0.85, 0.99):.3f}",
        "status": "Normalized",
    }


def run_pipeline(
    selected_ids: list[str],
    uploaded_files: dict,
    sources: list[dict] | None = None,
    parsed_statements: dict | None = None,
) -> dict:
    noise_percent = compute_noise(selected_ids, uploaded_files, sources)
    llm_active = noise_percent > 40
    rng = create_rng(f"pipeline:{','.join(selected_ids)}:{time.time()}")
    parsed_statements = parsed_statements or {}

    # One row per selected source: real numbers when we actually parsed
    # transactions out of that source's upload, simulated otherwise.
    processed_table = []
    for source_id in selected_ids:
        upload = uploaded_files.get(source_id)
        statement = parsed_statements.get(source_id)
        if upload and statement and statement["transactions"]:
            processed_table.append(_real_processed_row(source_id, upload, statement))
        else:
            processed_table.append(_simulated_processed_row(rng, source_id))

    return {
        "stages": PIPELINE_STAGES,
        "logs": MOCK_TRAINING_LOGS,
        "noisePercent": noise_percent,
        "llmActive": llm_active,
        "processedTable": processed_table,
        "storedFile": "processed_features_vector.csv",
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }


def train_models(algorithm: str) -> dict:
    algo = next((a for a in ML_ALGORITHMS if a["value"] == algorithm), ML_ALGORITHMS[0])
    rng = create_rng(f"train:{algorithm}:{time.time()}")

    models = []
    for tpl in MODEL_TEMPLATES:
        accuracy = max(60, min(99.9, tpl["baseAccuracy"] + algo["accuracyDelta"] + rand_range(rng, -0.4, 0.4)))
        models.append({
            "id": tpl["id"],
            "name": tpl["name"],
            "desc": tpl["desc"],
            "accuracy": f"{accuracy:.1f}%",
            "algorithm": algo["value"],
            "createdDate": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    return {"models": models, "algorithm": algo["value"]}
