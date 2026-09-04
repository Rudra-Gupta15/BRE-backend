"""AA Model Hub pipeline — the animated "process data" stage sequence.

run_pipeline(...) turns the uploaded bank statement(s) into the processed /
normalised / engineered / feature-selection tables the Model Hub UI walks
through. The GST equivalent is app.gst.pipeline.run_gst_pipeline.
"""

import csv
import os
import re
import time
from datetime import datetime, timezone

import numpy as np

from app.aa.model import extract_features
from app.common.rng import create_rng, rand_int, rand_range

UNKNOWN_SOURCE_COVERAGE = 70  # custom/unrecognized feeds: no known quality metric

_GENERATED_DIR = os.path.join(os.path.dirname(__file__), "_data", "generated")

# Domain-assumed bounds used for MinMax scaling (stage 3) and for putting
# features on a comparable [0,1] scale before variance ranking (stage 5).
# There's no real historical applicant population in this system, so these
# are documented business assumptions about plausible ranges, not fitted
# from data — real formula, assumed reference range.
FEATURE_BOUNDS = {
    "avg_daily_balance":    (1_000, 5_000_000),
    "balance_volatility":   (0.0, 5.0),
    "credit_debit_ratio":   (0.0, 10.0),
    "monthly_credit":       (1_000, 5_000_000),
    "monthly_debit":        (1_000, 5_000_000),
    "tx_velocity":          (0.0, 20.0),
    "max_drawdown_pct":     (0.0, 1.0),
    "large_tx_pct":         (0.0, 1.0),
    "irregular_gap_score":  (0.0, 1.0),
    "net_flow_ratio":       (-1.0, 1.0),
    "liquidity_runway_days": (0.0, 180.0),
}

# Assumed portfolio-level (mean, std) used as the Z-score reference, for the
# same reason as FEATURE_BOUNDS above — no historical population to fit on.
FEATURE_REFERENCE_STATS = {
    "avg_daily_balance":    (150_000, 120_000),
    "balance_volatility":   (0.35, 0.25),
    "credit_debit_ratio":   (1.1, 0.6),
    "monthly_credit":       (120_000, 90_000),
    "monthly_debit":        (100_000, 80_000),
    "tx_velocity":          (1.2, 0.8),
    "max_drawdown_pct":     (0.18, 0.15),
    "large_tx_pct":         (0.12, 0.1),
    "irregular_gap_score":  (0.3, 0.2),
    "net_flow_ratio":       (0.05, 0.3),
    "liquidity_runway_days": (25.0, 20.0),
}

BASE_FEATURE_NAMES  = [f for f in FEATURE_BOUNDS if f not in ("net_flow_ratio", "liquidity_runway_days")]
EXTRA_FEATURE_NAMES = ["net_flow_ratio", "liquidity_runway_days"]

FEATURE_LABELS = {
    "avg_daily_balance":     "Avg Daily Balance",
    "balance_volatility":    "Balance Volatility",
    "credit_debit_ratio":    "Credit / Debit Ratio",
    "monthly_credit":        "Monthly Credit",
    "monthly_debit":         "Monthly Debit",
    "tx_velocity":           "Transaction Velocity",
    "max_drawdown_pct":      "Max Drawdown %",
    "large_tx_pct":          "Large Transaction %",
    "irregular_gap_score":   "Irregular Gap Score",
    "net_flow_ratio":        "Net Cash-Flow Ratio",
    "liquidity_runway_days": "Liquidity Runway (Days)",
}


def _parse_coverage_percent(source: dict | None) -> float:
    coverage = source.get("coverage") if source else None
    if isinstance(coverage, str):
        m = re.search(r"[\d.]+", coverage)
        if m:
            return float(m.group())
    return UNKNOWN_SOURCE_COVERAGE


def compute_noise_by_source(selected_ids: list[str], uploaded_files: dict,
                             sources: list[dict] | None = None) -> dict[str, dict]:
    """Per-source "data noise" — never blended into one averaged number,
    since averaging a clean source against a dirty one hides the dirty one.
    A source with no uploaded file contributes 0 cleanliness (no data to be
    clean); one that was auto-filled (no real bytes to scan) falls back to
    that source's documented coverage metric. Cleanliness = 100 - noise; each
    source crossing noise > 40% needs its own LLM cleaning pass, matching the
    platform's documented threshold — independently of every other source."""
    sources = sources or []
    source_map = {s["id"]: s for s in sources}

    out: dict[str, dict] = {}
    for source_id in selected_ids:
        upload = uploaded_files.get(source_id)
        if not upload:
            cleanliness = 0.0
        elif isinstance(upload.get("cleanlinessPercent"), (int, float)):
            cleanliness = upload["cleanlinessPercent"]
        else:
            cleanliness = _parse_coverage_percent(source_map.get(source_id))
        noise = min(95, max(1, round(100 - cleanliness)))
        out[source_id] = {
            "label": source_map.get(source_id, {}).get("title", source_id),
            "cleanlinessPercent": round(cleanliness),
            "noisePercent": noise,
            "llmActive": noise > 40,
            "fileCount": upload.get("fileCount") if upload else 0,
            "worstFileName": upload.get("worstFileName") if upload else None,
        }
    return out


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _minmax(value: float, feat: str) -> float:
    lo, hi = FEATURE_BOUNDS[feat]
    if hi == lo:
        return 0.5
    return _clip01((value - lo) / (hi - lo))


def _zscore(value: float, feat: str) -> float:
    mean, std = FEATURE_REFERENCE_STATS[feat]
    if std == 0:
        return 0.0
    return round((value - mean) / std, 4)


def _extra_engineered_features(feat: dict) -> dict:
    """Two genuinely new engineered ratios beyond extract_features()'s base 9,
    computed from real monthly credit/debit and average balance."""
    m_credit = feat["monthly_credit"]
    m_debit  = feat["monthly_debit"]
    net_flow_ratio = max(-1.0, min(1.0, (m_credit - m_debit) / max(m_credit, 1.0)))
    liquidity_runway_days = min(180.0, feat["avg_daily_balance"] / max(m_debit / 30.0, 1.0))
    return {
        "net_flow_ratio":        round(net_flow_ratio, 4),
        "liquidity_runway_days": round(liquidity_runway_days, 1),
    }


def _chunk_transactions(txns: list[dict], n_chunks: int = 5, min_size: int = 3) -> list[list[dict]]:
    """Splits one source's cleaned transactions into sequential chunks so
    stage 5 can rank feature variance even when only a single data source
    is selected (variance across chunks of the same statement, instead of
    needing multiple sources to compare)."""
    if len(txns) < min_size * 2:
        return [txns] if txns else []
    chunk_size = max(min_size, len(txns) // n_chunks)
    return [
        txns[i:i + chunk_size]
        for i in range(0, len(txns), chunk_size)
        if len(txns[i:i + chunk_size]) >= min_size
    ]


def _detect_duplicate_transactions(txns: list[dict]) -> int:
    """Counts transactions that share the same (narration, amount, type) as
    another row in the same statement — a real, if simple, duplicate-charge
    signal computed from the actual parsed data (there is no live CERSAI
    registry this demo can query, so this replaces what was a hardcoded
    'Clean (0)' placeholder)."""
    seen: dict[tuple, int] = {}
    for t in txns:
        key = (t.get("narration"), t.get("amount"), t.get("type"))
        seen[key] = seen.get(key, 0) + 1
    return sum(count - 1 for count in seen.values() if count > 1)


# ── Stage 1: Data Gathering ────────────────────────────────────────────────────

def _stage1_gather(selected_ids: list[str], parsed_statements: dict) -> dict:
    t0 = time.perf_counter()
    gathered = {}
    total_tx = 0
    sources_without_data = []
    for sid in selected_ids:
        stmt = parsed_statements.get(sid)
        txns = list(stmt["transactions"]) if stmt and stmt.get("transactions") else []
        gathered[sid] = txns
        total_tx += len(txns)
        if not txns:
            sources_without_data.append(sid)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    detail = f"Gathered {total_tx} transaction(s) across {len(selected_ids)} source(s)."
    if sources_without_data:
        detail += f" {len(sources_without_data)} source(s) have no uploaded statement — will use simulated placeholders."
    return {"gathered": gathered, "totalTransactions": total_tx, "durationMs": duration_ms, "detail": detail}


# ── Stage 2: Preprocess Data ───────────────────────────────────────────────────

# A statement narration sometimes carries the amount even when the parser's
# dedicated amount column failed to extract it — e.g. "UPI/mmt/Rs.1,250.00 to
# XYZ". Real regex recovery, not a guess: only used when it actually matches.
_NARRATION_AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)


def _recover_from_narration(narration: str) -> float | None:
    m = _NARRATION_AMOUNT_RE.search(narration or "")
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return v if v > 0 else None


_DEBIT_HINTS  = ("wdl", "withdrawal", "purchase", "pos ", "payment", "paid", "debit", "emi", "bill", "atm")
_CREDIT_HINTS = ("salary", "credit", "interest", "refund", "reversal", "deposit", "received")


def _guess_type_from_narration(narration: str) -> str | None:
    low = (narration or "").lower()
    if any(k in low for k in _DEBIT_HINTS):
        return "DEBIT"
    if any(k in low for k in _CREDIT_HINTS):
        return "CREDIT"
    return None


def _clean_one_source(txns: list[dict], *, recover: bool) -> dict:
    """Baseline (recover=False): drop rows with no usable amount, forward-fill
    missing balance — unchanged behaviour for a source under the noise
    threshold. AI recovery pass (recover=True, only run for a source whose
    noise > 40%): before giving up on a row, try to reconstruct its amount
    from the running balance delta, then from the narration text (with a
    keyword-based debit/credit guess when the type is unknown too); a row is
    only dropped once neither recovery path finds anything — and every drop
    gets a specific, real reason instead of a bare count. The running balance
    updates from every row that carries one, even a dropped row — its own
    amount being unrecoverable doesn't make its balance snapshot any less
    real, and the next row's delta recovery depends on it."""
    last_balance = None
    clean_list, dropped_detail = [], []
    recovered_rows = imputed_rows = 0
    for t in txns:
        amount = t.get("amount")
        tx_type = t.get("type")
        row_balance = t.get("balance")
        has_amount = isinstance(amount, (int, float)) and amount > 0
        guessed_type = False

        if not has_amount and recover:
            method = None
            if isinstance(row_balance, (int, float)) and isinstance(last_balance, (int, float)):
                delta = round(row_balance - last_balance, 2)
                if delta != 0:
                    amount = abs(delta)
                    if not tx_type:
                        tx_type = "CREDIT" if delta > 0 else "DEBIT"
                    method = f"inferred from balance change ({last_balance:,.2f} → {row_balance:,.2f})"
            if amount is None or amount <= 0:
                narr_amount = _recover_from_narration(t.get("narration", ""))
                if narr_amount:
                    inferred_type = tx_type or _guess_type_from_narration(t.get("narration", ""))
                    if inferred_type:
                        amount = narr_amount
                        tx_type = inferred_type
                        guessed_type = not t.get("type")
                        method = f"amount pattern found in narration ({t.get('narration', '')[:60]!r})"
            if method:
                has_amount = True
                recovered_rows += 1

        # The balance snapshot on THIS row is real, observed data regardless
        # of whether we could work out an amount for it — carry it forward
        # so a later row can still delta against it.
        if isinstance(row_balance, (int, float)):
            last_balance = row_balance

        if not has_amount:
            reason = (
                "no amount value, and " + (
                    "the running balance shows no change to infer a delta"
                    if isinstance(row_balance, (int, float)) and isinstance(last_balance, (int, float))
                    else "no balance on this or an earlier row to infer a delta"
                ) + ", and no amount pattern in the narration"
                if recover else
                "missing or non-positive amount"
            )
            dropped_detail.append({
                "date": t.get("date") or "—",
                "narration": (t.get("narration") or "")[:80] or "(no narration)",
                "reason": reason,
            })
            continue

        balance = row_balance
        if balance is None:
            balance = last_balance
            if balance is not None:
                imputed_rows += 1
        entry = {**t, "amount": amount, "type": tx_type or t.get("type"), "balance": balance}
        if guessed_type:
            entry["_recoveryNote"] = "type guessed from narration wording"
        clean_list.append(entry)

    return {
        "cleaned": clean_list, "totalRows": len(txns), "droppedRows": len(dropped_detail),
        "recoveredRows": recovered_rows, "imputedRows": imputed_rows, "droppedDetail": dropped_detail,
    }


def _stage2_preprocess(gathered: dict, noise_by_source: dict | None = None) -> dict:
    t0 = time.perf_counter()
    noise_by_source = noise_by_source or {}
    cleaned, by_source = {}, {}
    total_rows = dropped_rows = imputed_rows = recovered_rows = 0
    for sid, txns in gathered.items():
        recover = bool(noise_by_source.get(sid, {}).get("llmActive"))
        r = _clean_one_source(txns, recover=recover)
        cleaned[sid] = r["cleaned"]
        total_rows += r["totalRows"]
        dropped_rows += r["droppedRows"]
        recovered_rows += r["recoveredRows"]
        imputed_rows += r["imputedRows"]
        by_source[sid] = {
            "mode": "ai_recovery" if recover else "baseline",
            "totalRows": r["totalRows"], "droppedRows": r["droppedRows"],
            "recoveredRows": r["recoveredRows"],
            "droppedDetail": r["droppedDetail"][:25],
            "droppedDetailTruncated": max(0, len(r["droppedDetail"]) - 25),
        }
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    ai_sources = [sid for sid, r in by_source.items() if r["mode"] == "ai_recovery"]
    detail = (
        f"Cleaned {total_rows} raw row(s) — dropped {dropped_rows} unusable, "
        f"forward-filled {imputed_rows} missing balance(s)."
    )
    if ai_sources:
        detail += (
            f" AI recovery pass ran on {len(ai_sources)} high-noise source(s), "
            f"recovering {recovered_rows} row(s) that baseline cleaning would have dropped."
        )
    return {
        "cleaned": cleaned, "totalRows": total_rows, "droppedRows": dropped_rows,
        "imputedRows": imputed_rows, "recoveredRows": recovered_rows,
        "bySource": by_source, "durationMs": duration_ms, "detail": detail,
    }


# ── Stage 3: Normalize Data ────────────────────────────────────────────────────

def _stage3_normalize(cleaned: dict) -> dict:
    t0 = time.perf_counter()
    per_source = {}
    for sid, txns in cleaned.items():
        if not txns:
            continue
        raw = extract_features({"transactions": txns, "summary": {}})
        minmax = {f: round(_minmax(v, f), 4) for f, v in raw.items() if f in FEATURE_BOUNDS}
        zscore = {f: _zscore(v, f) for f, v in raw.items() if f in FEATURE_REFERENCE_STATS}
        per_source[sid] = {"raw": raw, "minmax": minmax, "zscore": zscore}
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    detail = (
        f"Applied MinMax scaling (domain-bounded [0,1]) and Z-score standardization "
        f"(assumed portfolio reference μ/σ) to {len(per_source)} feature vector(s)."
    )
    return {"perSource": per_source, "durationMs": duration_ms, "detail": detail}


# ── Stage 4: Feature Engineering ───────────────────────────────────────────────

def _stage4_engineer(normalized_per_source: dict) -> dict:
    t0 = time.perf_counter()
    engineered = {}
    for sid, data in normalized_per_source.items():
        extra = _extra_engineered_features(data["raw"])
        full_raw = {**data["raw"], **extra}
        full_minmax = {**data["minmax"], **{f: round(_minmax(v, f), 4) for f, v in extra.items()}}
        full_zscore = {**data["zscore"], **{f: _zscore(v, f) for f, v in extra.items()}}
        engineered[sid] = {**data, "raw": full_raw, "minmax": full_minmax, "zscore": full_zscore}
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    detail = f"Derived 2 temporal ratio feature(s) (net cash-flow ratio, liquidity runway) for {len(engineered)} source(s)."
    return {"perSource": engineered, "durationMs": duration_ms, "detail": detail}


# ── Stage 5: Data Selection ────────────────────────────────────────────────────

def _stage5_select(cleaned: dict, engineered_per_source: dict, top_k: int = 6) -> dict:
    t0 = time.perf_counter()
    feature_names = list(FEATURE_BOUNDS.keys())

    # Prefer variance across chronological chunks *within* each source, so
    # ranking works even with a single selected source (the common case).
    samples = []
    for sid, txns in cleaned.items():
        for chunk in _chunk_transactions(txns):
            feat = extract_features({"transactions": chunk, "summary": {}})
            full = {**feat, **_extra_engineered_features(feat)}
            samples.append([_minmax(full[f], f) for f in feature_names])

    method = "intra-source chronological chunks"
    if len(samples) < 2:
        # Not enough chunks anywhere — fall back to comparing across the
        # selected sources' whole-statement feature vectors instead.
        samples = [
            [engineered_per_source[sid]["minmax"].get(f, 0.0) for f in feature_names]
            for sid in engineered_per_source
        ]
        method = "cross-source comparison"

    if len(samples) >= 2:
        X = np.array(samples)
        variances = X.var(axis=0)
        ranked = sorted(zip(feature_names, variances), key=lambda p: p[1], reverse=True)
        selected = [name for name, _ in ranked[:top_k]]
        variance_by_feature = {name: round(float(v), 5) for name, v in ranked}
    else:
        # A single data point — variance is mathematically undefined.
        # Keep every feature rather than fake a ranking.
        selected = feature_names
        variance_by_feature = {name: None for name in feature_names}
        method = "insufficient data for variance ranking (single sample) — all features retained"

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    detail = f"Ranked {len(feature_names)} feature(s) by variance ({method}); selected top {len(selected)}."
    return {
        "selectedFeatures": selected, "varianceByFeature": variance_by_feature,
        "durationMs": duration_ms, "detail": detail,
    }


# ── Processed table rows ───────────────────────────────────────────────────────

def _real_processed_row(source_id: str, cleaned_txns: list[dict], engineered_data: dict, selected_features: list[str]) -> dict:
    raw = engineered_data["raw"]
    minmax = engineered_data["minmax"]
    dup_count = _detect_duplicate_transactions(cleaned_txns)
    relevant = [minmax[f] for f in selected_features if f in minmax]
    norm_score = sum(relevant) / len(relevant) if relevant else 0.0

    return {
        "id": source_id,
        "adb": f"₹{round(raw['avg_daily_balance']):,}",
        "gstDelta": f"{min(9.99, raw['credit_debit_ratio']):.2f}",
        "upiVelocity": f"{round(len(cleaned_txns) / 30, 1)} / day",
        "cersai": "Clean (0)" if dup_count == 0 else f"{dup_count} Duplicate(s) Found",
        "normScore": f"{norm_score:.3f}",
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
        "status": "Simulated (no upload)",
    }


def _gst_processed_row(source_id: str, gst: dict) -> dict:
    """GST source → a row describing the GST-model scoring, not bank features."""
    seen = gst.get("returnsSeen") or {}
    seen_str = " + ".join(k for k, v in seen.items() if v) or (
        "GST summary" if gst.get("mode") == "summary" else "GST returns")
    risk = gst.get("riskCounts") or {}
    risk_str = " / ".join(f"{k} {v}" for k, v in risk.items()) or "—"
    avg = gst.get("avgUnderwritingScore")
    biz = gst.get("businesses") or gst.get("records") or 0
    return {
        "id": source_id,
        "kind": "gst",
        "adb": f"{biz} business{'' if biz == 1 else 'es'}",
        "gstDelta": f"score {avg}" if avg is not None else "—",
        "upiVelocity": seen_str,
        "cersai": risk_str,
        "normScore": f"{avg / 100:.3f}" if avg is not None else "—",
        "status": "GST scored",
    }


# ── UI table builders (one row per source×feature, for the frontend) ──────────

def _build_normalize_table(normalized_per_source: dict) -> list[dict]:
    """Stage 3 output: raw/MinMax/Z-score for the 9 base features, per source."""
    rows = []
    for sid, data in normalized_per_source.items():
        for f in BASE_FEATURE_NAMES:
            rows.append({
                "sourceId": sid,
                "feature":  FEATURE_LABELS.get(f, f),
                "raw":      round(data["raw"][f], 4),
                "minmax":   data["minmax"][f],
                "zscore":   data["zscore"][f],
            })
    return rows


def _build_engineered_table(engineered_per_source: dict) -> list[dict]:
    """Stage 4 output: the 2 newly-derived temporal-ratio features, per source."""
    rows = []
    for sid, data in engineered_per_source.items():
        for f in EXTRA_FEATURE_NAMES:
            rows.append({
                "sourceId": sid,
                "feature":  FEATURE_LABELS.get(f, f),
                "value":    round(data["raw"][f], 4),
                "minmax":   data["minmax"][f],
                "zscore":   data["zscore"][f],
            })
    return rows


def _build_selection_table(stage5: dict) -> list[dict]:
    """Stage 5 output: every candidate feature ranked by variance, flagged
    with whether it made the top-K cut. Dict insertion order already
    reflects the variance ranking computed in _stage5_select."""
    selected_set = set(stage5["selectedFeatures"])
    rows = []
    for rank, (f, variance) in enumerate(stage5["varianceByFeature"].items(), start=1):
        rows.append({
            "rank":     rank,
            "feature":  FEATURE_LABELS.get(f, f),
            "variance": variance,
            "selected": f in selected_set,
        })
    return rows


# ── Feature vector CSV persistence ─────────────────────────────────────────────

def _write_feature_vector_csv(selected_ids: list[str], engineered_per_source: dict, selected_features: list[str]) -> str:
    """Actually writes the processed feature vector to disk, so the UI's
    'Stored File' label points at a real file instead of a name with
    nothing behind it."""
    os.makedirs(_GENERATED_DIR, exist_ok=True)
    filename = "processed_features_vector.csv"
    path = os.path.join(_GENERATED_DIR, filename)
    feature_names = list(FEATURE_BOUNDS.keys())
    selected_set = set(selected_features)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id"] + feature_names + ["selected_by_variance"])
        for sid in selected_ids:
            data = engineered_per_source.get(sid)
            if not data:
                continue
            row = [sid] + [data["raw"].get(f, "") for f in feature_names]
            row.append(",".join(f for f in feature_names if f in selected_set))
            writer.writerow(row)

    return filename


# ── Orchestrator ────────────────────────────────────────────────────────────────

def run_pipeline(
    selected_ids: list[str],
    uploaded_files: dict,
    sources: list[dict] | None = None,
    parsed_statements: dict | None = None,
) -> dict:
    parsed_statements = parsed_statements or {}
    noise_by_source = compute_noise_by_source(selected_ids, uploaded_files, sources)
    # Back-compat single figures (worst source wins — never an average that
    # could hide one dirty source behind clean ones).
    noise_percent = max((v["noisePercent"] for v in noise_by_source.values()), default=1)
    llm_active = any(v["llmActive"] for v in noise_by_source.values())

    s1 = _stage1_gather(selected_ids, parsed_statements)
    s2 = _stage2_preprocess(s1["gathered"], noise_by_source)
    s3 = _stage3_normalize(s2["cleaned"])
    s4 = _stage4_engineer(s3["perSource"])
    s5 = _stage5_select(s2["cleaned"], s4["perSource"])

    rng = create_rng(f"pipeline:{','.join(selected_ids)}:{time.time()}")
    processed_table = []
    for source_id in selected_ids:
        gst_block = (parsed_statements.get(source_id) or {}).get("gst")
        if gst_block:
            processed_table.append(_gst_processed_row(source_id, gst_block))
            continue
        engineered = s4["perSource"].get(source_id)
        cleaned_txns = s2["cleaned"].get(source_id) or []
        if engineered and cleaned_txns:
            processed_table.append(_real_processed_row(source_id, cleaned_txns, engineered, s5["selectedFeatures"]))
        else:
            processed_table.append(_simulated_processed_row(rng, source_id))

    stored_file = _write_feature_vector_csv(selected_ids, s4["perSource"], s5["selectedFeatures"])
    selection_table = _build_selection_table(s5)

    stages = [
        {"id": 1, "name": "1. Data Gathering", "desc": "Fetch & aggregate feeds from selected sources",
         "durationMs": s1["durationMs"], "detail": s1["detail"]},
        {"id": 2, "name": "2. Preprocess Data", "desc": "Clean missing values & filter noise",
         "durationMs": s2["durationMs"], "detail": s2["detail"]},
        {"id": 3, "name": "3. Normalize Data", "desc": "MinMax scaling & Z-score standardization",
         "durationMs": s3["durationMs"], "detail": s3["detail"]},
        {"id": 4, "name": "4. Feature Engineering", "desc": "Generate variables & temporal ratios",
         "durationMs": s4["durationMs"], "detail": s4["detail"]},
        {"id": 5, "name": "5. Data Selection", "desc": "Select high-variance predictor features",
         "durationMs": s5["durationMs"], "detail": s5["detail"]},
    ]

    return {
        "stages": stages,
        "noisePercent": noise_percent,
        "llmActive": llm_active,
        "noiseBySource": noise_by_source,
        "cleaningBySource": s2["bySource"],
        "processedTable": processed_table,
        "storedFile": stored_file,
        "selectedFeatures": s5["selectedFeatures"],
        "normalizeTable": _build_normalize_table(s3["perSource"]),
        "engineeredTable": _build_engineered_table(s4["perSource"]),
        "selectionTable": selection_table,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
