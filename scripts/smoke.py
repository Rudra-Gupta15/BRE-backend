"""End-to-end route smoke test — the regression oracle for the package refactor.

Boots the FastAPI app in-process (TestClient), resets to a clean session, then
hits a fixed list of read-mostly endpoints and records a *normalized* snapshot of
(status_code, response_shape) for each. Volatile values (timestamps, durations,
uptimes, random ids) are blanked so two runs on identical code produce byte-equal
output.

Usage:
    python scripts/smoke.py            # write scripts/smoke_baseline.json
    python scripts/smoke.py --check    # diff current run vs the baseline, exit 1 on drift
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASELINE = _HERE / "smoke_baseline.json"

sys.path.insert(0, str(_HERE.parent))  # import app.*

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

# ── endpoints ────────────────────────────────────────────────────────────────
# (method, path). Only safe / side-effect-free calls (plus /reset up front).
GET_ENDPOINTS = [
    "/api/health",
    "/api/data-sources",
    "/api/data-sources/presets",
    "/api/data-sources/selection",
    "/api/data-sources/status",
    "/api/data-sources/account_aggregator",
    "/api/data-sources/gst_data",
    "/api/pipeline/uploads",
    "/api/pipeline/status",
    "/api/models",
    "/api/models/algorithms",
    "/api/models/dataset/status",
    "/api/models/evaluation",
    "/api/models/evaluation?model_id=gst_risk_flag_model",
    "/api/models/evaluation/summary",
    "/api/models/patterns",
    "/api/models/registry",
    "/api/inference/deployed-models",
    "/api/inference/history",
    "/api/bre-products",
    "/api/bre-products/active",
    "/api/bre-products/source-usage",
    "/api/bre-products/lap_sbl/sources",
    "/api/data-source-rules/account_aggregator",
    "/api/data-source-rules/gst_data",
    "/api/gst/model",
    "/api/gst/model/registry",
    "/api/gst/patterns",
    "/api/gst/pattern-match",
    "/api/gst/score-testing",
    "/api/ai-architecture",
    "/api/settings/rules",
    "/api/settings/scoring",
    "/api/settings/general",
    "/api/dashboard/kpis",
    "/api/dashboard/charts",
    "/api/dashboard/recent-statements",
    "/api/security/overview",
    "/api/security/events",
    "/api/security/batches",
    "/api/security/drift",
    "/api/security/models/integrity",
]

# POSTs with no lasting side effect beyond a fresh session (run in this order).
POST_ENDPOINTS = [
    ("/api/reset", {}),
    ("/api/inference/bre-rules", {"customId": "smoke", "sourceId": "account_aggregator"}),
    ("/api/inference/patterns", {"customId": "smoke", "sourceId": "account_aggregator"}),
    ("/api/gst/bre-evaluate", {}),
]

_TS_KEYS = {
    "evaluatedAt", "trainedAt", "lastRunAt", "scannedAt", "savedAt", "date",
    "timestamp", "uptime", "completedAt", "sessionTrainedAt", "gstTrainedAt",
    "createdDate", "generatedAt", "at", "ranAt", "startedAt",
}
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
# Absolute filesystem paths appear in a few responses (dataset/model file
# locations). They move with the refactor but the behavior is identical, so
# blank anything that looks like a path into or under the repo.
_PATH_RE = re.compile(r"[A-Za-z]:[\\/].*Backend|/.*/Backend|Backend[\\/]app[\\/]")


def _norm(obj):
    """Blank volatile leaves so the snapshot is stable across identical runs."""
    if isinstance(obj, dict):
        return {k: ("<ts>" if k in _TS_KEYS else _norm(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm(x) for x in obj]
    if isinstance(obj, str) and _TS_RE.search(obj):
        return "<ts>"
    if isinstance(obj, str) and _PATH_RE.search(obj):
        return "<path>"
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def run() -> dict:
    out: dict = {}
    with TestClient(app) as client:
        for path, body in POST_ENDPOINTS:
            try:
                r = client.post(path, json=body)
                out[f"POST {path}"] = {"status": r.status_code, "body": _norm(_safe_json(r))}
            except Exception as exc:  # noqa: BLE001
                out[f"POST {path}"] = {"error": type(exc).__name__ + ": " + str(exc)}
        for path in GET_ENDPOINTS:
            try:
                r = client.get(path)
                out[f"GET {path}"] = {"status": r.status_code, "body": _norm(_safe_json(r))}
            except Exception as exc:  # noqa: BLE001
                out[f"GET {path}"] = {"error": type(exc).__name__ + ": " + str(exc)}
    return out


def _safe_json(r):
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {"_text": r.text[:400]}


def _keys_only(obj):
    """Second, looser view: structure (keys / list-lengths) without values —
    catches shape drift even when values legitimately vary run to run."""
    if isinstance(obj, dict):
        return {k: _keys_only(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [f"list[{len(obj)}]"] + ([_keys_only(obj[0])] if obj else [])
    return type(obj).__name__


def main() -> int:
    check = "--check" in sys.argv
    current = run()
    shape = {k: (v if "error" in v else {"status": v["status"], "shape": _keys_only(v["body"])})
             for k, v in current.items()}
    payload = {"snapshot": current, "shape": shape}

    if not check:
        _BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
        print(f"baseline written: {_BASELINE}  ({len(current)} endpoints)")
        return 0

    if not _BASELINE.exists():
        print("no baseline — run `python scripts/smoke.py` first")
        return 1
    base = json.loads(_BASELINE.read_text("utf-8"))
    drift = []
    for k in sorted(set(base["snapshot"]) | set(current)):
        b = base["snapshot"].get(k)
        c = payload["snapshot"].get(k)
        if b != c:
            drift.append(k)
    if drift:
        print(f"DRIFT in {len(drift)} endpoint(s):")
        for k in drift:
            print(f"  - {k}")
            print(f"      baseline: {json.dumps(base['snapshot'].get(k))[:300]}")
            print(f"      current : {json.dumps(payload['snapshot'].get(k))[:300]}")
        return 1
    print(f"OK — {len(current)} endpoints match baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
