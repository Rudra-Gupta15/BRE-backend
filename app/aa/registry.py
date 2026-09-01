# Versioned, on-disk model registry.
#
# Every training run writes a NEW version (risk_v1.joblib, risk_v2.joblib, …).
# Old versions are never deleted — you can roll back or compare. registry.json
# tracks metrics per version and which one is "active" (used for inference).

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from app.common.security import serialization

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"


def _read_registry() -> list[dict]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        return json.loads(REGISTRY_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return []


def _write_registry(entries: list[dict]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(entries, indent=2), "utf-8")


def next_version() -> int:
    entries = _read_registry()
    return (max((e["version"] for e in entries), default=0)) + 1


def save_version(artifact: dict, *, algorithm: str, metrics: dict, n_samples: int,
                 lineage: list[int] | None = None, batches: list[int] | None = None,
                 golden_accuracy: float | None = None, may_promote: bool = True,
                 promotion_note: str | None = None) -> dict:
    """Persist a fitted artifact as a new version.

    Integrity: the artifact's SHA-256 + HMAC signature are recorded so a later
    load can detect tampering (secure model serialization).

    Promotion: `may_promote=False` (set by the data-poisoning guard when a
    candidate regresses the golden set) keeps the previous active model in
    place — the new version is saved for audit but not served."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = next_version()
    path = MODELS_DIR / f"risk_v{version}.joblib"
    joblib.dump(artifact, path, compress=3)

    sha = serialization.sha256_file(path)
    sig = serialization.sign(sha)

    entries = _read_registry()
    entry = {
        "version": version,
        "path": str(path.name),
        "algorithm": algorithm,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "nSamples": int(n_samples),
        "metrics": metrics,
        "lineage": lineage or [e["version"] for e in entries],
        "batches": batches or [],
        "goldenAccuracy": golden_accuracy,
        "promotionNote": promotion_note,
        "sha256": sha,
        "signature": sig,
        "active": False,
    }
    entries.append(entry)

    def _quality(e: dict) -> float:
        m = e.get("metrics", {})
        return 0.5 * m.get("accuracy", 0) + 0.3 * m.get("f1", 0) + 0.2 * m.get("scoreR2", 0)

    if may_promote:
        # Active = best version by composite score (never lose a better old one).
        best = max(entries, key=_quality)
    else:
        # Poison guard tripped — keep whatever was active, else fall back to best.
        best = next((e for e in entries if e.get("active")), None) or max(entries, key=_quality)
    for e in entries:
        e["active"] = e["version"] == best["version"]

    _write_registry(entries)
    logger.info(
        "Saved model v%d (%d samples, sha %s…). Active = v%d%s.",
        version, n_samples, sha[:8], best["version"],
        "" if may_promote else " (promotion BLOCKED by poison guard)",
    )
    return {**entry, "activeVersion": best["version"]}


def list_versions() -> list[dict]:
    return sorted(_read_registry(), key=lambda e: e["version"], reverse=True)


def active_meta() -> dict | None:
    return next((e for e in _read_registry() if e.get("active")), None)


def _safe_load(meta: dict) -> dict | None:
    """Verify the artifact's hash + signature, confine the path, then unpickle."""
    try:
        path = serialization.guard_path(meta["path"])
        serialization.verify_file(path, meta.get("sha256"), meta.get("signature"))
        return {"artifact": joblib.load(path), "meta": meta}
    except serialization.ModelIntegrityError as exc:
        logger.error("REFUSING to load model v%s — %s", meta.get("version"), exc)
        try:
            from app.common.persistence import log_security_event
            log_security_event("integrity", "block", f"model:v{meta.get('version')}", {"error": str(exc)})
        except Exception:  # noqa: BLE001
            pass
        return None
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Could not load model %s (%s).", meta.get("path"), exc)
        return None


def load_active() -> dict | None:
    meta = active_meta()
    return _safe_load(meta) if meta else None


def load_versions(versions: list[int]) -> list[dict]:
    """Load specific version artifacts (for ensembling the last N)."""
    out = []
    reg = {e["version"]: e for e in _read_registry()}
    for v in versions:
        meta = reg.get(v)
        if not meta:
            continue
        loaded = _safe_load(meta)
        if loaded:
            out.append(loaded)
    return out


def set_active(version: int) -> bool:
    entries = _read_registry()
    if not any(e["version"] == version for e in entries):
        return False
    for e in entries:
        e["active"] = e["version"] == version
    _write_registry(entries)
    return True


def backfill_hashes() -> int:
    """Record SHA-256 + signature for any pre-existing artifact that lacks them
    (so integrity verification has a baseline). Run once at startup."""
    entries = _read_registry()
    changed = 0
    for e in entries:
        if e.get("sha256"):
            continue
        p = MODELS_DIR / e["path"]
        if not p.exists():
            continue
        e["sha256"] = serialization.sha256_file(p)
        e["signature"] = serialization.sign(e["sha256"])
        changed += 1
    if changed:
        _write_registry(entries)
        logger.info("Backfilled integrity hashes for %d legacy model artifact(s).", changed)
    return changed


def reset() -> None:
    for p in MODELS_DIR.glob("risk_v*.joblib"):
        p.unlink(missing_ok=True)
    REGISTRY_FILE.unlink(missing_ok=True)
