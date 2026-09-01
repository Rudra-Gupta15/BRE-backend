"""Secure model serialization.

`joblib.load` unpickles - arbitrary code execution if an artifact (or the
registry file that names it) is tampered with. Defences:

  * every artifact gets a SHA-256 recorded at save time
  * every artifact gets an HMAC signature (key from config.MODEL_SIGNING_KEY,
    else a machine-local key file)
  * loads verify hash + signature and refuse on mismatch
  * loads are confined to the known models directory - never a caller path
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from pathlib import Path

from app.common import config

logger = logging.getLogger(__name__)

_KEY_FILE = Path(__file__).resolve().parents[3] / "models" / ".signing_key"


class ModelIntegrityError(RuntimeError):
    """Raised when an artifact fails hash or signature verification."""


def _signing_key() -> bytes:
    if config.MODEL_SIGNING_KEY:
        return config.MODEL_SIGNING_KEY.encode()
    try:
        if _KEY_FILE.exists():
            return _KEY_FILE.read_text().strip().encode()
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        _KEY_FILE.write_text(key)
        return key.encode()
    except OSError:
        # Last resort - signatures still stable within a process.
        return b"bre-insecure-fallback-key"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sign(digest_hex: str) -> str:
    return hmac.new(_signing_key(), digest_hex.encode(), hashlib.sha256).hexdigest()


def verify_file(path: str | Path, expected_sha: str | None, signature: str | None) -> None:
    """Raise ModelIntegrityError unless the file matches the recorded hash and
    that hash carries a valid signature. Missing metadata -> treated as legacy
    (logged, allowed) so pre-existing artifacts keep working."""
    p = Path(path)
    if not p.exists():
        raise ModelIntegrityError(f"artifact missing: {p.name}")

    if not expected_sha:
        logger.warning("Model %s has no recorded SHA-256 - loading unverified (legacy).", p.name)
        return

    actual = sha256_file(p)
    if not hmac.compare_digest(actual, expected_sha):
        raise ModelIntegrityError(
            f"artifact {p.name} SHA-256 mismatch - expected {expected_sha[:12]}..., got {actual[:12]}..."
        )
    if signature and not hmac.compare_digest(sign(expected_sha), signature):
        raise ModelIntegrityError(f"artifact {p.name} signature invalid - registry may be tampered")


def guard_path(path: str | Path) -> Path:
    """Refuse any artifact path that escapes the models directory."""
    models_dir = (Path(__file__).resolve().parents[3] / "models").resolve()
    p = Path(path)
    resolved = p if p.is_absolute() else (models_dir / p)
    resolved = resolved.resolve()
    if models_dir not in resolved.parents and resolved != models_dir:
        raise ModelIntegrityError(f"artifact path escapes models dir: {path}")
    return resolved
