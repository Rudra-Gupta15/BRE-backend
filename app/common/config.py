"""Environment configuration — reads Backend/.env (optional) + os.environ.

Ports, CORS origin, the bank-statement vision-LLM settings, ML-security
guardrail limits, and the assembled DATABASE_URL. Imported everywhere; imports
nothing from the app.
"""

import os
from pathlib import Path

try:  # optional — load Backend/.env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

PORT = int(os.environ.get("PORT", "4000"))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:5173")

# ── Bank-statement vision LLM (Ollama) ──────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
STATEMENT_LLM_MODEL = os.environ.get("STATEMENT_LLM_MODEL", "gemma4:31b-cloud")
STATEMENT_LLM_TIMEOUT = int(os.environ.get("STATEMENT_LLM_TIMEOUT", "120"))  # seconds per page
STATEMENT_LLM_DPI = int(os.environ.get("STATEMENT_LLM_DPI", "150"))
STATEMENT_PARSER_VERSION = os.environ.get("STATEMENT_PARSER_VERSION", "2026.08-vision")

# Runtime override of the vision model — set when the user picks a locally
# installed model on the AI Intelligence page. `None` → fall back to the env.
_RUNTIME_VISION_MODEL: str | None = None


def set_vision_model(name: str | None) -> None:
    global _RUNTIME_VISION_MODEL
    _RUNTIME_VISION_MODEL = (name or "").strip() or None


def active_vision_model() -> str:
    return _RUNTIME_VISION_MODEL or STATEMENT_LLM_MODEL

# ── ML-security guardrails ──────────────────────────────────────────────────
# Upload limits (0 disables a check).
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))  # 15 MB
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "60"))
MAX_TRANSACTIONS = int(os.environ.get("MAX_TRANSACTIONS", "5000"))
ALLOWED_UPLOAD_EXT = tuple(
    e.strip().lower() for e in os.environ.get("ALLOWED_UPLOAD_EXT", "pdf,csv,tsv,txt,json,md,xlsx").split(",") if e.strip()
)

# Training-data poisoning guard: a candidate model may not regress the active
# model on the frozen golden set by more than this (accuracy points, 0-1).
MODEL_PROMOTE_MAX_REGRESSION = float(os.environ.get("MODEL_PROMOTE_MAX_REGRESSION", "0.03"))

# Concept-drift alarm threshold on Population Stability Index.
DRIFT_PSI_WARN = float(os.environ.get("DRIFT_PSI_WARN", "0.10"))
DRIFT_PSI_ALERT = float(os.environ.get("DRIFT_PSI_ALERT", "0.25"))

# Secret used to HMAC-sign model artifacts (falls back to a machine-local value).
MODEL_SIGNING_KEY = os.environ.get("MODEL_SIGNING_KEY", "")


def _build_database_url() -> str:
    """Assembles the SQLAlchemy URL from the plain PG_* vars in .env (so the
    .env stays simple — no manual URL-encoding of '@' or spaces needed).
    A full DATABASE_URL, if given, wins. Empty string = run without a database."""
    if os.environ.get("DATABASE_URL", "").strip():
        return os.environ["DATABASE_URL"].strip()

    host = os.environ.get("INVENTORY_PG_HOST", "").strip()
    if not host:
        return ""

    from sqlalchemy import URL

    return URL.create(
        "postgresql+psycopg",
        username=os.environ.get("INVENTORY_PG_USER", "postgres").strip(),
        password=os.environ.get("INVENTORY_PG_PASSWORD", "").strip() or None,
        host=host,
        port=int(os.environ.get("INVENTORY_PG_PORT", "5432")),
        database=os.environ.get("INVENTORY_PG_DATABASE", "postgres").strip(),
    ).render_as_string(hide_password=False)


DATABASE_URL = _build_database_url()
