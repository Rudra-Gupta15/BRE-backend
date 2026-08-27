import os
from pathlib import Path

try:  # optional — load Backend/.env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

PORT = int(os.environ.get("PORT", "4000"))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:5173")

# ── Bank-statement vision LLM (Ollama) ──────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
STATEMENT_LLM_MODEL = os.environ.get("STATEMENT_LLM_MODEL", "gemma4:31b-cloud")
STATEMENT_LLM_TIMEOUT = int(os.environ.get("STATEMENT_LLM_TIMEOUT", "120"))  # seconds per page
STATEMENT_LLM_DPI = int(os.environ.get("STATEMENT_LLM_DPI", "150"))


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
