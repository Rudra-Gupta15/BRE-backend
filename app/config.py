import os
from pathlib import Path

try:  # optional — load Backend/.env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

PORT = int(os.environ.get("PORT", "4000"))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "http://localhost:5173")

# PostgreSQL connection. Set in Backend/.env, e.g.
#   DATABASE_URL=postgresql+psycopg://postgres:yourpassword@localhost:5432/bre
# Leave unset to run fully in-memory (state still cached to .session_cache.json).
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
