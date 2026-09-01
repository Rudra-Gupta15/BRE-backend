"""Quick PostgreSQL connectivity check. Run from Backend/:  python scripts/check_db.py"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from app.common.db import init_db  # noqa: E402

init_db()

from app.common.db import DB_ENABLED, engine  # noqa: E402

if not DB_ENABLED:
    print("\nFAIL: Not connected. Check DATABASE_URL in Backend/.env and that the server is reachable.")
    raise SystemExit(1)

from sqlalchemy import inspect  # noqa: E402

insp = inspect(engine)
tables = sorted(insp.get_table_names(schema="bre"))
print(f"\nOK: Connected: {engine.url.render_as_string(hide_password=True)}")
print(f"   schema 'bre' tables: {tables or '(none yet — will be created on first use)'}")
