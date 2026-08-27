"""Quick PostgreSQL connectivity check. Run from Backend/:  python check_db.py"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from app.db import init_db  # noqa: E402

init_db()

from app.db import DB_ENABLED, engine  # noqa: E402

if not DB_ENABLED:
    print("\n❌ Not connected. Check DATABASE_URL in Backend/.env and that the server is reachable.")
    raise SystemExit(1)

from sqlalchemy import inspect  # noqa: E402

insp = inspect(engine)
tables = sorted(insp.get_table_names(schema="bre"))
print(f"\n✅ Connected: {engine.url.render_as_string(hide_password=True)}")
print(f"   schema 'bre' tables: {tables or '(none yet — will be created on first use)'}")
