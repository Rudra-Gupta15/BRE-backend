# SQLAlchemy engine + session for the PostgreSQL persistence layer.
#
# The database is OPTIONAL. If DATABASE_URL is unset or the server is
# unreachable at startup, `DB_ENABLED` stays False and the app runs exactly as
# before (in-memory state + .session_cache.json). Every persistence call is a
# no-op in that mode, so nothing breaks.

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.common.config import DATABASE_URL

logger = logging.getLogger(__name__)


def _ensure_database_exists(url) -> None:
    """If the target database doesn't exist yet, create it by connecting to the
    server's default 'postgres' maintenance database."""
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
                logger.info("Created database '%s'.", url.database)
    finally:
        admin_engine.dispose()


# All BRE tables live in their own `bre` schema so they never collide with
# whatever else is in the target database (e.g. an existing auth/inventory app).
DB_SCHEMA = "bre"


class Base(DeclarativeBase):
    pass


Base.metadata.schema = DB_SCHEMA


engine = None
SessionLocal = None
DB_ENABLED = False


def init_db() -> None:
    """Called once on startup. Connects, creates tables, flips DB_ENABLED."""
    global engine, SessionLocal, DB_ENABLED

    if not DATABASE_URL:
        logger.info("DATABASE_URL not set — running without PostgreSQL (in-memory only).")
        return

    try:
        url = make_url(DATABASE_URL)
        try:
            engine = create_engine(url, pool_pre_ping=True, future=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except OperationalError:
            # Most likely the database doesn't exist yet — create it and retry.
            engine.dispose() if engine else None
            _ensure_database_exists(url)
            engine = create_engine(url, pool_pre_ping=True, future=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

        from app.common import db_models  # noqa: F401 — registers the ORM models on Base

        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}"'))
        Base.metadata.create_all(engine)
        _sync_missing_columns()
        DB_ENABLED = True
        logger.info("PostgreSQL connected — persistence enabled (%s).", engine.url.render_as_string(hide_password=True))
    except Exception as exc:  # noqa: BLE001
        engine = None
        SessionLocal = None
        DB_ENABLED = False
        logger.warning("Could not connect to PostgreSQL (%s) — falling back to in-memory only.", exc)


_SA_TO_SQL = {
    "INTEGER": "INTEGER", "BIGINT": "BIGINT", "SMALLINT": "SMALLINT",
    "BOOLEAN": "BOOLEAN", "FLOAT": "DOUBLE PRECISION", "DOUBLE PRECISION": "DOUBLE PRECISION",
    "DATETIME": "TIMESTAMP WITH TIME ZONE", "TEXT": "TEXT", "JSON": "JSON",
}


def _sync_missing_columns() -> None:
    """`create_all` makes missing TABLES but never adds a column to a table that
    already exists. This adds any column defined on the ORM models that the live
    table is missing (simple additive types only — no drops, no type changes)."""
    from sqlalchemy import inspect

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name, schema=DB_SCHEMA):
                continue
            live = {c["name"] for c in insp.get_columns(table.name, schema=DB_SCHEMA)}
            for col in table.columns:
                if col.name in live:
                    continue
                sa_type = str(col.type)
                sql_type = _SA_TO_SQL.get(sa_type.split("(")[0].upper())
                if sql_type is None:
                    if sa_type.upper().startswith(("VARCHAR", "NUMERIC")):
                        sql_type = sa_type
                    else:
                        logger.warning("Skipping ADD COLUMN %s.%s — unmapped type %s",
                                       table.name, col.name, sa_type)
                        continue
                conn.execute(text(
                    f'ALTER TABLE "{DB_SCHEMA}".{table.name} ADD COLUMN IF NOT EXISTS {col.name} {sql_type}'
                ))
                logger.info("Added column %s.%s (%s)", table.name, col.name, sql_type)


def get_session():
    """Context-manager style session. Returns None when the DB is disabled."""
    if not DB_ENABLED or SessionLocal is None:
        return None
    return SessionLocal()
