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

from app.config import DATABASE_URL

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

        from app import models_db  # noqa: F401 — registers the ORM models on Base

        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}"'))
        Base.metadata.create_all(engine)
        DB_ENABLED = True
        logger.info("PostgreSQL connected — persistence enabled (%s).", engine.url.render_as_string(hide_password=True))
    except Exception as exc:  # noqa: BLE001
        engine = None
        SessionLocal = None
        DB_ENABLED = False
        logger.warning("Could not connect to PostgreSQL (%s) — falling back to in-memory only.", exc)


def get_session():
    """Context-manager style session. Returns None when the DB is disabled."""
    if not DB_ENABLED or SessionLocal is None:
        return None
    return SessionLocal()
