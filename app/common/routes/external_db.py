"""/external-db — an ad-hoc, stateless connection probe for the bank's or
company's own database, so Model Hub's "Database Folder" / "Database File"
upload mode can browse it instead of a local file.

No sample bank DB exists yet — this is the real connection code for when one
is wired up. The connection string is never persisted; it's used once to
connect, list schemas/tables, then the connection is closed."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external-db", tags=["external-db"])

_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast", "sys", "mysql", "performance_schema"}


class ConnectBody(BaseModel):
    connectionString: str


@router.post("/connect")
async def connect(body: ConnectBody):
    """Try a real, short-lived connection to the given DB and list its
    schemas/tables (folders/files). Never raises on a bad/unreachable DB —
    returns {connected: false, detail} instead."""
    cs = (body.connectionString or "").strip()
    if not cs:
        raise HTTPException(422, "Connection string is required.")

    try:
        url = make_url(cs)
    except Exception as exc:  # noqa: BLE001
        return {"connected": False, "detail": f"Not a valid connection string: {exc}"}

    # Only the psycopg (v3) driver is installed — a plain "postgres://" or
    # "postgresql://" string (psycopg2's default) would fail with a
    # misleading "no module named psycopg2" instead of a real connection
    # result, so normalize it to the driver this app actually ships.
    if url.drivername in ("postgres", "postgresql"):
        url = url.set(drivername="postgresql+psycopg")

    engine = None
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 5}, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        insp = inspect(engine)
        try:
            schema_names = insp.get_schema_names()
        except NotImplementedError:
            schema_names = [None]

        schemas = []
        for schema in schema_names:
            if schema and schema.lower() in _SYSTEM_SCHEMAS:
                continue
            tables = insp.get_table_names(schema=schema)
            if tables:
                schemas.append({"schema": schema, "tables": sorted(tables)})

        return {
            "connected": True,
            "detail": f"Connected to {url.render_as_string(hide_password=True)}",
            "schemas": schemas,
        }
    except SQLAlchemyError as exc:
        logger.info("external-db connect failed: %s", exc)
        return {"connected": False, "detail": str(getattr(exc, "orig", None) or exc)}
    except Exception as exc:  # noqa: BLE001
        logger.info("external-db connect failed: %s", exc)
        return {"connected": False, "detail": str(exc)}
    finally:
        if engine is not None:
            engine.dispose()
