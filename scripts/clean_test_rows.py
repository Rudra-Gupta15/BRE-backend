"""One-off: remove the test artifacts left in bre.* during the persistence
bring-up (raw-filename applicants, duplicate RAMESH rows, orphan applicants).

Keeps: the one full statement (+ its 365 transactions) and the 12 real
historical summaries. Idempotent.

    ./venv/Scripts/python.exe scripts/clean_test_rows.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from sqlalchemy import text

import app.common.db as db
from app.common.db import init_db


def main() -> None:
    init_db()
    if not db.DB_ENABLED:
        raise SystemExit("DB not connected — use the venv Python and check Backend/.env.")

    with db.engine.begin() as c:
        # 1. Name the statement-linked applicant that came in with no holder.
        c.execute(text("""
            UPDATE bre.applicants SET name = 'Ramesh Babu Yadav'
            WHERE name IS NULL
              AND id IN (SELECT applicant_id FROM bre.statements)
        """))

        # 2. Drop inference runs (→ cascades anomalies/analytics) that are test
        #    duplicates: not linked to a statement AND the applicant is either a
        #    raw-filename row or the duplicate "Extreme Ramesh…" backfill row.
        n_runs = c.execute(text("""
            DELETE FROM bre.inference_runs
            WHERE statement_id IS NULL
              AND applicant_id IN (
                  SELECT id FROM bre.applicants
                  WHERE name LIKE 'stmt=_%' ESCAPE '='
                     OR name = 'Extreme Ramesh Babu Yadav'
              )
        """)).rowcount

        # 3. Remove applicants that now have no statement and no run.
        n_appl = c.execute(text("""
            DELETE FROM bre.applicants a
            WHERE NOT EXISTS (SELECT 1 FROM bre.statements s   WHERE s.applicant_id = a.id)
              AND NOT EXISTS (SELECT 1 FROM bre.inference_runs r WHERE r.applicant_id = a.id)
        """)).rowcount

    print(f"Deleted {n_runs} test inference runs and {n_appl} orphan applicants.")

    with db.engine.connect() as c:
        for t in ("applicants", "statements", "transactions", "inference_runs",
                  "anomalies", "analytics_months", "bre_evaluations"):
            print(f"  bre.{t:18}", c.execute(text(f"select count(*) from bre.{t}")).scalar())


if __name__ == "__main__":
    main()
