# In-memory session state for this single-tenant demo backend (no DB, no
# multi-user auth). A snapshot is mirrored to a local JSON file so an uploaded
# bank statement + pipeline result survive a `uvicorn --reload` restart —
# without it, every backend edit forces a re-upload.

import json
import logging
from pathlib import Path

from app.common.sources import DATA_SOURCES

logger = logging.getLogger(__name__)

# Backend/.session_cache.json  (gitignored)
_CACHE_FILE = Path(__file__).resolve().parents[3] / ".session_cache.json"
_PERSISTED_KEYS = (
    "selected_ids", "custom_sources", "uploaded_files", "parsed_statements",
    "pipeline", "inference_history",
    # Model Testing keeps its OWN uploads, completely separate from Model Hub's.
    "test_uploads", "test_statements",
)


def _default_pipeline():
    return {
        "status": "idle",  # idle | done
        "currentStage": 0,
        "noisePercent": 60,
        "llmActive": True,
        "processedTable": None,
        "lastRunAt": None,
    }


class SessionState:
    def __init__(self):
        self.selected_ids: list[str] = []
        self.custom_sources: list[dict] = []
        self.uploaded_files: dict[str, dict] = {}
        self.parsed_statements: dict[str, dict] = {}
        # Model Testing page uploads — never touched by Model Hub / the pipeline.
        self.test_uploads: dict[str, list] = {}
        self.test_statements: dict[str, list] = {}
        self.pipeline: dict = _default_pipeline()
        self.inference_history: list[dict] = []
        self._load()

    def all_data_sources(self) -> list[dict]:
        return [*DATA_SOURCES, *self.custom_sources]

    # ── Multi-file access ──────────────────────────────────────────────────
    # A source now holds a *folder* of files: uploaded_files[sid] and
    # parsed_statements[sid] are lists (one entry per file). These helpers
    # tolerate the legacy single-dict shape from an old cache.
    def _upload_store(self, scope: str) -> dict:
        return self.test_uploads if scope == "testing" else self.uploaded_files

    def _statement_store(self, scope: str) -> dict:
        return self.test_statements if scope == "testing" else self.parsed_statements

    def files_for(self, sid: str, scope: str = "hub") -> list[dict]:
        v = self._upload_store(scope).get(sid)
        if not v:
            return []
        return list(v) if isinstance(v, list) else [v]

    def statements_for(self, sid: str, scope: str = "hub") -> list[dict]:
        v = self._statement_store(scope).get(sid)
        if not v:
            return []
        return list(v) if isinstance(v, list) else [v]

    def all_statements(self) -> list[dict]:
        out: list[dict] = []
        for sid in self.parsed_statements:
            out.extend(self.statements_for(sid))
        return out

    def merged_statement_for(self, sid: str, scope: str = "hub") -> dict | None:
        """Concatenate every file's transactions for one source into a single
        statement (used by inference / BRE rules / the pipeline, which reason
        about one applicant per source). `scope="testing"` reads the Model
        Testing page's own uploads instead of Model Hub's."""
        stmts = [s for s in self.statements_for(sid, scope) if s]
        if not stmts:
            return None
        txns: list[dict] = []
        summary = {"totalDebit": 0.0, "totalCredit": 0.0,
                   "openingBalance": None, "closingBalance": None}
        for s in stmts:
            txns.extend(s.get("transactions") or [])
            ss = s.get("summary") or {}
            summary["totalDebit"] += ss.get("totalDebit") or 0
            summary["totalCredit"] += ss.get("totalCredit") or 0
            if summary["openingBalance"] is None:
                summary["openingBalance"] = ss.get("openingBalance")
            if ss.get("closingBalance") is not None:
                summary["closingBalance"] = ss.get("closingBalance")
            for k in ("bankName", "accountHolder", "transactionCount"):
                if not summary.get(k) and ss.get(k):
                    summary[k] = ss.get(k)
        summary["transactionCount"] = len(txns)
        return {"transactions": txns, "summary": summary, "fileCount": len(stmts)}

    # ── Persistence ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            for key in _PERSISTED_KEYS:
                if key in data:
                    setattr(self, key, data[key])
            if self.parsed_statements:
                logger.info(
                    "Restored session cache: %d parsed statement(s), %d upload(s).",
                    len(self.parsed_statements), len(self.uploaded_files),
                )
        except (OSError, ValueError) as exc:  # corrupt / unreadable → start fresh
            logger.warning("Could not read session cache (%s) — starting fresh.", exc)

    def persist(self) -> None:
        """Call after any change to an uploaded statement / pipeline result."""
        try:
            snapshot = {key: getattr(self, key) for key in _PERSISTED_KEYS}
            _CACHE_FILE.write_text(json.dumps(snapshot), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write session cache (%s).", exc)

    def reset(self):
        self.selected_ids = []
        self.custom_sources = []
        self.uploaded_files = {}
        self.parsed_statements = {}
        self.test_uploads = {}
        self.test_statements = {}
        self.pipeline = _default_pipeline()
        self.inference_history = []
        try:
            _CACHE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


session_state = SessionState()
