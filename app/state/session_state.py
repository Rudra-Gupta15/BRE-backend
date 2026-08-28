# In-memory session state for this single-tenant demo backend (no DB, no
# multi-user auth). A snapshot is mirrored to a local JSON file so an uploaded
# bank statement + pipeline result survive a `uvicorn --reload` restart —
# without it, every backend edit forces a re-upload.

import json
import logging
from pathlib import Path

from app.data.data_sources import DATA_SOURCES

logger = logging.getLogger(__name__)

# Backend/.session_cache.json  (gitignored)
_CACHE_FILE = Path(__file__).resolve().parents[2] / ".session_cache.json"
_PERSISTED_KEYS = (
    "selected_ids", "custom_sources", "uploaded_files", "parsed_statements",
    "pipeline", "inference_history",
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
        self.pipeline: dict = _default_pipeline()
        self.inference_history: list[dict] = []
        self._load()

    def all_data_sources(self) -> list[dict]:
        return [*DATA_SOURCES, *self.custom_sources]

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
        self.pipeline = _default_pipeline()
        self.inference_history = []
        try:
            _CACHE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


session_state = SessionState()
