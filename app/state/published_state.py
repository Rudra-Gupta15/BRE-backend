# Per-data-source publish status: 'published' | 'unpublished' | 'draft'.
# Survives /reset and a project restart. 'published' sources are the ones fed to
# the Model Hub pipeline (session_state.selected_ids is derived from them).

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[2] / ".published_sources_cache.json"
_VALID = {"published", "unpublished", "draft"}
_DEFAULT = "unpublished"


class PublishedState:
    def __init__(self):
        self.statuses: dict[str, str] = {}
        self._load()

    @property
    def published_ids(self) -> list[str]:
        return [sid for sid, st in self.statuses.items() if st == "published"]

    def status_of(self, sid: str) -> str:
        return self.statuses.get(sid, _DEFAULT)

    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            statuses = data.get("statuses")
            if isinstance(statuses, dict):
                self.statuses = {
                    str(k): v for k, v in statuses.items() if v in _VALID
                }
            else:  # migrate legacy {"published_ids": [...]}
                for sid in data.get("published_ids", []) or []:
                    self.statuses[str(sid)] = "published"
        except (OSError, ValueError) as exc:
            logger.warning("Could not read source-status cache (%s) — starting empty.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({"statuses": self.statuses}), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write source-status cache (%s).", exc)

    def set_status(self, sid: str, status: str) -> None:
        if status in _VALID:
            self.statuses[sid] = status
            self.persist()

    def set_many(self, mapping: dict[str, str]) -> None:
        for sid, status in mapping.items():
            if status in _VALID:
                self.statuses[str(sid)] = status
        self.persist()

    def set_published_ids(self, ids: list[str]) -> None:
        """Full-replace the published set (used by PUT /selection): given ids
        become 'published'; any source that *was* published and isn't in the
        list falls back to 'unpublished'. 'draft' sources are left alone."""
        keep = set(ids)
        for sid in list(self.statuses):
            if self.statuses[sid] == "published" and sid not in keep:
                self.statuses[sid] = "unpublished"
        for sid in ids:
            self.statuses[str(sid)] = "published"
        self.persist()


published_state = PublishedState()
