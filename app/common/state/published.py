# Per-data-source publish status: 'published' | 'unpublished' | 'draft'.
# Survives /reset and a project restart. Only ever changed from the Model Hub
# footer (Published / Unpublished / Draft) via set_status()/set_many() —
# selecting or deselecting a source on the Data Sources page never touches
# this, so a status sticks until someone deliberately changes it there.
# GET /data-sources/selection reads published_ids as the default working set
# on a fresh session, but nothing ever writes selection back into here.

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[3] / ".published_sources_cache.json"
_VALID = {"published", "unpublished", "draft"}
_DEFAULT = "unpublished"

# Starting point ONLY — used when the user has never set any status. As soon as
# they publish / unpublish / draft anything, their choices are what persist.
_SEED_PUBLISHED = ("account_aggregator", "gst_data")


class PublishedState:
    def __init__(self):
        self.statuses: dict[str, str] = {}
        self._load()
        if not self.statuses:  # fresh install — seed the two default feeds
            self.statuses = {sid: "published" for sid in _SEED_PUBLISHED}
            self.persist()

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


published_state = PublishedState()
