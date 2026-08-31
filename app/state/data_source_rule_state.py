# Per-data-source BRE rule enable/disable state (the "Data Source Rule" popup).
# Defaults every rule ON. Persisted to a cache file that survives /reset and a
# project restart.

import json
import logging
from pathlib import Path

from app.data.data_source_rules import DATA_SOURCE_RULES

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[2] / ".data_source_rules_cache.json"


def _default_enabled() -> dict[str, dict[str, bool]]:
    return {
        sid: {r["id"]: True for r in rules}
        for sid, rules in DATA_SOURCE_RULES.items()
    }


class DataSourceRuleState:
    def __init__(self):
        self.enabled: dict[str, dict[str, bool]] = _default_enabled()
        self._load()

    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            base = _default_enabled()
            for sid, rid_map in (data.get("enabled") or {}).items():
                if sid in base:
                    base[sid].update(
                        {k: bool(v) for k, v in rid_map.items() if k in base[sid]}
                    )
            self.enabled = base
        except (OSError, ValueError) as exc:
            logger.warning("Could not read data-source-rule cache (%s) — using defaults.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({"enabled": self.enabled}), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write data-source-rule cache (%s).", exc)

    def for_source(self, source_id: str) -> dict[str, bool]:
        return self.enabled.get(source_id, {})

    def set_rules(self, source_id: str, rule_map: dict[str, bool]) -> None:
        if source_id not in self.enabled:
            return
        for rid, on in rule_map.items():
            if rid in self.enabled[source_id]:
                self.enabled[source_id][rid] = bool(on)
        self.persist()

    def set_all(self, source_id: str, value: bool) -> None:
        if source_id in self.enabled:
            self.enabled[source_id] = {rid: value for rid in self.enabled[source_id]}
            self.persist()

    def reset_source(self, source_id: str) -> None:
        if source_id in DATA_SOURCE_RULES:
            self.enabled[source_id] = {r["id"]: True for r in DATA_SOURCE_RULES[source_id]}
            self.persist()


data_source_rule_state = DataSourceRuleState()
