# Editable overlay on the static per-data-source rule catalogue
# (app/common/source_rules.py) — lets Settings › BRE Signals rename a rule,
# give it a different threshold, or add a brand-new custom rule ("+ Signal").
# Persisted to a cache file that survives /reset and a restart.

import json
import logging
import re
from pathlib import Path

from app.common.source_rules import DATA_SOURCE_RULES

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[3] / ".rule_catalog_cache.json"


def _slug(label: str, salt: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "signal"
    return f"custom_{s}__{salt}"


class RuleCatalogState:
    def __init__(self):
        # {source_id: {rule_id: {"label"?, "threshold"?, "description"?}}}
        self.overrides: dict[str, dict[str, dict]] = {}
        # {source_id: [rule_id, ...]} — custom rules, in the order added
        self.custom_order: dict[str, list[str]] = {}
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            self.overrides = {
                sid: {rid: dict(patch) for rid, patch in (m or {}).items()}
                for sid, m in (data.get("overrides") or {}).items()
                if sid in DATA_SOURCE_RULES
            }
            self.custom_order = {
                sid: [rid for rid in (ids or []) if isinstance(rid, str)]
                for sid, ids in (data.get("customOrder") or {}).items()
                if sid in DATA_SOURCE_RULES
            }
            self._next_id = int(data.get("nextId") or 1)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read rule-catalog cache (%s) — using defaults.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({
                "overrides": self.overrides,
                "customOrder": self.custom_order,
                "nextId": self._next_id,
            }), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write rule-catalog cache (%s).", exc)

    def rules_for(self, source_id: str) -> list[dict]:
        patches = self.overrides.get(source_id, {})
        base = [
            {
                "id": r["id"],
                "label": patches.get(r["id"], {}).get("label", r["label"]),
                "threshold": patches.get(r["id"], {}).get("threshold", ""),
                "description": patches.get(r["id"], {}).get("description", ""),
            }
            for r in DATA_SOURCE_RULES.get(source_id, [])
        ]
        custom = [
            {
                "id": rid,
                "label": patches.get(rid, {}).get("label", ""),
                "threshold": patches.get(rid, {}).get("threshold", ""),
                "description": patches.get(rid, {}).get("description", ""),
                "custom": True,
            }
            for rid in self.custom_order.get(source_id, [])
        ]
        return base + custom

    def _rule_exists(self, source_id: str, rule_id: str) -> bool:
        if any(r["id"] == rule_id for r in DATA_SOURCE_RULES.get(source_id, [])):
            return True
        return rule_id in self.custom_order.get(source_id, [])

    def edit_rule(
        self, source_id: str, rule_id: str, *,
        label: str | None = None, threshold: str | None = None, description: str | None = None,
    ) -> bool:
        if source_id not in DATA_SOURCE_RULES or not self._rule_exists(source_id, rule_id):
            return False
        patch = self.overrides.setdefault(source_id, {}).setdefault(rule_id, {})
        if label is not None:
            patch["label"] = label
        if threshold is not None:
            patch["threshold"] = threshold
        if description is not None:
            patch["description"] = description
        self.persist()
        return True

    def add_rule(self, source_id: str, label: str, threshold: str = "", description: str = "") -> dict:
        rid = _slug(label, self._next_id)
        self._next_id += 1
        self.custom_order.setdefault(source_id, []).append(rid)
        self.overrides.setdefault(source_id, {})[rid] = {
            "label": label, "threshold": threshold, "description": description,
        }
        self.persist()
        return {"id": rid, "label": label, "threshold": threshold, "description": description, "custom": True}


rule_catalog_state = RuleCatalogState()
