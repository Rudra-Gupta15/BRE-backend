# Per-(loan-product × data-source) BRE rule enable/disable state.
# Settings › BRE Rule Setting › [product] › [data source] → rule checklist.
# The rule catalogue is the data-source catalogue (data_source_rules.py); the
# enabled map is tracked independently for every product. Defaults every rule ON.
# Persisted to a cache file that survives /reset and a restart.

import json
import logging
from pathlib import Path

from app.data.bre_product_rules import PRODUCT_RULE_IDS
from app.data.data_source_rules import DATA_SOURCE_RULES

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[2] / ".product_source_rules_cache.json"


def _default() -> dict[str, dict[str, dict[str, bool]]]:
    return {
        pid: {
            sid: {r["id"]: True for r in rules}
            for sid, rules in DATA_SOURCE_RULES.items()
        }
        for pid in PRODUCT_RULE_IDS
    }


def _default_active() -> dict[str, dict[str, bool]]:
    # A data source is "active" for a product if it has a rule catalogue.
    return {
        pid: {sid: (sid in DATA_SOURCE_RULES) for sid in DATA_SOURCE_RULES}
        for pid in PRODUCT_RULE_IDS
    }


class ProductSourceRuleState:
    def __init__(self):
        self.enabled = _default()
        self.source_active = _default_active()
        self._load()

    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            base = _default()
            for pid, per_source in (data.get("enabled") or {}).items():
                if pid not in base:
                    continue
                for sid, rid_map in (per_source or {}).items():
                    if sid in base[pid]:
                        base[pid][sid].update(
                            {k: bool(v) for k, v in rid_map.items() if k in base[pid][sid]}
                        )
            self.enabled = base

            act = _default_active()
            for pid, per_source in (data.get("source_active") or {}).items():
                if pid in act:
                    act[pid].update(
                        {sid: bool(v) for sid, v in per_source.items() if sid in act[pid]}
                    )
            self.source_active = act
        except (OSError, ValueError) as exc:
            logger.warning("Could not read product-source-rule cache (%s) — using defaults.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({
                "enabled": self.enabled,
                "source_active": self.source_active,
            }), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write product-source-rule cache (%s).", exc)

    # ── data-source on/off for a product ──────────────────────────────────
    def is_source_active(self, product_id: str, source_id: str) -> bool:
        return bool(self.source_active.get(product_id, {}).get(source_id, False))

    def set_source_active(self, product_id: str, source_id: str, value: bool) -> None:
        if product_id in self.source_active and source_id in DATA_SOURCE_RULES:
            self.source_active[product_id][source_id] = bool(value)
            self.persist()

    def usage_by_source(self) -> dict[str, dict[str, bool]]:
        """{ source_id: { product_id: active } } — for the Data Sources page."""
        out: dict[str, dict[str, bool]] = {sid: {} for sid in DATA_SOURCE_RULES}
        for pid, per_source in self.source_active.items():
            for sid, on in per_source.items():
                out.setdefault(sid, {})[pid] = bool(on)
        return out

    def for_ps(self, product_id: str, source_id: str) -> dict[str, bool]:
        return self.enabled.get(product_id, {}).get(source_id, {})

    def counts_for(self, product_id: str, source_id: str) -> tuple[int, int]:
        m = self.for_ps(product_id, source_id)
        return sum(1 for v in m.values() if v), len(m)

    def set_rules(self, product_id: str, source_id: str, rule_map: dict[str, bool]) -> None:
        m = self.enabled.get(product_id, {}).get(source_id)
        if m is None:
            return
        for rid, on in rule_map.items():
            if rid in m:
                m[rid] = bool(on)
        self.persist()

    def set_all(self, product_id: str, source_id: str, value: bool) -> None:
        m = self.enabled.get(product_id, {}).get(source_id)
        if m is not None:
            self.enabled[product_id][source_id] = {rid: value for rid in m}
            self.persist()

    def reset_ps(self, product_id: str, source_id: str) -> None:
        if source_id in DATA_SOURCE_RULES and product_id in self.enabled:
            self.enabled[product_id][source_id] = {
                r["id"]: True for r in DATA_SOURCE_RULES[source_id]
            }
            self.persist()


product_source_rule_state = ProductSourceRuleState()
