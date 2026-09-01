# Per-loan-product BRE rule config: which rules are enabled for each product,
# and which product is currently "active" (chosen in the BRE Rule Training
# modal — used by the Model Testing BRE tab). Persisted so it survives reload.

import json
import logging
from pathlib import Path

from app.aa.product_rules import PRODUCT_RULE_IDS, RULES

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[2] / ".bre_products_cache.json"


def _default_enabled() -> dict[str, dict[str, bool]]:
    # Computable rules ON by default; rules needing an external feed OFF.
    return {
        pid: {rid: (rid in RULES) for rid in rids}
        for pid, rids in PRODUCT_RULE_IDS.items()
    }


class BreProductState:
    def __init__(self):
        self.enabled: dict[str, dict[str, bool]] = _default_enabled()
        self.active_product: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            base = _default_enabled()
            for pid, rid_map in (data.get("enabled") or {}).items():
                if pid in base:
                    base[pid].update({k: bool(v) for k, v in rid_map.items() if k in base[pid]})
            self.enabled = base
            ap = data.get("active_product")
            self.active_product = ap if ap in PRODUCT_RULE_IDS else None
        except (OSError, ValueError) as exc:
            logger.warning("Could not read BRE-product cache (%s) — using defaults.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({
                "enabled": self.enabled,
                "active_product": self.active_product,
            }), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write BRE-product cache (%s).", exc)

    def for_product(self, product_id: str) -> dict[str, bool]:
        return self.enabled.get(product_id, {})

    def set_rules(self, product_id: str, rule_map: dict[str, bool]) -> None:
        if product_id not in self.enabled:
            return
        for rid, on in rule_map.items():
            if rid in self.enabled[product_id]:
                self.enabled[product_id][rid] = bool(on)
        self.persist()

    def set_all(self, product_id: str, value: bool) -> None:
        if product_id in self.enabled:
            self.enabled[product_id] = {rid: value for rid in self.enabled[product_id]}
            self.persist()

    def reset_product(self, product_id: str) -> None:
        if product_id in PRODUCT_RULE_IDS:
            self.enabled[product_id] = {rid: (rid in RULES) for rid in PRODUCT_RULE_IDS[product_id]}
            self.persist()

    def set_active(self, product_id: str | None) -> None:
        self.active_product = product_id if product_id in PRODUCT_RULE_IDS else None
        self.persist()

    def reset(self) -> None:
        self.enabled = _default_enabled()
        self.active_product = None
        try:
            _CACHE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


bre_product_state = BreProductState()
