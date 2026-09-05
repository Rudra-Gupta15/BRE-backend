# Per-loan-product BRE rule config: which of AA's own bank-statement rules
# are enabled for each product. Persisted so it survives reload.
#
# "Which product is currently active" (chosen in the BRE Rule Training modal)
# used to live on this class too, but every data source's rules.py (gst/bbps/
# upi, not just AA) needs to read it — so that piece moved to
# app.common.active_product and this class now delegates to it, keeping the
# same `.active_product` / `.set_active()` surface for existing AA call sites.

import json
import logging
from pathlib import Path

from app.aa.product_rules import PRODUCT_RULE_IDS, RULES
from app.common.active_product import active_product_state

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
        self._load()

    @property
    def active_product(self) -> str | None:
        return active_product_state.active_product

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
        except (OSError, ValueError) as exc:
            logger.warning("Could not read BRE-product cache (%s) — using defaults.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({"enabled": self.enabled}), "utf-8")
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
        active_product_state.set_active(product_id)

    def reset(self) -> None:
        self.enabled = _default_enabled()
        active_product_state.reset()
        try:
            _CACHE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


bre_product_state = BreProductState()
