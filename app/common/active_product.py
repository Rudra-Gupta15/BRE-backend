# Which loan product is currently "active" (selected in the BRE Rule Training
# modal). This is read by EVERY data source's rules.py (aa/gst/bbps/upi) to
# know which product's rule set to check a statement against, so it lives
# here rather than inside any one domain package. Persisted so it survives
# /reset and a restart.

import json
import logging
from pathlib import Path

from app.common.products import PRODUCT_IDS

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[2] / ".active_product_cache.json"


class ActiveProductState:
    def __init__(self):
        self.active_product: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            ap = data.get("active_product")
            self.active_product = ap if ap in PRODUCT_IDS else None
        except (OSError, ValueError) as exc:
            logger.warning("Could not read active-product cache (%s) — using default.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({"active_product": self.active_product}), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write active-product cache (%s).", exc)

    def set_active(self, product_id: str | None) -> None:
        self.active_product = product_id if product_id in PRODUCT_IDS else None
        self.persist()

    def reset(self) -> None:
        self.active_product = None
        try:
            _CACHE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


active_product_state = ActiveProductState()
