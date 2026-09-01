"""AA settings state — the scorecard weights + BRE rule on/off map.

Singleton `settings_state` behind the /settings routes; persisted to
.settings_cache.json so it survives a backend --reload.
"""

import json
import logging
from pathlib import Path

from app.aa.scoring_config import ScoringConfig
from app.aa.rule_catalog import UNDERWRITING_RULE_CATEGORIES

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[2] / ".settings_cache.json"


def build_default_rule_map() -> dict[str, bool]:
    rule_map = {}
    for category in UNDERWRITING_RULE_CATEGORIES:
        for rule in category["rules"]:
            rule_map[rule["id"]] = rule.get("defaultEnabled", True) is not False
    return rule_map


def _default_general() -> dict:
    return {"defaultLLM": "gemma", "threshold": "60", "environment": "production"}


class SettingsState:
    def __init__(self):
        self.general = _default_general()
        self.enabled_rules = build_default_rule_map()
        self.scoring = ScoringConfig()
        self.saved_at: str | None = None
        self._load()

    # ── Persistence (survives backend --reload) ────────────────────────────
    def _load(self) -> None:
        try:
            if not _CACHE_FILE.exists():
                return
            data = json.loads(_CACHE_FILE.read_text("utf-8"))
            self.general = {**_default_general(), **data.get("general", {})}
            self.enabled_rules = {**build_default_rule_map(), **data.get("enabled_rules", {})}
            self.scoring.apply(data.get("scoring", {}))
            self.saved_at = data.get("saved_at")
        except (OSError, ValueError) as exc:
            logger.warning("Could not read settings cache (%s) — using defaults.", exc)

    def persist(self) -> None:
        try:
            _CACHE_FILE.write_text(json.dumps({
                "general": self.general,
                "enabled_rules": self.enabled_rules,
                "scoring": self.scoring.to_dict(),
                "saved_at": self.saved_at,
            }), "utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("Could not write settings cache (%s).", exc)

    def reset(self):
        self.general = _default_general()
        self.enabled_rules = build_default_rule_map()
        self.scoring = ScoringConfig()
        self.saved_at = None
        try:
            _CACHE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


settings_state = SettingsState()
