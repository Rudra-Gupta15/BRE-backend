from app.data.underwriting_rules import UNDERWRITING_RULE_CATEGORIES


def build_default_rule_map() -> dict[str, bool]:
    rule_map = {}
    for category in UNDERWRITING_RULE_CATEGORIES:
        for rule in category["rules"]:
            rule_map[rule["id"]] = rule.get("defaultEnabled", True) is not False
    return rule_map


class SettingsState:
    def __init__(self):
        self.general = {"defaultLLM": "gemma", "threshold": "60", "environment": "production"}
        self.enabled_rules = build_default_rule_map()
        self.saved_at: str | None = None

    def reset(self):
        self.general = {"defaultLLM": "gemma", "threshold": "60", "environment": "production"}
        self.enabled_rules = build_default_rule_map()
        self.saved_at = None


settings_state = SettingsState()
