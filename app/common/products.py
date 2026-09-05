# The 4 loan products every data source's BRE rules can be scoped to
# (Settings > BRE Signals > Products). This is the ONE piece of the
# per-product BRE layer that's genuinely cross-domain — every source
# (aa/gst/bbps/upi) needs to know which product ids exist and their display
# names. The actual per-product RULE CONTENT (which AA bank-statement rules
# apply to "lap_sbl", their pass/fail logic, etc.) stays in app.aa.product_rules
# — that's real AA-specific rule logic, not shared, so it does NOT belong here.

PRODUCT_NAMES: dict[str, str] = {
    "lap_sbl": "LAP / SBL",
    "machine": "Machine Loan",
    "vehicle": "Vehicle Loan",
    "msme": "MSME Loan",
}

PRODUCT_IDS: list[str] = list(PRODUCT_NAMES)
