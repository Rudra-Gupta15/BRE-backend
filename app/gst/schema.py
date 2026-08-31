"""Canonical GST field names and feature groupings — the single source of truth
shared by the parser and the model."""

# The two prediction targets in the dataset.
SCORE_TARGET = "gst_underwriting_score"   # 0-100 regression
FLAG_TARGET = "gst_risk_flag"             # LOW / MEDIUM / HIGH classification
RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]

# Categorical inputs (one-hot encoded before training).
CATEGORICAL = [
    "gst_status", "return_type", "return_status",
    "filing_frequency", "buyer_concentration_level",
]

# Columns never fed to the model (identifiers / free-text dates / the targets).
DROP = {
    "customer_id", "gstin", "legal_name", "trade_name",
    "gst_registration_date", "gst_return_period", "return_filing_date",
    "gstr1_vs_gstr3b_mismatch_pct",  # carried for BRE rules, not a model feature
    SCORE_TARGET, FLAG_TARGET,
}

# Every field the parser recognises (a record may carry any subset).
CANONICAL = {
    "customer_id", "gstin", "legal_name", "trade_name", "gst_status",
    "gst_registration_date", "gst_return_period", "return_type",
    "return_filing_date", "return_status", "filing_frequency",
    "filing_delay_days", "gstr1_sales_value", "gstr3b_taxable_outward_supply",
    "gstr3b_net_tax_liability", "total_taxable_turnover", "b2b_sales_amount",
    "b2c_sales_amount", "b2b_sales_percentage", "b2c_sales_percentage",
    "export_sales_amount", "sez_sales_amount", "reverse_charge_sales_amount",
    "igst_amount", "cgst_amount", "sgst_amount", "cess_amount",
    "itc_available_amount", "itc_claimed_amount", "itc_reversed_amount",
    "net_itc_amount", "unique_buyer_count", "unique_b2b_buyer_count",
    "top_buyer_sales_percentage", "buyer_concentration_level", "monthly_turnover",
    "quarterly_turnover", "turnover_growth_mom", "turnover_growth_qoq",
    "turnover_growth_yoy", "turnover_decline_percentage",
    "consecutive_declining_quarters", "filing_regularity_percentage",
    "missed_return_count", "late_return_count", "business_vintage_years",
    "annualised_gst_turnover", "proposed_loan_amount",
    "gst_turnover_to_loan_ratio", "maximum_loan_by_gst_rule",
    "gst_data_completeness_score", FLAG_TARGET, SCORE_TARGET,
}
