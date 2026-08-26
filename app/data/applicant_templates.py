# Building blocks the inference engine (services/inference_engine.py) draws
# from to synthesize a deterministic applicant profile for any customId.

TRANSACTION_TEMPLATES = [
    {"narration": "NEFT STARK RETAIL SALARY", "type": "CREDIT", "category": "Salary", "amountRange": (30000, 60000)},
    {"narration": "RENT PAYMENT TO LANDLORD", "type": "DEBIT", "category": "Rent", "amountRange": (5000, 12000)},
    {"narration": "HOME LOAN MONTHLY INSTALLMENT", "type": "DEBIT", "category": "Loan Repayment", "amountRange": (8000, 16000)},
    {"narration": "LOCAL RESTAURANT", "type": "DEBIT", "category": "Food", "amountRange": (500, 3000)},
    {"narration": "MOBILE BILL PAYMENT", "type": "DEBIT", "category": "Utilities", "amountRange": (800, 2500)},
    {"narration": "BROADBAND BILL", "type": "DEBIT", "category": "Utilities", "amountRange": (600, 1800)},
    {"narration": "AMAZON SHOPPING", "type": "DEBIT", "category": "Shopping", "amountRange": (500, 6000)},
    {"narration": "ELECTRICITY BILL PAYMENT", "type": "DEBIT", "category": "Utilities", "amountRange": (700, 2200)},
    {"narration": "SWIGGY FOOD DELIVERY", "type": "DEBIT", "category": "Food", "amountRange": (200, 1200)},
    {"narration": "SPOTIFY SUBSCRIPTION", "type": "DEBIT", "category": "Entertainment", "amountRange": (150, 400)},
    {"narration": "GST PAYMENT NET BANKING", "type": "DEBIT", "category": "Tax", "amountRange": (4000, 15000)},
    {"narration": "ATM CASH WITHDRAWAL", "type": "DEBIT", "category": "Cash Withdrawal", "amountRange": (1000, 8000)},
    {"narration": "UPI MERCHANT SETTLEMENT", "type": "CREDIT", "category": "Revenue", "amountRange": (10000, 45000)},
    {"narration": "IRCTC TICKET", "type": "DEBIT", "category": "Travel", "amountRange": (400, 2500)},
    {"narration": "FLIPKART ORDER", "type": "DEBIT", "category": "Shopping", "amountRange": (800, 13000)},
]

ANOMALY_REASONS = ["UNUSUAL_AMOUNT", "STATISTICAL_OUTLIER", "VELOCITY_SPIKE", "NEW_COUNTERPARTY"]

MODEL_ANALYTICS_META = {
    "risk_model": {
        "unit": "%",
        "dataKey": "PDRiskPct",
        "secondaryKey": "CreditScore",
        "chartType": "area",
        "yDomain": [0, 10],
        "chartColor": "#059669",
        "chartTitleSuffix": "1-Year Probability of Default (PD %) & Risk Trajectory",
        "tableColumns": ["Month", "Default Risk (PD %)", "Credit Score (300-900)", "Inflow (₹)", "Outflow (₹)", "Avg Daily Balance (ADB)", "Risk Grade"],
    },
    "cashflow_model": {
        "unit": "",
        "dataKey": "NetCashflow",
        "secondaryKey": None,
        "chartType": "bar",
        "chartColor": "#059669",
        "chartTitleSuffix": "1-Year Forward Monthly Cashflow & Revenue Projection",
        "tableColumns": ["Month", "Monthly Inflow (₹)", "Monthly Outflow (₹)", "Net Cashflow (₹)", "Cash Runway", "DSCR Coverage", "Health"],
    },
    "fraud_model": {
        "unit": "",
        "dataKey": "AnomalyIndex",
        "secondaryKey": None,
        "chartType": "bar",
        "chartColor": "#2563eb",
        "chartTitleSuffix": "1-Year Transaction Anomaly Index & Velocity",
        "tableColumns": ["Month", "Tx Volume", "UPI Velocity / Day", "CERSAI Check", "Anomaly Score", "Flagged Tx", "Verdict"],
    },
    "money_balance_model": {
        "unit": "",
        "dataKey": "ADBScore",
        "secondaryKey": None,
        "chartType": "area",
        "chartColor": "#7c3aed",
        "chartTitleSuffix": "1-Year Average Daily Balance (ADB) Stability Trajectory",
        "tableColumns": ["Month", "ADB (₹)", "Min Balance (₹)", "Peak Balance (₹)", "Volatility Index", "NACH Bounces", "Stability Rating"],
    },
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
