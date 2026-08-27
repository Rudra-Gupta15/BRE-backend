# The 4 sub-models trained per pipeline run (mirrors Page2Pipeline's
# modelsList) plus the version options shown in the deployment table.

MODEL_TEMPLATES = [
    {"id": "risk_model", "name": "Risk Model", "desc": "Evaluates credit risk & default probability.", "baseAccuracy": 94.8, "color": "emerald"},
    {"id": "cashflow_model", "name": "Cashflow Model", "desc": "Projects 12-month forward revenue & cash runway.", "baseAccuracy": 92.4, "color": "blue"},
    {"id": "fraud_model", "name": "Fraud Model", "desc": "Detects anomalous transactions & duplicate pledges.", "baseAccuracy": 98.9, "color": "rose"},
    {"id": "money_balance_model", "name": "Money Balance Model", "desc": "Evaluates daily balance stability & cash volatility.", "baseAccuracy": 91.6, "color": "amber"},
]

VERSION_OPTIONS = [
    {"value": "v3.4", "label": "v3.4 (Current)"},
    {"value": "v3.3", "label": "v3.3 (Old)"},
    {"value": "v3.2", "label": "v3.2 (Old)"},
    {"value": "v3.1", "label": "v3.1 (Old)"},
    {"value": "v3.0", "label": "v3.0 (Old)"},
]

ML_ALGORITHMS = [
    {"value": "gradient_boosting", "label": "Gradient Boosting", "accuracyDelta": 0},
    {"value": "random_forest", "label": "Random Forest", "accuracyDelta": -0.6},
    {"value": "logistic_regression", "label": "Logistic Regression", "accuracyDelta": -3.2},
    {"value": "svm", "label": "SVM (Support Vector Machine)", "accuracyDelta": -1.8},
]
