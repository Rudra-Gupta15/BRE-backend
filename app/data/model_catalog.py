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

PIPELINE_STAGES = [
    {"id": 1, "name": "1. Data Gathering", "desc": "Fetch & aggregate feeds from selected sources", "duration": 1500},
    {"id": 2, "name": "2. Preprocess Data", "desc": "Clean missing values & filter noise", "duration": 1800},
    {"id": 3, "name": "3. Normalize Data", "desc": "MinMax scaling & Z-score standardization", "duration": 2200},
    {"id": 4, "name": "4. Feature Engineering", "desc": "Generate variables & temporal ratios", "duration": 1600},
    {"id": 5, "name": "5. Data Selection", "desc": "Select high-variance predictor features", "duration": 2500},
]

MOCK_TRAINING_LOGS = [
    "[SYS] Initializing Distributed ETL Cluster...",
    "[INGEST] Pulled active data vectors across historical loan records.",
    "[STAGE 1] Cleaning & Imputing: Resolved missing values using KNN Imputer.",
    "[STAGE 1] Outlier Detection: Winsorized extreme transaction spikes at 99th percentile.",
    "[STAGE 2] Normalization: Computed Z-scores for ADB, Turnover, and Utility Punctuality.",
    "[STAGE 2] Applying Box-Cox Transformation to UPI transaction velocity features.",
    "[STAGE 3] Feature Engineering: Created candidate features from multi-source join.",
    "[STAGE 3] Formed interaction ratio: (GST_Turnover / AA_Bank_Inflow).",
    "[STAGE 4] Feature Selection: Applied Boruta algorithm + SHAP Importance ranking.",
    "[STAGE 4] Reduced feature dimension to high-signal predictor variables.",
    "[STAGE 5] Model Training: Hyperparameter tuning across all sub-models.",
    "[SYS] Model Training Complete!",
]
