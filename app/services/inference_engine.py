import re
import statistics
from datetime import date

from app.data.applicant_templates import ANOMALY_REASONS, MODEL_ANALYTICS_META, MONTHS, TRANSACTION_TEMPLATES
from app.data.model_catalog import MODEL_TEMPLATES
from app.data.underwriting_rules import CREDIT_SCORE_GATE_RULE_ID, CREDIT_SCORE_GATE_THRESHOLD
from app.services.rng import create_rng, pick, rand_int, rand_range
from app.state.settings_state import settings_state


def inr(n: float) -> str:
    return f"₹{round(n):,}"


def score_to_grade(score: int) -> str:
    if score >= 700:
        return "LOW"
    if score >= 550:
        return "MEDIUM"
    return "HIGH"


def decision_for_grade(grade: str) -> str:
    if grade == "LOW":
        return "APPROVED"
    if grade == "MEDIUM":
        return "CONDITIONAL APPROVAL"
    return "REJECTED"


def apply_rule_engine(risk: dict) -> dict:
    """Overlays whichever BRE rules are actually enabled (see Settings page)
    on top of the raw score. Right now the Credit Score Gate is the only
    rule wired to a real computation — everything else in
    underwriting_rules.py is descriptive only and doesn't affect the
    decision. When the gate is on (its default state), it replaces the
    3-tier LOW/MEDIUM/HIGH grading with a hard binary cutoff: score above
    the threshold passes, at-or-below it is rejected — nothing in between."""
    if not settings_state.enabled_rules.get(CREDIT_SCORE_GATE_RULE_ID):
        return risk

    passed = risk["score"] > CREDIT_SCORE_GATE_THRESHOLD
    return {
        **risk,
        "grade": "LOW" if passed else "HIGH",
        "decision": "APPROVED" if passed else "REJECTED",
        "gateRule": {
            "id": CREDIT_SCORE_GATE_RULE_ID,
            "threshold": CREDIT_SCORE_GATE_THRESHOLD,
            "passed": passed,
        },
    }


def compute_credit_score(custom_id: str) -> dict:
    """Deterministic 300-900 credit score + PD derived from the customId seed."""
    rng = create_rng(f"score:{custom_id}")
    score = rand_int(rng, 480, 900)
    pd = max(0.4, round(((900 - score) / 900) * 20, 1))
    grade = score_to_grade(score)
    return {"score": score, "pd": pd, "grade": grade, "decision": decision_for_grade(grade)}


def generate_transactions(custom_id: str, count: int = 12) -> list[dict]:
    """Generates a synthetic but stable transaction list for a customId."""
    rng = create_rng(f"tx:{custom_id}")
    start_year, start_month = 2026, 2
    txs = []
    for i in range(count):
        tpl = pick(rng, TRANSACTION_TEMPLATES)
        day = rand_int(rng, 1, 27)
        month_offset = i // 4
        month = start_month + month_offset
        year = start_year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        tx_date = date(year, month, min(day, 28))
        amount = rand_range(rng, *tpl["amountRange"])
        txs.append({
            "date": tx_date.isoformat(),
            "narration": tpl["narration"],
            "type": tpl["type"],
            "amount": f"{amount:.2f}",
            "category": tpl["category"],
            "merchant": "-",
            "stage": "RULE",
            "confidence": rand_int(rng, 88, 99),
        })
    return sorted(txs, key=lambda t: t["date"])


def generate_anomalies(custom_id: str, transactions: list[dict]) -> list[dict]:
    """Flags a subset of transactions as anomalies with a risk score."""
    rng = create_rng(f"anomaly:{custom_id}")
    flag_count = max(2, round(len(transactions) * 0.3))
    shuffled = sorted(transactions, key=lambda _: rng())[:flag_count]

    result = []
    for tx in shuffled:
        score = rand_range(rng, 40, 95)
        result.append({
            "date": tx["date"],
            "narration": tx["narration"],
            "amount": tx["amount"],
            "score": f"{score:.1f}%",
            "level": "HIGH" if score >= 70 else "MEDIUM",
            "reasons": pick(rng, ANOMALY_REASONS),
        })
    return sorted(result, key=lambda a: float(a["score"].rstrip("%")), reverse=True)


BOUNCE_RE = re.compile(r"bounce|return|dishonour|dishonor|insufficient funds|ecs\s*ret", re.IGNORECASE)
CASH_RE = re.compile(r"\batm\b|cash\s*w(ith)?d(rawal)?", re.IGNORECASE)


def _std_dev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def map_real_transactions(parsed: dict) -> list[dict]:
    """Maps a real parsed statement's transactions into the shape the UI renders."""
    result = []
    for t in parsed["transactions"]:
        narration = t["narration"]
        category = "Bounce" if BOUNCE_RE.search(narration) else "Cash Withdrawal" if CASH_RE.search(narration) else "Other"
        result.append({
            "date": t.get("date") or "",
            "narration": narration,
            "type": t["type"],
            "amount": f"{t['amount']:.2f}",
            "category": category,
            "merchant": "-",
            "stage": "REAL",
            "confidence": 100,
        })
    return result


def compute_real_feature_vector(parsed: dict) -> dict:
    """Derives real underwriting ratios directly from a parsed statement — no randomness."""
    transactions = parsed["transactions"]
    summary = parsed["summary"]
    total_credit = summary.get("totalCredit") or 0
    total_debit = summary.get("totalDebit") or 0
    balances = [t["balance"] for t in transactions if isinstance(t.get("balance"), (int, float))]
    amounts = [t["amount"] for t in transactions]
    credits = [t["amount"] for t in transactions if t["type"] == "CREDIT"]

    nach_bounce_count = sum(1 for t in transactions if BOUNCE_RE.search(t["narration"]))
    cash_withdrawal_total = sum(t["amount"] for t in transactions if t["type"] == "DEBIT" and CASH_RE.search(t["narration"]))

    mean_amount = (sum(amounts) / len(amounts)) if amounts else 0
    volatility = (_std_dev(amounts) / mean_amount) if mean_amount > 0 else 0
    if len(credits) > 1:
        credit_mean = sum(credits) / len(credits)
        credit_stability = 1 - min(1, _std_dev(credits) / credit_mean) if credit_mean else 0.7
    else:
        credit_stability = 0.7

    dscr_ratio = min(9.99, round(total_credit / total_debit, 2)) if total_debit > 0 else (3 if total_credit > 0 else 0)

    return {
        "account_age_days": None,  # not derivable from a single statement
        "avg_monthly_inflow": round(total_credit),
        "nach_bounce_count_90d": nach_bounce_count,
        "dscr_ratio": dscr_ratio,
        "cash_withdrawal_ratio": round(cash_withdrawal_total / total_debit, 4) if total_debit > 0 else 0,
        "transaction_volatility": round(volatility, 4),
        "minimum_balance": round(min(balances)) if balances else round(summary.get("minBalance") or 0),
        "foir_ratio": round((total_debit / total_credit) * 100, 2) if total_credit > 0 else 0,
        "income_stability": round(max(0, min(1, credit_stability)), 3),
    }


def compute_real_credit_score(fv: dict) -> dict:
    """Deterministic (non-random) credit score built from real statement ratios."""
    score = 520
    score += min(150, fv["dscr_ratio"] * 45)
    score += min(100, fv["income_stability"] * 120)
    score -= min(220, fv["nach_bounce_count_90d"] * 45)
    score -= min(120, fv["cash_withdrawal_ratio"] * 300)
    score -= 150 if fv["minimum_balance"] < 0 else 0
    score -= min(80, fv["transaction_volatility"] * 20)

    score = max(300, min(900, round(score)))
    pd = max(0.4, round(((900 - score) / 900) * 20, 1))
    grade = score_to_grade(score)
    return {"score": score, "pd": pd, "grade": grade, "decision": decision_for_grade(grade)}


def _badge_for(model_id: str, chart: list[dict]) -> str:
    last = chart[-1]
    if model_id == "risk_model":
        grade = score_to_grade(last["CreditScore"])
        return "Low Risk (Approved)" if grade == "LOW" else "Medium Risk (Review)" if grade == "MEDIUM" else "High Risk (Declined)"
    if model_id == "cashflow_model":
        return f"Healthy Cashflow ({(last['NetCashflow'] / 1000):.1f}K Net)"
    if model_id == "fraud_model":
        return f"Clean Account (Anomaly Index {last['AnomalyIndex']})"
    if model_id == "money_balance_model":
        return f"Stable Balance (ADB {inr(last['ADBScore'])})"
    return "Evaluated"


def _metric_for(model_id: str, chart: list[dict]) -> str:
    last = chart[-1]
    if model_id == "risk_model":
        return f"Credit Score: {last['CreditScore']} / 900 | PD: {last['PDRiskPct']}%"
    if model_id == "cashflow_model":
        return f"Metric: {inr(last['Inflow'] - last['Outflow'])} Net Inflow Projection"
    if model_id == "fraud_model":
        return f"Metric: {last['AnomalyIndex']}% Anomaly Index (latest month)"
    if model_id == "money_balance_model":
        return f"Metric: ADB {inr(last['ADBScore'])}"
    return ""


def generate_analytics(
    custom_id: str, model_id: str, version: str,
    real_risk_score: dict | None = None, real_feature_vector: dict | None = None,
) -> dict:
    """12-month chart + numeric table for one sub-model, seeded per
    customId+modelId. When `real_risk_score` / `real_feature_vector` are
    supplied (from an uploaded statement), the trajectory's ending month is
    anchored to those real numbers instead of an unrelated synthetic score —
    otherwise this chart and the Credit Score tab could show two different,
    contradictory numbers for the same customer."""
    meta = MODEL_ANALYTICS_META.get(model_id, MODEL_ANALYTICS_META["risk_model"])
    template = next((m for m in MODEL_TEMPLATES if m["id"] == model_id), MODEL_TEMPLATES[0])
    rng = create_rng(f"analytics:{custom_id}:{model_id}")

    chart = []
    table_rows = []

    if model_id == "cashflow_model":
        final_inflow = real_feature_vector["avg_monthly_inflow"] if real_feature_vector else None
        inflow = rand_range(rng, 40000, 55000) if final_inflow is None else final_inflow * 0.75
        for i in range(12):
            if final_inflow is not None:
                inflow += (final_inflow - inflow) / (12 - i)
            else:
                inflow += rand_range(rng, -1500, 3000)
            outflow = inflow * rand_range(rng, 0.6, 0.68)
            net = inflow - outflow
            chart.append({"month": MONTHS[i], "NetCashflow": round(net), "Inflow": round(inflow), "Outflow": round(outflow)})
            runway_months = round(12 + i * 0.4, 1)
            dscr = round(1.9 + i * 0.07, 1)
            table_rows.append({
                "col1": f"Month {i + 1} ({MONTHS[i]})", "col2": inr(inflow), "col3": inr(outflow), "col4": inr(net),
                "col5": f"{runway_months} Mos", "col6": f"{dscr}x", "col7": "Strong" if net > inflow * 0.3 else "Healthy",
            })
    elif model_id == "fraud_model":
        for i in range(12):
            anomaly_index = max(0, round(rand_range(rng, 0, 20) - i))
            velocity = rand_int(rng, 280, 520)
            tx_volume = rand_int(rng, 60, 220)
            chart.append({"month": MONTHS[i], "AnomalyIndex": anomaly_index, "Velocity": velocity})
            table_rows.append({
                "col1": f"Month {i + 1} ({MONTHS[i]})", "col2": f"{tx_volume} Tx", "col3": f"{velocity} / day", "col4": "Clean (0)",
                "col5": f"{anomaly_index}%",
                "col6": "2 Flagged" if anomaly_index > 10 else "1 Flagged" if anomaly_index > 4 else "0 Flagged",
                "col7": "Low Risk" if anomaly_index > 10 else "Clean",
            })
    elif model_id == "money_balance_model":
        final_adb = (real_feature_vector["minimum_balance"] / 0.10) if real_feature_vector else None
        adb = rand_range(rng, 140000, 220000) if final_adb is None else final_adb * 0.8
        for i in range(12):
            if final_adb is not None:
                adb += (final_adb - adb) / (12 - i)
            else:
                adb += rand_range(rng, -8000, 15000)
            min_bal = adb * rand_range(rng, 0.08, 0.12)
            peak_bal = adb * rand_range(rng, 1.6, 1.9)
            volatility = round(2.0 - i * 0.07, 2)
            chart.append({"month": MONTHS[i], "ADBScore": round(adb), "MinBal": round(min_bal)})
            table_rows.append({
                "col1": f"Month {i + 1} ({MONTHS[i]})", "col2": inr(adb), "col3": inr(min_bal), "col4": inr(peak_bal),
                "col5": f"{volatility}", "col6": "0", "col7": "Very Stable" if volatility < 1.4 else "Stable",
            })
    else:  # risk_model (default)
        final = real_risk_score or compute_credit_score(custom_id)
        final_score, final_pd = final["score"], final["pd"]
        score = max(600, final_score - 65)
        pd = min(6, final_pd + 2)
        for i in range(12):
            remaining = 12 - i
            score += (final_score - score) / remaining
            pd += (final_pd - pd) / remaining
            inflow = rand_range(rng, 45000, 76000)
            outflow = inflow * rand_range(rng, 0.62, 0.68)
            adb = inflow * rand_range(rng, 30, 38)
            chart.append({"month": MONTHS[i], "PDRiskPct": round(pd, 1), "CreditScore": round(score)})
            grade = score_to_grade(score)
            table_rows.append({
                "col1": f"Month {i + 1} ({MONTHS[i]})", "col2": f"{pd:.1f}%", "col3": f"{round(score)}", "col4": inr(inflow),
                "col5": inr(outflow), "col6": inr(adb),
                "col7": "Low Risk" if grade == "LOW" else "Medium Risk" if grade == "MEDIUM" else "High Risk",
            })

    return {
        "modelId": model_id,
        "name": template["name"],
        "badge": _badge_for(model_id, chart),
        "metric": _metric_for(model_id, chart),
        "chartTitle": f"{meta['chartTitleSuffix']} (Version {version})",
        "chartType": meta["chartType"],
        "dataKey": meta["dataKey"],
        "yDomain": meta.get("yDomain"),
        "unit": meta["unit"],
        "chartColor": meta["chartColor"],
        "chart": chart,
        "tableColumns": meta["tableColumns"],
        "tableRows": table_rows,
    }


def generate_evaluation(custom_id: str, model_id: str) -> dict:
    """Model performance metrics + 5-fold CV, jittered off each model's base accuracy."""
    template = next((m for m in MODEL_TEMPLATES if m["id"] == model_id), MODEL_TEMPLATES[0])
    rng = create_rng(f"eval:{custom_id}:{model_id}")
    base = template["baseAccuracy"] / 100

    r2 = min(0.999, base + rand_range(rng, -0.01, 0.01))
    precision = min(0.999, base + rand_range(rng, 0, 0.015))
    recall = min(0.999, base - rand_range(rng, 0, 0.02))
    mse = round(1 - base, 4) * rand_range(rng, 0.3, 0.5)
    mae = mse * rand_range(rng, 0.6, 0.75)
    f1 = (2 * precision * recall) / (precision + recall)

    eval_metrics = {
        "r2Score": f"{r2:.3f}", "mse": f"{mse:.4f}", "precision": f"{precision * 100:.1f}%",
        "recall": f"{recall * 100:.1f}%", "mae": f"{mae:.4f}", "f1Score": f"{f1:.3f}",
    }

    cv_folds = []
    for i in range(5):
        jitter = lambda: rand_range(rng, -0.01, 0.01)  # noqa: E731
        fold_r2 = min(0.999, r2 + jitter())
        fold_precision = min(0.999, precision + jitter())
        fold_recall = min(0.999, recall + jitter())
        cv_folds.append({
            "fold": f"Fold {i + 1}", "r2": f"{fold_r2:.3f}", "mse": f"{max(0.0001, mse + jitter() * 0.5):.4f}",
            "precision": f"{fold_precision * 100:.1f}%", "recall": f"{fold_recall * 100:.1f}%",
            "mae": f"{max(0.0001, mae + jitter() * 0.3):.4f}", "status": "PASSED",
        })

    return {"evalMetrics": eval_metrics, "cvFolds": cv_folds}


def generate_bre_payload(
    custom_id: str, bank_name: str | None, model_name: str, version: str, transactions: list[dict],
    real_feature_vector: dict | None = None, real_risk_score: dict | None = None,
) -> dict:
    """Assembles the final BRE decision payload for the BRE Payload tab. Pass
    `real_feature_vector` + `real_risk_score` (from a parsed statement upload)
    to ground this in the user's actual data; otherwise falls back to a
    customId-seeded synthetic profile."""
    risk = real_risk_score or compute_credit_score(custom_id)
    score, pd, grade = risk["score"], risk["pd"], risk["grade"]
    rng = create_rng(f"payload:{custom_id}")

    feature_vector = real_feature_vector or {
        "account_age_days": rand_int(rng, 90, 900),
        "avg_monthly_inflow": rand_int(rng, 35000, 62000),
        "nach_bounce_count_90d": rand_int(rng, 1, 5) if grade == "HIGH" else 0,
        "dscr_ratio": round(1.1 + rand_range(rng, 0, 1.6), 2),
        "cash_withdrawal_ratio": round(rand_range(rng, 0.01, 0.12), 4),
        "transaction_volatility": round(rand_range(rng, 0.8, 2.4), 4),
        "minimum_balance": rand_int(rng, 5000, 30000),
        "foir_ratio": round(rand_range(rng, 18, 55), 2),
        "income_stability": round(rand_range(rng, 0.7, 0.99), 3),
    }

    negative_factors = []
    if feature_vector["cash_withdrawal_ratio"] > 0.08:
        negative_factors.append(f"Cash withdrawal ratio ({feature_vector['cash_withdrawal_ratio']})")
    if feature_vector["transaction_volatility"] > 1.8:
        negative_factors.append(f"Transaction volatility ({feature_vector['transaction_volatility']})")
    if feature_vector["minimum_balance"] < 10000:
        negative_factors.append(f"Minimum balance ({feature_vector['minimum_balance']})")
    if feature_vector["foir_ratio"] > 40:
        negative_factors.append(f"FOIR ({feature_vector['foir_ratio']})")
    if feature_vector["income_stability"] < 0.85:
        negative_factors.append(f"Income stability ({feature_vector['income_stability']})")
    if feature_vector["nach_bounce_count_90d"] > 0:
        negative_factors.append(f"NACH bounces in 90d ({feature_vector['nach_bounce_count_90d']})")

    return {
        "statement_id": custom_id,
        "bank": bank_name or "Not Specified",
        "total_transactions": len(transactions),
        "status": "ANALYZED",
        "data_source": "UPLOADED_STATEMENT" if real_feature_vector else "SIMULATED",
        "model_metadata": {
            "selected_model": model_name,
            "version": version,
            "credit_score": score,
            "risk_grade": grade,
            "probability_of_default": round(pd / 100, 4),
        },
        "feature_vector": feature_vector,
        "negative_factors": negative_factors,
    }
