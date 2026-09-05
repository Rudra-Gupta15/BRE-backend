"""AA feature-vector + credit-score engine, plus the deterministic simulation.

Real path: compute_real_feature_vector / compute_real_credit_score turn a parsed
statement into the 11 underwriting features and a score/grade/PD. Simulated path:
generate_transactions / generate_analytics / generate_anomalies / generate_evaluation
produce a customId-seeded demo when no statement was uploaded. Consumed by the
/inference and /bre-products routes.
"""

import math
import re
import statistics
from collections import Counter, OrderedDict
from datetime import date

from app.aa.templates import ANOMALY_REASONS, MODEL_ANALYTICS_META, MONTHS, TRANSACTION_TEMPLATES
from app.aa.catalog import MODEL_TEMPLATES
from app.aa.rule_catalog import CREDIT_SCORE_GATE_RULE_ID
from app.common.dates import parse_tx_date as _parse_tx_date
from app.common.rng import create_rng, pick, rand_int, rand_range
from app.aa.settings_state import settings_state


def inr(n: float) -> str:
    return f"₹{round(n):,}"


def score_to_grade(score: int) -> str:
    sc = settings_state.scoring
    if score >= sc.grade_low_min:
        return "LOW"
    if score >= sc.grade_medium_min:
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

    threshold = settings_state.scoring.gate_threshold
    passed = risk["score"] > threshold
    return {
        **risk,
        "gradeRaw": risk["grade"],  # the underlying 3-tier LOW/MEDIUM/HIGH band
        "grade": "LOW" if passed else "HIGH",  # gate collapses it to pass/fail
        "decision": "APPROVED" if passed else "REJECTED",
        "gateRule": {
            "id": CREDIT_SCORE_GATE_RULE_ID,
            "threshold": threshold,
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
        })
    return sorted(txs, key=lambda t: t["date"])


def generate_anomalies(
    custom_id: str, transactions: list[dict],
    real: bool = False, opening_balance: float | None = None,
) -> list[dict]:
    """Flags anomalous transactions with a risk score.

    `real=True` (uploaded statement): genuine detection via `_real_anomaly_rows`
    — robust amount-outlier test + rule flags, real reasons, no RNG. Pass the
    *raw* parsed transactions (float amounts + balance column), not the
    UI-mapped ones.

    `real=False`: deterministic customId-seeded simulation."""
    if real:
        return _real_anomaly_rows(transactions, opening_balance)

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
        result.append({
            "date": t.get("date") or "",
            "narration": t["narration"],
            "type": t["type"],
            "amount": f"{t['amount']:.2f}",
        })
    return result


def _statement_months(transactions: list[dict]) -> float:
    """How many months the statement spans, from the transaction dates.
    Falls back to a volume-based guess when dates can't be parsed."""
    dates = sorted(d for d in (_parse_tx_date(t.get("date")) for t in transactions) if d)
    if len(dates) >= 2:
        span_days = (dates[-1] - dates[0]).days
        return max(1.0, round((span_days + 1) / 30.44, 2))
    return max(1.0, round(len(transactions) / 30, 2))


def _income_stability(transactions: list[dict]) -> float:
    """0..1 — regularity of month-to-month inflow. Steady salary-like credits
    every month → near 1; erratic / lumpy inflow → near 0. Judged on the
    coefficient of variation of each month's total credits."""
    by_month: dict = {}
    for t in transactions:
        if t["type"] != "CREDIT":
            continue
        d = _parse_tx_date(t.get("date"))
        key = (d.year, d.month) if d else "?"
        by_month[key] = by_month.get(key, 0.0) + t["amount"]
    totals = [v for v in by_month.values() if v > 0]
    if len(totals) < 2:
        return 0.6  # not enough months to judge — neutral
    mean = sum(totals) / len(totals)
    cv = (_std_dev(totals) / mean) if mean > 0 else 1.0
    return max(0.0, 1.0 - cv)


def compute_real_feature_vector(parsed: dict) -> dict:
    """Derives real underwriting ratios directly from a parsed statement — no randomness."""
    transactions = parsed["transactions"]
    summary = parsed["summary"]
    total_credit = summary.get("totalCredit") or sum(t["amount"] for t in transactions if t["type"] == "CREDIT")
    total_debit = summary.get("totalDebit") or sum(t["amount"] for t in transactions if t["type"] == "DEBIT")

    months = _statement_months(transactions)
    opening = summary.get("openingBalance")
    running = [b for b in _running_balances(transactions, opening) if b is not None]

    nach_bounce_count = sum(1 for t in transactions if BOUNCE_RE.search(t["narration"]))
    cash_withdrawal_total = sum(
        t["amount"] for t in transactions if t["type"] == "DEBIT" and CASH_RE.search(t["narration"])
    )

    avg_monthly_inflow = total_credit / months
    avg_monthly_debit = total_debit / months

    # Balance volatility — coefficient of variation of the *running balance*.
    # Bounded and meaningful, unlike a per-transaction amount CV (which mixes a
    # ₹50k salary with ₹200 coffees and always looks huge).
    if len(running) >= 2:
        bal_mean = sum(running) / len(running)
        balance_volatility = (_std_dev(running) / bal_mean) if bal_mean > 0 else 1.0
    else:
        balance_volatility = 0.3
    balance_volatility = round(min(balance_volatility, 5.0), 4)

    dscr_ratio = min(9.99, round(total_credit / total_debit, 2)) if total_debit > 0 else (3.0 if total_credit > 0 else 0.0)
    min_balance = round(min(running)) if running else round(summary.get("minBalance") or 0)

    return {
        "account_age_days": None,  # not derivable from a single statement
        "statement_months": months,
        "avg_monthly_inflow": round(avg_monthly_inflow),
        "avg_monthly_debit": round(avg_monthly_debit),
        "nach_bounce_count_90d": nach_bounce_count,
        "dscr_ratio": dscr_ratio,
        "cash_withdrawal_ratio": round(cash_withdrawal_total / total_debit, 4) if total_debit > 0 else 0.0,
        "balance_volatility": balance_volatility,
        "transaction_volatility": balance_volatility,  # back-compat alias (now = balance volatility)
        "minimum_balance": min_balance,
        "avg_balance": round(sum(running) / len(running)) if running else 0,
        "foir_ratio": round((total_debit / total_credit) * 100, 2) if total_credit > 0 else 0.0,
        "income_stability": round(_income_stability(transactions), 3),
    }


def _scorecard(fv: dict) -> float:
    """The hand-tuned points-based scorecard. Weights come from
    settings_state.scoring (editable via /api/settings/scoring)."""
    c = settings_state.scoring
    score = c.baseline

    # 1. Surplus — does monthly inflow cover outflow, with headroom?
    inflow = fv["avg_monthly_inflow"]
    debit = fv.get("avg_monthly_debit", inflow)
    surplus_ratio = ((inflow - debit) / inflow) if inflow > 0 else -1.0
    score += max(c.surplus_floor, min(c.surplus_cap, surplus_ratio * c.surplus_weight))
    if surplus_ratio < c.overspend_trigger:
        score -= c.overspend_extra_penalty

    # 2. Income regularity (0..1)
    score += (fv["income_stability"] - c.income_stability_pivot) * c.income_stability_weight

    # 3. Cheque / NACH / ECS bounces
    n = fv["nach_bounce_count_90d"]
    if n > 0:
        score -= min(c.bounce_cap, c.bounce_first + (n - 1) * c.bounce_each_after)

    # 4. Minimum balance / overdraft
    mb = fv["minimum_balance"]
    if mb < 0:
        score -= c.overdraft_penalty
    elif mb < 2000:
        score -= c.low_balance_2k_penalty
    elif mb < 10000:
        score -= c.low_balance_10k_penalty
    elif mb > c.high_balance_threshold:
        score += c.high_balance_bonus

    # 5. Erratic balances
    score -= max(0.0, min(1.2, fv["balance_volatility"] - c.volatility_pivot)) * c.volatility_weight

    # 6. Heavy cash-withdrawal dependence
    score -= max(0.0, min(0.4, fv["cash_withdrawal_ratio"] - c.cash_withdrawal_pivot)) * c.cash_withdrawal_weight

    # 7. Strong (business-style) debt-service coverage
    if fv["dscr_ratio"] >= c.dscr_bonus_threshold:
        score += min(c.dscr_bonus_cap, (fv["dscr_ratio"] - c.dscr_bonus_threshold) * c.dscr_bonus_weight)

    return score


def _model_score(parsed: dict) -> float | None:
    """Runs the real statement's features through the trained sklearn risk
    classifier and maps its P(high-risk) to a 300-900 score. None when no
    model has been trained on the Model Hub page."""
    try:
        from app.aa.model import extract_features
        from app.aa.model_state import models_state
    except Exception:  # noqa: BLE001
        return None

    trained = models_state.trained_sklearn_map.get("risk_model")
    if not trained:
        return None
    try:
        import numpy as np

        feats = extract_features(parsed)
        names = trained.get("feat_names") or list(feats.keys())
        x = np.array([[feats[k] for k in names]], dtype=float)
        x = trained["scaler"].transform(x)
        model = trained["model"]
        if hasattr(model, "predict_proba"):
            p_high = float(model.predict_proba(x)[0, 1])
        else:
            p_high = float(model.predict(x)[0])
        return 900.0 - p_high * 600.0  # p_high 0 → 900, 1 → 300
    except Exception:  # noqa: BLE001
        return None


def _registry_model_score(fv: dict) -> dict | None:
    """The population model trained on the uploaded dataset(s) (model_registry)."""
    try:
        from app.aa.model import score_statement_with_registry

        return score_statement_with_registry(fv)
    except Exception:  # noqa: BLE001
        return None


def compute_real_credit_score(fv: dict, parsed: dict | None = None) -> dict:
    """Credit score from the real statement:

      • a transparent points scorecard (weights in settings_state.scoring), and
      • an ML component blended in at `ml_blend_weight` — the population model
        trained on the uploaded dataset (model_registry) if one exists, else
        the per-session model trained on the Model Hub page.

    A clean salaried account lands ~780-850; bounces / overdrafts / overspending
    / erratic inflow pull it down."""
    c = settings_state.scoring
    card = _scorecard(fv)

    ml, ml_source = None, None
    reg = _registry_model_score(fv)
    if reg is not None:
        ml, ml_source = reg["modelScore"], f"dataset-v{reg['version']}"
    elif parsed:
        m = _model_score(parsed)
        if m is not None:
            ml, ml_source = m, "session-model"

    if ml is not None and 0.0 < c.ml_blend_weight <= 1.0:
        final = (1.0 - c.ml_blend_weight) * card + c.ml_blend_weight * ml
        blended = True
    else:
        final = card
        blended = False

    score = max(300, min(900, round(final)))
    pd = max(c.pd_floor, round(((c.pd_scale_denom - score) / c.pd_scale_denom) * c.pd_scale_max, 1))
    grade = score_to_grade(score)
    return {
        "score": score, "pd": pd, "grade": grade, "decision": decision_for_grade(grade),
        "scorecardScore": round(card), "modelScore": round(ml) if ml is not None else None,
        "modelSource": ml_source, "mlBlended": blended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real-data helpers: everything below derives from the actual parsed statement
# (no RNG). Used by generate_analytics / generate_anomalies when an uploaded
# statement is available. _parse_tx_date itself now lives in app.common.dates
# (imported above) since gst/bbps/upi all need the same real-world date parser.
# ─────────────────────────────────────────────────────────────────────────────


def _days_in_month(y: int, m: int) -> int:
    if not (1 <= m <= 12):
        return 30
    if m == 12:
        return 31
    return (date(y, m + 1, 1) - date(y, m, 1)).days


def _bucket_transactions(transactions: list[dict]) -> list[dict]:
    """Groups the real transactions into ordered periods. Uses real calendar
    months when >=60% of rows have a parseable date; otherwise falls back to
    sequential chunks (still real transactions, just period-labelled)."""
    if not transactions:
        return []
    dated = [(_parse_tx_date(t.get("date")), t) for t in transactions]
    parseable = sum(1 for d, _ in dated if d)

    if parseable / len(transactions) >= 0.6:
        groups: "OrderedDict[tuple, list]" = OrderedDict()
        last_key = None
        for d, t in dated:
            if d:
                last_key = (d.year, d.month)
            key = last_key or (0, 0)
            groups.setdefault(key, []).append(t)
        out = []
        for (y, mth), txns in groups.items():
            label = f"{MONTHS[mth - 1]} {y}" if 1 <= mth <= 12 else "Period"
            out.append({"label": label, "txns": txns, "days": _days_in_month(y, mth)})
        return out

    n = len(transactions)
    periods = max(1, min(12, round(n / 10) or 1))
    size = math.ceil(n / periods)
    return [
        {"label": f"Period {i // size + 1}", "txns": transactions[i:i + size], "days": 30}
        for i in range(0, n, size)
    ]


def _running_balances(transactions: list[dict], opening: float | None) -> list[float | None]:
    """Real running balance per row — uses the statement's own balance column
    where present, else reconstructs it from the opening balance + flows."""
    out: list[float | None] = []
    bal = opening
    for t in transactions:
        b = t.get("balance")
        if isinstance(b, (int, float)):
            bal = float(b)
        elif bal is not None:
            bal += t["amount"] if t["type"] == "CREDIT" else -t["amount"]
        out.append(bal)
    return out


def _sub_summary(txns: list[dict]) -> dict:
    tc = round(sum(t["amount"] for t in txns if t["type"] == "CREDIT"), 2)
    td = round(sum(t["amount"] for t in txns if t["type"] == "DEBIT"), 2)
    bals = [t["balance"] for t in txns if isinstance(t.get("balance"), (int, float))]
    return {
        "totalCredit": tc, "totalDebit": td,
        "openingBalance": bals[0] if bals else None,
        "closingBalance": bals[-1] if bals else None,
        "minBalance": min(bals) if bals else None,
        "maxBalance": max(bals) if bals else None,
    }


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def _duplicate_count(txns: list[dict]) -> int:
    seen = Counter((t["narration"].strip().lower(), round(t["amount"], 2)) for t in txns)
    return sum(c - 1 for c in seen.values() if c > 1)


_NARR_STOPWORDS = {
    "UPI", "NEFT", "IMPS", "RTGS", "ACH", "ECS", "TRANSFER", "PAYMENT", "PAYMT",
    "TXN", "REF", "FROM", "TO", "THE", "AND", "FOR", "VIA", "BANK", "LTD", "PVT",
    "INDIA", "PRIVATE", "LIMITED", "ACCOUNT",
}
_HEADER_JUNK_RE = re.compile(
    r"statement of account|account holder|opening balance|closing balance|"
    r"a/c\s*:\s*x|page \d+ of \d+|brought forward|carried forward",
    re.IGNORECASE,
)


def _narr_sig(narration: str) -> str:
    """A short counterparty signature used to tell recurring transactions apart:
    salary from the same employer, monthly rent, a regular UPI merchant, etc."""
    words = [w for w in re.findall(r"[A-Za-z]{3,}", narration.upper()) if w not in _NARR_STOPWORDS]
    return " ".join(words[:3])


def _real_anomaly_rows(transactions: list[dict], opening_balance: float | None = None) -> list[dict]:
    """Conservative transaction-level anomaly detection from the parsed statement.

    Flags only genuine outliers, never routine activity:
      • recurring transactions (salary, rent, a regular payee — same signature
        seen 2-3+ times) are always excluded;
      • an amount is 'unusual' only if it exceeds 4x the account's own
        95th-percentile for that type (floored at ₹50k);
      • cheque/NACH/ECS returns and negative balances are always flagged.

    A clean salaried statement produces zero anomalies — nothing is random."""
    rows_in = [
        (i, t) for i, t in enumerate(transactions)
        if isinstance(t.get("amount"), (int, float)) and t["amount"] > 0
        and not _HEADER_JUNK_RE.search(t.get("narration", ""))
    ]
    if len(rows_in) < 8:
        return []

    running = _running_balances(transactions, opening_balance)

    sig_count = Counter(_narr_sig(t["narration"]) for _, t in rows_in)

    def _recurring(t: dict) -> bool:
        s = _narr_sig(t["narration"])
        if not s:
            return False
        if sig_count[s] >= 3:
            return True
        # same counterparty AND a similar amount, at least twice → expected
        similar = sum(
            1 for _, o in rows_in
            if _narr_sig(o["narration"]) == s and abs(o["amount"] - t["amount"]) <= 0.15 * max(t["amount"], 1)
        )
        return sig_count[s] >= 2 and similar >= 2

    # Per-type upper bound: 4x a robust "normal high" for that type — the 90th
    # percentile after dropping the top 5% (so a lone outlier can't inflate its
    # own baseline) — floored at ₹50k so nothing modest is ever called "large".
    # A txn is unusual only if it clears this AND isn't a recurring counterparty.
    fence: dict[str, float | None] = {}
    for typ in ("DEBIT", "CREDIT"):
        amts = sorted(t["amount"] for _, t in rows_in if t["type"] == typ)
        if len(amts) >= 4:
            cutoff = _percentile(amts, 95)
            pool = [a for a in amts if a <= cutoff] or amts
            fence[typ] = max(50000.0, 4.0 * _percentile(pool, 90))
        else:
            fence[typ] = None

    out = []
    for i, t in rows_in:
        amt, typ = t["amount"], t["type"]
        reasons: list[str] = []
        score = 0.0

        if BOUNCE_RE.search(t["narration"]):
            reasons.append("Cheque / NACH / ECS return or bounce")
            score = max(score, 80.0)

        bal = running[i] if i < len(running) else None
        if bal is not None and bal < 0:
            reasons.append(f"Balance went negative (₹{bal:,.0f})")
            score = max(score, 82.0)

        f = fence.get(typ)
        if f is not None and amt > f and not _recurring(t):
            ratio = amt / f
            reasons.append(f"{typ.title()} far outside this account's usual range (₹{amt:,.0f})")
            score = max(score, min(88.0, 48.0 + 22.0 * math.log10(ratio + 1.0)))

        if not reasons or score < 55:
            continue
        score = min(99.0, score)
        out.append({
            "date": t.get("date") or "",
            "narration": t["narration"],
            "amount": f"{amt:.2f}",
            "score": f"{score:.1f}%",
            "level": "HIGH" if score >= 72 else "MEDIUM",
            "reasons": "; ".join(reasons),
        })

    out.sort(key=lambda r: float(r["score"].rstrip("%")), reverse=True)
    return out[:12]


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


def _real_analytics(
    model_id: str, version: str, meta: dict, template: dict,
    statement: dict, real_risk_score: dict | None,
) -> dict | None:
    """Builds the analytics chart + table entirely from the parsed statement."""
    transactions = statement.get("transactions") or []
    summary = statement.get("summary", {})
    opening = summary.get("openingBalance")
    buckets = _bucket_transactions(transactions)
    if not buckets:
        return None

    chart: list[dict] = []
    table_rows: list[dict] = []
    cumulative: list[dict] = []

    for idx, bucket in enumerate(buckets):
        txns = bucket["txns"]
        cumulative = cumulative + txns
        label = bucket["label"]
        is_last = idx == len(buckets) - 1

        inflow = round(sum(t["amount"] for t in txns if t["type"] == "CREDIT"))
        outflow = round(sum(t["amount"] for t in txns if t["type"] == "DEBIT"))
        bals = [b for b in _running_balances(txns, opening) if b is not None]
        adb = round(sum(bals) / len(bals)) if bals else 0
        min_bal = round(min(bals)) if bals else 0
        peak_bal = round(max(bals)) if bals else 0

        if model_id == "cashflow_model":
            net = inflow - outflow
            dscr = round(inflow / outflow, 2) if outflow else (9.99 if inflow else 0.0)
            runway = round(bals[-1] / outflow, 1) if bals and outflow else 0.0
            chart.append({"month": label, "NetCashflow": net, "Inflow": inflow, "Outflow": outflow})
            table_rows.append({
                "col1": label, "col2": inr(inflow), "col3": inr(outflow), "col4": inr(net),
                "col5": f"{runway} Mos", "col6": f"{dscr}x",
                "col7": "Strong" if net > inflow * 0.3 else "Healthy" if net >= 0 else "Tight",
            })
        elif model_id == "fraud_model":
            flagged = len(_real_anomaly_rows(txns, opening))
            vol = len(txns)
            per_day = round(vol / max(bucket["days"], 1), 2)
            dup = _duplicate_count(txns)
            index = round(100 * flagged / vol) if vol else 0
            chart.append({"month": label, "AnomalyIndex": index, "Velocity": vol})
            table_rows.append({
                "col1": label, "col2": f"{vol} Tx", "col3": f"{per_day} / day",
                "col4": f"{dup} duplicate" if dup else "Clean (0)",
                "col5": f"{index}%", "col6": f"{flagged} Flagged",
                "col7": "Review" if index > 10 else "Watch" if flagged else "Clean",
            })
        elif model_id == "money_balance_model":
            mean_bal = (sum(bals) / len(bals)) if bals else 0
            vol_idx = round(_std_dev(bals) / mean_bal, 3) if mean_bal else 0.0
            nach = sum(1 for t in txns if BOUNCE_RE.search(t["narration"]))
            chart.append({"month": label, "ADBScore": adb, "MinBal": min_bal})
            table_rows.append({
                "col1": label, "col2": inr(adb), "col3": inr(min_bal), "col4": inr(peak_bal),
                "col5": f"{vol_idx}", "col6": f"{nach}",
                "col7": "Very Stable" if vol_idx < 0.15 else "Stable" if vol_idx < 0.35 else "Volatile",
            })
        else:  # risk_model
            if is_last and real_risk_score:
                score, pd = real_risk_score["score"], real_risk_score["pd"]
            else:
                fv = compute_real_feature_vector(
                    {"transactions": list(cumulative), "summary": _sub_summary(cumulative)}
                )
                rs = compute_real_credit_score(fv)
                score, pd = rs["score"], rs["pd"]
            grade = score_to_grade(score)
            chart.append({
                "month": label, "PDRiskPct": round(pd, 1), "CreditScore": round(score),
                "Inflow": inflow, "Outflow": outflow, "NetCashflow": inflow - outflow,
                "ADB": adb, "MinBal": min_bal,
            })
            table_rows.append({
                "col1": label, "col2": f"{pd:.1f}%", "col3": f"{round(score)}",
                "col4": inr(inflow), "col5": inr(outflow), "col6": inr(adb),
                "col7": "Low Risk" if grade == "LOW" else "Medium Risk" if grade == "MEDIUM" else "High Risk",
            })

    n = len(buckets)
    coverage = f"{n}-Month" if n > 1 else "Statement-Period"

    y_domain = meta.get("yDomain")
    if model_id == "risk_model" and chart:
        peak_pd = max(row["PDRiskPct"] for row in chart)
        y_domain = [0, max(10, math.ceil(peak_pd + 1))]

    return {
        "modelId": model_id,
        "name": template["name"],
        "badge": _badge_for(model_id, chart),
        "metric": _metric_for(model_id, chart),
        "chartTitle": f"{meta['chartTitleSuffix'].replace('1-Year', coverage)} (Version {version})",
        "chartType": meta["chartType"],
        "dataKey": meta["dataKey"],
        "yDomain": y_domain,
        "unit": meta["unit"],
        "chartColor": meta["chartColor"],
        "chart": chart,
        "tableColumns": meta["tableColumns"],
        "tableRows": table_rows,
        "periodLabel": f"{n} Month{'s' if n != 1 else ''} from statement",
        "dataSource": "UPLOADED_STATEMENT",
    }


def generate_analytics(
    custom_id: str, model_id: str, version: str,
    real_risk_score: dict | None = None, real_feature_vector: dict | None = None,
    real_statement: dict | None = None,
) -> dict:
    """Month-by-month chart + numeric table for one sub-model.

    When `real_statement` has transactions, every row is computed from the
    actual statement — real per-month inflow/outflow/balance, and a real
    cumulative credit-score trajectory that ends exactly on the Credit Score
    tab's number. Only the months the statement actually covers are shown.

    With no uploaded statement it falls back to the deterministic
    customId+modelId-seeded simulation (the ending month still anchored to
    `real_risk_score` so the chart and Credit Score tab never disagree)."""
    meta = MODEL_ANALYTICS_META.get(model_id, MODEL_ANALYTICS_META["risk_model"])
    template = next((m for m in MODEL_TEMPLATES if m["id"] == model_id), MODEL_TEMPLATES[0])

    if (real_statement or {}).get("transactions"):
        real = _real_analytics(model_id, version, meta, template, real_statement, real_risk_score)
        if real:
            return real

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
    """Real 5-fold cross-validation metrics for the sklearn model trained on
    the Model Hub page. The heavy CV is run once at training time
    (app.aa.model.evaluate_trained) and cached on models_state; this just serves
    it. Falls back to the synthetic estimate when no model has been trained
    yet."""
    from app.aa.model_state import models_state

    cached = models_state.evaluation_cache.get(model_id)
    if cached:
        return cached
    return _synthetic_evaluation(custom_id, model_id)


def _synthetic_evaluation(custom_id: str, model_id: str) -> dict:
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
    score, pd = risk["score"], risk["pd"]
    grade = risk.get("gradeRaw") or risk["grade"]  # 3-tier band; `decision` holds the gate outcome
    rng = create_rng(f"payload:{custom_id}")

    _inflow = rand_int(rng, 35000, 62000)
    feature_vector = real_feature_vector or {
        "account_age_days": rand_int(rng, 90, 900),
        "avg_monthly_inflow": _inflow,
        "avg_monthly_debit": round(_inflow * rand_range(rng, 0.6, 1.05)),
        "nach_bounce_count_90d": rand_int(rng, 1, 5) if grade == "HIGH" else 0,
        "dscr_ratio": round(1.1 + rand_range(rng, 0, 1.6), 2),
        "cash_withdrawal_ratio": round(rand_range(rng, 0.01, 0.12), 4),
        "balance_volatility": round(rand_range(rng, 0.2, 0.9), 4),
        "transaction_volatility": round(rand_range(rng, 0.2, 0.9), 4),
        "minimum_balance": rand_int(rng, 5000, 30000),
        "foir_ratio": round(rand_range(rng, 60, 105), 2),
        "income_stability": round(rand_range(rng, 0.7, 0.99), 3),
    }

    fv = feature_vector
    inflow = fv.get("avg_monthly_inflow", 0)
    debit = fv.get("avg_monthly_debit", inflow)
    bal_vol = fv.get("balance_volatility", fv.get("transaction_volatility", 0))

    negative_factors = []
    if fv["nach_bounce_count_90d"] > 0:
        negative_factors.append(f"Cheque/NACH/ECS bounces in 90d: {fv['nach_bounce_count_90d']}")
    if fv["minimum_balance"] < 0:
        negative_factors.append(f"Account went overdrawn (min balance ₹{fv['minimum_balance']:,})")
    elif fv["minimum_balance"] < 2000:
        negative_factors.append(f"Very low minimum balance (₹{fv['minimum_balance']:,})")
    if inflow > 0 and debit > inflow * 1.05:
        negative_factors.append(f"Spending exceeds income (outflow {round(debit / inflow * 100)}% of inflow)")
    if fv["cash_withdrawal_ratio"] > 0.20:
        negative_factors.append(f"Heavy cash-withdrawal dependence ({fv['cash_withdrawal_ratio'] * 100:.0f}% of debits)")
    if bal_vol > 1.0:
        negative_factors.append(f"Erratic balance (volatility {bal_vol})")
    if fv["income_stability"] < 0.45:
        negative_factors.append(f"Irregular / lumpy monthly inflow (stability {fv['income_stability']})")

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
