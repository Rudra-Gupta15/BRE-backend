"""Real PASS/FAIL/SKIP evaluators for the 16 BBPS rules listed in
app.common.source_rules — every one keyed off a field app.bbps.analysis's
analyze_bbps() actually computes. Twin of app.aa.product_engine, scoped to
this one data source instead of a whole loan product.

`evaluate()` / `payload()` wrap evaluate_bbps_rules() into the BRE-payload
shape the Model Testing "BRE payload" tab expects — product-level rule
enable/disable toggles, a decision, the JSON payload panel. Twin of
app.gst.rules.evaluate / .payload."""

from __future__ import annotations

from app.common.source_rules import DATA_SOURCE_RULES


def _row(rule_id: str, label: str, status: str, detail: str) -> dict:
    return {"id": rule_id, "label": label, "status": status, "detail": detail}


def evaluate_bbps_rules(result: dict) -> dict:
    """`result` is analyze_bbps()'s output. Returns {results, passed, failed,
    skipped, decision}."""
    if not result or not result.get("available"):
        skip_detail = (result or {}).get("message", "No BBPS activity found.")
        ids = [
            ("bbps_data_presence", "BBPS Data Presence Rule"),
            ("utility_account_count", "Utility Account Count Rule"),
            ("utility_type_diversity", "Utility Type Diversity Rule"),
            ("electricity_bill_payment", "Electricity Bill Payment Rule"),
            ("water_bill_payment", "Water Bill Payment Rule"),
            ("gas_bill_payment", "Gas Bill Payment Rule"),
            ("broadband_bill_payment", "Broadband Bill Payment Rule"),
            ("mobile_dth_bill_payment", "Mobile / DTH Bill Payment Rule"),
            ("recurring_utility_payment", "Recurring Utility Payment Rule"),
            ("utility_bill_punctuality", "Utility Bill Punctuality Rule"),
            ("missed_utility_payment", "Missed Utility Payment Rule"),
            ("on_time_payment_ratio", "On-Time Payment Ratio Rule"),
            ("average_bill_amount_consistency", "Average Bill Amount Consistency Rule"),
            ("statement_span_sufficiency", "Statement Span Sufficiency Rule"),
            ("utility_payment_frequency", "Utility Payment Frequency Rule"),
            ("final_bbps_underwriting_signal", "Final BBPS Underwriting Signal Rule"),
        ]
        results = [_row(rid, label, "SKIP", skip_detail) for rid, label in ids]
        return {"results": results, "passed": 0, "failed": 0, "skipped": len(results), "decision": "NO DATA"}

    by_type = {r["utilityType"]: r for r in result.get("byType", [])}
    accounts = result["utilityAccounts"]
    span = result["spanMonths"]
    on_time = result["onTimePaymentRatio"]
    missed = result["missedPaymentCount"]
    punctuality = result["utilityBillPunctualityIndex"]
    payments = result["paymentsLast12m"]

    results = [_row("bbps_data_presence", "BBPS Data Presence Rule", "PASS",
                     f"{payments} BBPS/utility payment(s) found across {span} month(s).")]

    results.append(_row("utility_account_count", "Utility Account Count Rule",
                         "PASS" if accounts >= 1 else "FAIL",
                         f"{accounts} distinct utility account(s) on record."))

    results.append(_row("utility_type_diversity", "Utility Type Diversity Rule",
                         "PASS" if accounts >= 2 else "FAIL",
                         f"{accounts} utility type(s) — " + ("diversified." if accounts >= 2 else "only one type, thin file.")))

    for kind, rid, label in (
        ("ELECTRICITY", "electricity_bill_payment", "Electricity Bill Payment Rule"),
        ("WATER", "water_bill_payment", "Water Bill Payment Rule"),
        ("GAS", "gas_bill_payment", "Gas Bill Payment Rule"),
        ("BROADBAND", "broadband_bill_payment", "Broadband Bill Payment Rule"),
        ("MOBILE_DTH", "mobile_dth_bill_payment", "Mobile / DTH Bill Payment Rule"),
    ):
        t = by_type.get(kind)
        if t:
            results.append(_row(rid, label, "PASS",
                                 f"{t['paymentCount']} payment(s), avg ₹{t['averageBillAmount']:,.0f}."))
        else:
            results.append(_row(rid, label, "SKIP", "No payment of this utility type found."))

    recurring_types = [t for t in by_type.values() if t["recurring"]]
    results.append(_row("recurring_utility_payment", "Recurring Utility Payment Rule",
                         "PASS" if recurring_types else "FAIL",
                         f"{len(recurring_types)} of {accounts} utility type(s) billed on a recurring monthly cadence."))

    results.append(_row("utility_bill_punctuality", "Utility Bill Punctuality Rule",
                         "PASS" if punctuality >= 70 else "FAIL",
                         f"Punctuality index {punctuality}/100 (≥ 70)." if punctuality >= 70
                         else f"Punctuality index {punctuality}/100 (< 70)."))

    results.append(_row("missed_utility_payment", "Missed Utility Payment Rule",
                         "PASS" if missed == 0 else "FAIL",
                         "No missed utility payments." if missed == 0
                         else f"{missed} month(s) a recurring bill went unpaid."))

    results.append(_row("on_time_payment_ratio", "On-Time Payment Ratio Rule",
                         "PASS" if on_time >= 0.9 else "FAIL",
                         f"On-time ratio {on_time * 100:.0f}% (≥ 90%)." if on_time >= 0.9
                         else f"On-time ratio {on_time * 100:.0f}% (< 90%)."))

    inconsistent = [
        t["utilityType"] for t in by_type.values()
        if t["paymentCount"] >= 3 and t["totalPaid"] > 0
        and (t["averageBillAmount"] * t["paymentCount"]) > 0
        and t["averageBillAmount"] > 0
        and (max(0.0, t["totalPaid"] / t["paymentCount"] - t["averageBillAmount"]) / t["averageBillAmount"]) > 0.5
    ]
    results.append(_row("average_bill_amount_consistency", "Average Bill Amount Consistency Rule",
                         "PASS" if not inconsistent else "FAIL",
                         "Bill amounts are stable month to month." if not inconsistent
                         else f"Irregular bill amounts detected for: {', '.join(inconsistent)}."))

    results.append(_row("statement_span_sufficiency", "Statement Span Sufficiency Rule",
                         "PASS" if span >= 3 else "SKIP",
                         f"{span} month(s) of statement history." + ("" if span >= 3 else " Too short to judge regularity confidently.")))

    results.append(_row("utility_payment_frequency", "Utility Payment Frequency Rule",
                         "PASS" if payments >= accounts else "FAIL",
                         f"{payments} total utility payment(s) across {accounts} account(s)."))

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    if failed == 0:
        decision, final_status = "STABLE UTILITY PAYMENT HISTORY", "PASS"
    elif failed <= 2:
        decision, final_status = "MINOR UTILITY PAYMENT CONCERNS", "PASS"
    else:
        decision, final_status = "UNSTABLE UTILITY PAYMENT HISTORY", "FAIL"
    results.append(_row("final_bbps_underwriting_signal", "Final BBPS Underwriting Signal Rule",
                         final_status, f"{decision} — {passed} passed, {failed} failed, {skipped} skipped."))

    return {
        "results": results,
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "skipped": sum(1 for r in results if r["status"] == "SKIP"),
        "decision": decision,
    }


# ── BRE-payload shape for the Model Testing "BRE payload" tab ──────────────

_SERIOUS_LABELS = {
    "Utility Account Count Rule",
    "Missed Utility Payment Rule",
    "On-Time Payment Ratio Rule",
    "Final BBPS Underwriting Signal Rule",
}


def _label_to_id() -> dict[str, str]:
    return {r["label"]: r["id"] for r in DATA_SOURCE_RULES.get("bbps_utility", [])}


def evaluate(analysis_result: dict, heads: dict, *, product_id: str | None = None) -> dict:
    """Run the 16 bbps_utility rules against one scored statement, respecting
    per-product rule enable/disable toggles (Settings page). Twin of
    app.gst.rules.evaluate."""
    from app.aa.product_source_state import product_source_rule_state
    from app.aa.product_state import bre_product_state

    product_id = product_id or bre_product_state.active_product
    enabled_map = product_source_rule_state.for_ps(product_id, "bbps_utility") if product_id else {}
    lid = _label_to_id()

    raw = evaluate_bbps_rules(analysis_result)
    results: list[dict] = []
    passed = failed = skipped = 0
    serious_flags: list[str] = []

    for r in raw["results"]:
        rid = lid.get(r["label"], r["id"])
        if enabled_map and enabled_map.get(rid) is False:
            continue  # rule switched off for this product
        serious = r["label"] in _SERIOUS_LABELS
        results.append({"id": rid, "label": r["label"], "status": r["status"],
                        "serious": serious, "detail": r["detail"]})
        if r["status"] == "PASS":
            passed += 1
        elif r["status"] == "FAIL":
            failed += 1
            if serious:
                serious_flags.append(r["label"])
        else:
            skipped += 1

    score = float((heads.get("bbps_utility_expense_stability_model") or {}).get("value") or 0)
    risk = str((heads.get("bbps_utility_payment_risk_model") or {}).get("label") or "").upper()
    if serious_flags:
        decision = "REJECTED"
    elif failed > 0 or risk == "HIGH":
        decision = "CONDITIONAL APPROVAL"
    elif risk == "LOW" and score >= 70:
        decision = "APPROVED"
    else:
        decision = "APPROVED WITH NOTES"

    return {
        "available": True,
        "decision": decision,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "enabledCount": passed + failed + skipped,
        "seriousFlags": serious_flags,
        "creditScore": round(score),
        "gateThreshold": 70,
        "applicantProfile": "BBPS-only",
        "productName": None,
        "results": results,
    }


def payload(analysis_result: dict, heads: dict, prediction: dict) -> dict:
    """The JSON blob shown in the BRE Output Payload panel."""
    keys = ["spanMonths", "utilityAccounts", "paymentsLast12m", "onTimePaymentRatio",
            "missedPaymentCount", "averageBillAmount", "utilityBillPunctualityIndex"]
    return {
        "source": "bbps_utility",
        "profile": {k: analysis_result[k] for k in keys if k in analysis_result},
        "byType": analysis_result.get("byType", []),
        "modelOutputs": {
            "stabilityScore": prediction.get("stabilityScore"),
            "riskFlag": prediction.get("riskFlag"),
            "riskProbability": prediction.get("riskProbability"),
            "paymentDisciplineScore": (heads.get("bbps_payment_discipline_score_model") or {}).get("value"),
            "billPaymentBehaviour": (heads.get("bbps_bill_payment_behaviour_model") or {}).get("label"),
            "modelVersion": prediction.get("modelVersion"),
        },
        "topFactors": prediction.get("topFactors", []),
    }
