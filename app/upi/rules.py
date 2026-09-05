"""Real PASS/FAIL/SKIP evaluators for the 16 UPI rules listed in
app.common.source_rules — every one keyed off a field app.upi.analysis's
analyze_upi() actually computes. Twin of app.bbps.rules, scoped to this one
data source instead of a whole loan product.

`evaluate()` / `payload()` wrap evaluate_upi_rules() into the BRE-payload
shape the Model Testing "BRE payload" tab expects — product-level rule
enable/disable toggles, a decision, the JSON payload panel. Twin of
app.gst.rules.evaluate / .payload / app.bbps.rules.evaluate / .payload."""

from __future__ import annotations

from app.common.source_rules import DATA_SOURCE_RULES


def _row(rule_id: str, label: str, status: str, detail: str) -> dict:
    return {"id": rule_id, "label": label, "status": status, "detail": detail}


def evaluate_upi_rules(result: dict) -> dict:
    """`result` is analyze_upi()'s output. Returns {results, passed, failed,
    skipped, decision}."""
    if not result or not result.get("available"):
        skip_detail = (result or {}).get("message", "No UPI activity found.")
        ids = [
            ("upi_data_presence", "UPI Data Presence Rule"),
            ("transaction_volume", "Transaction Volume Rule"),
            ("payment_success_rate", "Payment Success Rate Rule"),
            ("failed_transaction", "Failed Transaction Rule"),
            ("p2p_counterparty_diversity", "P2P Counterparty Diversity Rule"),
            ("p2m_merchant_diversity", "P2M Merchant Diversity Rule"),
            ("recurring_payee", "Recurring Payee Rule"),
            ("high_risk_mcc_exposure", "High-Risk MCC Exposure Rule"),
            ("weekend_spend_concentration", "Weekend Spend Concentration Rule"),
            ("average_ticket_size", "Average Ticket Size Rule"),
            ("p2p_lending_velocity", "P2P Lending Velocity Rule"),
            ("p2p_borrowing_velocity", "P2P Borrowing Velocity Rule"),
            ("daily_transaction_consistency", "Daily Transaction Consistency Rule"),
            ("statement_span_sufficiency", "Statement Span Sufficiency Rule"),
            ("network_stability", "Network Stability Rule"),
            ("final_upi_underwriting_signal", "Final UPI Underwriting Signal Rule"),
        ]
        results = [_row(rid, label, "SKIP", skip_detail) for rid, label in ids]
        return {"results": results, "passed": 0, "failed": 0, "skipped": len(results), "decision": "NO DATA"}

    total = result["totalTransactions"]
    span = result["spanMonths"]
    success = result["successRatio"]
    failed_ct = result["failedCount"]
    payers = result["uniquePayers"]
    payees = result["uniquePayees"]
    recurring = result["recurringPayeeCount"]
    high_risk_pct = result["highRiskMccSpendPct"]
    weekend_pct = result["weekendSpendPct"]
    avg_ticket = result["avgTicketSize"]
    p2p_debit_vel = result["p2pDebitVelocity"]
    p2p_credit_vel = result["p2pCreditVelocity"]
    daily_avg = result["dailyAvgTransactions"]
    p2p_count = result["p2pCount"]
    p2m_count = result["p2mCount"]

    results = [_row("upi_data_presence", "UPI Data Presence Rule", "PASS",
                     f"{total} UPI transaction(s) found across {span} month(s).")]

    results.append(_row("transaction_volume", "Transaction Volume Rule",
                         "PASS" if total >= 10 else "FAIL",
                         f"{total} transaction(s) on record (≥ 10)." if total >= 10
                         else f"Only {total} transaction(s) on record (< 10) — thin file."))

    results.append(_row("payment_success_rate", "Payment Success Rate Rule",
                         "PASS" if success >= 0.9 else "FAIL",
                         f"{success * 100:.0f}% of payments succeeded (≥ 90%)." if success >= 0.9
                         else f"Only {success * 100:.0f}% of payments succeeded (< 90%)."))

    results.append(_row("failed_transaction", "Failed Transaction Rule",
                         "PASS" if failed_ct <= 3 else "FAIL",
                         f"{failed_ct} failed transaction(s) (≤ 3)." if failed_ct <= 3
                         else f"{failed_ct} failed transaction(s) (> 3) — worth a closer look."))

    if p2p_count > 0:
        results.append(_row("p2p_counterparty_diversity", "P2P Counterparty Diversity Rule",
                             "PASS" if payers >= 1 else "FAIL",
                             f"{payers} distinct person(s) have paid this applicant via UPI."))
    else:
        results.append(_row("p2p_counterparty_diversity", "P2P Counterparty Diversity Rule",
                             "SKIP", "No P2P transactions in this file."))

    if p2m_count > 0:
        results.append(_row("p2m_merchant_diversity", "P2M Merchant Diversity Rule",
                             "PASS" if payees >= 2 else "FAIL",
                             f"{payees} distinct merchant(s)/QR code(s) paid — " +
                             ("diversified." if payees >= 2 else "only one, thin file.")))
    else:
        results.append(_row("p2m_merchant_diversity", "P2M Merchant Diversity Rule",
                             "SKIP", "No P2M (merchant/QR) transactions in this file."))

    results.append(_row("recurring_payee", "Recurring Payee Rule",
                         "PASS" if recurring >= 1 else "FAIL",
                         f"{recurring} payee(s) paid across 2+ calendar months." if recurring >= 1
                         else "No payee paid more than once across different months."))

    if p2m_count > 0:
        results.append(_row("high_risk_mcc_exposure", "High-Risk MCC Exposure Rule",
                             "PASS" if high_risk_pct <= 5 else "FAIL",
                             f"{high_risk_pct:.1f}% of merchant spend is high-risk MCC (≤ 5%)." if high_risk_pct <= 5
                             else f"{high_risk_pct:.1f}% of merchant spend is high-risk MCC (> 5%)."))
        results.append(_row("weekend_spend_concentration", "Weekend Spend Concentration Rule",
                             "PASS" if weekend_pct <= 60 else "FAIL",
                             f"{weekend_pct:.1f}% of merchant spend falls on a weekend (≤ 60%)." if weekend_pct <= 60
                             else f"{weekend_pct:.1f}% of merchant spend falls on a weekend (> 60%)."))
    else:
        results.append(_row("high_risk_mcc_exposure", "High-Risk MCC Exposure Rule",
                             "SKIP", "No P2M (merchant/QR) transactions in this file."))
        results.append(_row("weekend_spend_concentration", "Weekend Spend Concentration Rule",
                             "SKIP", "No P2M (merchant/QR) transactions in this file."))

    results.append(_row("average_ticket_size", "Average Ticket Size Rule", "PASS",
                         f"Average transaction size ₹{avg_ticket:,.0f}."))

    results.append(_row("p2p_lending_velocity", "P2P Lending Velocity Rule",
                         "PASS" if p2p_debit_vel <= 20000 else "FAIL",
                         f"₹{p2p_debit_vel:,.0f}/month sent to individuals (≤ ₹20,000)." if p2p_debit_vel <= 20000
                         else f"₹{p2p_debit_vel:,.0f}/month sent to individuals (> ₹20,000) — notable informal outflow."))

    results.append(_row("p2p_borrowing_velocity", "P2P Borrowing Velocity Rule", "PASS",
                         f"₹{p2p_credit_vel:,.0f}/month received from individuals."))

    results.append(_row("daily_transaction_consistency", "Daily Transaction Consistency Rule",
                         "PASS" if daily_avg >= 0.05 else "FAIL",
                         f"{daily_avg:.2f} transactions/day on average (≥ 0.05)." if daily_avg >= 0.05
                         else f"{daily_avg:.2f} transactions/day on average (< 0.05) — sparse activity."))

    results.append(_row("statement_span_sufficiency", "Statement Span Sufficiency Rule",
                         "PASS" if span >= 3 else "SKIP",
                         f"{span} month(s) of transaction history." + ("" if span >= 3 else " Too short to judge regularity confidently.")))

    network = payees + payers
    results.append(_row("network_stability", "Network Stability Rule",
                         "PASS" if network >= 3 else "FAIL",
                         f"{network} distinct counterpart(y/ies) across P2P + P2M (≥ 3)." if network >= 3
                         else f"Only {network} distinct counterpart(y/ies) (< 3) — thin network."))

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    if failed == 0:
        decision, final_status = "RELIABLE UPI TRANSACTION HISTORY", "PASS"
    elif failed <= 2:
        decision, final_status = "MINOR UPI SIGNAL CONCERNS", "PASS"
    else:
        decision, final_status = "UNRELIABLE UPI TRANSACTION HISTORY", "FAIL"
    results.append(_row("final_upi_underwriting_signal", "Final UPI Underwriting Signal Rule",
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
    "Payment Success Rate Rule",
    "Failed Transaction Rule",
    "High-Risk MCC Exposure Rule",
    "Final UPI Underwriting Signal Rule",
}


def _label_to_id() -> dict[str, str]:
    return {r["label"]: r["id"] for r in DATA_SOURCE_RULES.get("upi_enrichment", [])}


def evaluate(analysis_result: dict, heads: dict, *, product_id: str | None = None) -> dict:
    """Run the 16 upi_enrichment rules against one scored transaction log,
    respecting per-product rule enable/disable toggles (Settings page). Twin
    of app.gst.rules.evaluate / app.bbps.rules.evaluate."""
    from app.common.product_source_state import product_source_rule_state
    from app.common.active_product import active_product_state

    product_id = product_id or active_product_state.active_product
    enabled_map = product_source_rule_state.for_ps(product_id, "upi_enrichment") if product_id else {}
    lid = _label_to_id()

    raw = evaluate_upi_rules(analysis_result)
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

    score = float((heads.get("upi_network_stability_model") or {}).get("value") or 0)
    risk = str((heads.get("upi_transaction_risk_model") or {}).get("label") or "").upper()
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
        "applicantProfile": "UPI-only",
        "productName": None,
        "results": results,
    }


def payload(analysis_result: dict, heads: dict, prediction: dict) -> dict:
    """The JSON blob shown in the BRE Output Payload panel."""
    keys = ["spanMonths", "totalTransactions", "p2pRatio", "p2mRatio", "successRatio",
            "failedRatio", "uniquePayees", "uniquePayers", "recurringPayeeCount",
            "dailyAvgTransactions", "weekendSpendPct", "avgTicketSize",
            "highRiskMccSpendPct", "p2pDebitVelocity", "p2pCreditVelocity"]
    return {
        "source": "upi_enrichment",
        "profile": {k: analysis_result[k] for k in keys if k in analysis_result},
        "byMcc": analysis_result.get("byMcc", []),
        "modelOutputs": {
            "stabilityScore": prediction.get("stabilityScore"),
            "riskFlag": prediction.get("riskFlag"),
            "riskProbability": prediction.get("riskProbability"),
            "paymentReliabilityScore": (heads.get("upi_payment_reliability_score_model") or {}).get("value"),
            "spendBehaviour": (heads.get("upi_spend_behaviour_model") or {}).get("label"),
            "modelVersion": prediction.get("modelVersion"),
        },
        "topFactors": prediction.get("topFactors", []),
    }
