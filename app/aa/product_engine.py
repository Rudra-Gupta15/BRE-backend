# Runs a loan product's BRE rules against a real applicant (bank statement +
# feature vector + credit score) and returns PASS/FAIL/SKIP per rule plus an
# overall decision.
#
# Every rule in the product's list is checked when enabled. Rules with a real
# evaluator (RULES) produce PASS/FAIL/SKIP from the data; rules that need an
# external feed we don't ingest (EXTERNAL) return SKIP with the reason.

from datetime import datetime, timezone

from app.aa.product_rules import (
    EXTERNAL,
    PRODUCT_NAMES,
    PRODUCT_RULE_IDS,
    RULES,
)
from app.aa.rules import build_context
from app.aa.settings_state import settings_state


def evaluate_product_rules(
    product_id: str,
    enabled: dict[str, bool],
    fv: dict,
    risk: dict,
    transactions: list[dict],
    opening_balance: float | None,
) -> dict:
    c = build_context(fv, risk, transactions, opening_balance)
    salaried = c["is_salaried"]
    ids = PRODUCT_RULE_IDS.get(product_id, [])

    results: list[dict] = []
    passed = failed = skipped = 0
    serious_fail_ids: set[str] = set()
    gate_enabled = False
    gate_passed = True

    def _is_on(rid: str, computable: bool) -> bool:
        return enabled.get(rid, computable)   # computable → default ON, else default OFF

    for rid in ids:
        if rid in RULES:
            meta = RULES[rid]
            if not _is_on(rid, True) or meta["fn"] is None:
                continue
            if meta.get("business_only") and salaried:
                status, detail = "SKIP", "Not applicable — salaried applicant."
            else:
                try:
                    status, detail = meta["fn"](c, fv, risk)
                except Exception as exc:  # noqa: BLE001
                    status, detail = "SKIP", f"Could not evaluate ({exc})."
            serious = bool(meta.get("serious"))
            if status == "FAIL" and serious:
                serious_fail_ids.add(rid)
            if meta.get("gate"):
                gate_enabled = True
                gate_passed = status == "PASS"
        elif rid in EXTERNAL:
            if not _is_on(rid, False):
                continue
            status, detail = "SKIP", EXTERNAL[rid]["reason"]
            serious = False
        else:
            continue

        label = RULES[rid]["label"] if rid in RULES else EXTERNAL[rid]["label"]
        results.append({"id": rid, "label": label, "status": status, "detail": detail, "serious": serious})
        passed += status == "PASS"
        failed += status == "FAIL"
        skipped += status == "SKIP"

    _BOUNCE = {"bounce_count"}
    distinct_serious = len(serious_fail_ids - _BOUNCE) + (1 if serious_fail_ids & _BOUNCE else 0)

    if gate_enabled and not gate_passed:
        decision = "REJECTED"
    elif distinct_serious >= 2:
        decision = "REJECTED"
    elif distinct_serious == 1:
        decision = "CONDITIONAL APPROVAL"
    elif failed > 0:
        decision = "APPROVED WITH NOTES"
    else:
        decision = "APPROVED"

    outcome = {
        "manual_review": ("FAIL", "Sent to manual review — borderline / conditional.")
        if decision in ("CONDITIONAL APPROVAL", "APPROVED WITH NOTES")
        else ("PASS", "No manual review required."),
        "auto_approval": ("PASS", "Eligible for straight-through approval.")
        if decision == "APPROVED" else ("FAIL", f"Not auto-approvable ({decision})."),
        "auto_rejection": ("FAIL", f"Auto-rejected ({decision}).")
        if decision == "REJECTED" else ("PASS", "Not auto-rejected."),
    }
    for rid, (status, detail) in outcome.items():
        if rid in ids and enabled.get(rid, True):
            results.append({"id": rid, "label": RULES[rid]["label"], "status": status,
                            "detail": detail, "serious": False})
            passed += status == "PASS"
            failed += status == "FAIL"

    total_in_product = len(ids)
    return {
        "product": product_id,
        "productName": PRODUCT_NAMES.get(product_id, product_id),
        "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "creditScore": risk["score"],
        "riskGrade": risk["grade"],
        "gateThreshold": settings_state.scoring.gate_threshold,
        "gateEnabled": gate_enabled,
        "applicantProfile": "SALARIED" if salaried else "BUSINESS / SELF-EMPLOYED",
        "enabledCount": sum(1 for rid in ids if enabled.get(rid, rid in RULES)),
        "totalRules": total_in_product,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "seriousFlags": [r["label"] for r in results if r["status"] == "FAIL" and r["serious"]],
        "decision": decision,
        "results": results,
    }
