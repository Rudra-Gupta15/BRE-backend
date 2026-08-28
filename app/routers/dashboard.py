from fastapi import APIRouter

from app.services.persistence import dashboard_snapshot
from app.state.session_state import session_state

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Shown only before any analysis has been run (empty DB / fresh session) so the
# dashboard isn't blank on first load. Once real runs exist they take over.
_SEED_RECENT = [
    {"id": "—", "bank": "No analyses yet", "date": None, "txCount": 0, "riskScore": None, "grade": None, "status": "—"},
]


def _kpi(id_, label, value, desc, badge, sub=None):
    d = {"id": id_, "label": label, "value": value, "desc": desc, "badge": badge}
    if sub:
        d["sub"] = sub
    return d


@router.get("/kpis")
async def get_kpis():
    snap = dashboard_snapshot()
    if snap is None:
        # No database — derive what we can from this session's run history.
        hist = session_state.inference_history
        scores = [h["riskScore"] for h in hist if isinstance(h.get("riskScore"), (int, float))]
        avg = round(sum(scores) / len(scores), 1) if scores else None
        snap = {
            "analyzed": len(hist),
            "processed": sum(h.get("txCount", 0) for h in hist),
            "avgScore": avg,
            "anomalies": 0,
            "pending": sum(1 for h in hist if h.get("grade") == "MEDIUM"),
        }

    return {
        "kpis": [
            _kpi("analyzed", "ANALYZED", snap["analyzed"], "Statements analyzed", "Live"),
            _kpi("processed", "PROCESSED", snap["processed"], "Transactions processed", "Live"),
            _kpi("avg_score", "AVG SCORE",
                 snap["avgScore"] if snap["avgScore"] is not None else "—",
                 "Average credit score", "/ 900", sub="/ 900"),
            _kpi("pending", "PENDING", snap["pending"], "Conditional / review cases", "Queued"),
            _kpi("anomalies", "ANOMALIES", snap["anomalies"], "Anomalies flagged", "Flagged"),
        ]
    }


@router.get("/charts")
async def get_charts():
    snap = dashboard_snapshot()
    if snap is None:
        hist = session_state.inference_history
        grades = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for h in hist:
            if h.get("grade") in grades:
                grades[h["grade"]] += 1
        return {
            "byStatus": [{"status": "ANALYZED", "count": len(hist)}],
            "byRiskGrade": [
                {"name": "LOW Risk", "value": grades["LOW"]},
                {"name": "MEDIUM Risk", "value": grades["MEDIUM"]},
                {"name": "HIGH Risk", "value": grades["HIGH"]},
            ],
        }

    g = snap["byRiskGrade"]
    by_status = [{"status": k, "count": v} for k, v in sorted(snap["byDecision"].items()) if k]
    return {
        "byStatus": by_status or [{"status": "ANALYZED", "count": snap["analyzed"]}],
        "byRiskGrade": [
            {"name": "LOW Risk", "value": g["LOW"]},
            {"name": "MEDIUM Risk", "value": g["MEDIUM"]},
            {"name": "HIGH Risk", "value": g["HIGH"]},
        ],
    }


@router.get("/recent-statements")
async def get_recent_statements():
    snap = dashboard_snapshot()
    if snap is not None and snap["recent"]:
        return {"recentStatements": snap["recent"]}

    live = [
        {"id": h["id"], "bank": h["bank"], "date": h["date"], "txCount": h["txCount"],
         "riskScore": h["riskScore"], "grade": h["grade"], "status": h["status"]}
        for h in session_state.inference_history
    ]
    return {"recentStatements": (live or _SEED_RECENT)[:8]}
