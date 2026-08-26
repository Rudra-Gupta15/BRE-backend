from fastapi import APIRouter

from app.state.session_state import session_state

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

BASELINE_RECENT_STATEMENTS = [
    {"id": "STMT-2026-0891", "bank": "Axis Bank AA Feed", "date": "2026-08-25T16:45:00.000Z", "txCount": 142, "riskScore": 885, "grade": "LOW", "status": "ANALYZED"},
    {"id": "STMT-2026-0890", "bank": "HDFC Bank Statement", "date": "2026-08-25T16:20:00.000Z", "txCount": 69, "riskScore": 851, "grade": "LOW", "status": "ANALYZED"},
    {"id": "STMT-2026-0889", "bank": "ICICI Corporate Feed", "date": "2026-08-25T15:50:00.000Z", "txCount": 310, "riskScore": 720, "grade": "MEDIUM", "status": "ANALYZED"},
    {"id": "STMT-2026-0888", "bank": "State Bank of India", "date": "2026-08-25T14:15:00.000Z", "txCount": 88, "riskScore": 912, "grade": "LOW", "status": "ANALYZED"},
    {"id": "STMT-2026-0887", "bank": "Kotak Mahindra Feed", "date": "2026-08-25T13:00:00.000Z", "txCount": 18, "riskScore": 490, "grade": "HIGH", "status": "FAILED"},
]


@router.get("/kpis")
async def get_kpis():
    live_runs = len(session_state.inference_history)
    return {
        "kpis": [
            {"id": "analyzed", "label": "ANALYZED", "value": 89 + live_runs, "desc": "Statements analyzed", "badge": "+12.4%"},
            {"id": "processed", "label": "PROCESSED", "value": 16668, "desc": "Transactions processed", "badge": "+8.2%"},
            {"id": "avg_score", "label": "AVG SCORE", "value": 877.9, "sub": "/ 900", "desc": "Average risk score", "badge": "Optimal"},
            {"id": "pending", "label": "PENDING", "value": 571, "desc": "Pending human reviews", "badge": "Queued"},
            {"id": "anomalies", "label": "ANOMALIES", "value": 4540, "desc": "Anomalies flagged", "badge": "Flagged"},
        ]
    }


@router.get("/charts")
async def get_charts():
    return {
        "byStatus": [
            {"status": "ANALYZED", "count": 76},
            {"status": "FAILED", "count": 7},
            {"status": "NORMALIZED", "count": 6},
        ],
        "byRiskGrade": [
            {"name": "LOW Risk", "value": 75},
            {"name": "MEDIUM Risk", "value": 8},
            {"name": "HIGH Risk", "value": 6},
        ],
    }


@router.get("/recent-statements")
async def get_recent_statements():
    live_statements = [
        {"id": h["id"], "bank": h["bank"], "date": h["date"], "txCount": h["txCount"], "riskScore": h["riskScore"], "grade": h["grade"], "status": h["status"]}
        for h in session_state.inference_history
    ]
    merged = (live_statements + BASELINE_RECENT_STATEMENTS)[:8]
    return {"recentStatements": merged}
