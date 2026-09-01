"""Synthetic GST return generator for demos / testing.

Emits realistic GSTR-1, GSTR-3B, GSTR-2A and GSTR-2B files — 12-24 monthly rows
per business — with a controlled risk spread (strong / average / weak). The
files feed the same pipeline a real GST-portal pull would.

    python -m app.gst.demo --businesses 12 --months 18 --out ../demo_gst

Produces:  gstr1.csv  gstr3b.csv  gstr2a.csv  gstr2b.csv   (+ .json copies)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

_STATES = [("27", "Maharashtra"), ("29", "Karnataka"), ("07", "Delhi"),
           ("33", "Tamil Nadu"), ("24", "Gujarat"), ("06", "Haryana"),
           ("09", "Uttar Pradesh"), ("19", "West Bengal"), ("36", "Telangana")]
_SECTORS = ["Trading", "Manufacturing", "IT Services", "Retail", "Logistics",
            "Pharma", "Food", "Textiles", "Construction", "Agriculture"]
_NAMES = ["Sharma", "Patel", "Krishna", "Gupta", "Mehta", "Reddy", "Singh",
          "Nair", "Joshi", "Banerjee", "Iyer", "Das", "Kulkarni", "Rao", "Bose"]


def _gstin(i: int) -> str:
    sc, _ = _STATES[i % len(_STATES)]
    pan = f"{chr(65 + i % 26)}{chr(66 + i % 25)}{chr(67 + i % 24)}P{1000 + i:04d}{chr(65 + i % 26)}"
    return f"{sc}{pan}{(i % 9) + 1}Z{chr(65 + i % 26)}"


def _period_list(months: int, end: date) -> list[str]:
    out = []
    y, m = end.year, end.month
    for _ in range(months):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return list(reversed(out))


def _due_date(period: str) -> date:
    y, m = map(int, period.split("-"))
    m += 1
    if m == 13:
        m, y = 1, y + 1
    return date(y, m, 20)   # GSTR-3B due ~20th of next month


def _profile(i: int, rng: random.Random) -> dict:
    tier = "strong" if i % 10 < 5 else ("average" if i % 10 < 8 else "weak")
    base_turnover = rng.choice([2.5e5, 4e5, 6e5, 9e5, 1.5e6, 3e6])
    trend = {"strong": rng.uniform(0.004, 0.02),
             "average": rng.uniform(-0.004, 0.008),
             "weak": rng.uniform(-0.03, -0.005)}[tier]
    b2b_pct = rng.uniform(35, 90)
    on_time_prob = {"strong": 0.97, "average": 0.8, "weak": 0.45}[tier]
    mismatch = {"strong": 0.01, "average": 0.03, "weak": 0.09}[tier]
    reg_years = {"strong": rng.uniform(5, 12), "average": rng.uniform(3, 8),
                 "weak": rng.uniform(2, 5)}[tier]
    name = f"{rng.choice(_NAMES)} {rng.choice(_SECTORS)} {'Pvt Ltd' if i % 3 else 'Enterprises'}"
    return {
        "gstin": _gstin(i),
        "legal_name": name,
        "trade_name": name.replace("Pvt Ltd", "Traders"),
        "gst_status": "CANCELLED" if (tier == "weak" and i % 10 == 9) else "ACTIVE",
        "constitution": "Private Limited Company" if i % 3 else "Proprietorship",
        "filing_frequency": "QUARTERLY" if i % 5 == 0 else "MONTHLY",
        "registration_date": (date.today() - timedelta(days=int(reg_years * 365))).isoformat(),
        "_tier": tier, "_base": base_turnover, "_trend": trend,
        "_b2b": b2b_pct, "_ontime": on_time_prob, "_mismatch": mismatch,
    }


def _rows_for_business(p: dict, periods: list[str], rng: random.Random):
    g1, g3b, g2a, g2b = [], [], [], []
    turn = p["_base"]
    common = {k: p[k] for k in ("gstin", "legal_name", "trade_name", "gst_status",
                                "constitution", "filing_frequency", "registration_date")}
    for per in periods:
        turn *= (1 + p["_trend"] + rng.uniform(-0.04, 0.04))
        turn = max(30000, turn)
        b2b = turn * p["_b2b"] / 100
        b2c = turn - b2b
        exp = turn * rng.uniform(0, 0.10)
        sez = turn * rng.uniform(0, 0.03)
        igst = turn * rng.uniform(0.02, 0.05)
        cgst = turn * rng.uniform(0.015, 0.035)
        sgst = cgst
        cess = turn * rng.uniform(0, 0.008)
        itc_avail = turn * rng.uniform(0.03, 0.10)
        itc_claim = itc_avail * rng.uniform(0.7, 1.0)
        itc_rev = itc_claim * rng.uniform(0, 0.15)
        buyers = int(rng.uniform(5, 250))
        top_buyer = turn * rng.uniform(0.06, 0.60)

        due = _due_date(per)
        filed_on_time = rng.random() < p["_ontime"]
        delay = 0 if filed_on_time else rng.randint(1, 20)
        fdate = (due + timedelta(days=delay)).isoformat()

        g1_taxable = turn * (1 + rng.uniform(-p["_mismatch"], p["_mismatch"]))
        g1.append({**common, "period": per, "return_type": "GSTR1",
                   "filing_date": fdate, "due_date": due.isoformat(),
                   "gross_total_value": round(g1_taxable + b2c * 0.05, 2),
                   "taxable_value": round(g1_taxable, 2),
                   "b2b_value": round(b2b, 2), "b2c_value": round(b2c, 2),
                   "export_value": round(exp, 2), "sez_value": round(sez, 2),
                   "nil_rated_value": 0, "cdnr_value": round(turn * 0.01, 2),
                   "total_igst": round(igst, 2), "total_cgst": round(cgst, 2),
                   "total_sgst": round(sgst, 2), "total_cess": round(cess, 2),
                   "b2b_invoice_count": int(buyers * 0.6),
                   "unique_buyer_count": buyers,
                   "top_buyer_value": round(top_buyer, 2)})

        g3b.append({**common, "period": per, "return_type": "GSTR3B",
                    "filing_date": fdate, "due_date": due.isoformat(),
                    "return_status": "FILED" if filed_on_time else "LATE",
                    "outward_taxable_value": round(turn, 2),
                    "outward_igst": round(igst, 2), "outward_cgst": round(cgst, 2),
                    "outward_sgst": round(sgst, 2), "outward_cess": round(cess, 2),
                    "zero_rated_value": round(exp + sez, 2), "exempt_nil_value": 0,
                    "itc_igst": round(itc_claim * 0.4, 2),
                    "itc_cgst": round(itc_claim * 0.3, 2),
                    "itc_sgst": round(itc_claim * 0.3, 2), "itc_cess": 0,
                    "itc_total": round(itc_claim, 2),
                    "itc_reversed": round(itc_rev, 2),
                    "net_itc": round(itc_claim - itc_rev, 2),
                    "tax_payable": round(igst + cgst + sgst + cess, 2),
                    "tax_paid_cash": round((igst + cgst + sgst) * 0.6, 2),
                    "tax_paid_credit": round((igst + cgst + sgst) * 0.4, 2)})

        sup = int(rng.uniform(4, 120))
        g2a.append({**common, "period": per, "return_type": "GSTR2A",
                    "supplier_count": sup,
                    "total_invoice_value": round(itc_avail / 0.15, 2),
                    "total_taxable_value": round(itc_avail / 0.18, 2),
                    "itc_igst": round(itc_avail * 0.4, 2),
                    "itc_cgst": round(itc_avail * 0.3, 2),
                    "itc_sgst": round(itc_avail * 0.3, 2), "itc_cess": 0,
                    "top_supplier_value": round(itc_avail * rng.uniform(0.1, 0.5), 2)})

        g2b.append({**common, "period": per, "return_type": "GSTR2B",
                    "supplier_count": sup,
                    "itc_available_igst": round(itc_avail * 0.4, 2),
                    "itc_available_cgst": round(itc_avail * 0.3, 2),
                    "itc_available_sgst": round(itc_avail * 0.3, 2),
                    "itc_available_cess": 0,
                    "itc_available_total": round(itc_avail, 2),
                    "itc_not_available_total": round(itc_avail * rng.uniform(0, 0.1), 2)})
    return g1, g3b, g2a, g2b


def generate(businesses: int = 12, months: int = 18, seed: int = 42):
    rng = random.Random(seed)
    periods = _period_list(months, date.today().replace(day=1) - timedelta(days=1))
    out = {"gstr1": [], "gstr3b": [], "gstr2a": [], "gstr2b": []}
    for i in range(businesses):
        p = _profile(i, rng)
        g1, g3b, g2a, g2b = _rows_for_business(p, periods, rng)
        out["gstr1"] += g1
        out["gstr3b"] += g3b
        out["gstr2a"] += g2a
        out["gstr2b"] += g2b
    return out


def write(out_dir: str, data: dict):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        if not rows:
            continue
        cols = list({k for r in rows for k in r})
        with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        (d / f"{name}.json").write_text(
            json.dumps({"returns": rows}, indent=2), encoding="utf-8")
    print(f"wrote {sum(len(v) for v in data.values())} return rows to {d.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--businesses", type=int, default=12)
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="demo_gst")
    a = ap.parse_args()
    write(a.out, generate(a.businesses, a.months, a.seed))
