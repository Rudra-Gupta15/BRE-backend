# Real-world bank/transaction statement date parsing — shared by every data
# source (aa/gst/bbps/upi) because every real export uses its own mix of
# date shapes. Lives in app.common (not app.aa) since it's pure date logic,
# not bank-statement-specific.

import re
from datetime import date, datetime

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_MONTH_IDX = {m.lower(): i + 1 for i, m in enumerate(_MONTHS)}
_MONTH_IDX.update({
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "sept": 9,
})

_DATE_FORMATS = (
    "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%Y/%m/%d",
    "%d.%m.%Y", "%d.%m.%y", "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%b-%y",
    "%b %d %Y", "%B %d %Y", "%d %b %y",
)


def parse_tx_date(raw) -> date | None:
    """Best-effort parse of the many date shapes real statements use.
    Indian statements are day-first, so DD/MM is preferred over MM/DD."""
    if not raw or not isinstance(raw, str):
        return None
    s = re.sub(r"\s+", " ", raw.strip().replace(",", " ")).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{1,2})[ /\-]([A-Za-z]{3,})[ /\-](\d{2,4})", s)
    if m:
        mon = _MONTH_IDX.get(m.group(2).lower())
        if mon:
            day, yr = int(m.group(1)), int(m.group(3))
            yr += 2000 if yr < 100 else 0
            try:
                return date(yr, mon, min(day, 28))
            except ValueError:
                return None
    return None
