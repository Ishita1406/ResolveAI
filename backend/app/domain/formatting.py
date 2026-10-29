"""Speech-friendly formatting for amounts and dates."""

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def format_inr(paise: int) -> str:
    """₹ with Indian digit grouping; paise shown only when non-zero. 8499900 -> '₹84,999'."""
    rupees, rem = divmod(abs(paise), 100)
    digits = str(rupees)
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    grouped = ",".join([*groups, tail])
    sign = "-" if paise < 0 else ""
    return f"{sign}₹{grouped}" + (f".{rem:02d}" if rem else "")


def format_date(dt: datetime) -> str:
    """Calendar date in IST, e.g. '28 October 2026'."""
    return dt.astimezone(IST).strftime("%-d %B %Y")
