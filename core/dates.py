# core/dates.py
from __future__ import annotations

from datetime import date
from enum import Enum


class DayCount(str, Enum):
    ACT_365F = "ACT/365F"
    ACT_360 = "ACT/360"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"


def year_fraction(d0: date, d1: date, dc: DayCount = DayCount.ACT_365F) -> float:
    if d1 < d0:
        raise ValueError(f"maturity_date {d1} is before pricing_date {d0}")
    if d1 == d0:
        return 0.0

    days = (d1 - d0).days

    if dc == DayCount.ACT_365F:
        return days / 365.0

    if dc == DayCount.ACT_360:
        return days / 360.0

    if dc == DayCount.ACT_ACT:
        # Simple version: denominator depends on start year (ok for coursework)
        denom = 366.0 if _is_leap(d0.year) else 365.0
        return days / denom

    if dc == DayCount.THIRTY_360:
        # Simple 30/360 (US-like basic)
        d0d = min(d0.day, 30)
        d1d = min(d1.day, 30)
        return (
            (d1.year - d0.year) * 360
            + (d1.month - d0.month) * 30
            + (d1d - d0d)
        ) / 360.0

    raise ValueError(f"Unsupported day count: {dc}")


def _is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
