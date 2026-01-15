# core/options.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import date
from typing import Optional

import numpy as np

from core.dates import DayCount, year_fraction


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


def payoff(option_type: OptionType, S: np.ndarray, K: float) -> np.ndarray:
    if option_type == OptionType.CALL:
        return np.maximum(S - K, 0.0)
    if option_type == OptionType.PUT:
        return np.maximum(K - S, 0.0)
    raise ValueError(f"Unknown option type: {option_type}")


@dataclass(frozen=True)
class _DatedOptionMixin:
    """
    Allow either:
      - maturity (float, years)  [legacy mode]
      - or (pricing_date, maturity_date, day_count)  [date mode]
    """

    maturity: Optional[float] = None  # in years (float)
    pricing_date: Optional[date] = None
    maturity_date: Optional[date] = None
    day_count: DayCount = DayCount.ACT_365F

    def maturity_years(self) -> float:
        if self.maturity is not None:
            if self.maturity < 0:
                raise ValueError("maturity must be >= 0")
            return float(self.maturity)

        if self.pricing_date is None or self.maturity_date is None:
            raise ValueError(
                "Provide either `maturity` (float years) OR "
                "`pricing_date` + `maturity_date`."
            )

        return float(year_fraction(self.pricing_date, self.maturity_date, self.day_count))

    @property
    def T(self) -> float:
        return self.maturity_years()


@dataclass(frozen=True)
class EuropeanOption(_DatedOptionMixin):
    option_type: OptionType = OptionType.CALL
    strike: float = 0.0

    def payoff(self, ST: np.ndarray) -> np.ndarray:
        return payoff(self.option_type, ST, float(self.strike))


@dataclass(frozen=True)
class AmericanOption(_DatedOptionMixin):
    option_type: OptionType = OptionType.PUT
    strike: float = 0.0

    def payoff(self, S: np.ndarray) -> np.ndarray:
        return payoff(self.option_type, S, float(self.strike))
