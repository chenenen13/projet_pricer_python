from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Market:
    """Market environment for risk-neutral pricing."""
    r: float  # risk-free continuously compounded rate
    q: float = 0.0  # dividend yield (continuous)
