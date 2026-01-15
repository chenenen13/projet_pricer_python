from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PriceResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    notes: str = ""

    def to_dict(self):
        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

def confidence_interval(x: np.ndarray, alpha: float = 0.01) -> tuple[float, float, float, float]:
    """Returns (mean, std_error, ci_low, ci_high) for a two-sided normal approx CI."""
    x = np.asarray(x).reshape(-1)
    n = x.size
    if n < 2:
        m = float(x.mean()) if n == 1 else float("nan")
        return m, float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n)
    z = 1.959963984540054  # approx N^{-1}(1-alpha/2) for alpha=0.05
    return m, se, m - z * se, m + z * se
