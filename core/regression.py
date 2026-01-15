from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RegressionResult:
    beta: np.ndarray  # (n_features,)
    y_hat: np.ndarray  # fitted values
    residuals: np.ndarray
    r2: float

def ols_fit(X: np.ndarray, y: np.ndarray) -> RegressionResult:
    """Ordinary least squares with a stable solver."""
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of rows")

    # Solve min ||X b - y||_2
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if y.size > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return RegressionResult(beta=beta, y_hat=y_hat, residuals=resid, r2=r2)

class Basis:
    """Interface for regression bases."""
    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

@dataclass(frozen=True)
class PolyBasis(Basis):
    degree: int = 2
    include_intercept: bool = True

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        cols = []
        if self.include_intercept:
            cols.append(np.ones_like(x))
        for d in range(1, self.degree + 1):
            cols.append(x**d)
        return np.column_stack(cols)

@dataclass(frozen=True)
class LaguerreBasis(Basis):
    """First (degree+1) Laguerre polynomials L_0..L_degree (physicists' convention).

    Note: In the LSM literature, one often uses *weighted* Laguerre functions. For a first clean
    implementation, polynomials alone work fine; you can later add weight exp(-x/2) if desired.
    """
    degree: int = 2
    include_intercept: bool = True  # if True, L0 already acts as intercept

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        # L0=1, L1=1-x, L2=1-2x+x^2/2, etc.
        # recurrence: (n+1)L_{n+1} = (2n+1-x)L_n - n L_{n-1}
        L = []
        L0 = np.ones_like(x)
        L.append(L0)
        if self.degree >= 1:
            L1 = 1.0 - x
            L.append(L1)
        for n in range(1, self.degree):
            Ln = L[n]
            Lnm1 = L[n-1]
            Lnp1 = ((2*n + 1 - x) * Ln - n * Lnm1) / (n + 1)
            L.append(Lnp1)
        M = np.column_stack(L[: self.degree + 1])
        # if include_intercept=False, drop first column
        return M if self.include_intercept else M[:, 1:]
