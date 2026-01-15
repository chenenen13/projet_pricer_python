# core/brownian.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BrownianMotion:
    """
    Brownian motion simulator on a uniform time grid.

    Generates:
      - Z ~ N(0,1) (shape: n_paths x n_steps)
      - dW = sqrt(dt) * Z
      - W_t = cumulative sum of dW
      - time grid t in [0, T] of length n_steps+1

    Parameters
    ----------
    T : float
        Maturity / horizon in years.
    n_steps : int
        Number of time steps (dt = T/n_steps).
    n_paths : int
        Number of simulated paths.
    seed : int | None
        RNG seed.
    antithetic : bool
        If True, use antithetic variates for Z.
    normal_method : str
        "standard" -> rng.standard_normal
        "ppf"      -> inverse CDF transform (rng.uniform + norm.ppf)
    eps : float
        Small number to avoid 0/1 in uniform draws for ppf method.
    """

    T: float
    n_steps: int
    n_paths: int
    seed: int | None = None
    antithetic: bool = False
    normal_method: str = "standard"  # "standard" or "ppf"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be >= 1")
        if self.n_paths <= 0:
            raise ValueError("n_paths must be >= 1")
        if self.T < 0:
            raise ValueError("T must be >= 0")
        if self.normal_method not in ("standard", "ppf"):
            raise ValueError("normal_method must be 'standard' or 'ppf'")

    @property
    def dt(self) -> float:
        return 0.0 if self.n_steps == 0 else (self.T / self.n_steps)

    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, float(self.T), int(self.n_steps) + 1)

    def draw_Z(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n_paths = int(self.n_paths)
        n_steps = int(self.n_steps)

        if self.antithetic:
            half = (n_paths + 1) // 2
            shape_half = (half, n_steps)

            if self.normal_method == "ppf":
                U_half = rng.uniform(self.eps, 1.0 - self.eps, size=shape_half)
                Z_half = norm.ppf(U_half)
            else:
                Z_half = rng.standard_normal(size=shape_half)

            Z = np.vstack([Z_half, -Z_half])[:n_paths, :]
            return Z

        # no antithetic
        if self.normal_method == "ppf":
            U = rng.uniform(self.eps, 1.0 - self.eps, size=(n_paths, n_steps))
            return norm.ppf(U)

        return rng.standard_normal(size=(n_paths, n_steps))

    def simulate(self, return_Z: bool = False):
        """
        Returns
        -------
        t : (n_steps+1,)
        dW : (n_paths, n_steps)
        W : (n_paths, n_steps)  cumulative (starts at dt, no W0 column)
        Z : (n_paths, n_steps)  optional
        """
        t = self.time_grid()
        if float(self.T) == 0.0:
            # Degenerate case: T=0 => dt=0 => dW=W=0
            dW = np.zeros((int(self.n_paths), int(self.n_steps)), dtype=float)
            W = np.zeros_like(dW)
            Z = np.zeros_like(dW)
            return (t, dW, W, Z) if return_Z else (t, dW, W)

        Z = self.draw_Z()
        dt = float(self.dt)
        dW = np.sqrt(dt) * Z
        W = np.cumsum(dW, axis=1)

        return (t, dW, W, Z) if return_Z else (t, dW, W)
