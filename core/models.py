# core/models.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .market import Market
from .brownian import BrownianMotion


@dataclass(frozen=True)
class GBMParams:
    s0: float
    sigma: float


class GBMModel:
    def __init__(self, market: Market, params: GBMParams):
        self.market = market
        self.params = params

    def simulate_paths(
        self,
        maturity: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
        antithetic: bool = False,
        return_brownian: bool = False,
        normal_method: str = "standard",  # "standard" or "ppf"
    ):
        """
        Simulate GBM paths under risk-neutral measure:

            dS_t / S_t = (r - q) dt + sigma dW_t

        Returns
        -------
        t : (n_steps+1,)
        S : (n_paths, n_steps+1)
        W : (n_paths, n_steps) optional if return_brownian=True
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")
        if n_paths <= 0:
            raise ValueError("n_paths must be >= 1")
        if maturity < 0:
            raise ValueError("maturity must be >= 0")

        r, q = float(self.market.r), float(self.market.q)
        s0, sigma = float(self.params.s0), float(self.params.sigma)

        # --- Brownian motion block (NEW) ---
        bm = BrownianMotion(
            T=float(maturity),
            n_steps=int(n_steps),
            n_paths=int(n_paths),
            seed=seed,
            antithetic=bool(antithetic),
            normal_method=str(normal_method),
        )

        if return_brownian:
            t, dW, W, Z = bm.simulate(return_Z=True)
        else:
            t, dW, W = bm.simulate(return_Z=False)

        dt = float(bm.dt)

        # If T=0 -> dt=0 -> increments are zero -> S is constant at s0
        if float(maturity) == 0.0:
            S = np.full((int(n_paths), int(n_steps) + 1), s0, dtype=float)
            if return_brownian:
                return t, S, W
            return t, S

        # GBM log increments
        # log(S_{t+dt}/S_t) = (r-q-0.5*sigma^2)dt + sigma dW
        drift = (r - q - 0.5 * sigma * sigma) * dt
        log_increments = drift + sigma * dW  # shape (n_paths, n_steps)

        log_S = np.cumsum(log_increments, axis=1)
        S = np.empty((int(n_paths), int(n_steps) + 1), dtype=float)
        S[:, 0] = s0
        S[:, 1:] = s0 * np.exp(log_S)

        if return_brownian:
            return t, S, W
        return t, S
