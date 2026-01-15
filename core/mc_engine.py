# core/mc_engine.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .models import GBMModel
from .options import EuropeanOption, AmericanOption
from .results import PriceResult, confidence_interval
from .regression import Basis, PolyBasis, ols_fit


def _apply_control_variate(
    disc_payoffs: np.ndarray,
    disc_ST: np.ndarray,
    disc_ST_expectation: float,
) -> np.ndarray:
    """
    Control variate using discounted terminal price:
      X = discounted payoff
      C = discounted ST with known expectation E[C] = S0 * exp(-qT)

    Adjusted sample:
      X_cv = X - beta * (C - E[C])
      beta = Cov(X,C)/Var(C)
    """
    X = np.asarray(disc_payoffs).reshape(-1)
    C = np.asarray(disc_ST).reshape(-1)

    varC = float(np.var(C, ddof=1))
    if not np.isfinite(varC) or varC <= 0:
        return X

    covXC = float(np.cov(X, C, ddof=1)[0, 1])
    beta = covXC / varC

    X_cv = X - beta * (C - disc_ST_expectation)
    return X_cv


@dataclass(frozen=True)
class EuropeanMCPricer:
    model: GBMModel

    def price_scalar(
        self,
        option: EuropeanOption,
        n_paths: int,
        seed: int | None = None,
        antithetic: bool = False,
        control_variate: bool = False,
    ) -> PriceResult:
        """
        Scalar (loop) MC for one-step European pricing.
        Optional control variate based on discounted ST (model-consistent).
        """
        mkt = self.model.market
        s0, sigma = float(self.model.params.s0), float(self.model.params.sigma)
        r, q = float(mkt.r), float(mkt.q)

        T = float(option.T)
        if T < 0:
            raise ValueError("Option maturity T must be >= 0")

        rng = np.random.default_rng(seed)

        disc_payoffs = np.empty(int(n_paths), dtype=float)
        disc_ST = np.empty(int(n_paths), dtype=float) if control_variate else None

        disc = np.exp(-r * T)

        for i in range(int(n_paths)):
            if antithetic:
                z = rng.standard_normal()
                z2 = -z

                ST1 = s0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
                ST2 = s0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z2)

                payoff_val = 0.5 * (option.payoff(np.array([ST1]))[0] + option.payoff(np.array([ST2]))[0])
                disc_payoffs[i] = disc * payoff_val

                if control_variate:
                    disc_ST[i] = disc * (0.5 * (ST1 + ST2))
            else:
                z = rng.standard_normal()
                ST = s0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

                disc_payoffs[i] = disc * option.payoff(np.array([ST]))[0]
                if control_variate:
                    disc_ST[i] = disc * ST

        if control_variate:
            # E[ discounted ST ] = S0 * exp(-qT)
            disc_ST_exp = s0 * np.exp(-q * T)
            disc_payoffs = _apply_control_variate(disc_payoffs, disc_ST, disc_ST_exp)
            note = "European MC (scalar, 1-step)" + (" + antithetic" if antithetic else "") + " + control variate(ST)"
        else:
            note = "European MC (scalar, 1-step)" + (" + antithetic" if antithetic else "")

        mean, se, lo, hi = confidence_interval(disc_payoffs)
        return PriceResult(price=mean, std_error=se, ci_low=lo, ci_high=hi, n_paths=int(n_paths), notes=note)

    def price_vector(
        self,
        option: EuropeanOption,
        n_paths: int,
        seed: int | None = None,
        antithetic: bool = False,
        control_variate: bool = False,
    ) -> PriceResult:
        """
        Vectorized (NumPy) one-step European pricing.
        Optional control variate based on discounted ST (model-consistent).
        """
        mkt = self.model.market
        s0, sigma = float(self.model.params.s0), float(self.model.params.sigma)
        r, q = float(mkt.r), float(mkt.q)

        # IMPORTANT: use option.T (computed), not option.maturity
        T = float(option.T)
        if T < 0:
            raise ValueError("Option maturity T must be >= 0")

        rng = np.random.default_rng(seed)

        if antithetic:
            half = (int(n_paths) + 1) // 2
            Z_half = rng.standard_normal(half)
            Z = np.concatenate([Z_half, -Z_half])[: int(n_paths)]
        else:
            Z = rng.standard_normal(int(n_paths))

        ST = s0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        disc = np.exp(-r * T)
        disc_payoffs = disc * option.payoff(ST)

        if control_variate:
            disc_ST = disc * ST
            disc_ST_exp = s0 * np.exp(-q * T)
            disc_payoffs = _apply_control_variate(disc_payoffs, disc_ST, disc_ST_exp)
            note = f"European MC (vector, 1-step){' + antithetic' if antithetic else ''} + control variate(ST)"
        else:
            note = f"European MC (vector, 1-step){' + antithetic' if antithetic else ''}"

        mean, se, lo, hi = confidence_interval(disc_payoffs)
        return PriceResult(price=mean, std_error=se, ci_low=lo, ci_high=hi, n_paths=int(n_paths), notes=note)


@dataclass(frozen=True)
class LongstaffSchwartzPricer:
    model: GBMModel
    basis: Basis = PolyBasis(degree=2, include_intercept=True)

    def price(
        self,
        option: AmericanOption,
        n_paths: int,
        n_steps: int,
        seed: int | None = None,
        antithetic: bool = False,
        return_diagnostics: bool = False,
    ):
        """
        Longstaff–Schwartz (LSM) pricer (Bermudan on grid).
        """
        mkt = self.model.market
        r = float(mkt.r)

        # IMPORTANT: use option.T (computed), not option.maturity
        T = float(option.T)
        if T <= 0:
            raise ValueError("For LSM, maturity T must be > 0")

        dt = T / int(n_steps)
        disc = np.exp(-r * dt)

        t_grid, S = self.model.simulate_paths(
            maturity=T,
            n_steps=int(n_steps),
            n_paths=int(n_paths),
            seed=seed,
            antithetic=antithetic,
        )

        CF = option.payoff(S[:, -1])  # payoff at T

        diag = {"regression_r2": [], "n_itm": []}

        for k in range(int(n_steps) - 1, 0, -1):
            S_k = S[:, k]
            IV = option.payoff(S_k)

            # discount continuation one step back to t_k
            Y = CF * disc

            itm = IV > 0
            n_itm = int(np.sum(itm))
            diag["n_itm"].append(n_itm)

            if n_itm >= 3:
                X = (S_k[itm] / float(option.strike))
                y = Y[itm]

                Phi = self.basis.design_matrix(X)
                fit = ols_fit(Phi, y)

                C_hat = fit.y_hat
                exercise_itm = IV[itm] > C_hat

                CF_new_itm = np.where(exercise_itm, IV[itm], Y[itm])
                CF_updated = CF.copy()
                CF_updated[itm] = CF_new_itm
                CF = CF_updated

                diag["regression_r2"].append(float(fit.r2))
            else:
                CF = Y
                diag["regression_r2"].append(float("nan"))

        PV_paths = CF * disc  # discount from t1 -> t0

        mean, se, lo, hi = confidence_interval(PV_paths)
        res = PriceResult(
            price=mean,
            std_error=se,
            ci_low=lo,
            ci_high=hi,
            n_paths=int(n_paths),
            notes=f"LSM American (n_steps={int(n_steps)}){' + antithetic' if antithetic else ''}",
        )

        if return_diagnostics:
            diag["regression_r2"] = diag["regression_r2"][::-1]
            diag["n_itm"] = diag["n_itm"][::-1]
            diag["time_index"] = list(range(1, int(n_steps)))
            return res, diag
        return res
