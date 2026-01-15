# etudes_convergence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

from core.mc_engine import EuropeanMCPricer
from core.options import EuropeanOption
from utils import black_scholes_price


# ============================================================
# Global plotting style (Streamlit-friendly, small fonts)
# ============================================================

# Change these 2 lines if you want even smaller/bigger figures/fonts.
DEFAULT_FIGSIZE = (4.5, 2.6)   # width, height in inches
FONT_BASE = 6                 # base font size for axes/ticks

# Derived sizes
TITLE_SIZE = FONT_BASE + 2
LABEL_SIZE = FONT_BASE + 1
TICK_SIZE = FONT_BASE
LEGEND_SIZE = FONT_BASE

GRID_LW = 0.4
LINE_LW = 1.0
MARKER_SIZE = 4


def _apply_small_style(ax: plt.Axes) -> None:
    """Apply consistent small-font styling to current axes."""
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
    ax.grid(True, which="both", linestyle="--", linewidth=GRID_LW, alpha=0.8)


def _new_fig(figsize: Optional[tuple] = None) -> plt.Figure:
    """Create a new figure with Streamlit-friendly sizing."""
    fig = plt.figure(figsize=figsize or DEFAULT_FIGSIZE, dpi=130)
    return fig


# ============================
# Study 1: Distribution over seeds (fixed N)
# ============================

@dataclass(frozen=True)
class SeedStudyResult:
    prices: np.ndarray          # shape (n_seeds,)
    seeds: np.ndarray           # shape (n_seeds,)
    mean: float
    std: float
    q05: float
    q50: float
    q95: float
    bs_price: float
    N: int
    antithetic: bool
    control_variate: bool


def study_price_distribution_over_seeds(
    pricer: EuropeanMCPricer,
    option: EuropeanOption,
    N: int,
    n_seeds: int = 100,
    seed0: int = 1,
    antithetic: bool = True,
    control_variate: bool = False,
    bs_price: Optional[float] = None,
) -> SeedStudyResult:
    """
    Run MC pricing for many different seeds (fixed N) and study the distribution of estimates.
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be >= 1")
    if N <= 0:
        raise ValueError("N must be >= 1")

    seeds = np.arange(seed0, seed0 + n_seeds, dtype=int)
    prices = np.empty(n_seeds, dtype=float)

    # Compute BS only once if not provided
    if bs_price is None:
        mkt = pricer.model.market
        pars = pricer.model.params
        bs_price = float(
            black_scholes_price(
                S0=float(pars.s0),
                K=float(option.strike),
                T=float(option.T),
                r=float(mkt.r),
                q=float(mkt.q),
                sigma=float(pars.sigma),
                opt_type=option.option_type,
            )
        )

    for i, sd in enumerate(seeds):
        res = pricer.price_vector(
            option,
            n_paths=int(N),
            seed=int(sd),
            antithetic=bool(antithetic),
            control_variate=bool(control_variate),
        )
        prices[i] = float(res.price)

    mean = float(np.mean(prices))
    std = float(np.std(prices, ddof=1)) if n_seeds > 1 else 0.0
    q05, q50, q95 = (float(x) for x in np.quantile(prices, [0.05, 0.50, 0.95]))

    return SeedStudyResult(
        prices=prices,
        seeds=seeds,
        mean=mean,
        std=std,
        q05=q05,
        q50=q50,
        q95=q95,
        bs_price=float(bs_price),
        N=int(N),
        antithetic=bool(antithetic),
        control_variate=bool(control_variate),
    )


def plot_seed_distribution(
    res: SeedStudyResult,
    bins: int = 20,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot histogram of MC prices across seeds + BS as vertical line.
    """
    fig = _new_fig(figsize)
    ax = plt.gca()

    ax.hist(res.prices, bins=int(bins), alpha=0.85)
    ax.axvline(res.bs_price, linestyle="--", linewidth=1.5)

    ax.set_xlabel("MC price estimate")
    ax.set_ylabel("Frequency")

    if title is None:
        title = (
            f"Distribution over seeds (N={res.N:,}) "
            f"{'antithetic' if res.antithetic else ''}"
            f"{' + CV(ST)' if res.control_variate else ''}"
        )
    ax.set_title(title)

    _apply_small_style(ax)
    fig.tight_layout(pad=0.6)
    return fig


# ============================
# Study 1bis: Std(price over seeds) vs N (fixed n_seeds)
# ============================

@dataclass(frozen=True)
class SeedStdVsNResult:
    N_grid: np.ndarray           # shape (m,)
    std_over_seeds: np.ndarray   # shape (m,)
    scaled_std: np.ndarray       # std * sqrt(N)
    mean_over_seeds: np.ndarray  # shape (m,)
    bs_price: float
    n_seeds: int
    seed0: int
    antithetic: bool
    control_variate: bool


def study_std_over_seeds_vs_N(
    pricer: EuropeanMCPricer,
    option: EuropeanOption,
    N_grid: Iterable[int],
    n_seeds: int = 100,
    seed0: int = 1,
    antithetic: bool = True,
    control_variate: bool = False,
    bs_price: Optional[float] = None,
) -> SeedStdVsNResult:
    """
    For each N in N_grid:
      - run pricing across n_seeds different seeds
      - compute std(prices) across seeds

    Theory: std_over_seeds ~ c / sqrt(N) for large N.
    So scaled_std = std_over_seeds * sqrt(N) should be ~ flat.
    """
    N_list = sorted(set(int(n) for n in N_grid))
    if len(N_list) == 0:
        raise ValueError("N_grid is empty")
    if any(n <= 0 for n in N_list):
        raise ValueError("All N must be >= 1")
    if n_seeds <= 0:
        raise ValueError("n_seeds must be >= 1")

    # BS once
    if bs_price is None:
        mkt = pricer.model.market
        pars = pricer.model.params
        bs_price = float(
            black_scholes_price(
                S0=float(pars.s0),
                K=float(option.strike),
                T=float(option.T),
                r=float(mkt.r),
                q=float(mkt.q),
                sigma=float(pars.sigma),
                opt_type=option.option_type,
            )
        )

    stds = np.empty(len(N_list), dtype=float)
    means = np.empty(len(N_list), dtype=float)

    seeds = np.arange(seed0, seed0 + n_seeds, dtype=int)

    for j, N in enumerate(N_list):
        prices = np.empty(n_seeds, dtype=float)
        for i, sd in enumerate(seeds):
            res = pricer.price_vector(
                option,
                n_paths=int(N),
                seed=int(sd),
                antithetic=bool(antithetic),
                control_variate=bool(control_variate),
            )
            prices[i] = float(res.price)

        means[j] = float(np.mean(prices))
        stds[j] = float(np.std(prices, ddof=1)) if n_seeds > 1 else 0.0

    N_arr = np.array(N_list, dtype=float)
    scaled = stds * np.sqrt(N_arr)

    return SeedStdVsNResult(
        N_grid=np.array(N_list, dtype=int),
        std_over_seeds=stds,
        scaled_std=scaled,
        mean_over_seeds=means,
        bs_price=float(bs_price),
        n_seeds=int(n_seeds),
        seed0=int(seed0),
        antithetic=bool(antithetic),
        control_variate=bool(control_variate),
    )


def plot_std_over_seeds_vs_N(
    res: SeedStdVsNResult,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot std(prices over seeds) vs N.
    Expected ~ 1/sqrt(N).
    """
    fig = _new_fig(figsize)
    ax = plt.gca()

    ax.plot(res.N_grid, res.std_over_seeds, marker="o", linewidth=LINE_LW, markersize=MARKER_SIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (log)")
    ax.set_ylabel("Std over seeds (log)")

    if title is None:
        title = (
            f"Std over seeds vs N (n_seeds={res.n_seeds}) "
            f"{'antithetic' if res.antithetic else ''}"
            f"{' + CV(ST)' if res.control_variate else ''}"
        )
    ax.set_title(title)

    _apply_small_style(ax)
    fig.tight_layout(pad=0.6)
    return fig


def plot_scaled_std_over_seeds_vs_N(
    res: SeedStdVsNResult,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot std(prices over seeds) * sqrt(N) vs N.
    Expected ~ flat for large N.
    """
    fig = _new_fig(figsize)
    ax = plt.gca()

    ax.plot(res.N_grid, res.scaled_std, marker="o", linewidth=LINE_LW, markersize=MARKER_SIZE)
    ax.set_xscale("log")
    ax.set_xlabel("N (log)")
    ax.set_ylabel("Std over seeds * sqrt(N)")

    if title is None:
        title = (
            f"Scaled std vs N (n_seeds={res.n_seeds}) "
            f"{'antithetic' if res.antithetic else ''}"
            f"{' + CV(ST)' if res.control_variate else ''}"
        )
    ax.set_title(title)

    _apply_small_style(ax)
    fig.tight_layout(pad=0.6)
    return fig


# ============================
# Study 2: Error vs N and sqrt(N) scaling
# ============================

@dataclass(frozen=True)
class NStudyResult:
    N_grid: np.ndarray              # shape (m,)
    mc_prices: np.ndarray           # shape (m,)
    abs_error: np.ndarray           # shape (m,)
    signed_error: np.ndarray        # shape (m,)
    scaled_abs_error: np.ndarray    # abs_error * sqrt(N)
    bs_price: float
    seed: int
    antithetic: bool
    control_variate: bool


def study_error_vs_N(
    pricer: EuropeanMCPricer,
    option: EuropeanOption,
    N_grid: Iterable[int],
    seed: int = 1,
    antithetic: bool = True,
    control_variate: bool = False,
    bs_price: Optional[float] = None,
) -> NStudyResult:
    """
    For a fixed seed, compute MC price for each N in N_grid and compare to BS.
    Also compute sqrt(N) scaling: abs_error * sqrt(N) should be roughly stable for large N.
    """
    N_list = sorted(set(int(n) for n in N_grid))
    if any(n <= 0 for n in N_list):
        raise ValueError("All N must be >= 1")
    m = len(N_list)
    if m == 0:
        raise ValueError("N_grid is empty")

    if bs_price is None:
        mkt = pricer.model.market
        pars = pricer.model.params
        bs_price = float(
            black_scholes_price(
                S0=float(pars.s0),
                K=float(option.strike),
                T=float(option.T),
                r=float(mkt.r),
                q=float(mkt.q),
                sigma=float(pars.sigma),
                opt_type=option.option_type,
            )
        )

    mc_prices = np.empty(m, dtype=float)

    for i, N in enumerate(N_list):
        res = pricer.price_vector(
            option,
            n_paths=int(N),
            seed=int(seed),
            antithetic=bool(antithetic),
            control_variate=bool(control_variate),
        )
        mc_prices[i] = float(res.price)

    bs = float(bs_price)
    signed_error = mc_prices - bs
    abs_error = np.abs(signed_error)
    scaled_abs_error = abs_error * np.sqrt(np.array(N_list, dtype=float))

    return NStudyResult(
        N_grid=np.array(N_list, dtype=int),
        mc_prices=mc_prices,
        abs_error=abs_error,
        signed_error=signed_error,
        scaled_abs_error=scaled_abs_error,
        bs_price=bs,
        seed=int(seed),
        antithetic=bool(antithetic),
        control_variate=bool(control_variate),
    )


def plot_error_vs_N(
    res: NStudyResult,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot |MC - BS| vs N (log-log).
    Expect approx O(1/sqrt(N)).
    """
    fig = _new_fig(figsize)
    ax = plt.gca()

    ax.plot(res.N_grid, res.abs_error, marker="o", linewidth=LINE_LW, markersize=MARKER_SIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (log)")
    ax.set_ylabel("|MC - BS| (log)")

    if title is None:
        title = (
            f"Absolute error vs N (seed={res.seed}) "
            f"{'antithetic' if res.antithetic else ''}"
            f"{' + CV(ST)' if res.control_variate else ''}"
        )
    ax.set_title(title)

    _apply_small_style(ax)
    fig.tight_layout(pad=0.6)
    return fig


def plot_scaled_error_vs_N(
    res: NStudyResult,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot |MC - BS| * sqrt(N) vs N (log x).
    For large N, should be ~flat if CLT scaling holds.
    """
    fig = _new_fig(figsize)
    ax = plt.gca()

    ax.plot(res.N_grid, res.scaled_abs_error, marker="o", linewidth=LINE_LW, markersize=MARKER_SIZE)
    ax.set_xscale("log")
    ax.set_xlabel("N (log)")
    ax.set_ylabel("|MC - BS| * sqrt(N)")

    if title is None:
        title = (
            f"Scaled error vs N (seed={res.seed}) "
            f"{'antithetic' if res.antithetic else ''}"
            f"{' + CV(ST)' if res.control_variate else ''}"
        )
    ax.set_title(title)

    _apply_small_style(ax)
    fig.tight_layout(pad=0.6)
    return fig
