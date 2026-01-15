# utils.py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

from core.options import OptionType


# ---------------------------------------------------------------------
# Global Matplotlib style (safe defaults for Streamlit + CLI)
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ---------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------
def print_block(title: str, width: int = 90) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def fmt_price_result(res) -> str:
    return (
        f"price={res.price:.6f} | "
        f"SE={res.std_error:.6f} | "
        f"95% CI=[{res.ci_low:.6f}, {res.ci_high:.6f}] | "
        f"N={res.n_paths} | {res.notes}"
    )


# ---------------------------------------------------------------------
# Black-Scholes closed form
# ---------------------------------------------------------------------
def black_scholes_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    opt_type: OptionType
) -> float:
    if T <= 0:
        return max(S0 - K, 0.0) if opt_type == OptionType.CALL else max(K - S0, 0.0)

    if sigma <= 0:
        F = S0 * math.exp((r - q) * T)
        disc = math.exp(-r * T)
        if opt_type == OptionType.CALL:
            return disc * max(F - K, 0.0)
        return disc * max(K - F, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if opt_type == OptionType.CALL:
        return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)


# ---------------------------------------------------------------------
# MC convergence table builder
# ---------------------------------------------------------------------
def mc_convergence_rows(pricer, option, N_list, seed=1, antithetic=True, control_variate=False):
    rows = []
    for i, N in enumerate(N_list):
        res = pricer.price_vector(
            option,
            n_paths=int(N),
            seed=int(seed) + i,
            antithetic=bool(antithetic),
            control_variate=bool(control_variate),
        )
        rows.append((int(N), float(res.price), float(res.ci_low), float(res.ci_high), float(res.std_error)))
    return rows


def print_variance_reduction(rows_plain, rows_cv, label: str):
    print(f"\nVariance reduction summary — {label}")
    print("N       | SE_plain  | SE_CV     | Gain(SE) | N_equiv (approx)")
    for (N, _, _, _, se_p), (_, _, _, _, se_cv) in zip(rows_plain, rows_cv):
        gain = se_p / se_cv if se_cv > 0 else float("nan")
        n_equiv = N * (gain ** 2) if np.isfinite(gain) else float("nan")
        print(f"{N:7d} | {se_p:8.6f} | {se_cv:8.6f} | {gain:7.3f} | {n_equiv:,.0f}")


def print_convergence_table(rows, bs_price: float, label: str) -> None:
    print(f"\n{label}")
    print(f"Black-Scholes: {bs_price:.6f}")
    for (N, mc, lo, hi, se) in rows:
        inside = (lo <= bs_price <= hi)
        print(f"N={N:>7d} | MC={mc:.6f} | SE={se:.6f} | CI=[{lo:.6f}, {hi:.6f}] | BS in CI? {inside}")


# ---------------------------------------------------------------------
# Plotting (Matplotlib) — returns fig for Streamlit (more stable)
# ---------------------------------------------------------------------
def plot_convergence_compare(rows_plain, rows_cv, bs_price: float, title: str):
    """
    Two MC curves on same plot:
    - plain antithetic
    - antithetic + control variate(ST)
    with CI bands + BS line

    Returns:
        fig (matplotlib Figure)
    """
    def unpack(rows):
        N = np.array([r[0] for r in rows], dtype=float)
        mc = np.array([r[1] for r in rows], dtype=float)
        lo = np.array([r[2] for r in rows], dtype=float)
        hi = np.array([r[3] for r in rows], dtype=float)
        return N, mc, lo, hi

    N1, mc1, lo1, hi1 = unpack(rows_plain)
    N2, mc2, lo2, hi2 = unpack(rows_cv)

    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=110)

    ax.plot(N1, mc1, marker="o", linewidth=1.4, markersize=4, label="MC (antithetic)")
    ax.fill_between(N1, lo1, hi1, alpha=0.12)

    ax.plot(N2, mc2, marker="o", linewidth=1.4, markersize=4, label="MC + control variate (ST)")
    ax.fill_between(N2, lo2, hi2, alpha=0.12)

    ax.axhline(bs_price, linestyle="--", linewidth=1.2, label="Black–Scholes")

    ax.set_xscale("log")
    ax.set_xlabel("Number of paths N (log scale)")
    ax.set_ylabel("Option price")
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    fig.tight_layout(pad=0.6)
    return fig


def plot_se_compare(rows_plain, rows_cv, title: str):
    """
    Standard error vs N (plain vs control variate). Returns fig.
    """
    N = np.array([r[0] for r in rows_plain], dtype=float)
    se_plain = np.array([r[4] for r in rows_plain], dtype=float)
    se_cv = np.array([r[4] for r in rows_cv], dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=110)

    ax.plot(N, se_plain, marker="o", linewidth=1.4, markersize=4, label="SE (antithetic)")
    ax.plot(N, se_cv, marker="o", linewidth=1.4, markersize=4, label="SE (control variate)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Number of paths N (log scale)")
    ax.set_ylabel("Standard error (log scale)")
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    fig.tight_layout(pad=0.6)
    return fig
