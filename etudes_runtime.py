# etudes_runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RuntimeStudyResult:
    df: pd.DataFrame
    bs_price: float
    scalar_cap: int
    scalar_scaled: bool


def study_runtime_vs_N(
    pricer,
    option,
    N_grid: List[int],
    bs_price: float,
    seed: int = 1,
    antithetic: bool = True,
    scalar_cap: int = 50_000,
    scalar_scaled: bool = True,
) -> RuntimeStudyResult:
    """
    Measure runtime + price vs N for:
      - Vectorized MC: pricer.price_vector(...)
      - Scalar MC: pricer.price_scalar(...)

    If scalar_scaled=True:
      - run scalar at n_scalar = min(N, scalar_cap)
      - scale runtime to N by (N / n_scalar) (assume linear in N)
      - price is still computed at n_scalar (note: not same estimator variance as at N)
    If scalar_scaled=False:
      - run scalar at full N (can be slow!)
    """
    rows: List[Dict[str, Any]] = []

    for i, N in enumerate(N_grid):
        N = int(N)
        # ----- Vectorized -----
        t0 = __import__("time").perf_counter()
        res_vec = pricer.price_vector(
            option,
            n_paths=N,
            seed=int(seed) + i,
            antithetic=bool(antithetic),
            control_variate=False,
        )
        t_vec = __import__("time").perf_counter() - t0

        # ----- Scalar -----
        if scalar_scaled:
            n_scalar = int(min(N, int(scalar_cap)))
        else:
            n_scalar = int(N)

        t1 = __import__("time").perf_counter()
        res_sca = pricer.price_scalar(
            option,
            n_paths=n_scalar,
            seed=int(seed) + i,
            antithetic=bool(antithetic),
        )
        t_sca = __import__("time").perf_counter() - t1

        # scale scalar runtime to N if asked
        if scalar_scaled:
            t_sca_scaled = t_sca * (N / max(n_scalar, 1))
        else:
            t_sca_scaled = t_sca

        # Extract prices (be robust to your result object)
        p_vec = float(getattr(res_vec, "price", res_vec["price"] if isinstance(res_vec, dict) else res_vec))
        p_sca = float(getattr(res_sca, "price", res_sca["price"] if isinstance(res_sca, dict) else res_sca))

        rows.append(
            {
                "N": N,
                "price_vec": p_vec,
                "time_vec_s": float(t_vec),
                "n_scalar_used": n_scalar,
                "price_sca": p_sca,
                "time_sca_s": float(t_sca),
                "time_sca_scaled_to_N_s": float(t_sca_scaled),
                "bs_price": float(bs_price),
            }
        )

    df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
    return RuntimeStudyResult(df=df, bs_price=float(bs_price), scalar_cap=int(scalar_cap), scalar_scaled=bool(scalar_scaled))


def plot_runtime_analysis(result: RuntimeStudyResult):
    """
    Single matplotlib figure:
      - X: N (log)
      - Left Y: runtime (vector + scalar)
      - Right Y: prices (vector + scalar + BS dashed)
    """
    df = result.df
    N = df["N"].values.astype(float)

    fig, ax_time = plt.subplots(figsize=(8, 5))
    ax_price = ax_time.twinx()

    # ---- Runtime (left y) ----
    ax_time.plot(N, df["time_vec_s"].values, marker="o", linewidth=1, label="Runtime — Vectorized (s)")

    if result.scalar_scaled:
        ax_time.plot(
            N,
            df["time_sca_scaled_to_N_s"].values,
            marker="o",
            linewidth=1,
            label=f"Runtime — Scalar scaled to N (cap={result.scalar_cap:,})",
        )
    else:
        ax_time.plot(N, df["time_sca_s"].values, marker="o", linewidth=1, label="Runtime — Scalar (s)")

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Number of paths N (log scale)")
    ax_time.set_ylabel("Runtime (seconds, log scale)")
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)

    # ---- Prices (right y) ----
    ax_price.plot(N, df["price_vec"].values, marker="x", linewidth=1, label="Price — Vectorized")
    ax_price.plot(N, df["price_sca"].values, marker="x", linewidth=1, label="Price — Scalar")
    ax_price.axhline(result.bs_price, linestyle="--", linewidth=1, label="Black–Scholes (fixed)")
    ax_price.set_ylabel("Option price")

    # ---- Combine legends ----
    lines1, labels1 = ax_time.get_legend_handles_labels()
    lines2, labels2 = ax_price.get_legend_handles_labels()
    ax_time.legend(lines1 + lines2, labels1 + labels2, loc="best")

    title = "Runtime vs N (Vectorized vs Scalar) + Price vs N (with Black–Scholes)"
    ax_time.set_title(title)
    fig.tight_layout()
    return fig
