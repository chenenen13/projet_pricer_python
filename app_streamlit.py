# app_streamlit.py
import time
from datetime import date

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from core import Market, GBMModel
from core.models import GBMParams
from core.options import EuropeanOption, AmericanOption, OptionType
from core.mc_engine import EuropeanMCPricer, LongstaffSchwartzPricer
from core.regression import PolyBasis, LaguerreBasis
from core.dates import DayCount, year_fraction

from utils import (
    black_scholes_price,
    mc_convergence_rows,
    plot_convergence_compare,
    plot_se_compare,
)

# Studies
from etudes_convergence import (
    study_price_distribution_over_seeds,
    plot_seed_distribution,
    study_std_over_seeds_vs_N,
    plot_std_over_seeds_vs_N,
    plot_scaled_std_over_seeds_vs_N,
    study_error_vs_N,
    plot_error_vs_N,
    plot_scaled_error_vs_N,
)

# Runtime study
from etudes_runtime import (
    study_runtime_vs_N,
    plot_runtime_analysis,
)

st.set_page_config(page_title="MC & LSM Pricer", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _opt_enum(callput_str: str) -> OptionType:
    return OptionType.CALL if callput_str.upper().startswith("C") else OptionType.PUT


def _basis_from_choice(name: str):
    if name == "Polynomial (deg 2)":
        return PolyBasis(degree=2)
    if name == "Polynomial (deg 3)":
        return PolyBasis(degree=3)
    if name == "Laguerre (deg 2)":
        return LaguerreBasis(degree=2)
    return LaguerreBasis(degree=3)


def _to_result_table(res):
    if hasattr(res, "to_frame"):
        df = res.to_frame()
    elif hasattr(res, "to_dict"):
        df = pd.DataFrame([res.to_dict()]).T
        df.columns = ["value"]
    else:
        df = pd.DataFrame([res]).T
        df.columns = ["value"]
    return df


def _default_N_list():
    return [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]


def _time_call(fn, *args, **kwargs):
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


def _dc_label_to_enum(label: str) -> DayCount:
    mapping = {
        "ACT/365F": DayCount.ACT_365F,
        "ACT/360": DayCount.ACT_360,
        "ACT/ACT": DayCount.ACT_ACT,
        "30/360": DayCount.THIRTY_360,
    }
    return mapping[label]


def _build_N_grid_from_max_step(N_max: int, step: int, N_start: int = 0) -> list[int]:
    """
    Build N grid from (N_max, step, N_start).
    Note: N=0 is invalid for MC, so if N_start<1 we start at 'step' (or 1 if step==0).
    """
    N_max = int(N_max)
    step = int(step)
    N_start = int(N_start)

    if N_max <= 0:
        return []
    if step <= 0:
        return []

    start_eff = N_start if N_start >= 1 else step
    start_eff = max(1, start_eff)

    grid = list(np.arange(start_eff, N_max + 1, step, dtype=int))
    if len(grid) == 0:
        grid = [min(max(1, step), N_max)]
    return grid


# -----------------------------
# UI Header
# -----------------------------
st.title("Monte Carlo Pricer — European (MC) + American (LSM Longstaff–Schwartz)")


# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Market")
    r = st.number_input("r (risk-free, cont.)", value=0.03, step=0.005, format="%.4f")
    q = st.number_input("q (dividend yield)", value=0.00, step=0.005, format="%.4f")

    st.header("Underlying (GBM)")
    s0 = st.number_input("S0", value=100.0, step=1.0, format="%.4f")
    sigma = st.number_input("sigma", value=0.20, step=0.01, format="%.4f")

    st.header("Option")
    callput_str = st.selectbox("Call/Put", ["CALL", "PUT"])
    K = st.number_input("Strike K", value=100.0, step=1.0, format="%.4f")

    st.subheader("Dates")
    pricing_date = st.date_input("Pricing date", value=date.today())
    maturity_date = st.date_input(
        "Maturity date",
        value=date(pricing_date.year + 1, pricing_date.month, pricing_date.day),
    )
    dc_label = st.selectbox("Day count", ["ACT/365F", "ACT/360", "ACT/ACT", "30/360"], index=0)
    day_count = _dc_label_to_enum(dc_label)

    try:
        T = float(year_fraction(pricing_date, maturity_date, day_count))
        st.caption(f"Computed maturity T = **{T:.6f}** years ({dc_label})")
    except Exception as e:
        T = 0.0
        st.error(f"Invalid dates: {e}")

    st.header("Monte Carlo")
    antithetic = st.checkbox("Antithetic", value=True)
    seed = st.number_input("seed", value=1, step=1)
    n_paths = st.number_input("n_paths", value=100_000, step=50_000)

    st.header("Performance benchmark")
    enable_perf = st.checkbox("Show runtime benchmark", value=True)
    scalar_cap = st.number_input("Scalar cap (max paths)", value=50_000, step=10_000)

    st.divider()
    st.caption("Tip: run via `streamlit run app_streamlit.py` (not python).")


# Build model objects
market = Market(r=float(r), q=float(q))
model = GBMModel(market, GBMParams(s0=float(s0), sigma=float(sigma)))

tabs = st.tabs(["Pricing", "Convergence", "Runtime analysis", "LSM Diagnostics"])


# =====================================================================
# TAB 1 — Pricing
# =====================================================================
with tabs[0]:
    st.subheader("Single pricing")

    c1, c2 = st.columns([1, 1])
    with c1:
        product = st.radio("Product", ["European (MC)", "American (LSM)"], horizontal=True)
    with c2:
        use_cv = st.checkbox("Control variate using ST (European only)", value=False)

    opt_enum = _opt_enum(callput_str)

    if product.startswith("European"):
        option = EuropeanOption(
            option_type=opt_enum,
            strike=float(K),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            day_count=day_count,
        )
        pricer = EuropeanMCPricer(model)

        bs_price = black_scholes_price(
            S0=float(s0),
            K=float(K),
            T=float(option.T),
            r=float(r),
            q=float(q),
            sigma=float(sigma),
            opt_type=opt_enum,
        )

        run = st.button("Price European")

        if run:
            res_vec, t_vec = _time_call(
                pricer.price_vector,
                option,
                n_paths=int(n_paths),
                seed=int(seed),
                antithetic=bool(antithetic),
                control_variate=bool(use_cv),
            )

            n_scalar = int(min(int(n_paths), int(scalar_cap)))
            res_sca, t_sca = _time_call(
                pricer.price_scalar,
                option,
                n_paths=n_scalar,
                seed=int(seed),
                antithetic=bool(antithetic),
            )

            st.info(f"Black–Scholes (closed form): **{bs_price:.6f}**")

            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown("### Vectorized MC")
                st.dataframe(_to_result_table(res_vec), use_container_width=True)
            with colB:
                st.markdown(f"### Scalar MC (capped at {n_scalar:,})")
                st.dataframe(_to_result_table(res_sca), use_container_width=True)

            if enable_perf:
                st.markdown("### Performance benchmark (runtime)")
                n_vec = int(n_paths)
                t_sca_est = t_sca * (n_vec / max(n_scalar, 1))
                speedup_est = t_sca_est / max(t_vec, 1e-12)

                df_perf = pd.DataFrame(
                    [
                        {"engine": "Vectorized MC", "paths": n_vec, "time_s": t_vec, "time_s_scaled_to_vecN": t_vec},
                        {"engine": "Scalar MC", "paths": n_scalar, "time_s": t_sca, "time_s_scaled_to_vecN": t_sca_est},
                    ]
                )
                st.dataframe(df_perf, use_container_width=True)
                st.write(
                    {
                        "Speed-up (estimated @ same N)": round(speedup_est, 2),
                        "Note": "Scalar time is scaled to N_vec assuming ~linear runtime in N.",
                    }
                )

    else:
        option = AmericanOption(
            option_type=opt_enum,
            strike=float(K),
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            day_count=day_count,
        )

        st.markdown("### LSM settings")
        colA, colB = st.columns([1, 1])
        with colA:
            n_steps = st.number_input("n_steps (exercise grid)", value=50, step=10)
        with colB:
            basis_name = st.selectbox(
                "Regression basis",
                ["Polynomial (deg 2)", "Polynomial (deg 3)", "Laguerre (deg 2)", "Laguerre (deg 3)"],
            )

        basis = _basis_from_choice(basis_name)
        pricer = LongstaffSchwartzPricer(model, basis=basis)

        run = st.button("Price American (LSM)")

        if run:
            (res, diag), t_lsm = _time_call(
                pricer.price,
                option,
                n_paths=int(n_paths),
                n_steps=int(n_steps),
                seed=int(seed),
                antithetic=bool(antithetic),
                return_diagnostics=True,
            )

            st.markdown("### LSM result")
            st.dataframe(_to_result_table(res), use_container_width=True)

            if enable_perf:
                st.markdown("### Performance benchmark (runtime)")
                st.write({"LSM runtime (s)": round(t_lsm, 4)})

            st.markdown("### Diagnostics")
            df_diag = pd.DataFrame(
                {"k": diag["time_index"], "n_itm": diag["n_itm"], "regression_r2": diag["regression_r2"]}
            )
            st.dataframe(df_diag, use_container_width=True)
            st.line_chart(df_diag.set_index("k")[["n_itm"]])
            st.line_chart(df_diag.set_index("k")[["regression_r2"]])


# =====================================================================
# TAB 2 — Convergence + NEW STUDIES
# =====================================================================
with tabs[1]:
    st.subheader("Convergence (European MC vs Black–Scholes)")

    st.caption(
        "Tableau + graphes de convergence en fonction de N. "
        "Comparaison: MC (antithetic) vs MC + control variate(ST)."
    )

    # Build the European option/pricer once for this tab
    option_eu = EuropeanOption(
        option_type=_opt_enum(callput_str),
        strike=float(K),
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        day_count=day_count,
    )
    pricer_eu = EuropeanMCPricer(model)

    bs = black_scholes_price(
        S0=float(s0),
        K=float(K),
        T=float(option_eu.T),
        r=float(r),
        q=float(q),
        sigma=float(sigma),
        opt_type=_opt_enum(callput_str),
    )

    # ---- Existing convergence study (table + curves) ----
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        N_list = st.multiselect("N grid", options=_default_N_list(), default=_default_N_list(), key="conv_N_grid")
        N_list = sorted(list(set(int(x) for x in N_list)))
    with col2:
        do_cv = st.checkbox("Show control variate(ST)", value=True, key="conv_do_cv")
    with col3:
        show_se_plot = st.checkbox("Plot SE comparison", value=True, key="conv_show_se")

    run_conv = st.button("Run convergence study", key="btn_conv")

    if run_conv:
        rows_plain = mc_convergence_rows(
            pricer_eu,
            option_eu,
            N_list,
            seed=int(seed),
            antithetic=bool(antithetic),
            control_variate=False,
        )
        df_plain = pd.DataFrame(rows_plain, columns=["N", "MC", "CI_low", "CI_high", "SE"])
        df_plain["BS"] = bs
        st.markdown("### Results (plain)")
        st.dataframe(df_plain, use_container_width=True)

        if do_cv:
            rows_cv = mc_convergence_rows(
                pricer_eu,
                option_eu,
                N_list,
                seed=int(seed),
                antithetic=bool(antithetic),
                control_variate=True,
            )
            df_cv = pd.DataFrame(rows_cv, columns=["N", "MC", "CI_low", "CI_high", "SE"])
            df_cv["BS"] = bs
            st.markdown("### Results (control variate ST)")
            st.dataframe(df_cv, use_container_width=True)

            plot_convergence_compare(
                rows_plain,
                rows_cv,
                bs_price=bs,
                title=f"European {callput_str} — MC vs Control Variate(ST) vs BS (q={float(q):.2%})",
            )
            st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)

            if show_se_plot:
                plot_se_compare(
                    rows_plain,
                    rows_cv,
                    title=f"SE comparison — European {callput_str} (q={float(q):.2%})",
                )
                st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)

        else:
            N = df_plain["N"].values.astype(float)
            mc = df_plain["MC"].values.astype(float)
            lo = df_plain["CI_low"].values.astype(float)
            hi = df_plain["CI_high"].values.astype(float)

            plt.figure(figsize=(6, 4))
            plt.plot(N, mc, marker="o", linewidth=1, label="MC (antithetic)")
            plt.fill_between(N, lo, hi, alpha=0.15)
            plt.axhline(bs, linestyle="--", linewidth=1, label="Black–Scholes")
            plt.xscale("log")
            plt.xlabel("Number of paths N (log scale)")
            plt.ylabel("Option price")
            plt.title(f"European {callput_str} — MC convergence vs BS (q={float(q):.2%})")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)

    st.divider()
    st.markdown("## Complementary studies")

    # ---- NEW STUDY 1: Distribution over seeds ----
    st.markdown("### Study 1 — Distribution of MC estimates over many seeds (fixed N)")

    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        dist_N = st.number_input("N (fixed) for seed distribution", value=200_000, step=50_000, key="dist_N")
    with colB:
        n_seeds = st.number_input("Number of seeds", value=100, step=10, key="dist_n_seeds")
    with colC:
        seed_start = st.number_input("Seed start", value=1, step=1, key="dist_seed_start")
    with colD:
        dist_bins = st.number_input("Histogram bins", value=25, step=5, key="dist_bins")

    dist_use_cv = st.checkbox("Use control variate (ST) in seed distribution", value=False, key="dist_use_cv")
    run_dist = st.button("Run seed distribution study", key="btn_dist")

    if run_dist:
        with st.spinner("Running seed distribution study..."):
            res_seed = study_price_distribution_over_seeds(
                pricer=pricer_eu,
                option=option_eu,
                N=int(dist_N),
                n_seeds=int(n_seeds),
                seed0=int(seed_start),
                antithetic=bool(antithetic),
                control_variate=bool(dist_use_cv),
                bs_price=float(bs),
            )

        st.write(
            {
                "BS": round(res_seed.bs_price, 6),
                "MC mean": round(res_seed.mean, 6),
                "MC std (across seeds)": round(res_seed.std, 6),
                "q05": round(res_seed.q05, 6),
                "median": round(res_seed.q50, 6),
                "q95": round(res_seed.q95, 6),
                "N": int(res_seed.N),
                "n_seeds": int(len(res_seed.prices)),
            }
        )

        df_seed = pd.DataFrame({"seed": res_seed.seeds, "mc_price": res_seed.prices})
        st.dataframe(df_seed, use_container_width=True)

        fig = plot_seed_distribution(res_seed, bins=int(dist_bins))
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    
    st.divider()
    st.markdown("### Study 1bis — Std(prix MC sur seeds) en fonction de N (n_seeds fixe)")

    colS1, colS2, colS3, colS4 = st.columns([1, 1, 1, 1])
    with colS1:
        std_N_max = st.number_input("N max (std over seeds)", value=1_000_000, step=100_000, key="stdN_max", min_value=1)
    with colS2:
        std_N_step = st.number_input("ΔN (std over seeds)", value=50_000, step=10_000, key="stdN_step", min_value=1)
    with colS3:
        std_N_start = st.number_input("N start (can be 0)", value=0, step=10_000, key="stdN_start", min_value=0)
    with colS4:
        std_n_seeds = st.number_input("n_seeds (fixed)", value=int(n_seeds), step=10, key="std_n_seeds", min_value=1)

    std_seed0 = st.number_input("seed start (std over seeds)", value=int(seed_start), step=1, key="std_seed0", min_value=0)
    std_use_cv = st.checkbox("Use control variate (ST) for std-over-seeds study", value=bool(dist_use_cv), key="std_use_cv")

    # Build N grid like Study 2 (max + step)
    N_grid_std = _build_N_grid_from_max_step(
        N_max=int(std_N_max),
        step=int(std_N_step),
        N_start=int(std_N_start),
    )

    if len(N_grid_std) > 0:
        st.caption(f"N grid generated: {len(N_grid_std)} points (from {N_grid_std[0]:,} to {N_grid_std[-1]:,}).")
    else:
        st.caption("N grid generated: 0 points (check N max / ΔN).")

    run_stdN = st.button("Run std-over-seeds vs N", key="btn_stdN")

    if run_stdN:
        if len(N_grid_std) == 0:
            st.error("N grid is empty. Increase N max or decrease ΔN.")
        else:
            with st.spinner("Running std-over-seeds vs N..."):
                res_stdN = study_std_over_seeds_vs_N(
                    pricer=pricer_eu,
                    option=option_eu,
                    N_grid=N_grid_std,
                    n_seeds=int(std_n_seeds),
                    seed0=int(std_seed0),
                    antithetic=bool(antithetic),
                    control_variate=bool(std_use_cv),
                    bs_price=float(bs),
                )

            df_stdN = pd.DataFrame(
                {
                    "N": res_stdN.N_grid,
                    "mean_over_seeds": res_stdN.mean_over_seeds,
                    "std_over_seeds": res_stdN.std_over_seeds,
                    "std_over_seeds*sqrt(N)": res_stdN.scaled_std,
                }
            )
            st.dataframe(df_stdN, use_container_width=True)

            fig_std = plot_std_over_seeds_vs_N(res_stdN)
            st.pyplot(fig_std, clear_figure=True, use_container_width=True)

            fig_std_scaled = plot_scaled_std_over_seeds_vs_N(res_stdN)
            st.pyplot(fig_std_scaled, clear_figure=True, use_container_width=True)

            # Petit rappel interprétation
            st.info(
                "Théorie: std(prix MC) ~ c / sqrt(N). "
                "Donc std*sqrt(N) doit être ~constant quand N est grand."
            )


    st.divider()

    # ---- NEW STUDY 2: Error vs N + sqrt(N) scaling (UPDATED: N_max + step) ----
    st.markdown("### Study 2 — Error |MC - BS| vs N and sqrt(N) scaling")

    colX, colY, colZ, colW = st.columns([1, 1, 1, 1])
    with colX:
        N_max = st.number_input("N max", value=1_000_000, step=100_000, key="err_N_max", min_value=1)
    with colY:
        N_step = st.number_input("Increment step ΔN", value=50_000, step=10_000, key="err_N_step", min_value=1)
    with colZ:
        N_start = st.number_input("N start (can be 0)", value=0, step=10_000, key="err_N_start", min_value=0)
    with colW:
        err_seed = st.number_input("Seed for error-vs-N study", value=int(seed), step=1, key="err_seed")

    err_use_cv = st.checkbox("Use control variate (ST) for error-vs-N", value=False, key="err_use_cv")

    N_grid2 = _build_N_grid_from_max_step(N_max=int(N_max), step=int(N_step), N_start=int(N_start))
    if len(N_grid2) > 0:
        st.caption(f"N grid generated: {len(N_grid2)} points (from {N_grid2[0]:,} to {N_grid2[-1]:,}).")
    else:
        st.caption("N grid generated: 0 points (check N max / ΔN).")

    run_err = st.button("Run error vs N study", key="btn_err")

    if run_err:
        if len(N_grid2) == 0:
            st.error("N grid is empty. Increase N max or decrease ΔN.")
        else:
            with st.spinner("Running error vs N study..."):
                resN = study_error_vs_N(
                    pricer=pricer_eu,
                    option=option_eu,
                    N_grid=N_grid2,
                    seed=int(err_seed),
                    antithetic=bool(antithetic),
                    control_variate=bool(err_use_cv),
                    bs_price=float(bs),
                )

            df_err = pd.DataFrame(
                {
                    "N": resN.N_grid,
                    "MC": resN.mc_prices,
                    "BS": float(resN.bs_price),
                    "MC-BS": resN.signed_error,
                    "|MC-BS|": resN.abs_error,
                    "|MC-BS|*sqrt(N)": resN.scaled_abs_error,
                }
            )
            st.dataframe(df_err, use_container_width=True)

            fig_err = plot_error_vs_N(resN)
            st.pyplot(fig_err, clear_figure=True, use_container_width=True)

            fig_scaled = plot_scaled_error_vs_N(resN)
            st.pyplot(fig_scaled, clear_figure=True, use_container_width=True)

# =====================================================================
# TAB 3 — Runtime analysis (Vectorized vs Scalar) + Price vs N + BS line
# =====================================================================
with tabs[2]:
    st.subheader("Runtime analysis")
    st.caption(
        "Mesure du temps d'exécution en fonction de N (axe gauche) "
        "et prix estimé en fonction de N (axe droit), avec une ligne BS fixe. "
        "Axe X en log."
    )

    # European option/pricer for runtime study
    option_eu = EuropeanOption(
        option_type=_opt_enum(callput_str),
        strike=float(K),
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        day_count=day_count,
    )
    pricer_eu = EuropeanMCPricer(model)

    bs = black_scholes_price(
        S0=float(s0),
        K=float(K),
        T=float(option_eu.T),
        r=float(r),
        q=float(q),
        sigma=float(sigma),
        opt_type=_opt_enum(callput_str),
    )

    st.markdown("### Parameters (N grid + scalar cap)")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        rt_N_max = st.number_input("N max", value=1_000_000, step=100_000, min_value=1, key="rt_N_max")
    with c2:
        rt_step = st.number_input("ΔN step", value=50_000, step=10_000, min_value=1, key="rt_step")
    with c3:
        rt_N_start = st.number_input("N start (can be 0)", value=0, step=10_000, min_value=0, key="rt_N_start")
    with c4:
        rt_seed = st.number_input("seed", value=int(seed), step=1, key="rt_seed")

    c5, c6 = st.columns([1, 1])
    with c5:
        rt_scalar_cap = st.number_input("Scalar cap (max paths for scalar)", value=int(scalar_cap), step=10_000, key="rt_scalar_cap")
    with c6:
        rt_scalar_scaled = st.checkbox(
            "Scale scalar runtime to N (assume linear in N)",
            value=True,
            key="rt_scalar_scaled",
        )

    N_grid_rt = _build_N_grid_from_max_step(N_max=int(rt_N_max), step=int(rt_step), N_start=int(rt_N_start))
    if len(N_grid_rt) > 0:
        st.caption(f"N grid generated: {len(N_grid_rt)} points (from {N_grid_rt[0]:,} to {N_grid_rt[-1]:,}).")
    else:
        st.caption("N grid generated: 0 points (check N max / ΔN).")

    run_rt = st.button("Run runtime analysis", key="btn_runtime_analysis")

    if run_rt:
        if len(N_grid_rt) == 0:
            st.error("N grid is empty. Increase N max or decrease ΔN.")
        else:
            with st.spinner("Running runtime study (this may take time for large N)..."):
                rt_res = study_runtime_vs_N(
                    pricer=pricer_eu,
                    option=option_eu,
                    N_grid=N_grid_rt,
                    bs_price=float(bs),
                    seed=int(rt_seed),
                    antithetic=bool(antithetic),
                    scalar_cap=int(rt_scalar_cap),
                    scalar_scaled=bool(rt_scalar_scaled),
                )

            st.info(f"Black–Scholes (fixed): **{bs:.6f}**")

            # Show table
            df_rt = rt_res.df.copy()
            st.dataframe(df_rt, use_container_width=True)

            # Plot
            fig = plot_runtime_analysis(rt_res)
            st.pyplot(fig, clear_figure=True, use_container_width=True)

            if rt_scalar_scaled:
                st.warning(
                    "Scalar runtime is measured at min(N, scalar_cap) then scaled linearly to N. "
                    "Prices shown for scalar are computed at n_scalar_used (not at full N)."
                )


# =====================================================================
# TAB 3 — LSM Diagnostics (sweep n_steps)
# =====================================================================
with tabs[3]:
    st.subheader("LSM Diagnostics")
    st.caption("Visualiser l'impact du grid (n_steps) + diagnostics régression (R², ITM).")

    opt_enum = _opt_enum(callput_str)
    option = AmericanOption(
        option_type=opt_enum,
        strike=float(K),
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        day_count=day_count,
    )

    colA, colB = st.columns([1, 1])
    with colA:
        basis_name = st.selectbox(
            "Regression basis (LSM)",
            ["Polynomial (deg 2)", "Polynomial (deg 3)", "Laguerre (deg 2)", "Laguerre (deg 3)"],
            index=0,
            key="basis_lsm_diag",
        )
    with colB:
        n_steps_list = st.multiselect(
            "n_steps grid",
            options=[10, 25, 50, 75, 100, 150],
            default=[10, 25, 50, 100],
        )

    run_diag = st.button("Run LSM diagnostics")

    if run_diag:
        basis = _basis_from_choice(basis_name)
        pricer = LongstaffSchwartzPricer(model, basis=basis)

        rows = []
        diag_examples = []

        for i, ns in enumerate(sorted(n_steps_list)):
            res, diag = pricer.price(
                option,
                n_paths=int(n_paths),
                n_steps=int(ns),
                seed=int(seed) + i,
                antithetic=bool(antithetic),
                return_diagnostics=True,
            )
            rows.append(
                {
                    "n_steps": int(ns),
                    "price": float(res.price),
                    "std_error": float(res.std_error),
                    "ci_low": float(res.ci_low),
                    "ci_high": float(res.ci_high),
                }
            )
            diag_examples.append((ns, diag))

        df = pd.DataFrame(rows).sort_values("n_steps")
        st.markdown("### Price vs n_steps")
        st.dataframe(df, use_container_width=True)
        st.line_chart(df.set_index("n_steps")[["price"]])

        ns_max, diag = sorted(diag_examples, key=lambda x: x[0])[-1]
        st.markdown(f"### Detailed regression diagnostics (n_steps={ns_max})")

        df_diag = pd.DataFrame(
            {"k": diag["time_index"], "n_itm": diag["n_itm"], "regression_r2": diag["regression_r2"]}
        )
        st.dataframe(df_diag, use_container_width=True)
        st.line_chart(df_diag.set_index("k")[["n_itm"]])
        st.line_chart(df_diag.set_index("k")[["regression_r2"]])

st.caption("Core is UI-agnostic: you can later plug xlwings by importing the same core modules.")
