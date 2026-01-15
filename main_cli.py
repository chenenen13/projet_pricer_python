# main_cli.py
import time
from datetime import date

import matplotlib.pyplot as plt

from core import Market, GBMModel
from core.models import GBMParams
from core.options import EuropeanOption, AmericanOption, OptionType
from core.mc_engine import EuropeanMCPricer, LongstaffSchwartzPricer
from core.regression import PolyBasis
from core.dates import DayCount

from utils import (
    print_block,
    fmt_price_result,
    black_scholes_price,
    mc_convergence_rows,
    print_convergence_table,
    plot_convergence_compare,
    plot_se_compare,
    print_variance_reduction,
)

# NEW: complementary studies
from etudes_convergence import (
    study_price_distribution_over_seeds,
    plot_seed_distribution,
    study_error_vs_N,
    plot_error_vs_N,
    plot_scaled_error_vs_N,
)


def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


def main():
    # -------------------------
    # Global parameters
    # -------------------------
    S0 = 100.0
    K = 100.0
    sigma = 0.20
    r = 0.03
    seed = 1
    antithetic = True

    # Dates (instead of T)
    pricing_date = date(2026, 1, 15)
    maturity_date = date(2027, 1, 15)
    day_count = DayCount.ACT_365F

    # LSM settings
    steps = 50
    basis = PolyBasis(degree=2)

    # Convergence grid (existing)
    N_list = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
    N_big = N_list[-1]

    # NEW studies parameters
    seed_study_n_seeds = 100
    seed_study_seed0 = 1
    seed_study_N = 200_000  # fixed N for distribution across seeds

    error_study_N_grid = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
    error_study_seed = 1  # fixed seed for error vs N

    # -------------------------
    # PART A — European: MC vs BS + improvement via Control Variate(ST)
    # -------------------------
    for q in [0.00, 0.05]:
        print_block(f"EUROPEAN PRICING: MC convergence vs Black-Scholes (q={q:.2%})")

        market = Market(r=r, q=q)
        model = GBMModel(market, GBMParams(s0=S0, sigma=sigma))
        pricer = EuropeanMCPricer(model)

        euro_call = EuropeanOption(
            option_type=OptionType.CALL,
            strike=K,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            day_count=day_count,
        )
        euro_put = EuropeanOption(
            option_type=OptionType.PUT,
            strike=K,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            day_count=day_count,
        )

        # Use computed maturity (float years)
        T = float(euro_call.maturity_years())

        bs_call = black_scholes_price(S0, K, T, r, q, sigma, OptionType.CALL)
        bs_put = black_scholes_price(S0, K, T, r, q, sigma, OptionType.PUT)

        rows_call_plain = mc_convergence_rows(
            pricer, euro_call, N_list, seed=seed, antithetic=antithetic, control_variate=False
        )
        rows_put_plain = mc_convergence_rows(
            pricer, euro_put, N_list, seed=seed, antithetic=antithetic, control_variate=False
        )

        rows_call_cv = mc_convergence_rows(
            pricer, euro_call, N_list, seed=seed, antithetic=antithetic, control_variate=True
        )
        rows_put_cv = mc_convergence_rows(
            pricer, euro_put, N_list, seed=seed, antithetic=antithetic, control_variate=True
        )

        print("\n--- CALL (plain) ---")
        print_convergence_table(rows_call_plain, bs_call, label="CALL:")
        print("\n--- CALL (control variate ST) ---")
        print_convergence_table(rows_call_cv, bs_call, label="CALL:")

        print("\n--- PUT (plain) ---")
        print_convergence_table(rows_put_plain, bs_put, label="PUT:")
        print("\n--- PUT (control variate ST) ---")
        print_convergence_table(rows_put_cv, bs_put, label="PUT:")

        print_variance_reduction(rows_call_plain, rows_call_cv, "CALL")
        print_variance_reduction(rows_put_plain, rows_put_cv, "PUT")

        plot_convergence_compare(
            rows_call_plain, rows_call_cv, bs_call,
            title=f"European CALL — MC vs Control Variate(ST) vs BS (q={q:.0%})"
        )
        plot_convergence_compare(
            rows_put_plain, rows_put_cv, bs_put,
            title=f"European PUT — MC vs Control Variate(ST) vs BS (q={q:.0%})"
        )

        plot_se_compare(
            rows_call_plain, rows_call_cv,
            title=f"European CALL — Standard error vs N (q={q:.0%})"
        )
        plot_se_compare(
            rows_put_plain, rows_put_cv,
            title=f"European PUT — Standard error vs N (q={q:.0%})"
        )

        # -------------------------
        # Performance benchmark (European)
        # -------------------------
        print("\nPERFORMANCE BENCHMARK (European CALL)")
        n_vec = N_big
        n_scalar = 50_000  # keep reasonable for CLI

        res_vec, t_vec = _time_call(
            pricer.price_vector,
            euro_call,
            n_paths=n_vec,
            seed=seed,
            antithetic=antithetic,
            control_variate=True,
        )

        res_sca, t_sca = _time_call(
            pricer.price_scalar,
            euro_call,
            n_paths=n_scalar,
            seed=seed,
            antithetic=antithetic,
        )

        t_sca_est = t_sca * (n_vec / max(n_scalar, 1))
        speedup_est = t_sca_est / max(t_vec, 1e-12)

        print("Vectorized:", fmt_price_result(res_vec), f"| time={t_vec:.4f}s")
        print("Scalar    :", fmt_price_result(res_sca), f"| time={t_sca:.4f}s (for N={n_scalar:,})")
        print(f"Estimated speed-up @ N={n_vec:,}: {speedup_est:.2f}x (scalar time scaled to same N)")

        # -------------------------
        # NEW PART A.1 — Complementary studies (European)
        # -------------------------
        print_block(f"COMPLEMENTARY STUDIES (European, q={q:.2%})")

        # Study 1: Distribution over many seeds, fixed N
        print_block(f"STUDY 1 — Distribution over seeds (fixed N={seed_study_N:,}, n_seeds={seed_study_n_seeds})")

        # Plain
        res_seed_plain = study_price_distribution_over_seeds(
            pricer=pricer,
            option=euro_call,
            N=int(seed_study_N),
            n_seeds=int(seed_study_n_seeds),
            seed0=int(seed_study_seed0),
            antithetic=bool(antithetic),
            control_variate=False,
            bs_price=float(bs_call),
        )
        print("CALL plain | BS =", f"{bs_call:.6f}",
              "| mean =", f"{res_seed_plain.mean:.6f}",
              "| std(seeds) =", f"{res_seed_plain.std:.6f}",
              "| q05/median/q95 =", f"{res_seed_plain.q05:.6f}/{res_seed_plain.q50:.6f}/{res_seed_plain.q95:.6f}")

        plot_seed_distribution(res_seed_plain, bins=25, title=f"Seed distribution — CALL plain (q={q:.0%}, N={seed_study_N:,})")

        # CV(ST)
        res_seed_cv = study_price_distribution_over_seeds(
            pricer=pricer,
            option=euro_call,
            N=int(seed_study_N),
            n_seeds=int(seed_study_n_seeds),
            seed0=int(seed_study_seed0),
            antithetic=bool(antithetic),
            control_variate=True,
            bs_price=float(bs_call),
        )
        print("CALL CV(ST) | BS =", f"{bs_call:.6f}",
              "| mean =", f"{res_seed_cv.mean:.6f}",
              "| std(seeds) =", f"{res_seed_cv.std:.6f}",
              "| q05/median/q95 =", f"{res_seed_cv.q05:.6f}/{res_seed_cv.q50:.6f}/{res_seed_cv.q95:.6f}")

        plot_seed_distribution(res_seed_cv, bins=25, title=f"Seed distribution — CALL CV(ST) (q={q:.0%}, N={seed_study_N:,})")

        # Study 2: Error vs N and sqrt(N) scaling (fixed seed)
        print_block("STUDY 2 — Error |MC - BS| vs N and sqrt(N) scaling (fixed seed)")

        # Plain
        resN_plain = study_error_vs_N(
            pricer=pricer,
            option=euro_call,
            N_grid=error_study_N_grid,
            seed=int(error_study_seed),
            antithetic=bool(antithetic),
            control_variate=False,
            bs_price=float(bs_call),
        )
        plot_error_vs_N(resN_plain, title=f"|MC-BS| vs N — CALL plain (seed={error_study_seed}, q={q:.0%})")
        plot_scaled_error_vs_N(resN_plain, title=f"|MC-BS|*sqrt(N) vs N — CALL plain (seed={error_study_seed}, q={q:.0%})")

        # CV(ST)
        resN_cv = study_error_vs_N(
            pricer=pricer,
            option=euro_call,
            N_grid=error_study_N_grid,
            seed=int(error_study_seed),
            antithetic=bool(antithetic),
            control_variate=True,
            bs_price=float(bs_call),
        )
        plot_error_vs_N(resN_cv, title=f"|MC-BS| vs N — CALL CV(ST) (seed={error_study_seed}, q={q:.0%})")
        plot_scaled_error_vs_N(resN_cv, title=f"|MC-BS|*sqrt(N) vs N — CALL CV(ST) (seed={error_study_seed}, q={q:.0%})")

    # -------------------------
    # PART B — American: coherence tests (LSM)
    # -------------------------
    print_block("AMERICAN COHERENCE TESTS (LSM)")

    market_q0 = Market(r=r, q=0.0)
    model_q0 = GBMModel(market_q0, GBMParams(s0=S0, sigma=sigma))
    euro_pricer = EuropeanMCPricer(model_q0)
    lsm_pricer = LongstaffSchwartzPricer(model_q0, basis=basis)

    euro_call = EuropeanOption(OptionType.CALL, K, pricing_date=pricing_date, maturity_date=maturity_date, day_count=day_count)
    am_call = AmericanOption(OptionType.CALL, K, pricing_date=pricing_date, maturity_date=maturity_date, day_count=day_count)

    res_euro_call = euro_pricer.price_vector(euro_call, n_paths=N_big, seed=seed, antithetic=antithetic, control_variate=True)
    res_am_call, t_lsm_call = _time_call(lsm_pricer.price, am_call, n_paths=N_big, n_steps=steps, seed=seed, antithetic=antithetic)

    print("\nNo dividends (q=0): CALL")
    print("European CALL (with CV):", fmt_price_result(res_euro_call))
    print("American  CALL:", fmt_price_result(res_am_call))
    print(f"Difference (American - European) = {res_am_call.price - res_euro_call.price:.6f} (should be ~0)")
    print(f"LSM runtime (CALL) = {t_lsm_call:.4f}s")

    euro_put = EuropeanOption(OptionType.PUT, K, pricing_date=pricing_date, maturity_date=maturity_date, day_count=day_count)
    am_put = AmericanOption(OptionType.PUT, K, pricing_date=pricing_date, maturity_date=maturity_date, day_count=day_count)

    res_euro_put = euro_pricer.price_vector(euro_put, n_paths=N_big, seed=seed, antithetic=antithetic, control_variate=True)
    res_am_put, t_lsm_put = _time_call(lsm_pricer.price, am_put, n_paths=N_big, n_steps=steps, seed=seed, antithetic=antithetic)

    print("\nNo dividends (q=0): PUT")
    print("European PUT (with CV):", fmt_price_result(res_euro_put))
    print("American  PUT:", fmt_price_result(res_am_put))
    print(f"Premium (American - European) = {res_am_put.price - res_euro_put.price:.6f} (should be >= 0)")
    print(f"LSM runtime (PUT) = {t_lsm_put:.4f}s")

    q = 0.05
    market_q = Market(r=r, q=q)
    model_q = GBMModel(market_q, GBMParams(s0=S0, sigma=sigma))
    euro_pricer_q = EuropeanMCPricer(model_q)
    lsm_pricer_q = LongstaffSchwartzPricer(model_q, basis=basis)

    euro_call_q = EuropeanOption(OptionType.CALL, K, pricing_date=pricing_date, maturity_date=maturity_date, day_count=day_count)
    am_call_q = AmericanOption(OptionType.CALL, K, pricing_date=pricing_date, maturity_date=maturity_date, day_count=day_count)

    res_euro_call_q = euro_pricer_q.price_vector(euro_call_q, n_paths=N_big, seed=seed, antithetic=antithetic, control_variate=True)
    res_am_call_q = lsm_pricer_q.price(am_call_q, n_paths=N_big, n_steps=steps, seed=seed, antithetic=antithetic)

    print("\nWith dividends (q=5%): CALL")
    print("European CALL (with CV):", fmt_price_result(res_euro_call_q))
    print("American  CALL:", fmt_price_result(res_am_call_q))
    print(f"Premium (American - European) = {res_am_call_q.price - res_euro_call_q.price:.6f} (should be >= 0)")

    plt.show()


if __name__ == "__main__":
    main()
