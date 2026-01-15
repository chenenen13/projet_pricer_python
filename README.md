# LSM Monte Carlo Pricer (Core + Streamlit)

This repo contains a clean, modular core for:
- European option pricing by Monte Carlo (scalar + vectorized)
- Variance reduction (antithetic)
- Confidence intervals
- American option pricing by Longstaff–Schwartz (LSM)

The **core is UI-agnostic**: it returns plain Python objects / NumPy arrays / pandas DataFrames,
so you can plug it into Streamlit now and xlwings later without touching pricing logic.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Structure

- `core/market.py` : Market environment (r, q, daycount)
- `core/models.py` : GBM model + path simulation (vectorized)
- `core/options.py`: Option contracts (European / American, call/put)
- `core/regression.py`: Regression bases (poly, Laguerre) + OLS
- `core/mc_engine.py`: Monte Carlo pricing engines (Euro + LSM)
- `core/results.py`: Result dataclasses + helpers
- `app_streamlit.py`: Minimal Streamlit UI
- `main_cli.py`: CLI example run (useful for tests)

## Notes

- American pricing is Bermudan (discrete exercise dates). Increase `n_steps` to approximate continuous American.
- Default LSM basis: polynomial degree 2 (quadratic), consistent with the numerical example slides.
