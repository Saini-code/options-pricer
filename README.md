# options-pricer

A Python options pricing and risk engine, built from first principles.

This project implements option pricing models (starting with Black-Scholes-Merton),
Greeks, implied volatility solvers, and delta-hedging simulation without relying on
third-party options pricing libraries — all math is derived and coded from scratch,
with results cross-checked against known reference values in the test suite.

## Structure

```
src/options_pricer/
    black_scholes.py   # Black-Scholes-Merton pricing formulas
    greeks.py           # Option Greeks (delta, gamma, vega, theta, rho)
    implied_vol.py       # Implied volatility solvers
    data.py               # Market data retrieval and preparation
    hedging.py             # Delta hedging and portfolio risk simulation
tests/                       # pytest unit tests, one per pricing/risk function
notebooks/                     # exploratory analysis notebooks
plots/                           # generated PNG plots (150dpi, labelled axes)
```

See [CLAUDE.md](CLAUDE.md) for the coding and testing rules this repo follows.

## Setup

```bash
pip install -r requirements.txt
```

## Testing

```bash
pytest
```
