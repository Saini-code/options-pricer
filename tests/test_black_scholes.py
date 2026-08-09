"""Tests for src/options_pricer/black_scholes.py.

py_vollib is used here only as an independent cross-check library, per
CLAUDE.md — it must never be imported in src/.
"""

import numpy as np
import pytest
from py_vollib.black_scholes import black_scholes as vollib_black_scholes

from options_pricer.black_scholes import bsm_price, d1, d2


# ---------------------------------------------------------------------------
# Put-call parity: C - P = S*exp(-qT) - K*exp(-rT), for any valid inputs.
# ---------------------------------------------------------------------------


def test_put_call_parity_random_grid():
    rng = np.random.default_rng(seed=42)
    n = 200
    S = rng.uniform(1, 500, n)
    K = rng.uniform(1, 500, n)
    T = rng.uniform(0.01, 5, n)
    r = rng.uniform(-0.02, 0.15, n)
    sigma = rng.uniform(0.01, 2.0, n)
    q = rng.uniform(0.0, 0.10, n)

    call = bsm_price(S, K, T, r, sigma, q, option_type="call")
    put = bsm_price(S, K, T, r, sigma, q, option_type="put")

    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    np.testing.assert_allclose(lhs, rhs, atol=1e-8)


# ---------------------------------------------------------------------------
# Deep ITM / deep OTM limits.
# ---------------------------------------------------------------------------


def test_deep_itm_call_approaches_discounted_forward_minus_strike():
    S, K, T, r, sigma, q = 100_000.0, 100.0, 1.0, 0.05, 0.20, 0.02
    call = bsm_price(S, K, T, r, sigma, q, option_type="call")
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert call == pytest.approx(expected, rel=1e-6)


def test_deep_otm_call_approaches_zero():
    S, K, T, r, sigma, q = 0.0001, 100.0, 1.0, 0.05, 0.20, 0.02
    call = bsm_price(S, K, T, r, sigma, q, option_type="call")
    assert call == pytest.approx(0.0, abs=1e-6)


def test_deep_itm_put_approaches_discounted_strike_minus_forward():
    S, K, T, r, sigma, q = 100.0, 100_000.0, 1.0, 0.05, 0.20, 0.02
    put = bsm_price(S, K, T, r, sigma, q, option_type="put")
    expected = K * np.exp(-r * T) - S * np.exp(-q * T)
    assert put == pytest.approx(expected, rel=1e-6)


def test_deep_otm_put_approaches_zero():
    S, K, T, r, sigma, q = 100_000.0, 100.0, 1.0, 0.05, 0.20, 0.02
    put = bsm_price(S, K, T, r, sigma, q, option_type="put")
    assert put == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Hardcoded reference prices from published examples.
# ---------------------------------------------------------------------------


def test_reference_price_hull_example():
    # Hull, J.C., "Options, Futures, and Other Derivatives", 7th ed. (2009), p.294.
    # Non-dividend-paying stock: S0=42, K=40, r=10%, sigma=20%, T=0.5 years.
    # Published: c = 4.76, p = 0.81.
    call = bsm_price(42, 40, 0.5, 0.10, 0.20, option_type="call")
    put = bsm_price(42, 40, 0.5, 0.10, 0.20, option_type="put")
    assert call == pytest.approx(4.76, abs=0.01)
    assert put == pytest.approx(0.81, abs=0.01)


def test_reference_price_atm_benchmark():
    # Widely reproduced at-the-money benchmark (e.g. blackscholes-calculator.com
    # step-by-step tutorial): S=K=100, r=5%, sigma=20%, T=1 year, no dividend.
    # Published: d1=0.35, d2=0.15, call = 10.4506.
    call = bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="call")
    assert call == pytest.approx(10.4506, abs=1e-4)


def test_reference_price_mathworks_symbolic_example():
    # MathWorks Symbolic Math Toolbox documentation, "The Black-Scholes
    # Formula for Call Option Price": S=100, K=95, r=1%, sigma=50%, T=0.25
    # years, no dividend. Published: "the price of the option to 6
    # significant digits is $12.5279" (computed via exact symbolic
    # integration, i.e. an independent derivation of the closed-form price).
    call = bsm_price(100, 95, 0.25, 0.01, 0.50, option_type="call")
    assert call == pytest.approx(12.5279, abs=1e-4)


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_zero_time_to_expiry_returns_intrinsic_value():
    assert bsm_price(110, 100, 0.0, 0.05, 0.20, option_type="call") == pytest.approx(10.0)
    assert bsm_price(90, 100, 0.0, 0.05, 0.20, option_type="call") == pytest.approx(0.0)
    assert bsm_price(90, 100, 0.0, 0.05, 0.20, option_type="put") == pytest.approx(10.0)
    assert bsm_price(110, 100, 0.0, 0.05, 0.20, option_type="put") == pytest.approx(0.0)


def test_zero_volatility_returns_discounted_forward_payoff():
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
    call = bsm_price(S, K, T, r, 0.0, q, option_type="call")
    expected = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    assert call == pytest.approx(expected)


@pytest.mark.parametrize("bad_kwargs", [
    dict(S=-1, K=100, T=1, r=0.05, sigma=0.2),
    dict(S=100, K=-1, T=1, r=0.05, sigma=0.2),
    dict(S=100, K=100, T=-1, r=0.05, sigma=0.2),
    dict(S=100, K=100, T=1, r=0.05, sigma=-0.2),
    dict(S=0, K=100, T=1, r=0.05, sigma=0.2),   # S must be strictly positive
    dict(S=100, K=0, T=1, r=0.05, sigma=0.2),   # K must be strictly positive
    dict(S=0, K=0, T=1, r=0.05, sigma=0.2),     # would silently NaN via log(0/0) if unguarded
])
def test_negative_inputs_raise_value_error(bad_kwargs):
    with pytest.raises(ValueError):
        bsm_price(**bad_kwargs)


def test_invalid_option_type_raises_value_error():
    with pytest.raises(ValueError):
        bsm_price(100, 100, 1, 0.05, 0.2, option_type="straddle")


# ---------------------------------------------------------------------------
# Vectorisation.
# ---------------------------------------------------------------------------


def test_accepts_array_inputs_and_broadcasts():
    S = np.array([90.0, 100.0, 110.0])
    K = 100.0
    T = np.array([0.5, 1.0, 1.5])
    prices = bsm_price(S, K, T, r=0.05, sigma=0.2, option_type="call")
    assert prices.shape == (3,)
    for i in range(3):
        assert prices[i] == pytest.approx(
            bsm_price(S[i], K, T[i], r=0.05, sigma=0.2, option_type="call")
        )


# ---------------------------------------------------------------------------
# d1 / d2 helpers.
# ---------------------------------------------------------------------------


def test_d2_equals_d1_minus_sigma_sqrt_T():
    S, K, T, r, sigma, q = 100.0, 95.0, 0.75, 0.03, 0.25, 0.01
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)
    assert d2_val == pytest.approx(d1_val - sigma * np.sqrt(T))


# ---------------------------------------------------------------------------
# Independent cross-check against py_vollib (tests/ only, per CLAUDE.md).
# ---------------------------------------------------------------------------


def test_matches_py_vollib_random_grid():
    rng = np.random.default_rng(seed=7)
    n = 100
    S = rng.uniform(1, 500, n)
    K = rng.uniform(1, 500, n)
    T = rng.uniform(0.01, 5, n)
    r = rng.uniform(-0.02, 0.15, n)
    sigma = rng.uniform(0.01, 2.0, n)

    our_calls = bsm_price(S, K, T, r, sigma, option_type="call")
    our_puts = bsm_price(S, K, T, r, sigma, option_type="put")

    for i in range(n):
        vollib_call = vollib_black_scholes("c", S[i], K[i], T[i], r[i], sigma[i])
        vollib_put = vollib_black_scholes("p", S[i], K[i], T[i], r[i], sigma[i])
        assert our_calls[i] == pytest.approx(vollib_call, abs=1e-6)
        assert our_puts[i] == pytest.approx(vollib_put, abs=1e-6)
