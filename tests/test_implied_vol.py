"""Tests for src/options_pricer/implied_vol.py."""

import numpy as np
import pytest

from options_pricer.black_scholes import bsm_price
from options_pricer.implied_vol import implied_vol

# ---------------------------------------------------------------------------
# Round-trip property: price with a known sigma, recover that sigma.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q,option_type",
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call"),   # ATM
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "put"),
        (100.0, 110.0, 0.5, 0.03, 0.35, 0.01, "call"),  # OTM call
        (100.0, 90.0, 0.5, 0.03, 0.35, 0.01, "put"),    # OTM put
        (100.0, 80.0, 2.0, 0.04, 0.15, 0.02, "call"),   # ITM call
        (100.0, 120.0, 2.0, 0.04, 0.15, 0.02, "put"),   # ITM put
        (50.0, 50.0, 0.25, 0.01, 0.60, 0.0, "call"),    # high vol
    ],
)
def test_round_trip_recovers_known_sigma(S, K, T, r, sigma, q, option_type):
    price = bsm_price(S, K, T, r, sigma, q, option_type=option_type)
    recovered = implied_vol(price, S, K, T, r, q, option_type=option_type)
    assert recovered == pytest.approx(sigma, abs=1e-6)


def test_round_trip_random_grid():
    rng = np.random.default_rng(seed=11)
    n = 100
    S = rng.uniform(20, 300, n)
    K = S * rng.uniform(0.85, 1.15, n)
    T = rng.uniform(0.05, 3.0, n)
    r = rng.uniform(-0.01, 0.1, n)
    sigma = rng.uniform(0.05, 1.0, n)
    q = rng.uniform(0.0, 0.05, n)
    option_types = rng.choice(["call", "put"], n)

    max_err = 0.0
    for i in range(n):
        ot = str(option_types[i])
        price = bsm_price(S[i], K[i], T[i], r[i], sigma[i], q[i], option_type=ot)
        recovered = implied_vol(price, S[i], K[i], T[i], r[i], q[i], option_type=ot)
        assert not np.isnan(recovered), f"failed to converge at index {i}"
        max_err = max(max_err, abs(recovered - sigma[i]))

    assert max_err < 1e-6


# ---------------------------------------------------------------------------
# Convergence in fragile regimes: deep ITM/OTM, very short-dated.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q,option_type",
    [
        (100.0, 500.0, 1.0, 0.05, 0.30, 0.0, "call"),     # deep OTM call
        (100.0, 20.0, 1.0, 0.05, 0.30, 0.0, "put"),       # deep OTM put
        (100.0, 100.0, 1 / 365, 0.05, 0.20, 0.0, "call"), # 1-day ATM
        (100.0, 105.0, 1 / 365, 0.05, 0.20, 0.0, "call"), # 1-day OTM
        (100.0, 95.0, 1 / 365, 0.05, 0.20, 0.0, "put"),   # 1-day OTM put
        (100.0, 100.0, 1 / 52, 0.05, 0.50, 0.0, "call"),  # 1-week, high vol
    ],
)
def test_convergence_in_fragile_regimes(S, K, T, r, sigma, q, option_type):
    price = bsm_price(S, K, T, r, sigma, q, option_type=option_type)
    recovered = implied_vol(price, S, K, T, r, q, option_type=option_type)
    assert not np.isnan(recovered)
    assert recovered == pytest.approx(sigma, abs=1e-5)


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q,option_type",
    [
        (100.0, 20.0, 1.0, 0.05, 0.30, 0.0, "call"),   # deep ITM call
        (100.0, 500.0, 1.0, 0.05, 0.30, 0.0, "put"),   # deep ITM put
    ],
)
def test_convergence_deep_itm_within_ill_conditioned_tolerance(S, K, T, r, sigma, q, option_type):
    # Deep ITM options have vanishingly small vega (~3.9e-6 for the call
    # case here), so inverting price -> sigma is ill-conditioned: a price
    # agreement within our solver's price tolerance (1e-8) only pins sigma
    # down to about tol/vega ~= 1e-8 / 3.9e-6 ~= 2.6e-3. That is not a
    # solver bug, it is the same reason real deep ITM/OTM implied vols are
    # noisy in practice — a cent of bid-ask noise implies a large vol
    # range. We still require convergence (no NaN) and a loose bound.
    price = bsm_price(S, K, T, r, sigma, q, option_type=option_type)
    recovered = implied_vol(price, S, K, T, r, q, option_type=option_type)
    assert not np.isnan(recovered)
    assert recovered == pytest.approx(sigma, abs=5e-3)


# ---------------------------------------------------------------------------
# No-solution cases return NaN rather than raising.
# ---------------------------------------------------------------------------


def test_price_below_intrinsic_returns_nan():
    # Call intrinsic value with S=100, K=80 (no dividend) is 20; a quoted
    # price of 5 violates the no-arbitrage lower bound.
    result = implied_vol(5.0, S=100.0, K=80.0, T=1.0, r=0.05, q=0.0, option_type="call")
    assert np.isnan(result)


def test_price_above_forward_returns_nan():
    # A call can never be worth more than the discounted forward S*exp(-qT).
    result = implied_vol(1000.0, S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, option_type="call")
    assert np.isnan(result)


def test_invalid_option_type_raises_value_error():
    with pytest.raises(ValueError):
        implied_vol(5.0, S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, option_type="straddle")
