"""Tests for src/options_pricer/greeks.py."""

import numpy as np
import pytest

from options_pricer.black_scholes import bsm_price
from options_pricer.greeks import delta, gamma, numerical_greek, rho, theta, vega

# Shared base scenario for finite-difference cross-checks.
BASE = dict(S=100.0, K=95.0, T=0.75, r=0.03, sigma=0.25, q=0.02)


def _base(option_type: str) -> dict:
    return dict(BASE, option_type=option_type)


# ---------------------------------------------------------------------------
# Closed-form Greeks vs finite-difference bump-and-reprice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    closed_form = delta(**_base(option_type))
    numeric = numerical_greek(bsm_price, "S", bump=1e-3, **_base(option_type))
    assert closed_form == pytest.approx(numeric, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type):
    closed_form = vega(**BASE)
    numeric = numerical_greek(bsm_price, "sigma", bump=1e-4, **_base(option_type))
    assert closed_form == pytest.approx(numeric, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type):
    # Theta is defined as -dV/dT (value change as calendar time passes,
    # i.e. as T decreases), so it is the negative of the raw dV/dT bump.
    closed_form = theta(**_base(option_type))
    numeric_dV_dT = numerical_greek(bsm_price, "T", bump=1e-4, **_base(option_type))
    assert closed_form == pytest.approx(-numeric_dV_dT, abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    closed_form = rho(**_base(option_type))
    numeric = numerical_greek(bsm_price, "r", bump=1e-4, **_base(option_type))
    assert closed_form == pytest.approx(numeric, abs=1e-4)


def _numerical_gamma(bump: float = 0.05, **kwargs) -> float:
    """Second-order central difference: (C(S+h) - 2C(S) + C(S-h)) / h^2."""
    S = kwargs.pop("S")
    up = bsm_price(S=S + bump, **kwargs)
    mid = bsm_price(S=S, **kwargs)
    down = bsm_price(S=S - bump, **kwargs)
    return (up - 2.0 * mid + down) / bump**2


def test_gamma_matches_finite_difference():
    kwargs = {k: v for k, v in BASE.items() if k != "S"}
    closed_form = gamma(**BASE)
    numeric = _numerical_gamma(S=BASE["S"], option_type="call", **kwargs)
    assert closed_form == pytest.approx(numeric, abs=1e-4)


# ---------------------------------------------------------------------------
# Sign / bound sanity checks over a random grid.
# ---------------------------------------------------------------------------


def test_delta_bounds():
    rng = np.random.default_rng(seed=1)
    n = 100
    S = rng.uniform(1, 500, n)
    K = rng.uniform(1, 500, n)
    T = rng.uniform(0.01, 5, n)
    r = rng.uniform(-0.02, 0.15, n)
    sigma = rng.uniform(0.01, 2.0, n)
    q = rng.uniform(0.0, 0.10, n)

    call_delta = delta(S, K, T, r, sigma, q, option_type="call")
    put_delta = delta(S, K, T, r, sigma, q, option_type="put")

    assert np.all(call_delta >= 0.0) and np.all(call_delta <= 1.0)
    assert np.all(put_delta >= -1.0) and np.all(put_delta <= 0.0)


def test_gamma_and_vega_are_positive_for_long_options():
    # K is drawn relative to S (realistic moneyness) rather than independently
    # over the full range: with S and K fully independent, some random draws
    # push d1 far enough into the tail that phi(d1) underflows to exactly 0.0
    # in float64 (true gamma/vega there are positive but below float64
    # precision) — a numerical artefact of the test grid, not a formula bug.
    rng = np.random.default_rng(seed=2)
    n = 100
    S = rng.uniform(10, 500, n)
    K = S * rng.uniform(0.5, 1.5, n)
    T = rng.uniform(0.01, 5, n)
    r = rng.uniform(-0.02, 0.15, n)
    sigma = rng.uniform(0.05, 2.0, n)
    q = rng.uniform(0.0, 0.10, n)

    assert np.all(gamma(S, K, T, r, sigma, q) > 0.0)
    assert np.all(vega(S, K, T, r, sigma, q) > 0.0)


# ---------------------------------------------------------------------------
# Gamma and vega identity between call and put at the same strike.
#
# This must hold because of put-call parity: C - P = S*exp(-qT) - K*exp(-rT).
# The right-hand side is linear in S (so its second derivative w.r.t. S is
# zero => gamma_call = gamma_put) and does not depend on sigma at all
# (so d(C-P)/dsigma = 0 => vega_call = vega_put). We verify this directly
# by bump-and-reprice on the call price and the put price independently,
# rather than relying on the closed-form functions sharing one code path.
# ---------------------------------------------------------------------------


def test_gamma_identical_for_call_and_put():
    kwargs = {k: v for k, v in BASE.items() if k != "S"}
    numeric_gamma_call = _numerical_gamma(S=BASE["S"], option_type="call", **kwargs)
    numeric_gamma_put = _numerical_gamma(S=BASE["S"], option_type="put", **kwargs)
    assert numeric_gamma_call == pytest.approx(numeric_gamma_put, abs=1e-6)
    assert numeric_gamma_call == pytest.approx(gamma(**BASE), abs=1e-4)


def test_vega_identical_for_call_and_put():
    numeric_vega_call = numerical_greek(bsm_price, "sigma", bump=1e-4, **_base("call"))
    numeric_vega_put = numerical_greek(bsm_price, "sigma", bump=1e-4, **_base("put"))
    assert numeric_vega_call == pytest.approx(numeric_vega_put, abs=1e-6)
    assert numeric_vega_call == pytest.approx(vega(**BASE), abs=1e-4)


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_invalid_option_type_raises_value_error():
    with pytest.raises(ValueError):
        delta(**dict(BASE, option_type="straddle"))
    with pytest.raises(ValueError):
        theta(**dict(BASE, option_type="straddle"))
    with pytest.raises(ValueError):
        rho(**dict(BASE, option_type="straddle"))


def test_numerical_greek_rejects_unknown_param():
    with pytest.raises(ValueError):
        numerical_greek(bsm_price, "not_a_param", bump=1e-4, **_base("call"))
