"""Tests for src/options_pricer/hedging.py."""

import numpy as np
import pytest

from options_pricer.greeks import gamma
from options_pricer.hedging import delta_hedge_pnl, run_hedge_simulation, simulate_gbm_paths

S0, K, T, R, Q = 100.0, 100.0, 0.25, 0.05, 0.0
SIGMA = 0.20


# ---------------------------------------------------------------------------
# simulate_gbm_paths
# ---------------------------------------------------------------------------


def test_paths_start_at_S0_and_have_correct_shape():
    paths = simulate_gbm_paths(S0, R, SIGMA, T, n_steps=50, n_paths=100, seed=0)
    assert paths.shape == (100, 51)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_n_steps_below_one_raises_value_error():
    with pytest.raises(ValueError):
        simulate_gbm_paths(S0, R, SIGMA, T, n_steps=0, n_paths=100, seed=0)


def test_seed_reproducibility():
    p1 = simulate_gbm_paths(S0, R, SIGMA, T, n_steps=50, n_paths=100, seed=42)
    p2 = simulate_gbm_paths(S0, R, SIGMA, T, n_steps=50, n_paths=100, seed=42)
    np.testing.assert_array_equal(p1, p2)


def test_different_seeds_give_different_paths():
    p1 = simulate_gbm_paths(S0, R, SIGMA, T, n_steps=50, n_paths=100, seed=1)
    p2 = simulate_gbm_paths(S0, R, SIGMA, T, n_steps=50, n_paths=100, seed=2)
    assert not np.allclose(p1, p2)


def test_log_return_moments_match_gbm_theory():
    # E[ln(S_T/S0)] = (mu - sigma^2/2)*T, Var[ln(S_T/S0)] = sigma^2*T.
    n_paths = 200_000
    paths = simulate_gbm_paths(S0, R, SIGMA, T, n_steps=50, n_paths=n_paths, seed=7)
    log_returns = np.log(paths[:, -1] / S0)
    expected_mean = (R - 0.5 * SIGMA**2) * T
    expected_var = SIGMA**2 * T
    se_mean = np.sqrt(expected_var / n_paths)
    assert log_returns.mean() == pytest.approx(expected_mean, abs=5 * se_mean)
    assert log_returns.var() == pytest.approx(expected_var, rel=0.02)


# ---------------------------------------------------------------------------
# delta_hedge_pnl / run_hedge_simulation: core statistical properties.
# ---------------------------------------------------------------------------


def test_reproducible_given_seed():
    pnl1 = run_hedge_simulation(S0, K, T, R, SIGMA, SIGMA, n_steps=50, n_paths=500, seed=123)
    pnl2 = run_hedge_simulation(S0, K, T, R, SIGMA, SIGMA, n_steps=50, n_paths=500, seed=123)
    np.testing.assert_array_equal(pnl1, pnl2)


def test_matched_vol_zero_cost_pnl_is_approximately_unbiased():
    # When sigma_realised == sigma_hedge and there are no transaction
    # costs, delta-hedging replicates the option, so mean P&L should be
    # close to zero (up to Monte Carlo noise and discrete-rebalancing bias).
    pnl = run_hedge_simulation(S0, K, T, R, SIGMA, SIGMA, n_steps=252, n_paths=20_000, seed=1)
    se = pnl.std() / np.sqrt(len(pnl))
    assert abs(pnl.mean()) < 5 * se


def test_more_frequent_rebalancing_tightens_pnl_distribution():
    stds = []
    for n_steps in (4, 21, 63, 252):
        pnl = run_hedge_simulation(S0, K, T, R, SIGMA, SIGMA, n_steps=n_steps, n_paths=10_000, seed=2)
        stds.append(pnl.std())
    assert stds == sorted(stds, reverse=True)


def test_realised_vol_above_hedge_vol_loses_money_on_average():
    pnl = run_hedge_simulation(S0, K, T, R, SIGMA, sigma_realised=0.35, n_steps=252, n_paths=20_000, seed=3)
    se = pnl.std() / np.sqrt(len(pnl))
    assert pnl.mean() < -5 * se


def test_realised_vol_below_hedge_vol_makes_money_on_average():
    pnl = run_hedge_simulation(S0, K, T, R, SIGMA, sigma_realised=0.10, n_steps=252, n_paths=20_000, seed=4)
    se = pnl.std() / np.sqrt(len(pnl))
    assert pnl.mean() > 5 * se


def test_pnl_monotonically_decreasing_in_realised_vol():
    # Mirrors the gamma P&L relationship: P&L ~= -0.5*Gamma*S^2*(sigma_r^2 - sigma_h^2)*dt,
    # which is monotonically decreasing in sigma_realised.
    means = []
    for sigma_realised in (0.10, 0.15, 0.20, 0.25, 0.35):
        pnl = run_hedge_simulation(
            S0, K, T, R, SIGMA, sigma_realised, n_steps=252, n_paths=10_000, seed=5
        )
        means.append(pnl.mean())
    assert means == sorted(means, reverse=True)


def test_transaction_costs_reduce_mean_pnl():
    means = []
    for cost_bps in (0.0, 5.0, 20.0):
        pnl = run_hedge_simulation(
            S0, K, T, R, SIGMA, SIGMA, n_steps=252, n_paths=5_000, cost_bps=cost_bps, seed=6
        )
        means.append(pnl.mean())
    assert means == sorted(means, reverse=True)


# ---------------------------------------------------------------------------
# Cross-check against the theoretical gamma P&L approximation.
#
# For a continuously delta-hedged short option, P&L ~= -0.5 * integral of
# Gamma_t * S_t^2 * (sigma_realised^2 - sigma_hedge^2) dt. We approximate
# that integral along each simulated path using the closed-form gamma at
# sigma_hedge, and check it roughly agrees with the actual simulated P&L's
# mean - this validates that delta_hedge_pnl's mechanics actually produce
# the textbook gamma-driven relationship, not just the right sign.
# ---------------------------------------------------------------------------


def test_pnl_matches_gamma_theoretical_approximation():
    n_steps, n_paths = 252, 20_000
    sigma_realised = 0.30
    from options_pricer.hedging import simulate_gbm_paths

    paths = simulate_gbm_paths(S0, R, sigma_realised, T, n_steps, n_paths, seed=9)
    pnl = delta_hedge_pnl(paths, K, T, R, SIGMA, q=Q)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    gamma_s2_dt = np.zeros(n_paths)
    for i in range(n_steps):
        tau = T - times[i]
        S_i = paths[:, i]
        g = gamma(S_i, K, tau, R, SIGMA, Q)
        gamma_s2_dt += g * S_i**2 * dt

    theoretical_pnl = 0.5 * (SIGMA**2 - sigma_realised**2) * gamma_s2_dt

    se = pnl.std() / np.sqrt(n_paths)
    assert pnl.mean() == pytest.approx(theoretical_pnl.mean(), abs=10 * se)
