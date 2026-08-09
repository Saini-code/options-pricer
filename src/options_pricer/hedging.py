"""Discrete delta-hedging simulation for a short European call.

Scenario modelled: sell one European call for its Black-Scholes-Merton
premium (priced at `sigma_hedge`), then delta-hedge it by trading the
underlying at fixed intervals until expiry, financing/investing the cash
account at the risk-free rate. At expiry the option is settled and the
residual stock position liquidated; the sum is the hedging P&L.

The underlying path is simulated separately (`simulate_gbm_paths`) at its
own `sigma_realised`, which may differ from `sigma_hedge` - that gap is
the entire point of this module, since it is what drives hedging P&L (see
`delta_hedge_pnl`'s docstring for the mechanism).
"""

from __future__ import annotations

import numpy as np

from .black_scholes import bsm_price
from .greeks import delta


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate underlying price paths under geometric Brownian motion.

    Uses the exact GBM solution S_t = S0 * exp((mu - sigma^2/2)*t + sigma*W_t),
    sampled at n_steps equally spaced intervals, rather than an Euler
    discretisation - so this is exact at any step size, not an
    approximation that would improve with finer steps.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    mu : float
        Real-world (or risk-neutral, if you want a risk-neutral path)
        drift of the underlying.
    sigma : float
        Realised volatility used to generate the path. This is the
        "ground truth" volatility of the simulated world, as opposed to
        the volatility used for pricing/hedging (see `delta_hedge_pnl`).
    T : float
        Total time horizon, in years.
    n_steps : int
        Number of steps (so n_steps+1 grid points, including t=0). Also
        sets the hedge rebalancing frequency when paths are fed into
        `delta_hedge_pnl`.
    n_paths : int
        Number of independent paths to simulate.
    seed : int, optional
        Seed for the random number generator, for reproducibility.
        Default None (nondeterministic).

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1)
        Simulated price paths, one per row. Column 0 is S0 for every row.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return S0 * np.exp(log_paths)


def delta_hedge_pnl(
    paths: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma_hedge: float,
    q: float = 0.0,
    cost_per_share: float = 0.0,
    cost_bps: float = 0.0,
) -> np.ndarray:
    """Simulate discretely delta-hedging a short European call across paths.

    At t=0, sells the call for its BSM premium (priced at `sigma_hedge`)
    and buys `delta` shares to hedge, funding the purchase out of the
    premium (any shortfall/excess sits in the cash account). At each
    subsequent column of `paths`, recomputes delta (still at
    `sigma_hedge` - this is the vol the hedger *believes*, not the vol
    the path was generated with) and trades the difference, paying any
    transaction cost and accruing interest at `r` on the cash account
    between rebalances. At the final column (expiry), liquidates the
    remaining stock position and pays the option's intrinsic payoff.

    Because delta and the hedge trades are computed from `sigma_hedge`
    while `paths` may have been generated at a different volatility (see
    `simulate_gbm_paths`), the resulting P&L is not centred at zero in
    general - the gap between realised and hedging volatility is exactly
    what this function's P&L distribution reveals.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, n_steps + 1)
        Simulated underlying price paths (e.g. from `simulate_gbm_paths`).
        Column 0 is t=0, the last column is t=T; the number of columns
        sets the number of rebalances.
    K : float
        Strike price.
    T : float
        Time to expiry, in years (must match the horizon `paths` was
        simulated over).
    r : float
        Continuously compounded risk-free rate.
    sigma_hedge : float
        Volatility used to price the option at inception and to compute
        delta at every rebalance - i.e. the implied vol the option was
        sold at.
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.
    cost_per_share : float, optional
        Flat transaction cost per share traded, charged on every trade
        (initial hedge, each rebalance, and final liquidation).
        Default 0.0.
    cost_bps : float, optional
        Proportional transaction cost, in basis points of trade notional
        (bps * |shares traded| * price / 10000). Default 0.0.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Final hedging P&L per path.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    times = np.linspace(0.0, T, n_cols)

    def trade_cost(shares_traded: np.ndarray, price: np.ndarray) -> np.ndarray:
        return cost_per_share * np.abs(shares_traded) + cost_bps * 1e-4 * np.abs(shares_traded) * price

    S0 = paths[:, 0]
    premium = bsm_price(S0, K, T, r, sigma_hedge, q, option_type="call")
    position = delta(S0, K, T, r, sigma_hedge, q, option_type="call")
    cash = premium - position * S0 - trade_cost(position, S0)

    for i in range(1, n_steps):
        cash = cash * np.exp(r * dt)
        S_i = paths[:, i]
        tau = T - times[i]
        new_position = delta(S_i, K, tau, r, sigma_hedge, q, option_type="call")
        trade = new_position - position
        cash = cash - trade * S_i - trade_cost(trade, S_i)
        position = new_position

    cash = cash * np.exp(r * dt)
    S_T = paths[:, -1]
    cash = cash + position * S_T - trade_cost(position, S_T)
    payoff = np.maximum(S_T - K, 0.0)
    cash = cash - payoff

    return cash


def run_hedge_simulation(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_hedge: float,
    sigma_realised: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    mu: float | None = None,
    cost_per_share: float = 0.0,
    cost_bps: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate paths at `sigma_realised` and delta-hedge them at `sigma_hedge`.

    Convenience wrapper combining `simulate_gbm_paths` and
    `delta_hedge_pnl` - see both for parameter details.

    Parameters
    ----------
    mu : float, optional
        Drift used to simulate the underlying path. Defaults to `r`
        (risk-neutral drift) if not given, which isolates the volatility
        mismatch effect from any drift effect on hedging P&L.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Final hedging P&L per path.
    """
    if mu is None:
        mu = r
    paths = simulate_gbm_paths(S0, mu, sigma_realised, T, n_steps, n_paths, seed=seed)
    return delta_hedge_pnl(
        paths, K, T, r, sigma_hedge, q=q, cost_per_share=cost_per_share, cost_bps=cost_bps
    )
