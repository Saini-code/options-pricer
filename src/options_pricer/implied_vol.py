"""Implied volatility solvers for the Black-Scholes-Merton model."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq

from .black_scholes import bsm_price
from .greeks import vega

logger = logging.getLogger(__name__)

_SIGMA_LO = 1e-6
_SIGMA_HI = 5.0


def _brenner_subrahmanyam_guess(price: float, S: float, T: float) -> float:
    """Brenner & Subrahmanyam (1988) approximation, used only as a Newton starting point.

    Near the money, the Black-Scholes call price is approximately linear in
    sigma: C_ATM ~= S * sigma * sqrt(T / (2*pi)). Inverting that gives a
    closed-form volatility estimate:

        sigma ~= sqrt(2*pi / T) * (price / S)

    It's a crude, ATM-only approximation (it ignores K, r, q entirely), but
    it is cheap and lands close enough to the true implied vol to give
    Newton-Raphson a good starting point, cutting down the iterations
    needed versus starting from an arbitrary guess like sigma=0.2.

    Parameters
    ----------
    price : float
        Observed option price.
    S : float
        Spot price of the underlying.
    T : float
        Time to expiry, in years.

    Returns
    -------
    float
        Initial volatility guess, clipped to (`_SIGMA_LO`, `_SIGMA_HI`).
    """
    T_safe = max(T, 1e-8)
    guess = np.sqrt(2.0 * np.pi / T_safe) * (price / S)
    if not np.isfinite(guess) or guess <= 0:
        guess = 0.2
    return float(np.clip(guess, _SIGMA_LO, _SIGMA_HI))


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Solve for the Black-Scholes-Merton implied volatility of a single quote.

    Tries Newton-Raphson first, using the analytical vega for the step and
    the Brenner-Subrahmanyam approximation as the starting guess. Falls
    back to Brent's method (bisection-style, guaranteed to converge given a
    bracket with a sign change) if Newton fails to converge within
    `max_iter`, steps outside the admissible volatility range, or the local
    vega is too small to trust the step (this happens for deep ITM/OTM or
    very short-dated options, where the price surface is nearly flat in
    sigma and Newton is fragile).

    Operates on scalar inputs: the adaptive Newton/Brent fallback logic
    branches per-quote, so it does not vectorise cleanly the way bsm_price
    does. To solve an entire option chain, apply this row-by-row (see
    data.py / scripts/generate_vol_surface_plots.py for an example).

    Parameters
    ----------
    market_price : float
        Observed option price to invert.
    S, K, T : float
        Spot price, strike, time to expiry (years).
    r : float
        Continuously compounded risk-free rate.
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.
    option_type : {'call', 'put'}, optional
        Default 'call'.
    tol : float, optional
        Absolute price-error tolerance for convergence. Default 1e-8.
    max_iter : int, optional
        Maximum iterations for both Newton-Raphson and Brent's method.
        Default 100.

    Returns
    -------
    float
        Implied volatility, or NaN if no solution exists (e.g.
        market_price violates a no-arbitrage bound) or the solver fails to
        converge. A NaN return is always accompanied by a log message
        explaining why.
    """
    if S <= 0 or K <= 0 or T <= 0:
        logger.warning(
            "implied_vol: invalid inputs S=%s, K=%s, T=%s (all must be positive); returning NaN",
            S, K, T,
        )
        return float("nan")

    forward = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        intrinsic = max(forward - disc_K, 0.0)
        upper_bound = forward
    elif option_type == "put":
        intrinsic = max(disc_K - forward, 0.0)
        upper_bound = disc_K
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    if market_price < intrinsic - tol or market_price > upper_bound + tol:
        logger.warning(
            "implied_vol: market_price=%s is outside the no-arbitrage bounds [%s, %s] "
            "for S=%s, K=%s, T=%s, option_type=%s; no solution exists, returning NaN",
            market_price, intrinsic, upper_bound, S, K, T, option_type,
        )
        return float("nan")

    sigma = _brenner_subrahmanyam_guess(market_price, S, T)
    converged = False

    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, q, option_type=option_type)
        diff = price - market_price
        if abs(diff) < tol:
            converged = True
            break

        v = vega(S, K, T, r, sigma, q)
        if v < 1e-10:
            logger.info(
                "implied_vol: vega too small (%.2e) at sigma=%.4f for S=%s, K=%s, T=%s; "
                "Newton-Raphson unstable, falling back to Brent's method",
                v, sigma, S, K, T,
            )
            break

        new_sigma = sigma - diff / v
        if not (_SIGMA_LO < new_sigma < _SIGMA_HI):
            logger.info(
                "implied_vol: Newton-Raphson step left the admissible range "
                "(%.4f -> %.4f) for S=%s, K=%s, T=%s; falling back to Brent's method",
                sigma, new_sigma, S, K, T,
            )
            break
        sigma = new_sigma

    if converged:
        return sigma

    logger.info(
        "implied_vol: Newton-Raphson did not converge in %d iterations for "
        "S=%s, K=%s, T=%s, market_price=%s; falling back to Brent's method",
        max_iter, S, K, T, market_price,
    )

    def objective(s: float) -> float:
        return bsm_price(S, K, T, r, s, q, option_type=option_type) - market_price

    f_lo, f_hi = objective(_SIGMA_LO), objective(_SIGMA_HI)
    if f_lo * f_hi > 0:
        logger.warning(
            "implied_vol: no sign change in price(sigma) - market_price over "
            "[%s, %s] for S=%s, K=%s, T=%s, market_price=%s; no solution found, "
            "returning NaN",
            _SIGMA_LO, _SIGMA_HI, S, K, T, market_price,
        )
        return float("nan")

    try:
        return brentq(objective, _SIGMA_LO, _SIGMA_HI, xtol=tol, maxiter=max_iter)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "implied_vol: Brent's method failed for S=%s, K=%s, T=%s, "
            "market_price=%s: %s; returning NaN",
            S, K, T, market_price, exc,
        )
        return float("nan")
