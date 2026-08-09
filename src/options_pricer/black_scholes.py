"""Black-Scholes-Merton European option pricing formulas."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._types import ArrayLike


def _validate_inputs(S: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike) -> None:
    # S and K must be strictly positive: S=0 is a degenerate absorbing
    # state under GBM (never handled specially below) and K=0 is not a
    # meaningful strike. Rejecting both here avoids a silent log(S/K) =
    # log(0/0) = NaN in d1 if S and K were ever both exactly zero.
    if np.any(np.asarray(S) <= 0):
        raise ValueError("Spot price S must be strictly positive.")
    if np.any(np.asarray(K) <= 0):
        raise ValueError("Strike price K must be strictly positive.")
    if np.any(np.asarray(T) < 0):
        raise ValueError("Time to expiry T must be non-negative.")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("Volatility sigma must be non-negative.")


def d1(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0) -> ArrayLike:
    """Compute the Black-Scholes-Merton d1 term.

    ``N(d1)`` is the risk-neutral probability, under the stock-price measure,
    used to weight the expected stock value received if the option finishes
    in the money.

    Parameters
    ----------
    S : float or ndarray
        Spot price of the underlying.
    K : float or ndarray
        Strike price.
    T : float or ndarray
        Time to expiry, in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float or ndarray
        Volatility of the underlying (annualised).
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.

    Returns
    -------
    float or ndarray
        The d1 term. Undefined (nan/inf) where T == 0 or sigma == 0;
        callers handling those edge cases should not rely on this value there.

    Raises
    ------
    ValueError
        If S or K is not strictly positive, or T or sigma is negative.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    _validate_inputs(S, K, T, sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return result


def d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0) -> ArrayLike:
    """Compute the Black-Scholes-Merton d2 term.

    ``N(d2)`` is the risk-neutral probability that the option finishes in
    the money. ``d2 = d1 - sigma * sqrt(T)``.

    Parameters
    ----------
    S : float or ndarray
        Spot price of the underlying.
    K : float or ndarray
        Strike price.
    T : float or ndarray
        Time to expiry, in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float or ndarray
        Volatility of the underlying (annualised).
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.

    Returns
    -------
    float or ndarray
        The d2 term. Undefined (nan/inf) where T == 0 or sigma == 0;
        callers handling those edge cases should not rely on this value there.

    Raises
    ------
    ValueError
        If S or K is not strictly positive, or T or sigma is negative.
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bsm_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: str = "call",
) -> ArrayLike:
    """Price a European option under the Black-Scholes-Merton model.

    Parameters
    ----------
    S : float or ndarray
        Spot price of the underlying.
    K : float or ndarray
        Strike price.
    T : float or ndarray
        Time to expiry, in years. T == 0 returns intrinsic value.
    r : float
        Continuously compounded risk-free rate.
    sigma : float or ndarray
        Volatility of the underlying (annualised). sigma == 0 returns the
        discounted forward payoff (the deterministic, no-volatility price).
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.
    option_type : {'call', 'put'}, optional
        Option type. Default 'call'.

    Returns
    -------
    float or ndarray
        Option price(s), broadcast to the common shape of S, K, T, sigma.

    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put', or if S or K is not strictly
        positive, or T or sigma is negative.

    Notes
    -----
    S, K, T, sigma may be scalars or numpy arrays; they are broadcast
    together following standard numpy broadcasting rules.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    _validate_inputs(S, K, T, sigma)

    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)

    forward = S * np.exp(-q * T)
    strike_disc = K * np.exp(-r * T)

    if option_type == "call":
        intrinsic = np.maximum(S - K, 0.0)
        no_vol_price = np.maximum(forward - strike_disc, 0.0)
    else:
        intrinsic = np.maximum(K - S, 0.0)
        no_vol_price = np.maximum(strike_disc - forward, 0.0)

    expired = T == 0
    zero_vol = (sigma == 0) & ~expired
    standard = ~expired & ~zero_vol

    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)

    if option_type == "call":
        standard_price = forward * norm.cdf(d1_val) - strike_disc * norm.cdf(d2_val)
    else:
        standard_price = strike_disc * norm.cdf(-d2_val) - forward * norm.cdf(-d1_val)

    price = np.select([expired, zero_vol, standard], [intrinsic, no_vol_price, standard_price])

    return float(price) if price.ndim == 0 else price
