"""Option Greeks: sensitivities of the Black-Scholes-Merton price to its inputs.

Gamma and vega are identical for a call and a put at the same strike. This
follows directly from put-call parity, C - P = S*exp(-qT) - K*exp(-rT): the
right-hand side is linear in S and does not depend on sigma at all, so
d^2(C-P)/dS^2 = 0 (gamma_call = gamma_put) and d(C-P)/dsigma = 0
(vega_call = vega_put).
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.stats import norm

from .black_scholes import d1, d2

ArrayLike = Union[float, np.ndarray]


def _validate_option_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _scalar_if_0d(x: np.ndarray) -> ArrayLike:
    return float(x) if x.ndim == 0 else x


def delta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0,
          option_type: str = "call") -> ArrayLike:
    """Rate of change of option price with respect to spot price.

    Parameters
    ----------
    S, K, T, sigma : float or ndarray
        Spot price, strike, time to expiry (years), volatility.
    r : float
        Continuously compounded risk-free rate.
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.
    option_type : {'call', 'put'}, optional
        Default 'call'.

    Returns
    -------
    float or ndarray
        Delta: dV/dS, dimensionless. Call delta lies in [0, exp(-qT)]
        (i.e. [0, 1] when q=0); put delta lies in [-exp(-qT), 0].

    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put', or if any of S, K, T, sigma
        is negative.
    """
    _validate_option_type(option_type)
    T = np.asarray(T, dtype=float)
    disc_div = np.exp(-q * T)
    d1_val = d1(S, K, T, r, sigma, q)
    if option_type == "call":
        result = disc_div * norm.cdf(d1_val)
    else:
        result = disc_div * (norm.cdf(d1_val) - 1.0)
    return _scalar_if_0d(np.asarray(result))


def gamma(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0) -> ArrayLike:
    """Rate of change of delta with respect to spot price.

    Identical for a call and a put at the same strike (see module docstring).

    Parameters
    ----------
    S, K, T, sigma : float or ndarray
        Spot price, strike, time to expiry (years), volatility.
    r : float
        Continuously compounded risk-free rate.
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.

    Returns
    -------
    float or ndarray
        Gamma: d^2V/dS^2. Always positive for a long option (call or put).

    Raises
    ------
    ValueError
        If any of S, K, T, sigma is negative.
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1_val = d1(S, K, T, r, sigma, q)
    result = np.exp(-q * T) * norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
    return _scalar_if_0d(np.asarray(result))


def vega(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0) -> ArrayLike:
    """Rate of change of option price with respect to volatility.

    Identical for a call and a put at the same strike (see module docstring).

    Units convention: vega is the raw derivative dV/dsigma — the price
    change for a 1.00 (100 percentage point) absolute move in volatility,
    with sigma in decimal form (e.g. 0.20 = 20%). Divide by 100 to get the
    commonly quoted "price change per 1 vol point (1%)".

    Parameters
    ----------
    S, K, T, sigma : float or ndarray
        Spot price, strike, time to expiry (years), volatility.
    r : float
        Continuously compounded risk-free rate.
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.

    Returns
    -------
    float or ndarray
        Vega, per the units convention above. Always positive for a long
        option (call or put).

    Raises
    ------
    ValueError
        If any of S, K, T, sigma is negative.
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    d1_val = d1(S, K, T, r, sigma, q)
    result = S * np.exp(-q * T) * norm.pdf(d1_val) * np.sqrt(T)
    return _scalar_if_0d(np.asarray(result))


def theta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0,
          option_type: str = "call") -> ArrayLike:
    """Rate of change of option price with respect to the passage of calendar time.

    Units convention: theta is expressed per year, i.e. Theta = -dV/dT,
    where T is time to expiry in years (so calendar time passing means T
    decreasing). Divide by 365 for the commonly quoted "theta per calendar
    day", or by 252 for "theta per trading day".

    Parameters
    ----------
    S, K, T, sigma : float or ndarray
        Spot price, strike, time to expiry (years), volatility.
    r : float
        Continuously compounded risk-free rate.
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.
    option_type : {'call', 'put'}, optional
        Default 'call'.

    Returns
    -------
    float or ndarray
        Theta, per the units convention above.

    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put', or if any of S, K, T, sigma
        is negative.
    """
    _validate_option_type(option_type)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1_val = d1(S, K, T, r, sigma, q)
    d2_val = d2(S, K, T, r, sigma, q)
    disc_div = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    decay_term = -S * disc_div * norm.pdf(d1_val) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        result = decay_term - r * K * disc_r * norm.cdf(d2_val) + q * S * disc_div * norm.cdf(d1_val)
    else:
        result = decay_term + r * K * disc_r * norm.cdf(-d2_val) - q * S * disc_div * norm.cdf(-d1_val)
    return _scalar_if_0d(np.asarray(result))


def rho(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0,
        option_type: str = "call") -> ArrayLike:
    """Rate of change of option price with respect to the risk-free rate.

    Units convention: rho is the raw derivative dV/dr — the price change
    for a 1.00 (100 percentage point) absolute move in the risk-free rate.
    Divide by 100 to get the price change per 1 percentage point (0.01)
    move in rates.

    Parameters
    ----------
    S, K, T, sigma : float or ndarray
        Spot price, strike, time to expiry (years), volatility.
    r : float
        Continuously compounded risk-free rate.
    q : float, optional
        Continuously compounded dividend yield. Default 0.0.
    option_type : {'call', 'put'}, optional
        Default 'call'.

    Returns
    -------
    float or ndarray
        Rho, per the units convention above.

    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put', or if any of S, K, T, sigma
        is negative.
    """
    _validate_option_type(option_type)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    d2_val = d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if option_type == "call":
        result = K * T * disc_r * norm.cdf(d2_val)
    else:
        result = -K * T * disc_r * norm.cdf(-d2_val)
    return _scalar_if_0d(np.asarray(result))


def numerical_greek(func: Callable[..., ArrayLike], param: str, bump: float = 1e-4, **kwargs) -> ArrayLike:
    """Central finite-difference approximation of d(func)/d(param).

    Bumps the named keyword argument up and down by `bump`, reprices via
    `func`, and returns the symmetric difference quotient — a generic
    bump-and-reprice cross-check for any closed-form Greek.

    Parameters
    ----------
    func : callable
        Pricing function to differentiate, e.g. `bsm_price`. Must accept
        all of `kwargs` as keyword arguments.
    param : str
        Name of the keyword argument to bump (e.g. 'S', 'sigma', 'T', 'r').
    bump : float, optional
        Size of the symmetric bump applied to `param`. Default 1e-4.
    **kwargs
        Full set of keyword arguments to pass to `func`, including the
        base value of `param`.

    Returns
    -------
    float or ndarray
        Approximation of d(func)/d(param) at the given inputs.

    Raises
    ------
    ValueError
        If `param` is not among the supplied keyword arguments.
    """
    if param not in kwargs:
        raise ValueError(f"param {param!r} not found in the keyword arguments passed to func.")

    kwargs_up = dict(kwargs)
    kwargs_down = dict(kwargs)
    kwargs_up[param] = kwargs[param] + bump
    kwargs_down[param] = kwargs[param] - bump

    return (func(**kwargs_up) - func(**kwargs_down)) / (2.0 * bump)
