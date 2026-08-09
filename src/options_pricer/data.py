"""Market data retrieval and preparation.

Fetching functions here do network I/O (via yfinance) and so are not pure
in the sense the rest of this package is - the cleaning/transform functions
they feed into (`clean_option_chain`) are kept pure and independently
testable against a synthetic DataFrame, with no network access required.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_risk_free_rate(ticker: str = "^IRX") -> float:
    """Fetch a short-term risk-free rate proxy from the 13-week T-bill yield.

    Source: Yahoo Finance ticker ^IRX, the CBOE 13-Week Treasury Bill
    yield, quoted as an annualised discount yield in percentage points
    (e.g. a last price of 5.25 means 5.25%). Using the most recent close
    of a 3-month T-bill as a proxy for the continuously-compounded
    risk-free rate r is standard practice for short-dated equity option
    pricing, though it is technically a discount yield rather than a
    continuously-compounded rate - the distinction is a second-order
    effect at these maturities and rate levels.

    Returns
    -------
    float
        Annualised risk-free rate as a decimal (e.g. 0.0525 for 5.25%).
    """
    history = yf.Ticker(ticker).history(period="5d")
    if history.empty:
        raise RuntimeError(f"No data returned for {ticker}; cannot determine risk-free rate.")
    return float(history["Close"].iloc[-1]) / 100.0


def _normalize_dividend_yield(raw: float, plausible_decimal_yield_bound: float = 0.20) -> float:
    """Rescale a raw yfinance dividendYield value to a decimal fraction.

    Pulled out as its own pure function so the rescaling heuristic
    (documented in `get_dividend_yield`) is unit-testable without
    network access.
    """
    if raw > plausible_decimal_yield_bound:
        logger.info(
            "get_dividend_yield: raw dividendYield=%.4f exceeds %.0f%% as a decimal; "
            "treating it as percentage points and dividing by 100",
            raw, plausible_decimal_yield_bound * 100,
        )
        return raw / 100.0
    return raw


def get_dividend_yield(ticker: str) -> float:
    """Fetch the trailing dividend yield for `ticker`.

    Source: yfinance's Ticker.info['dividendYield']. As observed live
    against the current Yahoo Finance API (2026-08), this field is
    returned in percentage points (e.g. 1.01 meaning 1.01%), not as a
    decimal fraction - this is empirically observed behavior, not
    documented Yahoo/yfinance API behavior, so it is not guaranteed to
    stay this way (the yfinance version pin in requirements.txt already
    broke once against a Yahoo API change during this project - see
    README limitations). To avoid silently returning a yield 100x too
    small if that convention ever changes, we only rescale when the raw
    value would otherwise imply an implausible decimal yield (>20%,
    which essentially never happens for a real trailing dividend yield);
    otherwise we trust it is already a decimal.

    Parameters
    ----------
    ticker : str
        Underlying ticker symbol, e.g. 'SPY'.

    Returns
    -------
    float
        Continuously-compounded-equivalent dividend yield as a decimal.
    """
    info = yf.Ticker(ticker).info
    raw = info.get("dividendYield")
    if raw is None:
        logger.warning("get_dividend_yield: no dividend yield found for %s; defaulting to 0.0", ticker)
        return 0.0
    return _normalize_dividend_yield(float(raw))


def get_spot_price(ticker: str) -> float:
    """Fetch the most recent close price for `ticker`.

    Source: yfinance Ticker.history(period='1d')['Close'], the last
    regular-session close. Used as the spot price S for pricing and
    moneyness calculations elsewhere in this package.
    """
    history = yf.Ticker(ticker).history(period="1d")
    if history.empty:
        raise RuntimeError(f"No price data returned for {ticker}.")
    return float(history["Close"].iloc[-1])


def fetch_option_chain(ticker: str = "SPY", max_expiries: int | None = None) -> pd.DataFrame:
    """Fetch calls and puts across available expiries for `ticker`.

    Source: yfinance Ticker.option_chain(expiry), called once per expiry
    date in Ticker.options. This performs one network round-trip per
    expiry, so `max_expiries` can be used to cap it during development.

    Parameters
    ----------
    ticker : str, optional
        Underlying ticker symbol. Default 'SPY'.
    max_expiries : int, optional
        If given, only fetch the first `max_expiries` expiry dates
        (nearest-dated first). Default None (fetch all).

    Returns
    -------
    pd.DataFrame
        One row per contract, with yfinance's raw columns plus 'expiry'
        (the expiry date string as returned by yfinance) and 'option_type'
        ('call' or 'put').
    """
    handle = yf.Ticker(ticker)
    expiries = handle.options
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    frames = []
    for expiry in expiries:
        chain = handle.option_chain(expiry)
        calls = chain.calls.copy()
        calls["option_type"] = "call"
        puts = chain.puts.copy()
        puts["option_type"] = "put"
        combined = pd.concat([calls, puts], ignore_index=True)
        combined["expiry"] = expiry
        frames.append(combined)

    if not frames:
        raise RuntimeError(f"No option expiries returned for {ticker}.")
    return pd.concat(frames, ignore_index=True)


def clean_option_chain(
    chain: pd.DataFrame,
    spot: float,
    as_of: dt.datetime | None = None,
    moneyness_bounds: tuple[float, float] = (0.8, 1.2),
    max_staleness_days: float = 5.0,
) -> pd.DataFrame:
    """Clean a raw option chain for pricing/implied-vol work.

    Applies, in order: drop zero/missing bid or ask (illiquid contracts,
    since a 'last' trade price can be stale and unrepresentative of the
    current market - we use the bid/ask mid instead), drop contracts whose
    last trade is older than `max_staleness_days`, compute the bid-ask mid
    price, compute time to expiry T in years from `as_of` to the expiry
    date, drop non-positive T, and filter to `moneyness_bounds` (strike /
    spot).

    Parameters
    ----------
    chain : pd.DataFrame
        Raw contracts as returned by `fetch_option_chain`. Must have
        'bid', 'ask', 'lastTradeDate', 'strike', and 'expiry' columns.
    spot : float
        Current spot price of the underlying, used for the moneyness filter.
    as_of : datetime, optional
        Reference "now" used for time-to-expiry and staleness calculations.
        Defaults to the current UTC time.
    moneyness_bounds : tuple of float, optional
        Inclusive (low, high) bounds on strike/spot to keep. Default (0.8, 1.2).
    max_staleness_days : float, optional
        Drop contracts whose lastTradeDate is older than this many days.
        Default 5.0.

    Returns
    -------
    pd.DataFrame
        Cleaned contracts, with added 'mid' (bid-ask mid price) and 'T'
        (time to expiry in years) columns, sorted by expiry then strike.
    """
    as_of = as_of or dt.datetime.now(dt.timezone.utc)
    df = chain.copy()

    df = df[(df["bid"] > 0) & (df["ask"] > 0)]

    last_trade = pd.to_datetime(df["lastTradeDate"], utc=True)
    staleness_days = (as_of - last_trade).dt.total_seconds() / 86400.0
    df = df[staleness_days <= max_staleness_days]

    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    expiry = pd.to_datetime(df["expiry"], utc=True)
    df["T"] = (expiry - as_of).dt.total_seconds() / (365.0 * 86400.0)
    df = df[df["T"] > 0]

    moneyness = df["strike"] / spot
    df = df[(moneyness >= moneyness_bounds[0]) & (moneyness <= moneyness_bounds[1])]

    return df.sort_values(["expiry", "strike"]).reset_index(drop=True)
