"""Tests for src/options_pricer/data.py.

clean_option_chain and _normalize_dividend_yield are tested here against
synthetic input - the fetch_*/get_* functions that hit the network do live
I/O against Yahoo Finance and are exercised manually via
scripts/generate_vol_surface_plots.py rather than in the automated test
suite.
"""

import datetime as dt

import pandas as pd
import pytest

from options_pricer.data import _normalize_dividend_yield, clean_option_chain

AS_OF = dt.datetime(2026, 8, 9, 15, 0, tzinfo=dt.timezone.utc)
SPOT = 100.0


def _row(bid, ask, strike, days_to_expiry, days_since_last_trade, option_type="call"):
    return {
        "bid": bid,
        "ask": ask,
        "strike": strike,
        "expiry": (AS_OF + dt.timedelta(days=days_to_expiry)).strftime("%Y-%m-%d"),
        "lastTradeDate": AS_OF - dt.timedelta(days=days_since_last_trade),
        "option_type": option_type,
    }


def test_drops_zero_bid_contracts():
    chain = pd.DataFrame([
        _row(bid=0.0, ask=1.0, strike=100, days_to_expiry=30, days_since_last_trade=0),
        _row(bid=1.0, ask=1.2, strike=100, days_to_expiry=30, days_since_last_trade=0),
    ])
    cleaned = clean_option_chain(chain, spot=SPOT, as_of=AS_OF)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["bid"] == 1.0


def test_drops_stale_contracts():
    chain = pd.DataFrame([
        _row(bid=1.0, ask=1.2, strike=100, days_to_expiry=30, days_since_last_trade=10),
        _row(bid=1.0, ask=1.2, strike=101, days_to_expiry=30, days_since_last_trade=1),
    ])
    cleaned = clean_option_chain(chain, spot=SPOT, as_of=AS_OF, max_staleness_days=5.0)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["strike"] == 101


def test_computes_mid_price_and_time_to_expiry():
    chain = pd.DataFrame([
        _row(bid=2.0, ask=3.0, strike=100, days_to_expiry=365, days_since_last_trade=0),
    ])
    cleaned = clean_option_chain(chain, spot=SPOT, as_of=AS_OF)
    assert cleaned.iloc[0]["mid"] == pytest.approx(2.5)
    # Expiry strings are date-only (yfinance convention), so T is computed
    # against midnight on the expiry date - a few hours short of exactly
    # 1.0 years when as_of is later in the day, as it is here (15:00).
    assert cleaned.iloc[0]["T"] == pytest.approx(1.0, abs=0.01)


def test_filters_by_moneyness():
    chain = pd.DataFrame([
        _row(bid=1.0, ask=1.2, strike=50, days_to_expiry=30, days_since_last_trade=0),   # 0.5x, out
        _row(bid=1.0, ask=1.2, strike=90, days_to_expiry=30, days_since_last_trade=0),   # 0.9x, in
        _row(bid=1.0, ask=1.2, strike=110, days_to_expiry=30, days_since_last_trade=0),  # 1.1x, in
        _row(bid=1.0, ask=1.2, strike=200, days_to_expiry=30, days_since_last_trade=0),  # 2.0x, out
    ])
    cleaned = clean_option_chain(chain, spot=SPOT, as_of=AS_OF, moneyness_bounds=(0.8, 1.2))
    assert sorted(cleaned["strike"].tolist()) == [90, 110]


def test_drops_expired_contracts():
    chain = pd.DataFrame([
        _row(bid=1.0, ask=1.2, strike=100, days_to_expiry=-1, days_since_last_trade=0),
        _row(bid=1.0, ask=1.2, strike=100, days_to_expiry=30, days_since_last_trade=0),
    ])
    cleaned = clean_option_chain(chain, spot=SPOT, as_of=AS_OF)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["T"] > 0


# ---------------------------------------------------------------------------
# _normalize_dividend_yield
# ---------------------------------------------------------------------------


def test_normalize_dividend_yield_rescales_percentage_points():
    # Observed live yfinance behavior: SPY's dividendYield came back as
    # 1.01 (percentage points), which must become 0.0101 as a decimal.
    assert _normalize_dividend_yield(1.01) == pytest.approx(0.0101)


def test_normalize_dividend_yield_leaves_plausible_decimals_alone():
    # If yfinance ever returns an already-decimal yield (e.g. 0.015 for
    # 1.5%), it must NOT be divided by 100 again.
    assert _normalize_dividend_yield(0.015) == pytest.approx(0.015)


def test_normalize_dividend_yield_boundary():
    assert _normalize_dividend_yield(0.20) == pytest.approx(0.20)
    assert _normalize_dividend_yield(0.21) == pytest.approx(0.0021)
