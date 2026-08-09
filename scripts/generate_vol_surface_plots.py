"""Fetch a live SPY option chain, solve for implied vol, and plot the surface.

Lives outside src/options_pricer (per CLAUDE.md, no notebook-style scripts
in the package) since this is an executable script with side effects
(network I/O, writing PNGs), not a reusable pure function.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_pricer.data import (
    clean_option_chain,
    fetch_option_chain,
    get_dividend_yield,
    get_risk_free_rate,
    get_spot_price,
)
from options_pricer.implied_vol import implied_vol

logging.basicConfig(level=logging.WARNING)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
TICKER = "SPY"


def build_dataset(max_expiries: int = 8) -> tuple[pd.DataFrame, float, float, float]:
    # max_expiries=8 caps this to SPY's nearest ~8 expiries (its dense
    # weekly listings mean that's only a few weeks of tenor - see README
    # limitations). fetch_option_chain does one network round-trip per
    # expiry, and SPY has ~29 of them including LEAPS out to ~2 years;
    # raise this if you want a fuller term structure and don't mind the
    # slower fetch.
    spot = get_spot_price(TICKER)
    r = get_risk_free_rate()
    q = get_dividend_yield(TICKER)
    print(f"{TICKER} spot={spot:.2f}  r={r:.4f}  q={q:.4f}")

    raw = fetch_option_chain(TICKER, max_expiries=max_expiries)
    clean = clean_option_chain(raw, spot=spot)
    print(f"Raw contracts: {len(raw)}  ->  cleaned: {len(clean)}")

    ivs = []
    for row in clean.itertuples(index=False):
        iv = implied_vol(
            market_price=row.mid,
            S=spot,
            K=row.strike,
            T=row.T,
            r=r,
            q=q,
            option_type=row.option_type,
        )
        ivs.append(iv)
    clean = clean.copy()
    clean["implied_vol"] = ivs

    n_before = len(clean)
    clean = clean.dropna(subset=["implied_vol"])
    print(f"Implied vol solved for {len(clean)}/{n_before} cleaned contracts")

    return clean, spot, r, q


def plot_smile(df: pd.DataFrame, spot: float) -> Path:
    expiries = sorted(df["expiry"].unique())
    target_expiry = expiries[len(expiries) // 3] if len(expiries) > 2 else expiries[0]
    subset = df[(df["expiry"] == target_expiry) & (df["option_type"] == "call")]
    subset = subset.sort_values("strike")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(subset["strike"], subset["implied_vol"] * 100, marker="o", markersize=3)
    ax.axvline(spot, color="grey", linestyle="--", linewidth=0.8, label=f"spot = {spot:.0f}")
    ax.set_title(f"{TICKER} Volatility Smile/Skew — calls, expiry {target_expiry}")
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Implied volatility (%)")
    ax.legend()

    out_path = PLOTS_DIR / "vol_smile.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_term_structure(df: pd.DataFrame, spot: float) -> Path:
    df = df.copy()
    df["abs_moneyness"] = (df["strike"] - spot).abs()
    atm_idx = df.groupby("expiry")["abs_moneyness"].idxmin()
    atm = df.loc[atm_idx].sort_values("T")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(atm["T"] * 365, atm["implied_vol"] * 100, marker="o")
    ax.set_title(f"{TICKER} ATM Implied Volatility Term Structure")
    ax.set_xlabel("Time to expiry (days)")
    ax.set_ylabel("ATM implied volatility (%)")

    out_path = PLOTS_DIR / "vol_term_structure.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_surface(df: pd.DataFrame) -> Path:
    calls = df[df["option_type"] == "call"]
    pivot = calls.pivot_table(index="T", columns="strike", values="implied_vol", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolormesh(
        pivot.columns, pivot.index * 365, pivot.values * 100, shading="auto", cmap="viridis"
    )
    fig.colorbar(mesh, ax=ax, label="Implied volatility (%)")
    ax.set_title(f"{TICKER} Implied Volatility Surface (calls)")
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Time to expiry (days)")

    out_path = PLOTS_DIR / "vol_surface_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    df, spot, r, q = build_dataset()

    if df.empty:
        print("No contracts with a solved implied vol; cannot plot.")
        return

    for path in (plot_smile(df, spot), plot_term_structure(df, spot), plot_surface(df)):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
