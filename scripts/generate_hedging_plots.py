"""Run delta-hedging experiments and save plots to plots/.

Lives outside src/options_pricer (per CLAUDE.md, no notebook-style scripts
in the package) since this is an executable script with side effects
(running simulations, writing PNGs), not a reusable pure function.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from options_pricer.hedging import run_hedge_simulation

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

# Baseline scenario: ATM call, 3 months to expiry.
S0, K, T, R, Q = 100.0, 100.0, 0.25, 0.05, 0.0
SIGMA = 0.20
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5


def experiment_1_rebalance_frequency() -> Path:
    n_paths = 20_000
    n_days = int(round(T * TRADING_DAYS_PER_YEAR))
    n_hours = int(round(T * TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY))

    daily_pnl = run_hedge_simulation(S0, K, T, R, SIGMA, SIGMA, n_days, n_paths, q=Q, seed=101)
    hourly_pnl = run_hedge_simulation(S0, K, T, R, SIGMA, SIGMA, n_hours, n_paths, q=Q, seed=102)

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(
        min(daily_pnl.min(), hourly_pnl.min()), max(daily_pnl.max(), hourly_pnl.max()), 80
    )
    ax.hist(daily_pnl, bins=bins, alpha=0.6, density=True,
            label=f"Daily ({n_days} rebalances) — std={daily_pnl.std():.3f}")
    ax.hist(hourly_pnl, bins=bins, alpha=0.6, density=True,
            label=f"Hourly ({n_hours} rebalances) — std={hourly_pnl.std():.3f}")
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Delta-Hedging P&L Distribution: Daily vs Hourly Rebalancing\n"
                 "(sigma_realised = sigma_hedge, no transaction costs)")
    ax.set_xlabel("Hedging P&L ($)")
    ax.set_ylabel("Density")
    ax.legend()

    out_path = PLOTS_DIR / "hedging_pnl_rebalance_frequency.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def experiment_2_realised_vs_implied_vol() -> Path:
    n_paths = 20_000
    n_steps = int(round(T * TRADING_DAYS_PER_YEAR))
    realised_vols = np.linspace(0.05, 0.40, 15)

    means, ses = [], []
    for sigma_realised in realised_vols:
        pnl = run_hedge_simulation(
            S0, K, T, R, SIGMA, sigma_realised, n_steps, n_paths, q=Q, seed=201
        )
        means.append(pnl.mean())
        ses.append(pnl.std() / np.sqrt(n_paths))
    means, ses = np.array(means), np.array(ses)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(realised_vols * 100, means, yerr=2 * ses, marker="o", capsize=3,
                label="Mean hedging P&L (+/- 2 SE)")
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(SIGMA * 100, color="tab:red", linestyle=":", linewidth=1.2,
               label=f"sigma_hedge = {SIGMA * 100:.0f}%  (break-even)")
    ax.set_title("Short-Call Delta-Hedging P&L vs Realised Volatility\n"
                 "(daily rebalancing, no transaction costs)")
    ax.set_xlabel("Realised volatility of the underlying (%)")
    ax.set_ylabel("Mean hedging P&L ($)")
    ax.legend()

    out_path = PLOTS_DIR / "hedging_pnl_vs_realised_vol.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def experiment_3_transaction_costs_vs_frequency() -> Path:
    n_paths = 5_000
    step_counts = [4, 12, 26, 52, 126, 252, 504]
    cost_levels_bps = [0.0, 5.0, 20.0]
    risk_aversion = 2.0  # weight on P&L std in the risk-adjusted objective

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for cost_bps in cost_levels_bps:
        stds, neg_means, objective = [], [], []
        for n_steps in step_counts:
            pnl = run_hedge_simulation(
                S0, K, T, R, SIGMA, SIGMA, n_steps, n_paths, q=Q, cost_bps=cost_bps, seed=301
            )
            stds.append(pnl.std())
            neg_means.append(-pnl.mean())
            objective.append(-pnl.mean() + risk_aversion * pnl.std())

        label = f"{cost_bps:.0f} bps"
        axes[0].plot(step_counts, stds, marker="o", label=label)
        axes[1].plot(step_counts, neg_means, marker="o", label=label)
        axes[2].plot(step_counts, objective, marker="o", label=label)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Number of rebalances")
        ax.legend()

    axes[0].set_title("Hedging risk\n(P&L std)")
    axes[0].set_ylabel("Std of P&L ($)")

    axes[1].set_title("Transaction cost drag\n(-mean P&L)")
    axes[1].set_ylabel("-Mean P&L ($)")

    axes[2].set_title(f"Risk-adjusted total cost\n(-mean + {risk_aversion:.0f} x std)")
    axes[2].set_ylabel("Objective ($, lower is better)")

    fig.suptitle("Transaction Costs vs Rebalancing Frequency: the Optimal-Frequency Tradeoff")
    fig.tight_layout()

    out_path = PLOTS_DIR / "hedging_transaction_costs_vs_frequency.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    for path in (
        experiment_1_rebalance_frequency(),
        experiment_2_realised_vs_implied_vol(),
        experiment_3_transaction_costs_vs_frequency(),
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
