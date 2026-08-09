"""Generate Greek visualisation plots and save them to plots/.

Lives outside src/options_pricer (per CLAUDE.md, no notebook-style scripts
in the package) since this is an executable script with side effects
(writing PNGs), not a reusable pure function.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from options_pricer.greeks import delta, gamma, rho, theta, vega

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

K = 100.0
R = 0.05
SIGMA = 0.20
Q = 0.0


def plot_greeks_vs_spot(T: float = 0.5) -> Path:
    S = np.linspace(50, 150, 400)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    axes[0].plot(S, delta(S, K, T, R, SIGMA, Q, option_type="call"), label="call")
    axes[0].plot(S, delta(S, K, T, R, SIGMA, Q, option_type="put"), label="put")
    axes[0].set_title("Delta vs Spot")
    axes[0].set_xlabel("Spot price (S)")
    axes[0].set_ylabel("Delta")
    axes[0].axvline(K, color="grey", linestyle="--", linewidth=0.8)
    axes[0].legend()

    axes[1].plot(S, gamma(S, K, T, R, SIGMA, Q), color="tab:green", label="call = put")
    axes[1].set_title("Gamma vs Spot")
    axes[1].set_xlabel("Spot price (S)")
    axes[1].set_ylabel("Gamma")
    axes[1].axvline(K, color="grey", linestyle="--", linewidth=0.8)
    axes[1].legend()

    axes[2].plot(S, vega(S, K, T, R, SIGMA, Q), color="tab:purple", label="call = put")
    axes[2].set_title("Vega vs Spot")
    axes[2].set_xlabel("Spot price (S)")
    axes[2].set_ylabel("Vega (per 1.00 change in sigma)")
    axes[2].axvline(K, color="grey", linestyle="--", linewidth=0.8)
    axes[2].legend()

    axes[3].plot(S, theta(S, K, T, R, SIGMA, Q, option_type="call"), label="call")
    axes[3].plot(S, theta(S, K, T, R, SIGMA, Q, option_type="put"), label="put")
    axes[3].set_title("Theta vs Spot")
    axes[3].set_xlabel("Spot price (S)")
    axes[3].set_ylabel("Theta (per year)")
    axes[3].axvline(K, color="grey", linestyle="--", linewidth=0.8)
    axes[3].legend()

    axes[4].plot(S, rho(S, K, T, R, SIGMA, Q, option_type="call"), label="call")
    axes[4].plot(S, rho(S, K, T, R, SIGMA, Q, option_type="put"), label="put")
    axes[4].set_title("Rho vs Spot")
    axes[4].set_xlabel("Spot price (S)")
    axes[4].set_ylabel("Rho (per 1.00 change in r)")
    axes[4].axvline(K, color="grey", linestyle="--", linewidth=0.8)
    axes[4].legend()

    axes[5].axis("off")

    fig.suptitle(f"Black-Scholes-Merton Greeks vs Spot (K={K}, T={T}y, r={R}, sigma={SIGMA}, q={Q})")
    fig.tight_layout()
    out_path = PLOTS_DIR / "greeks_vs_spot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_delta_vs_spot_multi_expiry() -> Path:
    S = np.linspace(50, 150, 400)
    expiries = [1.0, 0.25, 1 / 52]  # 1 year, 3 months, 1 week

    fig, ax = plt.subplots(figsize=(8, 6))
    for T in expiries:
        ax.plot(S, delta(S, K, T, R, SIGMA, Q, option_type="call"), label=f"T = {T:.3f}y")
    ax.axvline(K, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Call Delta vs Spot, across time to expiry")
    ax.set_xlabel("Spot price (S)")
    ax.set_ylabel("Delta")
    ax.legend()

    out_path = PLOTS_DIR / "delta_vs_spot_multi_expiry.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_gamma_vega_vs_spot_multi_expiry() -> Path:
    S = np.linspace(50, 150, 400)
    expiries = [1.0, 0.25, 1 / 52]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for T in expiries:
        axes[0].plot(S, gamma(S, K, T, R, SIGMA, Q), label=f"T = {T:.3f}y")
        axes[1].plot(S, vega(S, K, T, R, SIGMA, Q), label=f"T = {T:.3f}y")

    axes[0].axvline(K, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_title("Gamma vs Spot, across time to expiry")
    axes[0].set_xlabel("Spot price (S)")
    axes[0].set_ylabel("Gamma")
    axes[0].legend()

    axes[1].axvline(K, color="grey", linestyle="--", linewidth=0.8)
    axes[1].set_title("Vega vs Spot, across time to expiry")
    axes[1].set_xlabel("Spot price (S)")
    axes[1].set_ylabel("Vega (per 1.00 change in sigma)")
    axes[1].legend()

    fig.tight_layout()
    out_path = PLOTS_DIR / "gamma_vega_vs_spot_multi_expiry.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    for path in (
        plot_greeks_vs_spot(),
        plot_delta_vs_spot_multi_expiry(),
        plot_gamma_vega_vs_spot_multi_expiry(),
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
