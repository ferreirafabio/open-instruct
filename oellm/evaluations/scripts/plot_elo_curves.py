#!/usr/bin/env python3
"""
Plot ELO training curves for Think and Instruct SFT checkpoints.

Usage:
    python oellm/evaluations/scripts/plot_elo_curves.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/figures")

# Think ELO curve data (20k LMArena battles, 100 bootstraps)
THINK_STEPS = [500, 1000, 2000, 3000, 4000, 5000, 7000, 8000, 11000, 13000,
               15000, 17000, 19000, 21000, 24000, 27000, 31000, 34000, 38000, 42856]
THINK_ELO = [980.4, 941.2, 916.5, 907.7, 937.2, 915.2, 947.0, 915.4,
             932.5, 949.2, 943.3, 928.2, 961.6, 955.0, 979.8, 973.4,
             1012.4, 1000.5, 991.4, 1002.2]
THINK_CI = [11.8, 7.7, 9.3, 9.2, 10.1, 9.1, 8.0, 10.6,
            7.0, 7.7, 9.7, 8.3, 7.0, 7.1, 8.2, 7.6,
            5.6, 12.5, 8.7, 13.0]
THINK_OFFICIAL = 1002.9  # official Think-SFT ELO

# Instruct ELO curve data
INSTRUCT_STEPS = [1000, 2000, 3000, 3252]
INSTRUCT_ELO = [889.0, 945.9, 954.2, 953.0]
INSTRUCT_CI = [7.6, 10.0, 9.2, 9.3]
INSTRUCT_OFFICIAL = 940.5  # official Instruct-SFT ELO


def plot_think_elo():
    fig, ax = plt.subplots(figsize=(12, 5))

    steps = np.array(THINK_STEPS)
    elo = np.array(THINK_ELO)
    ci = np.array(THINK_CI)

    ax.errorbar(steps, elo, yerr=ci, fmt="o-", color="#3498db", linewidth=1.8,
                markersize=5, capsize=3, capthick=1, label="Ours (per checkpoint)")
    ax.axhline(y=THINK_OFFICIAL, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Official Think-SFT ({THINK_OFFICIAL})")
    ax.fill_between(steps, elo - ci, elo + ci, alpha=0.15, color="#3498db")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("ELO Rating", fontsize=12)
    ax.set_title("Think SFT: ELO over Training (20k LMArena battles)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "think_elo_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_instruct_elo():
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = np.array(INSTRUCT_STEPS)
    elo = np.array(INSTRUCT_ELO)
    ci = np.array(INSTRUCT_CI)

    ax.errorbar(steps, elo, yerr=ci, fmt="o-", color="#3498db", linewidth=1.8,
                markersize=6, capsize=3, capthick=1, label="Ours (per checkpoint)")
    ax.axhline(y=INSTRUCT_OFFICIAL, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Official Instruct-SFT ({INSTRUCT_OFFICIAL})")
    ax.fill_between(steps, elo - ci, elo + ci, alpha=0.15, color="#3498db")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("ELO Rating", fontsize=12)
    ax.set_title("Instruct SFT: ELO over Training (20k LMArena battles)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "instruct_elo_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_think_elo()
    plot_instruct_elo()
    print(f"\nAll plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
