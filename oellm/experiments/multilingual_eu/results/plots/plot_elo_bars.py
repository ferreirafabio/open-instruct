#!/usr/bin/env python3
"""Generate per-language Elo bar plots for dolci-translated and datamix-9b results."""

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = "/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/multilingual_eu/results/plots"


def plot_dolci_translated():
    """Bar plot for dolci-translated A-75en vs A-25en vs baseline."""
    langs = ["en", "cs", "de", "es", "fi", "fr", "it", "sv"]

    data = {
        "A-75en": {
            "elo": [950, 714, 690, 746, 732, 743, 820, 722],
            "ci": [14, 19, 24, 18, 44, 17, 15, 35],
        },
        "A-25en": {
            "elo": [913, 745, 701, 763, 813, 756, 801, 756],
            "ci": [16, 15, 23, 18, 33, 17, 15, 33],
        },
        "Baseline": {
            "elo": [954, 647, 632, 745, 688, 695, 766, 777],
            "ci": [16, 23, 27, 20, 48, 23, 17, 33],
        },
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(langs))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#9E9E9E"]

    for i, (model, vals) in enumerate(data.items()):
        bars = ax.bar(
            x + i * width, vals["elo"], width,
            yerr=vals["ci"], capsize=3,
            label=model, color=colors[i], alpha=0.85,
        )

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Elo Rating", fontsize=12)
    ax.set_title("Per-language Elo: Dolci-Translated SFT (OLMo-3-7B)", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(langs, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(550, 1050)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=750, color="gray", linestyle="--", alpha=0.2)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/dolci_translated_per_language_elo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_datamix_9b():
    """Bar plot for datamix-9b vs OLMo A-75en vs OLMo baseline."""
    langs = ["en", "cs", "de", "es", "fi", "fr", "it", "sv"]

    data = {
        "datamix-9b-sft": {
            "elo": [879, 833, 792, 764, None, 790, 780, 820],
            "ci": [16, 15, 19, 20, None, 18, 18, 32],
        },
        "OLMo A-75en": {
            "elo": [950, 714, 690, 746, 732, 743, 820, 722],
            "ci": [14, 19, 24, 18, 44, 17, 15, 35],
        },
        "OLMo Baseline": {
            "elo": [954, 647, 632, 745, 688, 695, 766, 777],
            "ci": [16, 23, 27, 20, 48, 23, 17, 33],
        },
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(langs))
    width = 0.25
    colors = ["#4CAF50", "#2196F3", "#9E9E9E"]

    for i, (model, vals) in enumerate(data.items()):
        elo = [v if v is not None else 0 for v in vals["elo"]]
        ci = [v if v is not None else 0 for v in vals["ci"]]
        bars = ax.bar(
            x + i * width, elo, width,
            yerr=ci, capsize=3,
            label=model, color=colors[i], alpha=0.85,
        )
        # Mark missing data
        for j, v in enumerate(vals["elo"]):
            if v is None:
                ax.text(
                    x[j] + i * width, 600, "tba",
                    ha="center", va="bottom", fontsize=8, fontstyle="italic",
                )

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Elo Rating", fontsize=12)
    ax.set_title("Per-language Elo: datamix-9b vs OLMo-3-7B", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(langs, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(550, 1050)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/datamix_9b_per_language_elo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot_dolci_translated()
    plot_datamix_9b()
