#!/usr/bin/env python3
"""
Plot multilingual EU training curves from collected evaluation results.

Generates:
  - training_curve_winrate.png    Winrate over steps (m-arena-hard-EU + arena-hard)
  - training_curve_rubric.png     Rubric composite over steps

Usage:
    python oellm/experiments/multilingual_eu/plot_training_curve.py
    python oellm/experiments/multilingual_eu/plot_training_curve.py --csv results.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path(
    "/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/multilingual_eu"
)
PLOTS_DIR = EXPERIMENT_DIR / "plots"

# Color palette for experiments
COLORS = {
    "A1-90en": "#e74c3c",  # Red
    "A2-80en": "#3498db",  # Blue
    "A3-70en": "#2ecc71",  # Green
    "B1-90en": "#9b59b6",  # Purple
    "B2-80en": "#f39c12",  # Orange
}

MARKERS = {
    "A1-90en": "o",
    "A2-80en": "s",
    "A3-70en": "^",
    "B1-90en": "D",
    "B2-80en": "v",
}


def load_csv(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                row["step"] = int(row["step"])
            except ValueError:
                continue  # skip "final" for numeric plots
            row["value"] = float(row["value"])
            row["baseline_value"] = float(row["baseline_value"])
            rows.append(row)
    return rows


def get_series(
    rows: list[dict],
    experiment: str,
    dataset: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (steps, values, baseline_values) for a given experiment/dataset/metric."""
    filtered = [
        (r["step"], r["value"], r["baseline_value"])
        for r in rows
        if r["experiment"] == experiment
        and r["dataset"] == dataset
        and r["metric"] == metric
    ]
    if not filtered:
        return np.array([]), np.array([]), np.array([])
    filtered.sort(key=lambda x: x[0])
    steps, values, baselines = zip(*filtered)
    return np.array(steps), np.array(values), np.array(baselines)


def plot_winrate(rows: list[dict], output_dir: Path):
    """Plot winrate training curves for all experiments."""
    experiments = sorted(set(r["experiment"] for r in rows))
    datasets = ["m-arena-hard-EU", "arena-hard"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 6), sharey=True)
    fig.suptitle("Multilingual EU: Winrate Training Curves", fontsize=15, fontweight="bold")

    for ax, dataset in zip(axes, datasets):
        has_data = False
        for experiment in experiments:
            steps, values, _ = get_series(rows, experiment, dataset, "winrate")
            if len(steps) == 0:
                continue
            color = COLORS.get(experiment, "gray")
            marker = MARKERS.get(experiment, "o")
            ax.plot(
                steps, values, f"{marker}-",
                color=color, linewidth=1.8, markersize=5,
                label=experiment, alpha=0.9,
            )
            has_data = True

        if has_data:
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1, label="Baseline (0.5)")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Winrate vs Baseline")
            ax.set_title(dataset, fontsize=12)
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curve_winrate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_rubric(rows: list[dict], output_dir: Path):
    """Plot rubric composite training curves."""
    experiments = sorted(set(r["experiment"] for r in rows))
    datasets = ["m-arena-hard-EU", "arena-hard"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 6), sharey=True)
    fig.suptitle("Multilingual EU: Rubric Composite Training Curves", fontsize=15, fontweight="bold")

    for ax, dataset in zip(axes, datasets):
        has_data = False
        baseline_vals = []

        for experiment in experiments:
            steps, values, baselines = get_series(rows, experiment, dataset, "rubric_composite")
            if len(steps) == 0:
                continue
            color = COLORS.get(experiment, "gray")
            marker = MARKERS.get(experiment, "o")
            ax.plot(
                steps, values, f"{marker}-",
                color=color, linewidth=1.8, markersize=5,
                label=experiment, alpha=0.9,
            )
            baseline_vals.extend(baselines.tolist())
            has_data = True

        if has_data and baseline_vals:
            baseline_mean = np.mean(baseline_vals)
            ax.axhline(
                y=baseline_mean, color="gray", linestyle="--", linewidth=1.5,
                label=f"Baseline ({baseline_mean:.3f})",
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Composite Score (0-1)")
        ax.set_title(dataset, fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curve_rubric.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def load_lang_csv(csv_path: str) -> list[dict]:
    """Load per-language results CSV."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                row["step"] = int(row["step"])
            except ValueError:
                continue
            row["value"] = float(row["value"])
            row["num_battles"] = int(row["num_battles"])
            rows.append(row)
    return rows


# Language categories for coloring
TRAINED_LANGS = {"de", "es", "fr", "it", "pt", "pl", "nl", "cs"}
HELDOUT_LANGS = {"ro", "el"}
# EU language display order
LANG_ORDER = ["en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"]


def plot_per_language(lang_rows: list[dict], output_dir: Path):
    """Plot per-language winrate bar charts for each experiment."""
    experiments = sorted(set(r["experiment"] for r in lang_rows))

    for experiment in experiments:
        exp_rows = [r for r in lang_rows if r["experiment"] == experiment]
        steps = sorted(set(r["step"] for r in exp_rows))

        for step in steps:
            step_rows = [r for r in exp_rows if r["step"] == step]
            # Sort by LANG_ORDER
            lang_order_map = {l: i for i, l in enumerate(LANG_ORDER)}
            step_rows.sort(key=lambda x: lang_order_map.get(x["language"], 99))

            langs = [r["language"] for r in step_rows]
            values = [r["value"] for r in step_rows]
            n_battles = [r["num_battles"] for r in step_rows]

            fig, ax = plt.subplots(figsize=(12, 6))

            colors = []
            for lang in langs:
                if lang in TRAINED_LANGS:
                    colors.append("#3498db")  # blue = trained
                elif lang in HELDOUT_LANGS:
                    colors.append("#e74c3c")  # red = held-out
                elif lang == "en":
                    colors.append("#95a5a6")  # gray = english
                else:
                    colors.append("#f39c12")  # orange = other

            bars = ax.bar(langs, values, color=colors, edgecolor="white", linewidth=0.5)

            # Add value labels
            for bar, val, n in zip(bars, values, n_battles):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f"n={n}", ha="center", va="center", fontsize=7, color="white",
                )

            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1)
            ax.set_ylabel("Winrate vs Baseline")
            ax.set_xlabel("Language")
            ax.set_ylim(0.3, 0.7)
            ax.set_title(
                f"{experiment} step{step}: Per-Language Winrate (m-arena-hard-EU)",
                fontsize=13, fontweight="bold",
            )
            ax.grid(True, alpha=0.2, axis="y")

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#3498db", label="Trained language"),
                Patch(facecolor="#e74c3c", label="Held-out (zero-shot)"),
                Patch(facecolor="#95a5a6", label="English"),
                Patch(facecolor="#f39c12", label="Other"),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

            plt.tight_layout()
            path = output_dir / f"per_language_{experiment}_step{step}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
            plt.close()


def plot_per_language_comparison(lang_rows: list[dict], output_dir: Path):
    """Compare per-language winrates across experiments (single chart)."""
    experiments = sorted(set(r["experiment"] for r in lang_rows))
    if len(experiments) < 2:
        return

    # Use final step for each experiment
    exp_data = {}
    for experiment in experiments:
        exp_rows = [r for r in lang_rows if r["experiment"] == experiment]
        max_step = max(r["step"] for r in exp_rows)
        step_rows = [r for r in exp_rows if r["step"] == max_step]
        exp_data[experiment] = {r["language"]: r["value"] for r in step_rows}

    # Get all languages
    all_langs = []
    for lang in LANG_ORDER:
        if any(lang in d for d in exp_data.values()):
            all_langs.append(lang)

    if not all_langs:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_langs))
    width = 0.8 / len(experiments)

    for i, experiment in enumerate(experiments):
        values = [exp_data[experiment].get(lang, 0) for lang in all_langs]
        color = COLORS.get(experiment, "gray")
        offset = (i - len(experiments) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=experiment, color=color, alpha=0.85)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_ylabel("Winrate vs Baseline")
    ax.set_xlabel("Language")
    ax.set_xticks(x)
    ax.set_xticklabels(all_langs)
    ax.set_ylim(0.3, 0.7)
    ax.set_title(
        "Per-Language Winrate Comparison (m-arena-hard-EU, final step)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    path = output_dir / "per_language_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot multilingual EU training curves")
    parser.add_argument(
        "--csv", type=str, default=str(EXPERIMENT_DIR / "results.csv"),
        help="Path to results CSV",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(PLOTS_DIR),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    winrate_rows = [r for r in rows if r["metric"] == "winrate"]
    rubric_rows = [r for r in rows if r["metric"] == "rubric_composite"]

    if winrate_rows:
        plot_winrate(rows, output_dir)
    if rubric_rows:
        plot_rubric(rows, output_dir)

    # Per-language plots
    lang_csv = str(Path(args.csv).parent / "results_per_language.csv")
    lang_rows = load_lang_csv(lang_csv)
    if lang_rows:
        print(f"Loaded {len(lang_rows)} per-language rows from {lang_csv}")
        plot_per_language(lang_rows, output_dir)
        plot_per_language_comparison(lang_rows, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
