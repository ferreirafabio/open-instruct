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
            ax.set_ylim(0.35, 0.65)
            ax.set_title(dataset, fontsize=12)
            ax.legend(fontsize=9, loc="upper right")
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

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
