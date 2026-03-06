#!/usr/bin/env python3
"""
Plot Think SFT v2 training curves from collected evaluation results.

Generates:
  - alpaca-eval_training_curve.png   Winrate + rubric over steps (alpaca-eval)
  - arena-hard_training_curve.png    Winrate + rubric over steps (arena-hard)

Usage:
    python oellm/experiments/think_v2_checkpoint_eval/plot_training_curve.py
    python oellm/experiments/think_v2_checkpoint_eval/plot_training_curve.py --csv results.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path("/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/think_v2_checkpoint_eval")
PLOTS_DIR = EXPERIMENT_DIR / "plots"

# Boundary between v2 and horeka runs
V2_HOREKA_BOUNDARY = 36450  # last v2 step


def load_csv(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["step"] = int(row["step"])
            row["value"] = float(row["value"])
            row["baseline_value"] = float(row["baseline_value"])
            rows.append(row)
    return rows


def get_series(rows: list[dict], dataset: str, metric: str,
               swap_mode: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (steps, values, baseline_values) for a given dataset, metric, and optional swap_mode."""
    filtered = []
    for r in rows:
        if r["dataset"] != dataset or r["metric"] != metric:
            continue
        if swap_mode is not None and r.get("swap_mode") != swap_mode:
            continue
        filtered.append((r["step"], r["value"], r["baseline_value"]))
    if not filtered:
        return np.array([]), np.array([]), np.array([])
    filtered.sort(key=lambda x: x[0])
    steps, values, baselines = zip(*filtered)
    return np.array(steps), np.array(values), np.array(baselines)


def plot_dataset(rows: list[dict], dataset: str, output_dir: Path):
    """Create a 2-row stacked plot for one dataset: winrate on top, rubric on bottom."""
    fig, (ax_win, ax_rub) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Think SFT v2 Training Curve: {dataset}", fontsize=15, fontweight="bold")

    # --- Top subplot: Winrate ---
    has_winrate = False

    # Winrate with swap_mode=fixed
    steps_f, values_f, _ = get_series(rows, dataset, "winrate", swap_mode="fixed")
    if len(steps_f) > 0:
        ax_win.plot(steps_f, values_f, "o-", color="#e74c3c", linewidth=1.8, markersize=4,
                    label="Winrate (fixed)", alpha=0.9)
        has_winrate = True

    # Winrate with swap_mode=both
    steps_b, values_b, _ = get_series(rows, dataset, "winrate", swap_mode="both")
    if len(steps_b) > 0:
        ax_win.plot(steps_b, values_b, "s-", color="#3498db", linewidth=1.8, markersize=4,
                    label="Winrate (both)", alpha=0.9)
        has_winrate = True

    if has_winrate:
        ax_win.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1, label="Baseline (0.5)")
        ax_win.axvline(x=V2_HOREKA_BOUNDARY, color="orange", linestyle=":", alpha=0.4, linewidth=1)
        ax_win.set_ylabel("Winrate vs Baseline")
        ax_win.set_ylim(0.35, 0.65)
        ax_win.legend(fontsize=9, loc="upper right")
        ax_win.grid(True, alpha=0.3)
        ax_win.set_title("Winrate (higher = trained wins more)", fontsize=11)

    # --- Bottom subplot: Rubric composite ---
    steps_r, values_r, baselines_r = get_series(rows, dataset, "rubric_composite")
    has_rubric = len(steps_r) > 0

    if has_rubric:
        ax_rub.plot(steps_r, values_r, "o-", color="#2ecc71", linewidth=1.8, markersize=4,
                    label="Trained (v2)")
        if len(baselines_r) > 0:
            # Baseline is constant — draw as horizontal line using mean
            baseline_val = np.mean(baselines_r)
            ax_rub.axhline(y=baseline_val, color="#e74c3c", linestyle="--", linewidth=1.5,
                           label=f"Baseline ({baseline_val:.3f})")
        ax_rub.axvline(x=V2_HOREKA_BOUNDARY, color="orange", linestyle=":", alpha=0.4, linewidth=1)
        ax_rub.set_xlabel("Training Step")
        ax_rub.set_ylabel("Composite Score (0-1)")
        ax_rub.legend(fontsize=9, loc="upper right")
        ax_rub.grid(True, alpha=0.3)
        ax_rub.set_title("Rubric (absolute quality score)", fontsize=11)

    plt.tight_layout()
    path = output_dir / f"{dataset}_training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Think SFT v2 training curves")
    parser.add_argument("--csv", type=str, default=str(EXPERIMENT_DIR / "results.csv"),
                        help="Path to results CSV")
    parser.add_argument("--output-dir", type=str, default=str(PLOTS_DIR),
                        help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    for dataset in ["alpaca-eval", "arena-hard"]:
        plot_dataset(rows, dataset, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
