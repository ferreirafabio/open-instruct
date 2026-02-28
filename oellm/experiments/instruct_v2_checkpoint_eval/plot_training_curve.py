#!/usr/bin/env python3
"""
Plot Instruct SFT v2 training curves from collected evaluation results.

Generates two figures matching the think SFT plot layout:
  - training_curve_eval.png        Combined overview (winrate + rubric mean + per-criterion)
  - rubric_criteria_eval.png       Per-criterion 2x2 grid

Usage:
    python oellm/experiments/instruct_v2_checkpoint_eval/plot_training_curve.py
    python oellm/experiments/instruct_v2_checkpoint_eval/plot_training_curve.py --csv results.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

EXPERIMENT_DIR = Path("/work/dlclarge2/ferreira-oellm/open-instruct/oellm/experiments/instruct_v2_checkpoint_eval")
PLOTS_DIR = EXPERIMENT_DIR / "plots"
FIGURES_DIR = Path("/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/figures")

DATASETS = ["alpaca-eval", "arena-hard"]
CRITERIA = ["instruction_following", "naturalness", "coherence", "accuracy"]
CRITERIA_LABELS = {
    "instruction_following": "Instruction Following",
    "naturalness": "Naturalness",
    "coherence": "Coherence",
    "accuracy": "Accuracy",
}

# Colors matching think plots
COLOR_ALPACA = "#1f77b4"  # blue
COLOR_ARENA = "#ff7f0e"   # orange
COLOR_ALPACA_BASELINE = "#aec7e8"  # light blue
COLOR_ARENA_BASELINE = "#ffbb78"   # light orange


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


def plot_training_curve_eval(rows: list[dict], output_dir: Path):
    """Create the combined overview figure: 2 winrate plots (top) + 3 rubric plots (bottom)."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "OLMo-3-7B Instruct SFT Performance over Training Time: Reproduction vs. Baseline\n"
        "Higher is better",
        fontsize=15, fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.4)

    # Top row: 2 winrate subplots (each spanning 3 columns)
    ax_both = fig.add_subplot(gs[0, :3])
    ax_fixed = fig.add_subplot(gs[0, 3:])

    # Bottom row: 3 rubric subplots (each spanning 2 columns)
    ax_rubric_mean = fig.add_subplot(gs[1, :2])
    ax_rubric_alpaca = fig.add_subplot(gs[1, 2:4])
    ax_rubric_arena = fig.add_subplot(gs[1, 4:])

    # --- Top-left: Winrate (swap_mode=both) ---
    for dataset, color, marker in [("alpaca-eval", COLOR_ALPACA, "o"), ("arena-hard", COLOR_ARENA, "o")]:
        steps, values, _ = get_series(rows, dataset, "winrate", swap_mode="both")
        if len(steps) > 0:
            ax_both.plot(steps, values * 100, f"{marker}-", color=color, linewidth=1.8,
                         markersize=6, label=dataset, alpha=0.9)
    ax_both.axhline(y=50, color="gray", linestyle="--", alpha=0.7, linewidth=1, label="Parity (50%)")
    ax_both.set_title("Winrate (swap_mode=both)", fontsize=11)
    ax_both.set_xlabel("Training Step")
    ax_both.set_ylabel("Our Model Win Rate (%)")
    ax_both.legend(fontsize=9, loc="upper left")
    ax_both.grid(True, alpha=0.3)

    # --- Top-right: Winrate (swap_mode=fixed) ---
    for dataset, color, marker in [("alpaca-eval", COLOR_ALPACA, "o"), ("arena-hard", COLOR_ARENA, "o")]:
        steps, values, _ = get_series(rows, dataset, "winrate", swap_mode="fixed")
        if len(steps) > 0:
            ax_fixed.plot(steps, values * 100, f"{marker}-", color=color, linewidth=1.8,
                          markersize=6, label=dataset, alpha=0.9)
    ax_fixed.axhline(y=50, color="gray", linestyle="--", alpha=0.7, linewidth=1, label="Parity (50%)")
    ax_fixed.set_title("Winrate (swap_mode=fixed)", fontsize=11)
    ax_fixed.set_xlabel("Training Step")
    ax_fixed.set_ylabel("Our Model Win Rate (%)")
    ax_fixed.legend(fontsize=9, loc="lower right")
    ax_fixed.grid(True, alpha=0.3)

    # --- Bottom-left: Rubric Mean Score ---
    for dataset, color, bl_color in [
        ("alpaca-eval", COLOR_ALPACA, COLOR_ALPACA_BASELINE),
        ("arena-hard", COLOR_ARENA, COLOR_ARENA_BASELINE),
    ]:
        steps, values, baselines = get_series(rows, dataset, "rubric_composite")
        if len(steps) > 0:
            ax_rubric_mean.plot(steps, values, "o-", color=color, linewidth=1.8,
                                markersize=6, label=f"Ours ({dataset})", alpha=0.9)
            bl_val = np.mean(baselines)
            ax_rubric_mean.axhline(y=bl_val, color=bl_color, linestyle="--", linewidth=1.5,
                                   label=f"Baseline ({dataset}): {bl_val:.2f}")
    ax_rubric_mean.set_title("Rubric Mean Score (equally weighted)", fontsize=11)
    ax_rubric_mean.set_xlabel("Training Step")
    ax_rubric_mean.set_ylabel("Score (0-1)")
    ax_rubric_mean.legend(fontsize=8, loc="lower right")
    ax_rubric_mean.grid(True, alpha=0.3)

    # --- Bottom-center: Per-Criterion Scores: alpaca-eval ---
    _plot_per_criterion(ax_rubric_alpaca, rows, "alpaca-eval")
    ax_rubric_alpaca.set_title("Per-Criterion Scores: alpaca-eval", fontsize=11)

    # --- Bottom-right: Per-Criterion Scores: arena-hard ---
    _plot_per_criterion(ax_rubric_arena, rows, "arena-hard")
    ax_rubric_arena.set_title("Per-Criterion Scores: arena-hard", fontsize=11)

    path = output_dir / "training_curve_eval.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def _plot_per_criterion(ax, rows: list[dict], dataset: str):
    """Plot 4 criteria lines + baselines on a single axis for one dataset."""
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    baseline_colors = ["#aec7e8", "#98df8a", "#ffbb78", "#ff9896"]

    for criterion, color, bl_color in zip(CRITERIA, colors, baseline_colors):
        metric = f"rubric_{criterion}"
        steps, values, baselines = get_series(rows, dataset, metric)
        if len(steps) == 0:
            continue
        label = CRITERIA_LABELS[criterion]
        ax.plot(steps, values, "o-", color=color, linewidth=1.8, markersize=5,
                label=f"{label} (Ours)", alpha=0.9)
        bl_val = np.mean(baselines)
        ax.axhline(y=bl_val, color=bl_color, linestyle="--", linewidth=1.2, alpha=0.8,
                    label=f"{label} (Baseline): {bl_val:.2f}")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Score (1-7 Likert)")
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.3)


def plot_rubric_criteria_eval(rows: list[dict], output_dir: Path):
    """Create the 2x2 per-criterion figure with both datasets overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "OLMo-3-7B Instruct SFT Rubric: Per-Criterion Average Raw Likert Scores\n"
        "Higher is better",
        fontsize=15, fontweight="bold",
    )

    for idx, criterion in enumerate(CRITERIA):
        ax = axes[idx // 2][idx % 2]
        metric = f"rubric_{criterion}"

        for dataset, color, bl_color in [
            ("alpaca-eval", COLOR_ALPACA, COLOR_ALPACA_BASELINE),
            ("arena-hard", COLOR_ARENA, COLOR_ARENA_BASELINE),
        ]:
            steps, values, baselines = get_series(rows, dataset, metric)
            if len(steps) == 0:
                continue
            ax.plot(steps, values, "o-", color=color, linewidth=1.8, markersize=6,
                    label=f"Ours ({dataset})", alpha=0.9)
            bl_val = np.mean(baselines)
            ax.axhline(y=bl_val, color=bl_color, linestyle="--", linewidth=1.5,
                        label=f"Baseline avg ({dataset}): {bl_val:.2f}")

        ax.set_title(CRITERIA_LABELS[criterion], fontsize=12)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Score (1-7 Likert)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "rubric_criteria_eval.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Instruct SFT v2 training curves")
    parser.add_argument("--csv", type=str, default=str(EXPERIMENT_DIR / "results.csv"),
                        help="Path to results CSV")
    parser.add_argument("--output-dir", type=str, default=str(PLOTS_DIR),
                        help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    plot_training_curve_eval(rows, output_dir)
    plot_rubric_criteria_eval(rows, output_dir)

    # Copy to shared figures directory
    for name in ["training_curve_eval.png", "rubric_criteria_eval.png"]:
        src = output_dir / name
        dst = FIGURES_DIR / f"instruct_{name}"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
