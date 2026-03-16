"""
Plot per-category winrate curves: baseline vs our think checkpoints.

Uses cached judge results (no GPU needed) + LMArena category metadata.
One subplot per category, each showing baseline (horizontal line) vs
our training curve.

Usage:
    python oellm/experiments/baseline_repro/scripts/think_eval/plot_elo_by_category.py
"""

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
CACHE_DIR = PROJECT_ROOT / "data/openjury-eval-data/cache/elo"
FIGURES_DIR = PROJECT_ROOT / "oellm/experiments/baseline_repro/figures"

os.environ["HF_HOME"] = str(PROJECT_ROOT / "models/huggingface")
os.environ["HF_DATASETS_CACHE"] = str(PROJECT_ROOT / "data/huggingface")

# Add OpenJury to path for PairScore
sys.path.insert(
    0, str(PROJECT_ROOT / "oellm/evaluations/benchmarks/OpenJury")
)
from openjury.evaluate import PairScore

BASELINE_CACHE = CACHE_DIR / (
    "judge_LMArena_VLLM__work_dlclarge2_ferreira-oellm_open-instruct_"
    "models_eval_think-baseline_VLLM_Qwen_Qwen3-30B-A3B-Instruct-2507"
    "_20000_None.csv.zip"
)

CATEGORIES = {
    "Overall": None,  # no filter
    "Code": "is_code",
    "Math": "is_math",
    "Complexity": "complexity",
    "Creativity": "creativity",
    "Problem Solving": "problem_solving",
}


def load_arena_metadata() -> pd.DataFrame:
    """Load LMArena parquet and extract category metadata per instruction."""
    path = snapshot_download(
        repo_id="lmarena-ai/arena-human-preference-100k",
        repo_type="dataset",
        allow_patterns="*parquet",
        force_download=False,
    )
    df = pd.read_parquet(
        Path(path) / "data" / "arena-explorer-preference-100k.parquet"
    )

    # Filter to single-turn (same as estimate_elo_ratings.py)
    df["turns"] = df.apply(lambda row: len(row["conversation_a"]) - 1, axis=1)
    df["turns_b"] = df.apply(
        lambda row: len(row["conversation_b"]) - 1, axis=1
    )
    df = df.loc[(df.turns == 1) & (df.turns_b >= 1)]

    # Extract instruction text
    df["instruction"] = df["conversation_a"].apply(lambda c: c[0]["content"])

    # Extract category fields
    df["is_code"] = df["is_code"].astype(bool)
    df["is_math"] = df["category_tag"].apply(
        lambda ct: ct.get("math_v0.1", {}).get("math", False)
        if isinstance(ct, dict)
        else False
    )

    criteria_keys = [
        "complexity",
        "creativity",
        "problem_solving",
    ]
    for key in criteria_keys:
        df[key] = df["category_tag"].apply(
            lambda ct, k=key: ct.get("criteria_v0.1", {}).get(k, False)
            if isinstance(ct, dict)
            else False
        )

    df = df.head(20000)

    keep_cols = ["instruction", "is_code", "is_math"] + criteria_keys
    return df[keep_cols].reset_index(drop=True)


def load_judge_cache(cache_path: Path) -> pd.DataFrame:
    """Load a judge cache and parse win/loss/tie results."""
    df = pd.read_csv(cache_path)
    score_parser = PairScore()
    prefs = [
        score_parser.parse_model_raw(jc) for jc in df["judge_completion"]
    ]

    results = []
    for pref, is_pos_a in zip(prefs, df["our_model_is_position_a"]):
        if pref is None or pref == 0.5:
            results.append(0.5)
        elif pref < 0.5:
            results.append(1.0 if is_pos_a else 0.0)
        else:
            results.append(0.0 if is_pos_a else 1.0)

    return pd.DataFrame({"instruction": df["instruction"], "win": results})


def find_think_caches() -> list[tuple[int, Path]]:
    """Find all think checkpoint judge caches and extract step numbers."""
    pattern = re.compile(r"think-v2-step(\d+)")
    caches = []
    for p in sorted(CACHE_DIR.glob("judge_LMArena*think-v2-step*")):
        m = pattern.search(p.name)
        if m:
            caches.append((int(m.group(1)), p))
    caches.sort(key=lambda x: x[0])
    return caches


def compute_category_winrates(
    merged: pd.DataFrame,
) -> dict[str, float]:
    """Compute winrate for each category on a pre-merged dataframe."""
    winrates = {}
    for cat_label, col in CATEGORIES.items():
        if col is None:
            winrates[cat_label] = merged["win"].mean()
        else:
            mask = merged[col].astype(bool)
            if mask.sum() > 0:
                winrates[cat_label] = merged.loc[mask, "win"].mean()
    return winrates


def main():
    print("Loading LMArena metadata...")
    arena_meta = load_arena_metadata()
    print(f"  {len(arena_meta)} instructions with category labels")

    for cat_label, col in CATEGORIES.items():
        if col is not None:
            count = arena_meta[col].astype(bool).sum()
            print(f"  {cat_label}: {count} ({100 * count / len(arena_meta):.1f}%)")

    # Load baseline
    print("\nLoading baseline...")
    baseline_results = load_judge_cache(BASELINE_CACHE)
    baseline_merged = baseline_results.merge(
        arena_meta, on="instruction", how="inner"
    )
    baseline_wr = compute_category_winrates(baseline_merged)
    print(f"  Baseline winrates: {baseline_wr}")

    # Load all think checkpoints
    think_caches = find_think_caches()
    print(f"\nFound {len(think_caches)} think checkpoints")

    steps = []
    all_winrates = []

    for step, cache_path in think_caches:
        print(f"  Processing step {step}...")
        judge_results = load_judge_cache(cache_path)
        merged = judge_results.merge(arena_meta, on="instruction", how="inner")
        winrates = compute_category_winrates(merged)
        steps.append(step)
        all_winrates.append(winrates)

    df_wr = pd.DataFrame(all_winrates, index=steps)
    df_wr.index.name = "step"

    # --- Plot: one subplot per category ---
    cat_labels = list(CATEGORIES.keys())
    n_cats = len(cat_labels)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, cat_label in enumerate(cat_labels):
        ax = axes[i]
        # Our training curve
        ax.plot(
            steps,
            df_wr[cat_label].values,
            "-o",
            color="#1f77b4",
            markersize=4,
            linewidth=1.5,
            label="Ours (think-v2)",
        )
        # Baseline horizontal line
        ax.axhline(
            y=baseline_wr[cat_label],
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline ({baseline_wr[cat_label]:.1%})",
        )
        # 50% reference
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)

        ax.set_title(cat_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0.25, 0.65)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")

        if i >= ncols:
            ax.set_xlabel("Training Step")
        if i % ncols == 0:
            ax.set_ylabel("Win Rate vs Arena")

    fig.suptitle(
        "Think-SFT Training Curve by Category (LMArena 20k battles)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "think_elo_by_category.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # Save raw data
    csv_path = FIGURES_DIR / "think_elo_by_category.csv"
    df_wr.to_csv(csv_path)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
