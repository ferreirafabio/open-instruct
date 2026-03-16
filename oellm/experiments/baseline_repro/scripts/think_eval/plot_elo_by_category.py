"""
Plot per-category ELO/winrate curves across think training checkpoints.

Uses cached judge results (no GPU needed) + LMArena category metadata.
Categories: is_code, math, complexity, creativity, domain_knowledge,
            problem_solving, real_world, specificity, technical_accuracy.

Usage:
    python oellm/experiments/baseline_repro/scripts/think_eval/plot_elo_by_category.py
"""

import os
import re
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
import sys

sys.path.insert(
    0, str(PROJECT_ROOT / "oellm/evaluations/benchmarks/OpenJury")
)
from openjury.evaluate import PairScore


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
    df["turns_b"] = df.apply(lambda row: len(row["conversation_b"]) - 1, axis=1)
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
    df["if_score"] = df["category_tag"].apply(
        lambda ct: ct.get("if_v0.1", {}).get("score", None)
        if isinstance(ct, dict)
        else None
    )

    criteria_keys = [
        "complexity",
        "creativity",
        "domain_knowledge",
        "problem_solving",
        "real_world",
        "specificity",
        "technical_accuracy",
    ]
    for key in criteria_keys:
        df[key] = df["category_tag"].apply(
            lambda ct, k=key: ct.get("criteria_v0.1", {}).get(k, False)
            if isinstance(ct, dict)
            else False
        )

    # Keep first 20k after sorting by question_id (match the order used in eval)
    df = df.head(20000)

    keep_cols = [
        "instruction",
        "is_code",
        "is_math",
        "if_score",
        "language",
    ] + criteria_keys
    return df[keep_cols].reset_index(drop=True)


def load_judge_cache(cache_path: Path) -> pd.DataFrame:
    """Load a judge cache and parse win/loss/tie results."""
    df = pd.read_csv(cache_path)
    score_parser = PairScore()
    prefs = [
        score_parser.parse_model_raw(jc) for jc in df["judge_completion"]
    ]

    # Determine if our model won each battle
    results = []
    for pref, is_pos_a in zip(prefs, df["our_model_is_position_a"]):
        if pref is None or pref == 0.5:
            results.append(0.5)  # tie
        elif pref < 0.5:
            # position A won
            results.append(1.0 if is_pos_a else 0.0)
        else:
            # position B won
            results.append(0.0 if is_pos_a else 1.0)

    return pd.DataFrame(
        {"instruction": df["instruction"], "win": results}
    )


def find_think_caches() -> list[tuple[int, Path]]:
    """Find all think checkpoint judge caches and extract step numbers."""
    pattern = re.compile(r"think-v2-step(\d+)")
    caches = []
    for p in sorted(CACHE_DIR.glob("judge_LMArena*think-v2-step*")):
        m = pattern.search(p.name)
        if m:
            caches.append((int(m.group(1)), p))
    # Sort by step
    caches.sort(key=lambda x: x[0])
    return caches


def compute_category_winrates(
    arena_meta: pd.DataFrame,
    judge_results: pd.DataFrame,
    categories: dict[str, pd.Series],
) -> dict[str, float]:
    """Compute winrate for each category by joining on instruction text."""
    # Join on instruction
    merged = judge_results.merge(arena_meta, on="instruction", how="inner")

    winrates = {}
    winrates["overall"] = merged["win"].mean()

    for cat_name, cat_mask_col in categories.items():
        mask = merged[cat_mask_col].astype(bool)
        if mask.sum() > 0:
            winrates[cat_name] = merged.loc[mask, "win"].mean()
            winrates[f"NOT_{cat_name}"] = merged.loc[~mask, "win"].mean()

    return winrates


def main():
    print("Loading LMArena metadata...")
    arena_meta = load_arena_metadata()
    print(f"  {len(arena_meta)} instructions with category labels")

    # Print category distribution
    categories = {
        "is_code": "is_code",
        "is_math": "is_math",
        "complexity": "complexity",
        "creativity": "creativity",
        "domain_knowledge": "domain_knowledge",
        "problem_solving": "problem_solving",
    }
    for cat_name, col in categories.items():
        count = arena_meta[col].astype(bool).sum()
        print(f"  {cat_name}: {count} ({100*count/len(arena_meta):.1f}%)")

    # Find and load all think checkpoint caches
    think_caches = find_think_caches()
    print(f"\nFound {len(think_caches)} think checkpoints")

    steps = []
    all_winrates = []

    for step, cache_path in think_caches:
        print(f"  Processing step {step}...")
        judge_results = load_judge_cache(cache_path)
        winrates = compute_category_winrates(
            arena_meta, judge_results, categories
        )
        steps.append(step)
        all_winrates.append(winrates)

    # Convert to DataFrame for easy plotting
    df_wr = pd.DataFrame(all_winrates, index=steps)
    df_wr.index.name = "step"
    print(f"\nWinrate table:\n{df_wr.to_string()}")

    # --- Plot 1: Main categories ---
    fig, ax = plt.subplots(figsize=(12, 7))

    plot_cats = ["overall", "is_code", "is_math", "complexity", "creativity", "problem_solving"]
    colors = ["black", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    styles = ["-o", "-s", "-^", "-D", "-v", "-<"]

    for cat, color, style in zip(plot_cats, colors, styles):
        if cat in df_wr.columns:
            vals = df_wr[cat].values
            ax.plot(steps, vals, style, color=color, label=cat, markersize=5, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% (tie)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Win Rate vs Arena Opponent")
    ax.set_title("Think-SFT Training Curve by Category (LMArena 20k)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.7)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "think_elo_by_category.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # --- Plot 2: Code vs non-code, Math vs non-math ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cat in zip(axes, ["is_code", "is_math"]):
        cat_label = "Code" if cat == "is_code" else "Math"
        ax.plot(steps, df_wr[cat].values, "-o", color="#e41a1c", label=cat_label, markersize=5)
        ax.plot(steps, df_wr[f"NOT_{cat}"].values, "-s", color="#377eb8", label=f"Non-{cat_label}", markersize=5)
        ax.plot(steps, df_wr["overall"].values, "-", color="black", label="Overall", linewidth=1, alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"{cat_label} vs Non-{cat_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 0.7)

    fig.suptitle("Think-SFT: Category Split (LMArena 20k)", fontsize=14)
    fig.tight_layout()
    out_path2 = FIGURES_DIR / "think_elo_code_vs_math.png"
    fig.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.close()

    # Save raw data
    csv_path = FIGURES_DIR / "think_elo_by_category.csv"
    df_wr.to_csv(csv_path)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
