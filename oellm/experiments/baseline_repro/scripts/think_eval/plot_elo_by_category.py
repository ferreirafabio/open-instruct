"""
Plot per-category Bradley-Terry ELO curves: baseline vs our think checkpoints.

Uses cached judge results (no GPU needed) + LMArena category metadata.
Computes full BT ELO (not just winrate) per category to match the main
ELO rating plots in eval_results.md.

Usage:
    python oellm/experiments/baseline_repro/scripts/think_eval/plot_elo_by_category.py
"""

import os
import re
import sys
from pathlib import Path

# Force line-buffered stdout for progress visibility
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
CACHE_DIR = PROJECT_ROOT / "data/openjury-eval-data/cache/elo"
FIGURES_DIR = PROJECT_ROOT / "oellm/experiments/baseline_repro/figures"

os.environ["HF_HOME"] = str(PROJECT_ROOT / "models/huggingface")
os.environ["HF_DATASETS_CACHE"] = str(PROJECT_ROOT / "data/huggingface")

sys.path.insert(
    0, str(PROJECT_ROOT / "oellm/evaluations/benchmarks/OpenJury")
)
from openjury.evaluate import PairScore
from openjury.estimate_elo_ratings import compute_bradley_terry

BASELINE_CACHE = CACHE_DIR / (
    "judge_LMArena_VLLM__work_dlclarge2_ferreira-oellm_open-instruct_"
    "models_eval_think-baseline_VLLM_Qwen_Qwen3-30B-A3B-Instruct-2507"
    "_20000_None.csv.zip"
)

CATEGORIES = {
    "Overall": None,
    "Code": "is_code",
    "Math": "is_math",
    "Complexity": "complexity",
    "Creativity": "creativity",
    "Problem Solving": "problem_solving",
}

N_BOOTSTRAPS = 10  # Reduced for speed; 21 checkpoints × 6 categories × N fits
MIN_COMBINED_BATTLES = 100


def load_arena_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LMArena parquet. Returns (arena battles df, category metadata df).

    The arena df has model_a/model_b/winner for human battles.
    The metadata df maps instruction text -> category booleans (first 20k only).
    """
    path = snapshot_download(
        repo_id="lmarena-ai/arena-human-preference-100k",
        repo_type="dataset",
        allow_patterns="*parquet",
        force_download=False,
    )
    parquet_path = Path(path) / "data" / "arena-explorer-preference-100k.parquet"

    # Step 1: Load ONLY lightweight columns for arena battles
    print("  Loading arena battles (lightweight)...")
    df_battles = pd.read_parquet(
        parquet_path,
        columns=["model_a", "model_b", "winner", "turn"],
    )
    print(f"  Raw: {len(df_battles)} rows")
    df_battles = df_battles[df_battles["turn"] == 1]
    print(f"  After turn==1 filter: {len(df_battles)} rows")
    arena = df_battles[["model_a", "model_b", "winner"]].reset_index(drop=True)

    # Step 2: Load category metadata + conversations for first 20k only
    # We need conversation_a to extract instructions, but only for 20k rows
    print("  Loading category metadata (first 20k)...")
    df_meta = pd.read_parquet(
        parquet_path,
        columns=["turn", "conversation_a", "conversation_b", "is_code", "category_tag"],
    )
    # Apply same filter
    df_meta = df_meta[df_meta["turn"] == 1]
    # Filter rows with assistant response
    df_meta = df_meta[df_meta["conversation_a"].apply(len) >= 2]
    df_meta = df_meta.head(20000)

    instructions = df_meta["conversation_a"].apply(lambda c: c[0]["content"])
    cat_tags = df_meta["category_tag"]

    meta = pd.DataFrame({
        "instruction": instructions.values,
        "is_code": df_meta["is_code"].astype(bool).values,
        "is_math": [
            ct.get("math_v0.1", {}).get("math", False)
            if isinstance(ct, dict) else False
            for ct in cat_tags
        ],
        "complexity": [
            ct.get("criteria_v0.1", {}).get("complexity", False)
            if isinstance(ct, dict) else False
            for ct in cat_tags
        ],
        "creativity": [
            ct.get("criteria_v0.1", {}).get("creativity", False)
            if isinstance(ct, dict) else False
            for ct in cat_tags
        ],
        "problem_solving": [
            ct.get("criteria_v0.1", {}).get("problem_solving", False)
            if isinstance(ct, dict) else False
            for ct in cat_tags
        ],
    })

    return arena, meta


def load_judge_cache(cache_path: Path) -> pd.DataFrame:
    """Load a judge cache and parse into battle results with instruction."""
    df = pd.read_csv(cache_path)
    score_parser = PairScore()
    prefs = [
        score_parser.parse_model_raw(jc) for jc in df["judge_completion"]
    ]

    model_name = _extract_model_name(cache_path)
    battles = []
    for i, (pref, is_pos_a, opp_model) in enumerate(
        zip(prefs, df["our_model_is_position_a"], df["opponent_model"])
    ):
        if pref is None or pref == 0.5:
            winner = "tie"
        elif pref < 0.5:
            winner = "model_a"
        else:
            winner = "model_b"

        if is_pos_a:
            battles.append({
                "model_a": model_name,
                "model_b": opp_model,
                "winner": winner,
                "instruction": df["instruction"].iloc[i],
            })
        else:
            battles.append({
                "model_a": opp_model,
                "model_b": model_name,
                "winner": winner,
                "instruction": df["instruction"].iloc[i],
            })

    return pd.DataFrame(battles)


def _extract_model_name(cache_path: Path) -> str:
    """Extract the VLLM model path from cache filename."""
    name = cache_path.stem.replace(".csv", "")
    # Pattern: judge_LMArena_VLLM__..._VLLM_Qwen_...
    # We want the first VLLM/... part
    parts = name.split("_VLLM_")
    # Reconstruct: VLLM/ + path with / restored
    model_part = parts[1]  # work_dlclarge2_..._models_eval_think-v2-step500
    model_path = "VLLM/" + model_part.replace("_", "/")
    # Fix: the path uses / not _, but some parts of the name have legit underscores
    # Simpler: use the models/eval symlink pattern
    m = re.search(r"models/eval/([\w-]+)", model_path)
    if m:
        return f"VLLM//work/dlclarge2/ferreira-oellm/open-instruct/models/eval/{m.group(1)}"
    return model_path


def filter_arena_for_bt(df_arena: pd.DataFrame) -> pd.DataFrame:
    """Filter arena battles: keep models with >= 500 human battles."""
    battle_counts = pd.concat(
        [df_arena["model_a"], df_arena["model_b"]]
    ).value_counts()
    well_represented = set(battle_counts[battle_counts >= 500].index)
    return df_arena[
        df_arena["model_a"].isin(well_represented)
        & df_arena["model_b"].isin(well_represented)
    ]


def filter_min_battles(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Remove models with < MIN_COMBINED_BATTLES (the chocolatine-2 fix)."""
    battle_counts = {}
    for _, row in df.iterrows():
        battle_counts[row["model_a"]] = battle_counts.get(row["model_a"], 0) + 1
        battle_counts[row["model_b"]] = battle_counts.get(row["model_b"], 0) + 1
    keep = {m for m, c in battle_counts.items() if c >= MIN_COMBINED_BATTLES}
    keep.add(model_name)
    return df[df["model_a"].isin(keep) & df["model_b"].isin(keep)]


def compute_elo_for_category(
    df_llm_judge: pd.DataFrame,
    df_arena_filtered: pd.DataFrame,
    meta: pd.DataFrame,
    category_col: str | None,
    model_name: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute BT ELO rating (mean ± CI) for a given category.

    Only LLM judge battles are filtered by category. Human arena battles
    are kept in full to provide the BT graph structure (opponent-opponent
    comparisons).

    Returns (mean_elo, ci_half_width) for our model.
    """
    # Tag LLM judge battles with categories
    judge_with_cats = df_llm_judge.merge(meta, on="instruction", how="inner")

    # Filter LLM judge battles by category
    if category_col is not None:
        judge_filtered = judge_with_cats[judge_with_cats[category_col].astype(bool)]
    else:
        judge_filtered = judge_with_cats

    # Combine: filtered LLM judge + full arena (for graph connectivity)
    bt_cols = ["model_a", "model_b", "winner"]
    df_combined = pd.concat(
        [judge_filtered[bt_cols], df_arena_filtered[bt_cols]], ignore_index=True
    )

    # Apply min battles filter
    df_combined = filter_min_battles(df_combined, model_name)

    if len(df_combined) < 100:
        return float("nan"), float("nan")

    # Bootstrap BT
    ratings_list = []
    for _ in range(N_BOOTSTRAPS):
        df_sample = df_combined.sample(
            n=len(df_combined), replace=True,
            random_state=int(rng.integers(0, 2**31))
        )
        try:
            ratings = compute_bradley_terry(df_sample, winner_col="winner")
            if model_name in ratings:
                ratings_list.append(ratings[model_name])
        except Exception:
            pass

    if not ratings_list:
        return float("nan"), float("nan")

    mean_elo = np.mean(ratings_list)
    ci = 1.96 * np.std(ratings_list)
    return mean_elo, ci


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


def main():
    rng = np.random.default_rng(0)

    print("Loading LMArena data...")
    df_arena_full, meta = load_arena_data()
    print(f"  Arena battles: {len(df_arena_full)}, Instructions with metadata: {len(meta)}")

    for cat_label, col in CATEGORIES.items():
        if col is not None:
            count = meta[col].astype(bool).sum()
            print(f"  {cat_label}: {count} ({100 * count / len(meta):.1f}%)")

    # Pre-filter arena for BT (done once, reused for all checkpoints/categories)
    df_arena_filtered = filter_arena_for_bt(df_arena_full)
    print(f"  Arena after >= 500 filter: {len(df_arena_filtered)} battles")

    # Load baseline
    print("\nComputing baseline ELO per category...")
    baseline_judge = load_judge_cache(BASELINE_CACHE)
    baseline_name = _extract_model_name(BASELINE_CACHE)
    baseline_elos = {}
    for cat_label, col in CATEGORIES.items():
        elo, ci = compute_elo_for_category(
            baseline_judge, df_arena_filtered, meta, col, baseline_name, rng
        )
        baseline_elos[cat_label] = elo
        print(f"  {cat_label}: {elo:.1f} ± {ci:.1f}")

    # Load all think checkpoints
    think_caches = find_think_caches()
    print(f"\nFound {len(think_caches)} think checkpoints")

    steps = []
    all_elos = []
    all_cis = []

    for step, cache_path in think_caches:
        print(f"  Processing step {step}...")
        judge_battles = load_judge_cache(cache_path)
        model_name = _extract_model_name(cache_path)

        elos = {}
        cis = {}
        for cat_label, col in CATEGORIES.items():
            elo, ci = compute_elo_for_category(
                judge_battles, df_arena_filtered, meta, col, model_name, rng
            )
            elos[cat_label] = elo
            cis[cat_label] = ci

        steps.append(step)
        all_elos.append(elos)
        all_cis.append(cis)
        print(f"    Overall: {elos['Overall']:.1f} ± {cis['Overall']:.1f}")

    df_elo = pd.DataFrame(all_elos, index=steps)
    df_ci = pd.DataFrame(all_cis, index=steps)
    df_elo.index.name = "step"

    # --- Plot: one subplot per category ---
    cat_labels = list(CATEGORIES.keys())
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9), sharex=True)

    for i, cat_label in enumerate(cat_labels):
        ax = axes[i // ncols][i % ncols]
        elo_vals = df_elo[cat_label].values
        ci_vals = df_ci[cat_label].values

        # Our training curve with CI band
        ax.plot(
            steps, elo_vals, "-o",
            color="#1f77b4", markersize=4, linewidth=1.5,
            label="Ours (think-v2)",
        )
        ax.fill_between(
            steps, elo_vals - ci_vals, elo_vals + ci_vals,
            color="#1f77b4", alpha=0.15,
        )

        # Baseline horizontal line
        bl = baseline_elos[cat_label]
        ax.axhline(
            y=bl, color="#d62728", linestyle="--", linewidth=1.5,
            label=f"Baseline ({bl:.0f})",
        )

        ax.set_title(cat_label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")

        if i >= ncols:
            ax.set_xlabel("Training Step")
        if i % ncols == 0:
            ax.set_ylabel("ELO Rating")

    fig.suptitle(
        "Think-SFT ELO by Category (Bradley-Terry, LMArena 20k)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "think_elo_by_category.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # Save raw data
    csv_path = FIGURES_DIR / "think_elo_by_category.csv"
    df_elo.to_csv(csv_path)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
