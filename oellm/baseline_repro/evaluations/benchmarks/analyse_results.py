from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

"""
Analyze LLM-judge evaluation results.
Supports filtering by model type (instruct/think) and reading multiple timestamped result directories.
"""

# Model configurations for reference
MODEL_CONFIGS = {
    "instruct": {
        "baseline": "models/baselines/Olmo-3-7B-Instruct-SFT",
        "ours": "checkpoints/ferreira/olmo3-7b-sft/dolci-instruct-sft-hf",
    },
    "think": {
        "baseline": "models/baselines/Olmo-3-7B-Think-SFT",
        "ours": "checkpoints/ferreira/olmo3-7b-sft/dolci-think-sft-hf",
    },
}

DEFAULT_RESULTS_ROOT = Path(
    "/work/dlclarge2/ferreira-oellm/open-instruct/oellm/baseline_repro/evaluations/benchmarks/OpenJury/results"
).expanduser()
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def collect_result_dirs(cli_dirs: list[str] | None, latest_only: bool = False) -> list[Path]:
    """Return the list of result directories to scan."""
    if cli_dirs:
        return [Path(d).expanduser() for d in cli_dirs]

    if not DEFAULT_RESULTS_ROOT.exists():
        return []

    candidates = [d for d in DEFAULT_RESULTS_ROOT.iterdir() if d.is_dir()]
    
    if latest_only and candidates:
        # Sort by modification time, return only the most recent
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return [candidates[0]]

    # Return all directories
    return candidates if candidates else [DEFAULT_RESULTS_ROOT]


parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, choices=["instruct", "think"], default=None)
parser.add_argument(
    "--results-dir",
    action="append",
    help="Path to a results directory. Can be passed multiple times. "
    "Defaults to all subdirectories under the standard results root.",
)
parser.add_argument(
    "--latest",
    action="store_true",
    default=True,
    help="Only analyze the most recent results directory (by modification time). Default: True.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Analyze all results directories instead of just the latest.",
)
args = parser.parse_args()

# --all overrides --latest
latest_only = not args.all
result_dirs = collect_result_dirs(args.results_dir, latest_only=latest_only)
result_rows = []
all_judges = set()

# First pass: Load results and collect judges
for base_dir in result_dirs:
    for result in sorted(base_dir.rglob("*results-*.json")):
        # Filter by model type if specified
        if args.type and f"{args.type}-" not in result.name:
            continue

        with open(result, "r") as f:
            res = json.load(f)

            judge = res.get("judge_model", "unknown").split("/")[-1]
            all_judges.add(judge)

            # This is Model A's winrate
            res["winrate"] = float(
                (res["num_wins"] + 0.5 * res["num_ties"])
                / (res["num_ties"] + res["num_wins"] + res["num_losses"])
            )

            # Store metadata for printing later
            res["_mtime"] = datetime.fromtimestamp(result.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            res["_date_str"] = res.get("date", "")[:19]

            result_rows.append(res)

# Print Header Section
if args.type:
    config = MODEL_CONFIGS[args.type]
    print(f"\nAnalyzing: {args.type.upper()}")
    print("-" * 60)
    print(f"  baseline  -> {config['baseline']}")
    print(f"  ours      -> {config['ours']}")
    if all_judges:
        print(f"  judge     -> {', '.join(sorted(all_judges))}")
    print("-" * 60 + "\n")
elif all_judges:
    print(f"\nJudge: {', '.join(sorted(all_judges))}\n")

print("Results files loaded:")
print("-" * 80)
for res in result_rows:
    dataset = res.get("dataset", "unknown")
    n_battles = res.get("num_battles", 0)
    print(f"  {dataset:20s} | {n_battles:4d} battles | file: {res['_mtime']} | eval: {res['_date_str']}")
print("-" * 80)

if not result_rows:
    print("\nNo results found!")
else:
    df = pd.DataFrame(result_rows)

    # Use short names for models in the table (include parent for context, resolve symlinks)
    def short_model_name(x):
        # Strip provider prefix (e.g., "VLLM/")
        path_str = x.split("VLLM/", 1)[-1] if "VLLM/" in x else x
        p = Path(path_str)
        if p.exists():
            try:
                p = p.resolve()  # Follow symlinks
            except Exception:
                pass
        parts = p.parts
        return "/".join(parts[-2:]) if len(parts) >= 2 else p.name
    df["model_B_short"] = df["model_B"].apply(short_model_name)

    # Count runs per model/dataset
    counts = df.groupby(["model_B_short", "dataset"]).size().unstack()

    # Create pivot table (initially Model A winrates)
    df_pivot = df.pivot_table(index="model_B_short", columns="dataset", values="winrate")

    # Add baseline row (0.5) - use resolved model_A name
    if not df.empty:
        baseline_name = short_model_name(df["model_A"].iloc[0])
    else:
        baseline_name = f"{args.type}-baseline" if args.type else "instruct-baseline"
    if baseline_name not in df_pivot.index:
        df_pivot.loc[baseline_name] = 0.5

    # Flip perspective: 1 - Model_A_Winrate = Model_B_Winrate
    df_pivot = 1 - df_pivot

    # Compute average
    df_pivot["Average"] = df_pivot.mean(axis=1)
    df_pivot.sort_values(by="Average", inplace=True, ascending=True)

    print("\nWINRATES (higher = better than baseline)")
    print("==========================================")
    df_pivot.index.name = None  # Remove index name from output
    print(df_pivot.to_string(float_format="%.3f"))
    print()
