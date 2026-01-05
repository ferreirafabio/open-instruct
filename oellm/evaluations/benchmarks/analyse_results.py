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
    "/work/dlclarge2/ferreira-oellm/open-instruct/oellm/evaluations/benchmarks/OpenJury/results"
).expanduser()
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def collect_result_dirs(cli_dirs: list[str] | None) -> list[Path]:
    """Return the list of result directories to scan."""
    if cli_dirs:
        return [Path(d).expanduser() for d in cli_dirs]

    if not DEFAULT_RESULTS_ROOT.exists():
        return []

    candidates = [d for d in DEFAULT_RESULTS_ROOT.iterdir() if d.is_dir()]
    timestamped = [d for d in candidates if TIMESTAMP_PATTERN.match(d.name)]

    # If timestamped dirs exist, use them all; otherwise fall back to root for legacy runs.
    return timestamped if timestamped else [DEFAULT_RESULTS_ROOT]


parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, choices=["instruct", "think"], default=None)
parser.add_argument(
    "--results-dir",
    action="append",
    help="Path to a results directory. Can be passed multiple times. "
    "Defaults to all timestamped subdirectories under the standard results root, "
    "or the root itself if none are timestamped.",
)
args = parser.parse_args()

result_dirs = collect_result_dirs(args.results_dir)
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

    # Use short names for models in the table
    df["model_B_short"] = df["model_B"].apply(lambda x: x.split("/")[-1])

    # Count runs per model/dataset
    counts = df.groupby(["model_B_short", "dataset"]).size().unstack()

    # Create pivot table (initially Model A winrates)
    df_pivot = df.pivot_table(index="model_B_short", columns="dataset", values="winrate")

    # Add baseline row (0.5)
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
    print(df_pivot.to_string(float_format="%.3f"))
    print()
