#!/usr/bin/env python3
"""
Analyze LLM-judge evaluation results.

Reads results from slurmpilot jobs and computes winrates for each model
against the baseline across different benchmarks.

Usage:
    python analyse_results.py --path ~/slurmpilot/jobs/<jobname>
    python analyse_results.py  # Uses default path
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def compute_winrate(num_wins: int, num_ties: int, num_losses: int) -> float:
    """Compute winrate with ties counting as 0.5."""
    total = num_wins + num_ties + num_losses
    if total == 0:
        return 0.5
    return (num_wins + 0.5 * num_ties) / total


def load_results(results_path: Path) -> list[dict]:
    """Load all result JSON files from the job directory."""
    result_rows = []

    for result_file in results_path.rglob("*results-*.json"):
        # Filter to only include results in the expected structure
        if result_file.parent.parent.name == "results":
            print(f"Loading: {result_file}")
            with open(result_file, "r") as f:
                res = json.load(f)
                res["winrate"] = compute_winrate(
                    res.get("num_wins", 0),
                    res.get("num_ties", 0),
                    res.get("num_losses", 0),
                )
                result_rows.append(res)

    return result_rows


def format_model_name(full_name: str) -> str:
    """Extract short model name from full path."""
    return full_name.split("/")[-1]


def analyse_results(results_path: Path, baseline_name: str | None = None):
    """Analyze and display evaluation results."""
    print(f"\n{'='*60}")
    print(f"Analyzing results from: {results_path}")
    print(f"{'='*60}\n")

    # Load results
    result_rows = load_results(results_path)

    if not result_rows:
        print("No results found!")
        print(f"Expected results in: {results_path}/*/results/*results-*.json")
        return

    # Create DataFrame
    df = pd.DataFrame(result_rows)
    print("Raw results:")
    print(df[["dataset", "model_A", "model_B", "num_wins", "num_ties", "num_losses", "winrate"]].to_string())
    print()

    # Create pivot table: models vs datasets
    df_pivot = df.pivot_table(index="model_B", columns="dataset", values="winrate")
    df_pivot.index = [format_model_name(x) for x in df_pivot.index]

    # Add baseline row (winrate = 0.5 against itself)
    if baseline_name:
        baseline_short = format_model_name(baseline_name)
        if baseline_short not in df_pivot.index:
            df_pivot.loc[baseline_short] = 0.5

    # Compute average across datasets
    df_pivot["Average"] = df_pivot.mean(axis=1)
    df_pivot = df_pivot.sort_values(by="Average", ascending=False)

    print("\n" + "="*60)
    print("WINRATES (higher = better than baseline)")
    print("="*60)
    print(df_pivot.to_string(float_format="%.3f"))

    # Also show "loss rate" perspective (1 - winrate)
    print("\n" + "="*60)
    print("LOSS RATES (lower = better than baseline)")
    print("="*60)
    print((1 - df_pivot).to_string(float_format="%.3f"))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    best_model = df_pivot["Average"].idxmax()
    best_avg = df_pivot.loc[best_model, "Average"]
    print(f"Best model: {best_model} (avg winrate: {best_avg:.3f})")

    if best_avg > 0.5:
        print(f"  -> Outperforms baseline by {(best_avg - 0.5) * 100:.1f}%")
    elif best_avg < 0.5:
        print(f"  -> Underperforms baseline by {(0.5 - best_avg) * 100:.1f}%")
    else:
        print("  -> Ties with baseline")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM-judge evaluation results")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to slurmpilot job results directory",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="Olmo-3-7B-Instruct-SFT",
        help="Baseline model name (for display)",
    )
    args = parser.parse_args()

    if args.path:
        results_path = Path(args.path).expanduser()
    else:
        # Default: look for most recent job in slurmpilot directory
        slurmpilot_dir = Path("~/slurmpilot/jobs").expanduser()
        if slurmpilot_dir.exists():
            jobs = sorted(slurmpilot_dir.glob("llm-judge-olmo3*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if jobs:
                results_path = jobs[0]
                print(f"Using most recent job: {results_path.name}")
            else:
                print(f"No matching jobs found in {slurmpilot_dir}")
                print("Run with --path <path_to_job> to specify a job directory")
                return
        else:
            print(f"Slurmpilot directory not found: {slurmpilot_dir}")
            print("Run with --path <path_to_job> to specify a job directory")
            return

    analyse_results(results_path, baseline_name=args.baseline)


if __name__ == "__main__":
    main()

