#!/usr/bin/env python3
"""
Collect Instruct SFT v2 checkpoint evaluation results into a single CSV.

Scans OpenJury result directories for instruct-v2-curve-* results and also
pulls step3252 results from the existing instruct-v2-horeka-* directories.

Only includes alpaca-eval and arena-hard benchmarks (no m-arena-hard-EU).

Usage:
    python oellm/experiments/instruct_v2_checkpoint_eval/collect_results.py
    python oellm/experiments/instruct_v2_checkpoint_eval/collect_results.py --output results.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
RESULTS_ROOT = PROJECT_ROOT / "oellm/evaluations/benchmarks/OpenJury/results"
EXPERIMENT_DIR = PROJECT_ROOT / "oellm/experiments/instruct_v2_checkpoint_eval"

ALLOWED_DATASETS = {"alpaca-eval", "arena-hard"}

# Existing step3252 results (final checkpoint evals from run_evaluation.sh)
STEP3252_DIRS = {
    ("winrate", "fixed"): RESULTS_ROOT / "instruct-v2-horeka-Olmo-3-7B-Instruct-SFT-20260227_205046",
    ("winrate", "both"): RESULTS_ROOT / "instruct-v2-horeka-Olmo-3-7B-Instruct-SFT-20260228_192129",
    ("rubric", "fixed"): RESULTS_ROOT / "instruct-v2-horeka-Olmo-3-7B-Instruct-SFT-20260227_205048",
}


def find_result_dirs():
    """Find all instruct-v2-curve result directories, plus step3252 horeka dirs."""
    dirs = {}

    # Curve results:
    #   winrate: instruct-v2-curve-winrate-{swap_mode}-step{N}-*
    #   rubric:  instruct-v2-curve-rubric-step{N}-* (no swap_mode)
    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        # Try winrate pattern first
        m = re.match(r"instruct-v2-curve-winrate-(fixed|both)-step(\d+)-", d.name)
        if m:
            swap_mode = m.group(1)
            step = int(m.group(2))
            dirs[(step, "winrate", swap_mode)] = d
            continue
        # Try rubric pattern (no swap_mode in dir name)
        m = re.match(r"instruct-v2-curve-rubric-step(\d+)-", d.name)
        if m:
            step = int(m.group(1))
            dirs[(step, "rubric", "fixed")] = d

    # step3252 from existing horeka results
    for (eval_mode, swap_mode), d in STEP3252_DIRS.items():
        if d.exists():
            dirs[(3252, eval_mode, swap_mode)] = d

    return dirs


def parse_winrate_result(result_json: dict) -> dict:
    """Extract winrate metrics from a result JSON."""
    # winrate in JSON is baseline winrate; trained winrate = 1 - winrate
    trained_winrate = 1 - result_json["winrate"]
    return {
        "metric": "winrate",
        "value": round(trained_winrate, 4),
        "baseline_value": round(result_json["winrate"], 4),
        "num_battles": result_json.get("num_battles", 0),
    }


def parse_rubric_result(result_json: dict) -> list[dict]:
    """Extract rubric metrics from a result JSON."""
    rows = []

    # Composite score
    baseline_composite = result_json["model_A_scores"].get("composite_score", 0)
    trained_composite = result_json["model_B_scores"].get("composite_score", 0)
    rows.append({
        "metric": "rubric_composite",
        "value": round(trained_composite, 4),
        "baseline_value": round(baseline_composite, 4),
        "num_battles": result_json.get("num_instructions", 0),
    })

    # Individual criteria
    for criterion in result_json.get("criteria", []):
        key = f"{criterion}_score"
        baseline_score = result_json["model_A_scores"].get(key, 0)
        trained_score = result_json["model_B_scores"].get(key, 0)
        rows.append({
            "metric": f"rubric_{criterion}",
            "value": round(trained_score, 4),
            "baseline_value": round(baseline_score, 4),
            "num_battles": result_json.get("num_instructions", 0),
        })

    return rows


def collect_results(result_dirs: dict) -> list[dict]:
    """Parse all result JSONs and return rows for CSV."""
    all_rows = []

    for (step, eval_mode, swap_mode), result_dir in sorted(result_dirs.items()):
        for result_json_path in result_dir.rglob("results-*.json"):
            with open(result_json_path) as f:
                data = json.load(f)

            dataset = data["dataset"]

            # Filter to allowed datasets only
            if dataset not in ALLOWED_DATASETS:
                continue

            if eval_mode == "winrate":
                row = parse_winrate_result(data)
                all_rows.append({
                    "step": step,
                    "dataset": dataset,
                    "eval_mode": eval_mode,
                    "swap_mode": swap_mode,
                    **row,
                })
            elif eval_mode == "rubric":
                rows = parse_rubric_result(data)
                for row in rows:
                    all_rows.append({
                        "step": step,
                        "dataset": dataset,
                        "eval_mode": eval_mode,
                        "swap_mode": swap_mode,
                        **row,
                    })

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Collect Instruct SFT v2 checkpoint eval results")
    parser.add_argument("--output", type=str,
                        default=str(EXPERIMENT_DIR / "results.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    result_dirs = find_result_dirs()
    print(f"Found {len(result_dirs)} result directories")

    for (step, eval_mode, swap_mode), d in sorted(result_dirs.items()):
        print(f"  step{step:>5d} ({eval_mode}, {swap_mode}): {d.name}")

    rows = collect_results(result_dirs)
    print(f"\nCollected {len(rows)} result rows")

    if not rows:
        print("No results found. Run evaluations first.")
        return

    # Write CSV
    output_path = Path(args.output)
    fieldnames = ["step", "dataset", "eval_mode", "swap_mode", "metric", "value", "baseline_value", "num_battles"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to: {output_path}")

    # Print summary table
    print("\n--- Summary ---")
    steps_found = sorted(set(r["step"] for r in rows))
    datasets = sorted(set(r["dataset"] for r in rows))
    for dataset in datasets:
        for metric_swap in [("winrate", "fixed"), ("winrate", "both"), ("rubric_composite", None)]:
            metric, swap = metric_swap
            vals = {r["step"]: r["value"] for r in rows
                    if r["dataset"] == dataset and r["metric"] == metric
                    and (swap is None or r["swap_mode"] == swap)}
            if vals:
                label = f"{metric}" + (f" ({swap})" if swap else "")
                print(f"\n{dataset} / {label}:")
                for s in steps_found:
                    if s in vals:
                        print(f"  step{s:>5d}: {vals[s]:.4f}")


if __name__ == "__main__":
    main()
