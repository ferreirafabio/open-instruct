#!/usr/bin/env python3
"""
Collect multilingual EU evaluation results into a single CSV.

Scans OpenJury result directories for multilingual-eu-* results.

Usage:
    python oellm/experiments/multilingual_eu/collect_results.py
    python oellm/experiments/multilingual_eu/collect_results.py --output results.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path("/work/dlclarge2/ferreira-oellm/open-instruct")
RESULTS_ROOT = PROJECT_ROOT / "oellm/evaluations/benchmarks/OpenJury/results/multilingual_eu"
EXPERIMENT_DIR = PROJECT_ROOT / "oellm/experiments/multilingual_eu"


def find_result_dirs() -> dict:
    """Find all multilingual-eu result directories."""
    dirs = {}

    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        # Match: multilingual-eu-{experiment}-{eval_mode}-step{N}-*
        m = re.match(
            r"multilingual-eu-([A-Z]\d+-\d+en)-(winrate|rubric)-step(\w+)-",
            d.name,
        )
        if m:
            experiment = m.group(1)
            eval_mode = m.group(2)
            step = m.group(3)
            # Try to convert step to int, keep as string for "final"
            try:
                step = int(step)
            except ValueError:
                pass
            dirs[(experiment, step, eval_mode)] = d

    return dirs


def parse_winrate_result(result_json: dict) -> dict:
    """Extract winrate metrics from a result JSON."""
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

    baseline_composite = result_json["model_A_scores"].get("composite_score", 0)
    trained_composite = result_json["model_B_scores"].get("composite_score", 0)
    rows.append({
        "metric": "rubric_composite",
        "value": round(trained_composite, 4),
        "baseline_value": round(baseline_composite, 4),
        "num_battles": result_json.get("num_instructions", 0),
    })

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

    for (experiment, step, eval_mode), result_dir in sorted(result_dirs.items()):
        for result_json_path in result_dir.rglob("results-*.json"):
            with open(result_json_path) as f:
                data = json.load(f)

            dataset = data["dataset"]

            if eval_mode == "winrate":
                row = parse_winrate_result(data)
                all_rows.append({
                    "experiment": experiment,
                    "step": step,
                    "dataset": dataset,
                    "eval_mode": eval_mode,
                    **row,
                })
            elif eval_mode == "rubric":
                rows = parse_rubric_result(data)
                for row in rows:
                    all_rows.append({
                        "experiment": experiment,
                        "step": step,
                        "dataset": dataset,
                        "eval_mode": eval_mode,
                        **row,
                    })

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Collect multilingual EU eval results")
    parser.add_argument(
        "--output",
        type=str,
        default=str(EXPERIMENT_DIR / "results.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    result_dirs = find_result_dirs()
    print(f"Found {len(result_dirs)} result directories")

    for (experiment, step, eval_mode), d in sorted(result_dirs.items()):
        print(f"  {experiment} step{step} ({eval_mode}): {d.name}")

    rows = collect_results(result_dirs)
    print(f"\nCollected {len(rows)} result rows")

    if not rows:
        print("No results found. Run evaluations first.")
        return

    output_path = Path(args.output)
    fieldnames = [
        "experiment", "step", "dataset", "eval_mode",
        "metric", "value", "baseline_value", "num_battles",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to: {output_path}")

    # Print summary
    print("\n--- Summary ---")
    experiments = sorted(set(r["experiment"] for r in rows))
    datasets = sorted(set(r["dataset"] for r in rows))
    for experiment in experiments:
        print(f"\n{experiment}:")
        for dataset in datasets:
            winrate_rows = [
                r for r in rows
                if r["experiment"] == experiment
                and r["dataset"] == dataset
                and r["metric"] == "winrate"
            ]
            for r in sorted(winrate_rows, key=lambda x: (str(x["step"]))):
                print(f"  {dataset} step{r['step']}: winrate={r['value']:.4f}")


if __name__ == "__main__":
    main()
