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

import pandas as pd

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


def parse_per_language_winrates(result_dir: Path, dataset: str) -> list[dict]:
    """Extract per-language winrates from annotations CSV.

    The instruction_index in m-arena-hard-EU has format '{question_id}-{lang}',
    e.g. '0122ab60646b4961bc39e9c03bdf6bcc-cs'. We parse the language suffix
    and compute winrate per language.
    """
    rows = []
    # Find annotations CSV
    annotation_files = list(result_dir.rglob(f"{dataset}-*-annotations.csv"))
    if not annotation_files:
        return rows

    df = pd.read_csv(annotation_files[0])
    if "instruction_index" not in df.columns or "judge_completion" not in df.columns:
        return rows

    # Extract language from instruction_index
    df["lang"] = df["instruction_index"].astype(str).str.rsplit("-", n=1).str[-1]

    # Parse judge scores: look for score_A and score_B in judge_completion
    def parse_winner(judge_text):
        """Return 'A', 'B', or 'tie' based on judge scores."""
        if not isinstance(judge_text, str):
            return None
        score_a = re.search(r"score_A:\s*(\d+)", judge_text)
        score_b = re.search(r"score_B:\s*(\d+)", judge_text)
        if not score_a or not score_b:
            return None
        a, b = int(score_a.group(1)), int(score_b.group(1))
        if b > a:
            return "B"  # trained model wins
        elif a > b:
            return "A"  # baseline wins
        return "tie"

    df["winner"] = df["judge_completion"].apply(parse_winner)
    df = df.dropna(subset=["winner"])

    for lang, group in df.groupby("lang"):
        n = len(group)
        wins_b = (group["winner"] == "B").sum()
        ties = (group["winner"] == "tie").sum()
        winrate_trained = (wins_b + 0.5 * ties) / n if n > 0 else 0
        rows.append({
            "language": lang,
            "metric": "winrate",
            "value": round(winrate_trained, 4),
            "num_battles": n,
        })

    return rows


def collect_results(result_dirs: dict) -> tuple[list[dict], list[dict]]:
    """Parse all result JSONs and return rows for CSV.

    Returns:
        (aggregate_rows, per_language_rows)
    """
    all_rows = []
    lang_rows = []

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
                # Per-language breakdown for m-arena-hard-EU
                if "m-arena-hard" in dataset:
                    per_lang = parse_per_language_winrates(result_dir, dataset)
                    for lr in per_lang:
                        lang_rows.append({
                            "experiment": experiment,
                            "step": step,
                            "dataset": dataset,
                            "eval_mode": eval_mode,
                            **lr,
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

    return all_rows, lang_rows


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

    rows, lang_rows = collect_results(result_dirs)
    print(f"\nCollected {len(rows)} aggregate rows, {len(lang_rows)} per-language rows")

    if not rows:
        print("No results found. Run evaluations first.")
        return

    # Save aggregate results
    output_path = Path(args.output)
    fieldnames = [
        "experiment", "step", "dataset", "eval_mode",
        "metric", "value", "baseline_value", "num_battles",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved aggregate results to: {output_path}")

    # Save per-language results
    if lang_rows:
        lang_output = output_path.parent / "results_per_language.csv"
        lang_fieldnames = [
            "experiment", "step", "dataset", "eval_mode",
            "language", "metric", "value", "num_battles",
        ]
        with open(lang_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=lang_fieldnames)
            writer.writeheader()
            writer.writerows(lang_rows)
        print(f"Saved per-language results to: {lang_output}")

    # Print summary
    print("\n--- Aggregate Summary ---")
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

    # Print per-language summary
    if lang_rows:
        print("\n--- Per-Language Summary ---")
        # Group by experiment, show language breakdown
        trained_langs = {"de", "es", "fr", "it", "pt", "pl", "nl", "cs"}
        heldout_langs = {"ro", "el"}
        for experiment in experiments:
            exp_langs = [r for r in lang_rows if r["experiment"] == experiment]
            if not exp_langs:
                continue
            steps = sorted(set(r["step"] for r in exp_langs), key=str)
            for step in steps:
                step_langs = [r for r in exp_langs if r["step"] == step]
                step_langs.sort(key=lambda x: x["language"])
                print(f"\n{experiment} step{step} per-language winrates:")
                for r in step_langs:
                    lang = r["language"]
                    marker = ""
                    if lang in trained_langs:
                        marker = " [trained]"
                    elif lang in heldout_langs:
                        marker = " [held-out]"
                    elif lang == "en":
                        marker = " [english]"
                    print(f"  {lang}: {r['value']:.4f} (n={r['num_battles']}){marker}")


if __name__ == "__main__":
    main()
