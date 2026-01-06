#!/usr/bin/env python3
"""
Quick script to summarize preference distributions from OpenJury results.

Usage:
    python summarize_preferences.py
    python summarize_preferences.py --results-dir /path/to/results
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize preference distributions")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to results directory. Defaults to OpenJury/results.",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser()
    else:
        results_dir = Path(__file__).parent / "OpenJury" / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print(f"Scanning: {results_dir}\n")
    
    # Get model names from first result file
    result_files = list(sorted(results_dir.rglob("results-*.json")))
    if not result_files:
        print("No result files found!")
        return
    
    if result_files:
        with open(result_files[0]) as f:
            first_data = json.load(f)
        model_a = first_data.get("model_A", "unknown")
        model_b = first_data.get("model_B", "unknown")
        
        # Resolve symlinks and get actual model name
        def resolve_model_name(raw: str) -> str:
            # Strip provider prefix (e.g., "VLLM/")
            path_str = raw.split("VLLM/", 1)[-1] if "VLLM/" in raw else raw
            p = Path(path_str)
            if p.exists():
                try:
                    p = p.resolve()  # Follow symlinks
                except Exception:
                    pass
            return p.name
        
        model_a_short = resolve_model_name(model_a)
        model_b_short = resolve_model_name(model_b)
        print(f"Model A (baseline): {model_a_short}")
        print(f"Model B (ours):     {model_b_short}")
        print()
    
    print("=" * 96)
    print(f"{'Dataset':<25} {'A Wins':>10} {'B Wins':>10} {'Ties':>10} {'Total':>10} {'A Winrate':>12} {'B Winrate':>12}")
    print("=" * 96)

    b_winrates = []
    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)

        preferences = data.get("preferences", [])
        dataset = data.get("dataset", "unknown")

        # preference < 0.5 means Model A wins (higher score for A)
        # preference > 0.5 means Model B wins (higher score for B)
        a_wins = sum(1 for p in preferences if p is not None and p < 0.5)
        b_wins = sum(1 for p in preferences if p is not None and p > 0.5)
        ties = sum(1 for p in preferences if p is not None and p == 0.5)
        total = len(preferences)
        # Same formula as analyse_results_original.py: (num_wins + 0.5 * num_ties) / total
        a_winrate = (a_wins + 0.5 * ties) / total if total > 0 else 0
        b_winrate = 1 - a_winrate
        b_winrates.append(b_winrate)

        print(f"{dataset:<25} {a_wins:>10} {b_wins:>10} {ties:>10} {total:>10} {a_winrate:>11.2f} {b_winrate:>11.2f}")

    print("=" * 96)
    
    # Print average B winrate
    if b_winrates:
        avg_b_winrate = sum(b_winrates) / len(b_winrates)
        print(f"{'AVERAGE':<25} {'':>10} {'':>10} {'':>10} {'':>10} {1-avg_b_winrate:>11.2f} {avg_b_winrate:>11.2f}")
        print("=" * 96)
    
    print("\nNote: A wins = preference < 0.5, B wins = preference > 0.5, Ties = 0.5")
    print("      A Winrate = (A_wins + 0.5 * Ties) / Total  (same as analyse_results_original.py)")


if __name__ == "__main__":
    main()

