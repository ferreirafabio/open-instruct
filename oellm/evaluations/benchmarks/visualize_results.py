#!/usr/bin/env python3
"""
Visualize LLM-judge evaluation results.

Usage:
    python visualize_results.py                          # Auto-find results (timestamped dirs supported)
    python visualize_results.py --results-dir <path>     # Specify one or more results dirs
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_ROOT = Path(__file__).parent / "OpenJury" / "results"
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def collect_result_dirs(cli_dirs: list[str] | None) -> list[Path]:
    """Return result directories to scan."""
    if cli_dirs:
        return [Path(d).expanduser() for d in cli_dirs]

    if not DEFAULT_RESULTS_ROOT.exists():
        return []

    candidates = [d for d in DEFAULT_RESULTS_ROOT.iterdir() if d.is_dir()]
    timestamped = [d for d in candidates if TIMESTAMP_PATTERN.match(d.name)]

    # Use all timestamped runs if present; otherwise fall back to legacy root.
    return timestamped if timestamped else [DEFAULT_RESULTS_ROOT]


def load_results(results_dirs: list[Path]) -> list[dict]:
    """Load all result JSON files from the provided results directories."""
    results = []
    for results_dir in results_dirs:
        for result_file in results_dir.rglob("results-*.json"):
            with open(result_file) as f:
                data = json.load(f)
                # Add friendly names
                data["model_A_short"] = Path(data["model_A"]).name
                data["model_B_short"] = Path(data["model_B"]).name
                results.append(data)
    return results


def plot_winrate_bar(results: list[dict], output_path: Path):
    """Create a bar chart comparing model winrates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, res in enumerate(results):
        # Winrate for model B (trained) = 1 - winrate_A
        winrate_baseline = res["winrate"]
        winrate_trained = 1 - winrate_baseline
        
        x = np.arange(2)
        width = 0.35
        
        bars = ax.bar(x, [winrate_baseline * 100, winrate_trained * 100], 
                      width, color=['#ff6b6b', '#4ecdc4'])
        
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title(f'Model Comparison on {results[0]["dataset"]}\n(judged by {Path(results[0]["judge_model"]).name})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n(Olmo-3-7B-Instruct-SFT)', 'Trained\n(dolci-instruct-sft)'], fontsize=11)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Tie line')
    ax.legend()
    
    # Add summary text
    res = results[0]
    summary = f"Battles: {res['num_battles']} | Wins: {res['num_losses']} | Losses: {res['num_wins']} | Ties: {res['num_ties']}"
    ax.text(0.5, -0.12, summary, transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_preference_distribution(results: list[dict], output_path: Path):
    """Create a histogram of preference scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for res in results:
        preferences = np.array(res["preferences"])
        
        # Preferences > 0.5 favor model A (baseline), < 0.5 favor model B (trained)
        ax.hist(preferences, bins=20, range=(0, 1), 
                color='#667eea', edgecolor='white', alpha=0.8)
        
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Tie threshold')
        ax.axvline(x=preferences.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean: {preferences.mean():.3f}')
    
    ax.set_xlabel('Preference Score (< 0.5 = Trained wins, > 0.5 = Baseline wins)', fontsize=11)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Judge Preference Distribution\n{results[0]["dataset"]}', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(results: list[dict]):
    """Print a text summary of results."""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    for res in results:
        winrate_trained = 1 - res["winrate"]
        print(f"\nDataset: {res['dataset']}")
        print(f"Judge: {Path(res['judge_model']).name}")
        print(f"Date: {res['date'][:10]}")
        print()
        print(f"  Baseline (Olmo-3-7B-Instruct-SFT): {res['winrate']*100:.1f}%")
        print(f"  Trained (dolci-instruct-sft):      {winrate_trained*100:.1f}%")
        print()
        print(f"  Wins:  {res['num_losses']:3d} (trained)")
        print(f"  Losses:{res['num_wins']:3d} (baseline)")
        print(f"  Ties:  {res['num_ties']:3d}")
        print()
        
        diff = (winrate_trained - 0.5) * 100
        if diff > 0:
            print(f"  ✅ Trained model outperforms baseline by +{diff:.1f}%")
        elif diff < 0:
            print(f"  ❌ Trained model underperforms baseline by {diff:.1f}%")
        else:
            print(f"  🤝 Models are tied")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results-dir", type=str, default=None,
                        action="append",
                        help="Path to a results directory. Can be passed multiple times. "
                        "Defaults to all timestamped subdirectories under the standard results root, "
                        "or the root itself if none are timestamped.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()
    
    # Find results directories
    result_dirs = collect_result_dirs(args.results_dir)
    if not result_dirs:
        print(f"No results directories found under {DEFAULT_RESULTS_ROOT}")
        return
    
    # Load results
    results = load_results(result_dirs)
    if not results:
        print(f"No results found in: {', '.join(str(p) for p in result_dirs)}")
        return
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else result_dirs[0]
    
    # Print summary
    print_summary(results)
    
    # Generate plots
    plot_winrate_bar(results, output_dir / "winrate_comparison.png")
    plot_preference_distribution(results, output_dir / "preference_distribution.png")
    
    print(f"\n📈 Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()




