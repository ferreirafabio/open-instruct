"""Per-language winrate analysis from ELO judge caches.

Reads judge cache CSV files (from estimate_elo_ratings.py runs with Qwen3.5-27B)
and computes per-language winrate for each model. No GPU needed.

Usage:
    cd oellm/evaluations/benchmarks/OpenJury
    python ../analyze_per_language_elo.py

    # Or specific models:
    python ../analyze_per_language_elo.py --models A1-90en D1-90en
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Add OpenJury to path for PairScore import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "OpenJury"))

from judgearena.evaluate import PairScore

CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../data/openjury-eval-data/cache/elo",
)

EU_LANGS = ["en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"]

# Model symlink → display name mapping
MODEL_MAP = {
    "eu-A1-90en-step238": "A1-90en",
    "eu-A2-80en-step228": "A2-80en",
    "eu-A3-70en-step216": "A3-70en",
    "eu-B1-90en": "B1-90en",
    "eu-B2-80en": "B2-80en",
    "eu-C0-100en": "C0-100en",
    "eu-D1-90en": "D1-90en",
    "eu-D2-80en": "D2-80en",
    "eu-D3-70en": "D3-70en",
    "eu-E1-90en": "E1-90en",
    "eu-E2-80en": "E2-80en",
    "eu-E3-70en": "E3-70en",
    "instruct-v2-step3252": "Baseline",
}


def detect_language(text: str) -> str:
    """Detect language using fast_langdetect (same as estimate_elo_ratings.py)."""
    try:
        from fast_langdetect import detect_language as fld
        return fld(str(text)[:500]).lower()
    except Exception:
        return "unknown"


def compute_winrate(df: pd.DataFrame) -> float:
    """Compute winrate from judge cache dataframe."""
    score_parser = PairScore()

    wins = 0
    losses = 0
    ties = 0

    for _, row in df.iterrows():
        pref = score_parser.parse_model_raw(str(row["judge_completion"]))
        is_pos_a = row["our_model_is_position_a"]

        if pref is None or pref == 0.5:
            ties += 1
        elif pref < 0.5:
            # Judge prefers A
            if is_pos_a:
                wins += 1
            else:
                losses += 1
        else:
            # Judge prefers B
            if is_pos_a:
                losses += 1
            else:
                wins += 1

    total = wins + losses + ties
    if total == 0:
        return 0.0
    return wins / total


def analyze_model(cache_file: str) -> pd.DataFrame:
    """Analyze a single model's judge cache and return per-language winrates."""
    df = pd.read_csv(cache_file)

    # Detect language for each instruction
    df["lang"] = df["instruction"].apply(detect_language)

    score_parser = PairScore()

    results = []
    for _, row in df.iterrows():
        pref = score_parser.parse_model_raw(str(row["judge_completion"]))
        is_pos_a = row["our_model_is_position_a"]

        if pref is None or pref == 0.5:
            our_win = 0.5  # tie counts as half
        elif pref < 0.5:
            our_win = 1.0 if is_pos_a else 0.0
        else:
            our_win = 0.0 if is_pos_a else 1.0

        results.append({"lang": row["lang"], "our_win": our_win})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Per-language ELO winrate analysis")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Filter to specific models (e.g., A1-90en D1-90en). Default: all.",
    )
    parser.add_argument(
        "--cache-dir",
        default=CACHE_DIR,
        help="Path to cache directory",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    judge_files = sorted(glob.glob(os.path.join(cache_dir, "judge_*Qwen3.5*200*")))

    if not judge_files:
        print(f"No judge cache files found in {cache_dir}")
        return

    # Collect results
    all_results = {}

    for jf in judge_files:
        basename = os.path.basename(jf)
        # Extract model symlink from filename
        model_symlink = None
        for sym, name in MODEL_MAP.items():
            if sym.replace("/", "_") in basename or sym in basename:
                model_symlink = sym
                break

        if model_symlink is None:
            continue

        display_name = MODEL_MAP[model_symlink]

        # Filter if --models specified
        if args.models and display_name not in args.models:
            continue

        # Check row count
        df = pd.read_csv(jf)
        if len(df) < 1000:
            print(f"SKIP {display_name}: only {len(df)} rows (corrupted cache)")
            continue

        print(f"Analyzing {display_name} ({len(df)} battles)...", flush=True)
        battle_df = analyze_model(jf)
        all_results[display_name] = battle_df

    if not all_results:
        print("No valid results to analyze.")
        return

    # Print per-language winrate table
    print("\n" + "=" * 100)
    print("PER-LANGUAGE WINRATE (% of battles won by our model vs arena opponents)")
    print("=" * 100)

    # Header
    header = f"{'Model':<12}"
    for lang in EU_LANGS:
        header += f" {lang:>5}"
    header += f" {'ALL':>5}"
    print(header)
    print("-" * len(header))

    for name in [
        "Baseline",
        "A1-90en", "A2-80en", "A3-70en",
        "B1-90en", "B2-80en",
        "C0-100en",
        "D1-90en", "D2-80en", "D3-70en",
        "E1-90en", "E2-80en", "E3-70en",
    ]:
        if name not in all_results:
            continue

        battle_df = all_results[name]
        row = f"{name:<12}"

        for lang in EU_LANGS:
            subset = battle_df[battle_df["lang"] == lang]
            if len(subset) == 0:
                row += f"    —"
            else:
                wr = subset["our_win"].mean() * 100
                row += f" {wr:4.1f}%"

        overall_wr = battle_df["our_win"].mean() * 100
        row += f" {overall_wr:4.1f}%"
        print(row)

    # Print per-language battle counts
    print("\n" + "=" * 100)
    print("BATTLES PER LANGUAGE")
    print("=" * 100)

    first_model = list(all_results.values())[0]
    counts = first_model.groupby("lang").size()
    for lang in EU_LANGS:
        if lang in counts.index:
            print(f"  {lang}: {counts[lang]}")

    # A-track vs D-track comparison
    print("\n" + "=" * 100)
    print("A-TRACK vs D-TRACK COMPARISON (winrate difference: A - D)")
    print("=" * 100)

    a_models = [n for n in ["A1-90en", "A2-80en", "A3-70en"] if n in all_results]
    d_models = [n for n in ["D1-90en", "D2-80en", "D3-70en"] if n in all_results]

    if a_models and d_models:
        header = f"{'Comparison':<20}"
        for lang in EU_LANGS:
            header += f" {lang:>5}"
        header += f" {'ALL':>5}"
        print(header)
        print("-" * len(header))

        for a_name, d_name in zip(a_models, d_models):
            a_df = all_results[a_name]
            d_df = all_results[d_name]
            row = f"{a_name} - {d_name:<8}"

            for lang in EU_LANGS:
                a_sub = a_df[a_df["lang"] == lang]
                d_sub = d_df[d_df["lang"] == lang]
                if len(a_sub) == 0 or len(d_sub) == 0:
                    row += f"    —"
                else:
                    diff = (a_sub["our_win"].mean() - d_sub["our_win"].mean()) * 100
                    sign = "+" if diff > 0 else ""
                    row += f" {sign}{diff:3.1f}%"

            a_all = a_df["our_win"].mean() * 100
            d_all = d_df["our_win"].mean() * 100
            diff = a_all - d_all
            sign = "+" if diff > 0 else ""
            row += f" {sign}{diff:3.1f}%"
            print(row)


if __name__ == "__main__":
    main()
