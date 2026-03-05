#!/usr/bin/env python3
"""
Profile multilingual datasets for per-language sample counts.

Downloads each dataset from HuggingFace, counts samples per language,
and reports statistics. Output JSON for planning mixture compositions.

Usage:
    python oellm/pipelines/preprocessing/profile_multilingual_datasets.py
    python oellm/pipelines/preprocessing/profile_multilingual_datasets.py --datasets fusion-synth wildchat
    python oellm/pipelines/preprocessing/profile_multilingual_datasets.py --output data/datasets_multilingual_sft/profiles/
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# Datasets to profile
DATASETS = {
    "fusion-synth": {
        "id": "CohereLabs/fusion-synth-data-ufb",
        "split": "train",
        "language_col": "language_code",
    },
    "wildchat": {
        "id": "allenai/WildChat-1M",
        "split": "train",
        "language_col": "language",
        "filter_toxic": True,
    },
    "lmsys-chat": {
        "id": "lmsys/lmsys-chat-1m",
        "split": "train",
        "language_col": "language",
    },
    "oasst2": {
        "id": "OpenAssistant/oasst2",
        "split": "train+validation",
        "language_col": "lang",
    },
}

# Languages we care about for the multilingual EU experiment
TARGET_LANGUAGES = {"en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"}


def profile_dataset(name: str, config: dict) -> dict:
    """Profile a single dataset for language distribution."""
    print(f"\n{'='*60}")
    print(f"Profiling: {name} ({config['id']})")
    print(f"{'='*60}")

    try:
        ds = load_dataset(config["id"], split=config["split"], trust_remote_code=True)
    except (TypeError, ValueError):
        # Newer HF datasets versions don't support trust_remote_code
        ds = load_dataset(config["id"], split=config["split"])
    total_rows = len(ds)
    print(f"  Total rows: {total_rows:,}")

    lang_col = config["language_col"]
    if lang_col not in ds.column_names:
        print(f"  WARNING: Language column '{lang_col}' not found. Available: {ds.column_names}")
        return {"total": total_rows, "per_language": {}, "error": f"No '{lang_col}' column"}

    # Vectorized counting via pandas for speed on 1M+ row datasets
    df = ds.to_pandas()
    langs = df[lang_col].fillna("unknown").replace("", "unknown")

    # Filter toxic in a single pass if needed
    toxic_count = 0
    if config.get("filter_toxic") and "toxic" in df.columns:
        toxic_mask = df["toxic"].fillna(False).astype(bool)
        toxic_count = int(toxic_mask.sum())
        langs = langs[~toxic_mask]

    lang_counts = Counter(langs.value_counts().to_dict())

    result = {
        "total": sum(lang_counts.values()),
        "per_language": dict(lang_counts.most_common()),
        "num_languages": len(lang_counts),
    }

    if toxic_count:
        result["toxic_filtered"] = toxic_count

    # Print summary
    print(f"  Non-toxic total: {result['total']:,}")
    print(f"  Languages found: {result['num_languages']}")
    print(f"  Target language counts:")
    for lang in sorted(TARGET_LANGUAGES):
        count = lang_counts.get(lang, 0)
        if count > 0:
            pct = count / result["total"] * 100
            print(f"    {lang}: {count:>8,} ({pct:.1f}%)")

    return result


def merge_profiles(output_dir: Path) -> dict:
    """Merge per-dataset profile JSONs into a single file."""
    merged = {}
    for json_file in sorted(output_dir.glob("profile_*.json")):
        with open(json_file) as f:
            data = json.load(f)
        merged.update(data)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Profile multilingual datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to profile (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets_multilingual_sft/profiles"),
        help="Output directory for profile JSON",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge per-dataset profiles into dataset_profiles.json",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merged = merge_profiles(args.output)
        output_path = args.output / "dataset_profiles.json"
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged {len(merged)} profiles into: {output_path}")
        for name, data in merged.items():
            print(f"  {name}: {data['total']:,} samples, {data['num_languages']} languages")
        return

    datasets_to_profile = args.datasets or list(DATASETS.keys())

    for name in datasets_to_profile:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
            continue
        result = profile_dataset(name, DATASETS[name])

        # Save per-dataset profile (safe for parallel SLURM array tasks)
        per_dataset_path = args.output / f"profile_{name}.json"
        with open(per_dataset_path, "w") as f:
            json.dump({name: result}, f, indent=2)
        print(f"  Saved: {per_dataset_path}")

    # If running all datasets at once, also save merged
    if len(datasets_to_profile) == len(DATASETS):
        merged = merge_profiles(args.output)
        output_path = args.output / "dataset_profiles.json"
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\n{'='*60}")
        print(f"All profiles saved to: {output_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
