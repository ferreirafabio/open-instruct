#!/usr/bin/env python3
"""
Split preprocessed multilingual parquet files by language.

Takes preprocessed parquets (with 'messages' and 'language' columns) and splits
each into per-language parquet files.

Usage:
    python oellm/pipelines/preprocessing/split_by_language.py
    python oellm/pipelines/preprocessing/split_by_language.py --input-dir data/datasets_multilingual_sft/preprocessed
    python oellm/pipelines/preprocessing/split_by_language.py --datasets fusion-synth wildchat
"""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/datasets_multilingual_sft/preprocessed")
DEFAULT_OUTPUT_DIR = Path("data/datasets_multilingual_sft/by_language")


def split_parquet_by_language(
    input_path: Path,
    output_dir: Path,
    min_samples: int = 1,
) -> dict[str, int]:
    """Split a parquet file by language column.

    Args:
        input_path: Path to input parquet with 'language' column
        output_dir: Output directory for per-language parquets
        min_samples: Minimum samples per language to create a file

    Returns:
        Dict mapping language code to sample count
    """
    dataset_name = input_path.stem
    print(f"\nSplitting: {dataset_name}")

    # Read using pyarrow iter_batches to handle large nested parquets
    # (pq.read_table fails on large files with nested types due to chunked array bug)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(input_path)

    # Check for language column
    schema_names = [f.name for f in pf.schema_arrow]
    if "language" not in schema_names:
        raise ValueError(f"No 'language' column in {input_path}. Columns: {schema_names}")

    # First pass: get language distribution
    all_langs = []
    for batch in pf.iter_batches(batch_size=10000, columns=["language"]):
        all_langs.extend(batch.column("language").to_pylist())
    total_rows = len(all_langs)
    print(f"  Total rows: {total_rows:,}")

    if total_rows == 0:
        return {}

    # Build indices per language
    lang_indices: dict[str, list[int]] = {}
    for idx, lang in enumerate(all_langs):
        if lang:
            lang_indices.setdefault(lang, []).append(idx)

    # Create output directory for this dataset
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Second pass: read full data in batches and accumulate per-language tables
    lang_batches: dict[str, list[pa.RecordBatch]] = {lang: [] for lang in lang_indices}
    row_offset = 0
    for batch in pf.iter_batches(batch_size=10000):
        batch_size = batch.num_rows
        # For each language, find indices in this batch
        for lang, indices in lang_indices.items():
            # Binary search for indices in [row_offset, row_offset + batch_size)
            local_indices = [
                i - row_offset for i in indices
                if row_offset <= i < row_offset + batch_size
            ]
            if local_indices:
                lang_batches[lang].append(batch.take(local_indices))
        row_offset += batch_size

    # Write per-language parquets
    lang_counts = {}
    for lang, batches in sorted(lang_batches.items()):
        if len(batches) == 0 or len(lang_indices[lang]) < min_samples:
            continue

        table = pa.Table.from_batches(batches)
        output_path = dataset_output_dir / f"{lang}.parquet"
        # Write with row_group_size to avoid chunked array issues on read
        pq.write_table(table, output_path, row_group_size=10000)
        lang_counts[lang] = table.num_rows
        print(f"  {lang}: {table.num_rows:>8,} samples -> {output_path.name}")

    print(f"  Split into {len(lang_counts)} languages")
    return lang_counts


def main():
    parser = argparse.ArgumentParser(description="Split preprocessed parquets by language")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing preprocessed parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-language splits",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset names to split (default: all parquets in input dir)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples per language to create a file",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}")
        print("Run preprocess_datasets.py first to create preprocessed parquets.")
        return

    # Find parquet files to split
    if args.datasets:
        parquet_files = [args.input_dir / f"{name}.parquet" for name in args.datasets]
        parquet_files = [p for p in parquet_files if p.exists()]
    else:
        parquet_files = sorted(args.input_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {args.input_dir}")
        return

    print(f"Will split {len(parquet_files)} file(s):")
    for p in parquet_files:
        print(f"  - {p.name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_counts = {}
    for parquet_path in parquet_files:
        counts = split_parquet_by_language(
            parquet_path,
            args.output_dir,
            min_samples=args.min_samples,
        )
        all_counts[parquet_path.stem] = counts

    print(f"\n{'='*60}")
    print("Split complete!")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
