#!/usr/bin/env python3
"""
Assemble a multilingual dataset mixture from per-language parquet files.

Given a YAML config defining language ratios and data sources, this script:
1. Computes per-language sample counts
2. Samples from per-language parquets (with source priority)
3. Outputs a merged parquet ready for tokenization

Usage:
    python oellm/pipelines/preprocessing/assemble_mixture.py --config oellm/configs/multilingual_trackA_90en.yaml
    python oellm/pipelines/preprocessing/assemble_mixture.py --config oellm/configs/multilingual_trackA_90en.yaml --dry-run
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from oellm.utils.language_mixer import LanguageMixer

DEFAULT_BY_LANGUAGE_DIR = Path("data/datasets_multilingual_sft/by_language")
DEFAULT_OUTPUT_DIR = Path("data/datasets_multilingual_sft/assembled")


def load_language_data(
    language: str,
    sources: list[str],
    by_language_dir: Path,
    n_needed: int,
    seed: int = 42,
) -> pa.Table | None:
    """Load and sample data for a single language from multiple sources.

    Sources are tried in priority order. If the first source doesn't have
    enough data, we take what's available and try the next source.

    Uses pyarrow tables to handle nested data (messages column).

    Args:
        language: Language code (e.g., "de")
        sources: List of source dataset names in priority order
        by_language_dir: Root directory containing per-language parquets
        n_needed: Number of samples needed
        seed: Random seed for sampling

    Returns:
        pyarrow Table with sampled data, or None if no data found
    """
    collected = []
    remaining = n_needed

    for source in sources:
        if remaining <= 0:
            break

        parquet_path = by_language_dir / source / f"{language}.parquet"
        if not parquet_path.exists():
            continue

        # Read using iter_batches to handle large nested parquets
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        if total_rows == 0:
            continue

        if total_rows <= remaining:
            # Take all rows — read in batches and concat
            batches = list(pf.iter_batches(batch_size=10000))
            table = pa.Table.from_batches(batches)
            collected.append(table)
            remaining -= total_rows
        else:
            # Sample by selecting random indices
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(total_rows), remaining))
            # Read in batches and select the needed indices
            result_batches = []
            row_offset = 0
            idx_pos = 0
            for batch in pf.iter_batches(batch_size=10000):
                batch_size = batch.num_rows
                local_indices = []
                while idx_pos < len(indices) and indices[idx_pos] < row_offset + batch_size:
                    local_indices.append(indices[idx_pos] - row_offset)
                    idx_pos += 1
                if local_indices:
                    result_batches.append(batch.take(local_indices))
                row_offset += batch_size
                if idx_pos >= len(indices):
                    break
            collected.append(pa.Table.from_batches(result_batches))
            remaining = 0

    if not collected:
        return None

    # Unify column order across sources before concatenating
    target_schema = collected[0].schema
    unified = []
    for t in collected:
        if t.schema != target_schema:
            t = t.select([f.name for f in target_schema])
        unified.append(t)
    return pa.concat_tables(unified)


def assemble_mixture(
    config_path: Path,
    by_language_dir: Path = DEFAULT_BY_LANGUAGE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> Path | None:
    """Assemble a dataset mixture from a YAML config.

    Args:
        config_path: Path to YAML config
        by_language_dir: Directory with per-language parquet files
        output_dir: Output directory for assembled parquet
        dry_run: If True, compute counts but don't write output

    Returns:
        Path to output parquet, or None if dry run
    """
    mixer = LanguageMixer.from_yaml(config_path)
    config = mixer.config

    total_samples = config.total_samples
    if total_samples is None:
        raise ValueError("Config must specify 'total_samples'")

    sources = config.sources
    if not sources:
        raise ValueError("Config must specify 'sources' (list of dataset names)")

    name = config.name or config_path.stem
    seed = config.seed

    # Compute per-language sample counts
    samples_per_lang = mixer.compute_samples_per_language(total_samples)

    print(f"\n{'='*60}")
    print(f"Assembling mixture: {name}")
    print(f"{'='*60}")
    print(mixer.get_config_summary())
    print(f"\nPer-language targets:")
    for lang, count in sorted(samples_per_lang.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count:>8,}")
    print(f"  Total: {sum(samples_per_lang.values()):>8,}")

    if dry_run:
        print("\n[DRY RUN] Would assemble the above mixture")
        return None

    # Collect data for each language
    all_tables = []
    actual_counts = {}

    for lang, n_needed in sorted(samples_per_lang.items(), key=lambda x: -x[1]):
        if n_needed == 0:
            continue

        table = load_language_data(lang, sources, by_language_dir, n_needed, seed)
        actual = table.num_rows if table is not None else 0
        actual_counts[lang] = actual

        if actual < n_needed:
            warnings.warn(
                f"Language '{lang}': only {actual:,} samples available, "
                f"needed {n_needed:,} ({actual/n_needed*100:.0f}%)"
            )
        print(f"\n  {lang}: {actual:,}/{n_needed:,} samples collected")

        if table is not None and table.num_rows > 0:
            all_tables.append(table)

    if not all_tables:
        print("No data collected!")
        return None

    # Merge and shuffle — unify column order across languages.
    # Break large tables into small chunks (<=50k rows) to avoid
    # pyarrow 32-bit offset overflow on nested string arrays during take().
    target_schema = all_tables[0].schema
    MAX_CHUNK = 50000
    chunks = []  # list of small pa.Table
    for t in all_tables:
        if t.schema != target_schema:
            t = t.select([f.name for f in target_schema])
        for start in range(0, t.num_rows, MAX_CHUNK):
            chunks.append(t.slice(start, min(MAX_CHUNK, t.num_rows - start)))

    # Build global row index → (chunk_idx, local_row)
    row_map = []
    for cidx, c in enumerate(chunks):
        for i in range(c.num_rows):
            row_map.append((cidx, i))

    rng = random.Random(seed)
    rng.shuffle(row_map)

    # Stream shuffled chunks to parquet to avoid offset overflow
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.parquet"
    WRITE_SIZE = 10000

    with pq.ParquetWriter(output_path, target_schema, use_dictionary=False) as writer:
        for write_start in range(0, len(row_map), WRITE_SIZE):
            batch = row_map[write_start:write_start + WRITE_SIZE]
            # Group by chunk for efficient .take()
            chunk_groups: dict[int, list[tuple[int, int]]] = {}
            for pos, (cidx, local_row) in enumerate(batch):
                chunk_groups.setdefault(cidx, []).append((pos, local_row))

            # Extract rows, tagged with their position in batch
            parts: list[tuple[int, pa.Table]] = []
            for cidx, pos_rows in chunk_groups.items():
                local_indices = [r for _, r in pos_rows]
                positions = [p for p, _ in pos_rows]
                taken = chunks[cidx].take(local_indices)
                for i, pos in enumerate(positions):
                    parts.append((pos, taken.slice(i, 1)))

            parts.sort(key=lambda x: x[0])
            batch_table = pa.concat_tables([p for _, p in parts])
            writer.write_table(batch_table)

    # Re-read metadata for reporting (cheap — just reads footer)
    pf = pq.ParquetFile(output_path)
    merged_rows = pf.metadata.num_rows

    n_languages = len(actual_counts)

    print(f"\n  Merged: {merged_rows:,} samples")
    print(f"  Languages: {n_languages}")

    # Save metadata
    metadata = {
        "name": name,
        "config_path": str(config_path),
        "total_samples_target": total_samples,
        "total_samples_actual": merged_rows,
        "per_language_actual": actual_counts,
        "sources": sources,
        "seed": seed,
    }
    metadata_path = output_dir / f"{name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to: {output_path}")
    print(f"  Metadata: {metadata_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Assemble multilingual dataset mixture")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML mixture config",
    )
    parser.add_argument(
        "--by-language-dir",
        type=Path,
        default=DEFAULT_BY_LANGUAGE_DIR,
        help="Directory with per-language parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for assembled parquet",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be assembled without writing",
    )
    args = parser.parse_args()

    assemble_mixture(
        config_path=args.config,
        by_language_dir=args.by_language_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
