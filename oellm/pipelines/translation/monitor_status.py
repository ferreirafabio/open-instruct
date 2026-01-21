#!/usr/bin/env python3
"""Generate translation status table.

Creates a markdown table showing completion status for each language/dataset combination.

Usage:
    python oellm/pipelines/translation/monitor_status.py
    python oellm/pipelines/translation/monitor_status.py --output status.md
    watch -n 60 python oellm/pipelines/translation/monitor_status.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

# Languages (24 total: English + 23 EU languages)
LANGUAGES = [
    "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu",
    "ga", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv"
]

# Datasets to translate (17 total, excluding French and code-heavy datasets)
DATASETS = [
    "nvidia-Nemotron-Post-Training-Dataset-v2.parquet",
    "HuggingFaceTB-smoltalk2.parquet",
    "mlabonne-open-perfectblend.parquet",
    "lmsys-lmsys-chat-1m.parquet",
    "Anthropic-hh-rlhf.parquet",
    "stanfordnlp-SHP.parquet",
    "berkeley-nest-Nectar.parquet",
    "lmarena-ai-arena-human-preference-55k.parquet",
    "argilla-ultrafeedback-binarized-preferences-cleaned.parquet",
    "nvidia-HelpSteer2.parquet",
    "nvidia-Aegis-AI-Content-Safety-Dataset-2.0.parquet",
    "argilla-distilabel-intel-orca-dpo-pairs.parquet",
    "HumanLLMs-Human-Like-DPO-Dataset.parquet",
    "argilla-distilabel-capybara-dpo-7k-binarized.parquet",
    "lmsys-mt_bench_human_judgments.parquet",
    "argilla-distilabel-math-preference-dpo.parquet",
    "jondurbin-truthy-dpo-v0.1.parquet",
]
# EXCLUDED: ministere-culture-comparia-votes.parquet (French), microsoft-orca-agentinstruct-1M-v1.parquet (code)

# Short names for display
DATASET_SHORT = {
    "nvidia-Nemotron-Post-Training-Dataset-v2.parquet": "Nemotron",
    "HuggingFaceTB-smoltalk2.parquet": "Smoltalk",
    "mlabonne-open-perfectblend.parquet": "PerfectBlend",
    "lmsys-lmsys-chat-1m.parquet": "LMSYS-1M",
    "Anthropic-hh-rlhf.parquet": "HH-RLHF",
    "stanfordnlp-SHP.parquet": "SHP",
    "berkeley-nest-Nectar.parquet": "Nectar",
    "lmarena-ai-arena-human-preference-55k.parquet": "Arena-55k",
    "argilla-ultrafeedback-binarized-preferences-cleaned.parquet": "UltraFeedback",
    "nvidia-HelpSteer2.parquet": "HelpSteer2",
    "nvidia-Aegis-AI-Content-Safety-Dataset-2.0.parquet": "Aegis",
    "argilla-distilabel-intel-orca-dpo-pairs.parquet": "Orca-DPO",
    "HumanLLMs-Human-Like-DPO-Dataset.parquet": "HumanLLMs",
    "argilla-distilabel-capybara-dpo-7k-binarized.parquet": "Capybara",
    "lmsys-mt_bench_human_judgments.parquet": "MT-Bench",
    "argilla-distilabel-math-preference-dpo.parquet": "Math-DPO",
    "jondurbin-truthy-dpo-v0.1.parquet": "Truthy",
}


def get_status(base_dir: Path, lang: str, dataset: str) -> tuple[str, dict]:
    """Get status for a language/dataset combination.

    Returns:
        (status_symbol, info_dict)
        status_symbol: "done" | "progress" | "copy" | "none"
        info_dict: {"progress": float, "examples": int, ...}
    """
    lang_dir = base_dir / lang
    output_file = lang_dir / dataset
    checkpoint_dir = lang_dir / f"{dataset.replace('.parquet', '')}.checkpoint"
    stats_file = output_file.with_suffix(".stats.json")

    info = {}

    # Check if complete (requires .stats.json to confirm real translation)
    if output_file.exists():
        info["size_mb"] = output_file.stat().st_size / (1024 * 1024)
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    stats = json.load(f)
                    info["translated"] = stats.get("translated", 0)
                    info["total"] = stats.get("total", 0)
                return "done", info
            except (json.JSONDecodeError, OSError):
                pass
        # File exists but no stats = just a copy (e.g., English originals)
        return "copy", info

    # Check if in progress (checkpoint exists)
    if checkpoint_dir.exists():
        # Find latest metadata file
        meta_files = list(checkpoint_dir.glob("*.meta.json"))
        if meta_files:
            latest_meta = max(meta_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_meta) as f:
                    meta = json.load(f)
                    processed = meta.get("processed_count", 0)
                    # Estimate total from sampling (we don't know exact total here)
                    info["processed"] = processed
                    info["timestamp"] = meta.get("timestamp", "")
            except (json.JSONDecodeError, OSError):
                pass
        return "progress", info

    return "none", info


def generate_table(base_dir: Path) -> str:
    """Generate markdown table with translation status."""
    lines = []

    # Header
    lines.append(f"# Translation Status")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    lines.append(f"Legend: {chr(0x2705)} translated | {chr(0x1F7E1)} in progress | {chr(0x1F539)} legacy copy | {chr(0x2B1C)} not started")
    lines.append(f"")

    # Count stats
    total_complete = 0
    total_progress = 0
    total_copy = 0
    total_none = 0
    lang_stats = {lang: {"done": 0, "progress": 0, "copy": 0, "none": 0} for lang in LANGUAGES}

    # Table header
    header = "| Dataset |" + "|".join(f" {lang} " for lang in LANGUAGES) + "|"
    separator = "|" + "|".join(["---"] * (len(LANGUAGES) + 1)) + "|"
    lines.append(header)
    lines.append(separator)

    # Table rows
    for dataset in DATASETS:
        short_name = DATASET_SHORT.get(dataset, dataset[:15])
        row = f"| {short_name} |"

        for lang in LANGUAGES:
            status, info = get_status(base_dir, lang, dataset)

            if status == "done":
                symbol = chr(0x2705)  # Green check
                total_complete += 1
                lang_stats[lang]["done"] += 1
            elif status == "progress":
                symbol = chr(0x1F7E1)  # Yellow circle
                total_progress += 1
                lang_stats[lang]["progress"] += 1
            elif status == "copy":
                symbol = chr(0x1F539)  # Blue diamond (copy/original)
                total_copy += 1
                lang_stats[lang]["copy"] += 1
            else:
                symbol = chr(0x2B1C)  # White square
                total_none += 1
                lang_stats[lang]["none"] += 1

            row += f" {symbol} |"

        lines.append(row)

    # Summary (all 24 languages including English for cross-product translation)
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")
    # All 24 languages need translation (English = translate non-EN rows to EN)
    total_to_translate = len(LANGUAGES) * len(DATASETS)
    lines.append(f"- **Translated**: {total_complete}/{total_to_translate} ({100*total_complete/total_to_translate:.1f}%)")
    lines.append(f"- **In Progress**: {total_progress}")
    lines.append(f"- **Not Started**: {total_none}")
    lines.append(f"")

    # Per-language summary
    lines.append(f"## Per-Language Progress")
    lines.append(f"")
    lines.append(f"| Language | Complete | In Progress | Not Started |")
    lines.append(f"|----------|----------|-------------|-------------|")
    for lang in LANGUAGES:
        stats = lang_stats[lang]
        lines.append(f"| {lang} | {stats['done']}/{len(DATASETS)} | {stats['progress']} | {stats['none']} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate translation status table")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for translated datasets (e.g., .../eu24_600m_1pct)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate status for all eu24_* directories in the translated data folder"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: auto-generate based on base-dir name)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for status files (default: same as this script)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = Path("/work/dlclarge2/ferreira-oellm/open-instruct/data/datasets_mixture_sft_translated")
    output_dir = Path(args.output_dir) if args.output_dir else script_dir

    if args.all:
        # Find all eu24_* directories
        if not data_dir.exists():
            print(f"Error: Data directory does not exist: {data_dir}")
            return

        eu24_dirs = sorted(data_dir.glob("eu24_*"))
        if not eu24_dirs:
            print(f"No eu24_* directories found in {data_dir}")
            return

        print(f"Found {len(eu24_dirs)} translation directories")
        for base_dir in eu24_dirs:
            dir_name = base_dir.name  # e.g., "eu24_600m_1pct"
            output_path = output_dir / f"STATUS_{dir_name.replace('eu24_', '')}.md"

            table = generate_table(base_dir)
            output_path.write_text(table)
            print(f"  {dir_name} -> {output_path.name}")

    else:
        # Single directory mode
        if args.base_dir:
            base_dir = Path(args.base_dir)
        else:
            # Default to first eu24_* directory found
            eu24_dirs = sorted(data_dir.glob("eu24_*"))
            if eu24_dirs:
                base_dir = eu24_dirs[0]
            else:
                print(f"Error: No eu24_* directories found in {data_dir}")
                print("Specify --base-dir or ensure translation directories exist")
                return

        if not base_dir.exists():
            print(f"Warning: Base directory does not exist: {base_dir}")
            print(f"Creating empty status table...")

        table = generate_table(base_dir)

        if args.output:
            output_path = Path(args.output)
        else:
            # Auto-name based on directory
            dir_name = base_dir.name
            output_path = output_dir / f"STATUS_{dir_name.replace('eu24_', '')}.md"

        output_path.write_text(table)
        print(f"Status written to: {output_path}")


if __name__ == "__main__":
    main()
