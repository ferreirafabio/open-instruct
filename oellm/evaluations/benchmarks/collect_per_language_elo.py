"""Collect per-language Elo results from log files into a markdown table.

Parses elo_lang_*.log files to extract per-language Elo ratings for each model.

Usage:
    python oellm/evaluations/benchmarks/collect_per_language_elo.py
    python oellm/evaluations/benchmarks/collect_per_language_elo.py --logs-dir oellm/evaluations/logs
"""

import argparse
import glob
import os
import re


LANGUAGES = ["en", "de", "es", "fr", "it", "pt", "pl", "nl", "cs", "ro", "el", "uk"]

MODEL_ORDER = [
    "Baseline",
    "A1-90en", "A2-80en", "A3-70en",
    "B1-90en", "B2-80en",
    "C0-100en",
    "D1-90en", "D2-80en", "D3-70en",
    "E1-90en", "E2-80en", "E3-70en",
    "F1-100en", "F2-75en", "F3-50en", "F4-25en", "F5-0en",
    "G1-100en", "G2-75en", "G3-50en", "G4-25en", "G5-0en",
]

SYMLINK_TO_NAME = {
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
    "eu-F1-100en": "F1-100en",
    "eu-F2-75en": "F2-75en",
    "eu-F3-50en": "F3-50en",
    "eu-F4-25en": "F4-25en",
    "eu-F5-0en": "F5-0en",
    "eu-G1-100en": "G1-100en",
    "eu-G2-75en": "G2-75en",
    "eu-G3-50en": "G3-50en",
    "eu-G4-25en": "G4-25en",
    "eu-G5-0en": "G5-0en",
    "instruct-v2-step3252": "Baseline",
}


def parse_log(log_path: str) -> tuple[str | None, str | None, str | None]:
    """Parse a per-language Elo log file. Returns (model_name, language, elo_string)."""
    model_symlink = None
    language = None
    elo = None

    with open(log_path, errors="ignore") as f:
        content = f.read()

    # Extract model
    m = re.search(r"Model:\s+(\S+)", content)
    if m:
        model_symlink = m.group(1)

    # Extract language
    m = re.search(r"Language:\s+(\S+)", content)
    if m:
        language = m.group(1)

    # Extract Elo rating
    m = re.search(r"<-----.*?([-\d.]+)\s*±\s*([\d.]+)", content)
    if m:
        elo = f"{float(m.group(1)):.0f}±{float(m.group(2)):.0f}"

    # Map symlink to display name
    model_name = None
    if model_symlink:
        for sym, name in SYMLINK_TO_NAME.items():
            if sym in model_symlink:
                model_name = name
                break

    return model_name, language, elo


def main():
    parser = argparse.ArgumentParser(description="Collect per-language Elo results")
    parser.add_argument(
        "--logs-dir",
        default="oellm/evaluations/logs",
        help="Directory with elo_lang_*.log files",
    )
    args = parser.parse_args()

    # Find all per-language Elo logs
    log_files = sorted(glob.glob(os.path.join(args.logs_dir, "elo_lang_*.log")))
    if not log_files:
        print(f"No elo_lang_*.log files found in {args.logs_dir}")
        return

    # Parse all logs
    results: dict[str, dict[str, str]] = {}
    for log in log_files:
        model, lang, elo = parse_log(log)
        if model and lang and elo:
            if model not in results:
                results[model] = {}
            results[model][lang] = elo

    # Print markdown table
    header = "| Model |" + " | ".join(LANGUAGES) + " |"
    separator = "|---|" + "|".join(["---"] * len(LANGUAGES)) + "|"
    print(header)
    print(separator)

    for model in MODEL_ORDER:
        if model not in results:
            continue
        row = f"| **{model}** |"
        for lang in LANGUAGES:
            val = results[model].get(lang, "—")
            row += f" {val} |"
        print(row)

    # Summary
    print(f"\n{len(results)} models, {sum(len(v) for v in results.values())} results collected from {len(log_files)} logs")
    missing = []
    for model in MODEL_ORDER:
        for lang in LANGUAGES:
            if model not in results or lang not in results[model]:
                missing.append(f"{model}/{lang}")
    if missing:
        print(f"Missing: {len(missing)} — {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
