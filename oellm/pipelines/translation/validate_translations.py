#!/usr/bin/env python3
"""
Validate translation quality for EU language datasets.

This script performs automated quality checks on translated datasets:
1. Length ratio check (translated vs original)
2. Language detection (verify target language)
3. Completeness check (no empty translations)
4. Entity preservation (numbers, URLs)
5. Export random samples for manual review

Usage:
    python validate_translations.py --input-dir /path/to/translations --lang de
    python validate_translations.py --input-dir /path/to/translations --all

Output:
    - Validation report in JSON format
    - Random samples for manual review (optional)
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from language_codes import EU_LANGUAGES, get_language_name

# Optional: langdetect for language detection
try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 42  # Reproducible results
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed. Language detection will be skipped.")
    print("Install with: pip install langdetect")


class TranslationValidator:
    """Validate translation quality."""

    def __init__(self, english_dir: Path, translated_dir: Path, target_lang: str):
        """
        Initialize validator.

        Args:
            english_dir: Directory with English parquet files
            translated_dir: Directory with translated parquet files
            target_lang: Target language ISO code
        """
        self.english_dir = Path(english_dir)
        self.translated_dir = Path(translated_dir)
        self.target_lang = target_lang
        self.results = {}

    def validate_length_ratio(
        self,
        original: str,
        translated: str,
        min_ratio: float = 0.3,
        max_ratio: float = 4.0,
    ) -> tuple[bool, float]:
        """
        Check if length ratio is within acceptable bounds.

        Args:
            original: Original English text
            translated: Translated text
            min_ratio: Minimum acceptable ratio
            max_ratio: Maximum acceptable ratio

        Returns:
            (is_valid, ratio)
        """
        if not original or not translated:
            return False, 0.0

        ratio = len(translated) / len(original)
        is_valid = min_ratio <= ratio <= max_ratio
        return is_valid, ratio

    def detect_language(self, text: str) -> tuple[str | None, float]:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            (detected_language, confidence) or (None, 0) if detection fails
        """
        if not HAS_LANGDETECT or not text or len(text.strip()) < 20:
            return None, 0.0

        try:
            detected = detect(text)
            return detected, 1.0  # langdetect doesn't provide confidence
        except Exception:
            return None, 0.0

    def check_entity_preservation(self, original: str, translated: str) -> dict:
        """
        Check if entities (numbers, URLs) are preserved.

        Args:
            original: Original English text
            translated: Translated text

        Returns:
            Dict with preservation stats
        """
        # Extract numbers
        original_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", original))
        translated_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", translated))

        # Extract URLs
        url_pattern = r"https?://\S+"
        original_urls = set(re.findall(url_pattern, original))
        translated_urls = set(re.findall(url_pattern, translated))

        numbers_preserved = original_numbers <= translated_numbers or original_numbers == translated_numbers
        urls_preserved = original_urls == translated_urls

        return {
            "numbers_preserved": numbers_preserved,
            "urls_preserved": urls_preserved,
            "original_numbers": len(original_numbers),
            "translated_numbers": len(translated_numbers),
            "original_urls": len(original_urls),
            "translated_urls": len(translated_urls),
        }

    def validate_dataset(self, dataset_name: str, sample_size: int = 1000) -> dict:
        """
        Validate a single dataset.

        Args:
            dataset_name: Name of the parquet file
            sample_size: Number of examples to validate

        Returns:
            Validation results dict
        """
        english_path = self.english_dir / dataset_name
        translated_path = self.translated_dir / dataset_name

        if not english_path.exists():
            return {"error": f"English file not found: {english_path}"}

        if not translated_path.exists():
            return {"error": f"Translated file not found: {translated_path}"}

        print(f"\nValidating: {dataset_name}")

        # Load datasets
        en_df = pd.read_parquet(english_path)
        tr_df = pd.read_parquet(translated_path)

        if len(en_df) != len(tr_df):
            return {"error": f"Row count mismatch: {len(en_df)} vs {len(tr_df)}"}

        # Sample for validation
        total = len(en_df)
        indices = random.sample(range(total), min(sample_size, total))

        results = {
            "dataset": dataset_name,
            "total_examples": total,
            "validated_examples": len(indices),
            "length_ratio": {"valid": 0, "invalid": 0, "ratios": []},
            "language_detection": {"correct": 0, "incorrect": 0, "unknown": 0},
            "empty_translations": 0,
            "entity_preservation": {"numbers_ok": 0, "urls_ok": 0, "total_checked": 0},
            "samples": [],
        }

        for idx in tqdm(indices, desc="Validating"):
            en_messages = en_df.iloc[idx]["messages"]
            tr_messages = tr_df.iloc[idx]["messages"]

            # Convert numpy arrays if needed
            if hasattr(en_messages, "tolist"):
                en_messages = en_messages.tolist()
            if hasattr(tr_messages, "tolist"):
                tr_messages = tr_messages.tolist()

            for en_msg, tr_msg in zip(en_messages, tr_messages):
                en_content = en_msg.get("content", "")
                tr_content = tr_msg.get("content", "")

                # Skip system messages (kept in English)
                if en_msg.get("role") == "system":
                    continue

                # Check empty
                if not tr_content or not tr_content.strip():
                    results["empty_translations"] += 1
                    continue

                # Length ratio
                valid, ratio = self.validate_length_ratio(en_content, tr_content)
                if valid:
                    results["length_ratio"]["valid"] += 1
                else:
                    results["length_ratio"]["invalid"] += 1
                results["length_ratio"]["ratios"].append(ratio)

                # Language detection
                detected, _ = self.detect_language(tr_content)
                if detected is None:
                    results["language_detection"]["unknown"] += 1
                elif detected == self.target_lang:
                    results["language_detection"]["correct"] += 1
                else:
                    results["language_detection"]["incorrect"] += 1

                # Entity preservation
                entities = self.check_entity_preservation(en_content, tr_content)
                results["entity_preservation"]["total_checked"] += 1
                if entities["numbers_preserved"]:
                    results["entity_preservation"]["numbers_ok"] += 1
                if entities["urls_preserved"]:
                    results["entity_preservation"]["urls_ok"] += 1

            # Collect samples
            if len(results["samples"]) < 10:
                results["samples"].append(
                    {
                        "index": idx,
                        "english": en_messages,
                        "translated": tr_messages,
                    }
                )

        # Calculate summary stats
        if results["length_ratio"]["ratios"]:
            ratios = results["length_ratio"]["ratios"]
            results["length_ratio"]["mean"] = sum(ratios) / len(ratios)
            results["length_ratio"]["min"] = min(ratios)
            results["length_ratio"]["max"] = max(ratios)
        del results["length_ratio"]["ratios"]  # Remove raw data

        return results

    def validate_all(self, sample_size: int = 1000) -> dict:
        """
        Validate all datasets for this language.

        Args:
            sample_size: Number of examples to validate per dataset

        Returns:
            Combined validation results
        """
        datasets = list(self.translated_dir.glob("*.parquet"))

        if not datasets:
            return {"error": f"No parquet files found in {self.translated_dir}"}

        all_results = {
            "language": self.target_lang,
            "language_name": get_language_name(self.target_lang),
            "datasets": {},
            "summary": {
                "total_datasets": len(datasets),
                "passed": 0,
                "failed": 0,
            },
        }

        for dataset_path in datasets:
            dataset_name = dataset_path.name
            results = self.validate_dataset(dataset_name, sample_size)
            all_results["datasets"][dataset_name] = results

            # Check pass/fail
            if "error" in results:
                all_results["summary"]["failed"] += 1
            else:
                # Pass criteria
                lang_det = results["language_detection"]
                total_detected = lang_det["correct"] + lang_det["incorrect"]
                if total_detected > 0:
                    accuracy = lang_det["correct"] / total_detected
                else:
                    accuracy = 1.0  # No detection = assume OK

                length_ok = (
                    results["length_ratio"]["valid"]
                    / (results["length_ratio"]["valid"] + results["length_ratio"]["invalid"])
                    if (results["length_ratio"]["valid"] + results["length_ratio"]["invalid"]) > 0
                    else 1.0
                )

                if accuracy > 0.8 and length_ok > 0.9:
                    all_results["summary"]["passed"] += 1
                else:
                    all_results["summary"]["failed"] += 1

        return all_results


def export_samples(results: dict, output_path: Path, num_samples: int = 100):
    """
    Export random samples for manual review.

    Args:
        results: Validation results
        output_path: Path to output file
        num_samples: Number of samples to export
    """
    samples = []

    for dataset_name, dataset_results in results.get("datasets", {}).items():
        if "samples" in dataset_results:
            for sample in dataset_results["samples"][:10]:
                samples.append(
                    {
                        "dataset": dataset_name,
                        "index": sample["index"],
                        "english": sample["english"],
                        "translated": sample["translated"],
                    }
                )

    # Shuffle and limit
    random.shuffle(samples)
    samples = samples[:num_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(samples)} samples to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate translation quality")
    parser.add_argument("--input-dir", type=str, required=True, help="Base directory with translations")
    parser.add_argument("--lang", type=str, help="Language to validate (ISO code)")
    parser.add_argument("--all", action="store_true", help="Validate all languages")
    parser.add_argument("--english-dir", type=str, help="Directory with English originals (default: input-dir/en)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Examples to validate per dataset")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--export-samples", type=str, help="Export samples for manual review to this file")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    english_dir = Path(args.english_dir) if args.english_dir else input_dir / "en"

    if not english_dir.exists():
        print(f"Error: English directory not found: {english_dir}")
        sys.exit(1)

    # Determine languages to validate
    if args.all:
        languages = [d.name for d in input_dir.iterdir() if d.is_dir() and d.name != "en" and d.name in EU_LANGUAGES]
    elif args.lang:
        if args.lang not in EU_LANGUAGES:
            print(f"Error: Unknown language code '{args.lang}'")
            sys.exit(1)
        languages = [args.lang]
    else:
        print("Error: Specify --lang or --all")
        sys.exit(1)

    print("=" * 60)
    print("TRANSLATION VALIDATION")
    print("=" * 60)
    print(f"Input directory:   {input_dir}")
    print(f"English directory: {english_dir}")
    print(f"Languages:         {', '.join(languages)}")
    print(f"Sample size:       {args.sample_size}")
    print("=" * 60)

    all_results = {}

    for lang in languages:
        print(f"\n{'='*60}")
        print(f"VALIDATING: {lang} ({get_language_name(lang)})")
        print("=" * 60)

        translated_dir = input_dir / lang
        if not translated_dir.exists():
            print(f"Warning: Directory not found: {translated_dir}")
            continue

        validator = TranslationValidator(english_dir, translated_dir, lang)
        results = validator.validate_all(args.sample_size)
        all_results[lang] = results

        # Print summary
        summary = results.get("summary", {})
        print(f"\nSummary for {lang}:")
        print(f"  Datasets passed: {summary.get('passed', 0)}/{summary.get('total_datasets', 0)}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    # Export samples
    if args.export_samples:
        for lang, results in all_results.items():
            sample_path = Path(args.export_samples).with_stem(f"{Path(args.export_samples).stem}_{lang}")
            export_samples(results, sample_path)


if __name__ == "__main__":
    main()
