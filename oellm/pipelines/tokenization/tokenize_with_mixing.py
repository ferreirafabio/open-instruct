#!/usr/bin/env python3
"""
Tokenization pipeline with language mixing support.

This script tokenizes datasets with configurable language ratios:
- 100% English baseline
- 90% English + 10% EU languages
- 80% English + 20% EU languages
- Custom ratios via YAML config

Uses:
- CheckpointManager for resumable processing
- LanguageMixer for per-language sample distribution
- ChunkedWriter for memory-efficient output

Usage:
    # Using a config file
    python tokenize_with_mixing.py --config oellm/configs/mixture_eu24_90en_10eu.yaml \\
        --translated-dir data/datasets_eu24_translated/600m_10pct \\
        --output-dir data/datasets_eu24_tokenized/90en_10eu

    # Quick test with 1% sample
    python tokenize_with_mixing.py --config oellm/configs/mixture_eu24_90en_10eu.yaml \\
        --translated-dir data/datasets_eu24_translated/600m_10pct \\
        --output-dir /tmp/test_tokenized \\
        --sample 0.01
"""

import argparse
import json
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import from oellm.utils
from oellm.utils.checkpoint import CheckpointManager
from oellm.utils.language_mixer import LanguageMixer
from oellm.utils.chunking import ChunkedWriter

# Global flag for graceful shutdown
_shutdown_requested = False


def handle_shutdown_signal(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    print(f"\nReceived signal {signum}, requesting graceful shutdown...")
    _shutdown_requested = True


class TokenizationPipeline:
    """Pipeline for tokenizing multilingual datasets with language mixing.

    Features:
    - Configurable language ratios via YAML
    - Atomic checkpointing for resumability
    - Memory-efficient chunked output
    - Graceful shutdown handling
    """

    def __init__(
        self,
        config_path: Path,
        translated_dir: Path,
        output_dir: Path,
        tokenizer_name: str = "allenai/OLMo-2-1124-7B-Instruct",
        max_seq_length: int = 32768,
        sample_fraction: Optional[float] = None,
    ):
        """Initialize the tokenization pipeline.

        Args:
            config_path: Path to language mixing config YAML
            translated_dir: Directory containing translated parquet files
            output_dir: Directory to write tokenized output
            tokenizer_name: HuggingFace tokenizer name
            max_seq_length: Maximum sequence length for tokenization
            sample_fraction: Optional sampling fraction for testing
        """
        self.translated_dir = Path(translated_dir)
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.sample_fraction = sample_fraction

        # Load language mixer config
        self.mixer = LanguageMixer.from_yaml(config_path)
        print(self.mixer.get_config_summary())

        # Save config to output for reproducibility
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, self.output_dir / "config_used.yaml")

        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Set up checkpoint manager
        self.checkpoint_dir = self.output_dir / ".checkpoints"
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_dir, prefix="tokenization")

    def get_language_data_paths(self) -> dict[str, list[Path]]:
        """Get paths to translated data for each language.

        Returns:
            Dict mapping language code to list of parquet file paths
        """
        languages = self.mixer.get_languages_for_mixing()
        paths = {}

        for lang in languages:
            lang_dir = self.translated_dir / lang
            if lang_dir.exists():
                parquet_files = sorted(lang_dir.glob("*.parquet"))
                # Filter out checkpoint files
                parquet_files = [f for f in parquet_files if ".checkpoint" not in f.name]
                if parquet_files:
                    paths[lang] = parquet_files
                else:
                    print(f"Warning: No parquet files found for {lang} in {lang_dir}")
            else:
                print(f"Warning: Language directory not found: {lang_dir}")

        return paths

    def load_and_sample_language(
        self,
        lang: str,
        paths: list[Path],
        n_samples: int,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Load and sample data for a specific language.

        Args:
            lang: Language code
            paths: List of parquet file paths
            n_samples: Number of samples to take
            seed: Random seed

        Returns:
            Sampled DataFrame
        """
        # Load all files for this language
        dfs = []
        for path in paths:
            df = pd.read_parquet(path)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        total_available = len(combined)

        # Apply test sampling if specified
        if self.sample_fraction is not None:
            n_samples = int(n_samples * self.sample_fraction)

        # Sample
        if total_available <= n_samples:
            print(f"  {lang}: Using all {total_available} samples (requested {n_samples})")
            return combined
        else:
            print(f"  {lang}: Sampling {n_samples} from {total_available} available")
            return combined.sample(n=n_samples, random_state=seed)

    def tokenize_messages(self, messages: list[dict]) -> dict[str, Any]:
        """Tokenize a conversation using chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        # Create labels (mask non-assistant tokens with -100)
        labels = input_ids.copy()

        # Simple approach: mask everything before the last assistant response
        # A more sophisticated approach would mask each turn properly
        # For now, we'll use the full sequence as labels (SFT style)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def run(self, total_samples: int = 1000000) -> dict[str, Any]:
        """Run the tokenization pipeline.

        Args:
            total_samples: Total number of samples in final dataset

        Returns:
            Statistics dictionary
        """
        global _shutdown_requested

        # Set up signal handlers
        signal.signal(signal.SIGTERM, handle_shutdown_signal)
        signal.signal(signal.SIGINT, handle_shutdown_signal)

        # Check for resume
        resume_batch_idx, last_metadata = self.checkpoint_mgr.get_resume_point()
        if resume_batch_idx > 0 and last_metadata:
            print(f"Resuming from batch {resume_batch_idx}")
            completed_languages = last_metadata.get("completed_languages", [])
        else:
            completed_languages = []

        # Get paths for each language
        lang_paths = self.get_language_data_paths()
        if not lang_paths:
            raise ValueError("No language data found!")

        # Compute samples per language
        samples_per_lang = self.mixer.compute_samples_per_language(total_samples)
        print(f"\nSamples per language (total {total_samples}):")
        for lang, count in sorted(samples_per_lang.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {lang}: {count:,}")

        # Process each language
        all_tokenized = []
        stats = {
            "config": str(self.mixer.config),
            "total_requested": total_samples,
            "languages": {},
        }

        batch_idx = resume_batch_idx

        for lang, paths in tqdm(lang_paths.items(), desc="Processing languages"):
            if _shutdown_requested:
                print("\nShutdown requested, saving checkpoint...")
                break

            if lang in completed_languages:
                print(f"Skipping {lang} (already completed)")
                continue

            n_samples = samples_per_lang.get(lang, 0)
            if n_samples == 0:
                continue

            print(f"\nProcessing {lang}...")

            # Load and sample
            df = self.load_and_sample_language(
                lang, paths, n_samples, seed=self.mixer.config.seed
            )

            # Tokenize
            tokenized_samples = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {lang}"):
                if _shutdown_requested:
                    break

                messages = row["messages"]
                if hasattr(messages, "tolist"):
                    messages = messages.tolist()

                try:
                    tokenized = self.tokenize_messages(messages)
                    tokenized["language"] = lang
                    tokenized_samples.append(tokenized)
                except Exception as e:
                    print(f"Error tokenizing sample {idx}: {e}")
                    continue

            if _shutdown_requested:
                break

            # Save checkpoint for this language
            checkpoint_df = pd.DataFrame({
                "input_ids": [t["input_ids"] for t in tokenized_samples],
                "labels": [t["labels"] for t in tokenized_samples],
                "language": [t["language"] for t in tokenized_samples],
            })

            completed_languages.append(lang)
            self.checkpoint_mgr.save(
                checkpoint_df,
                batch_idx=batch_idx,
                metadata={
                    "language": lang,
                    "samples": len(tokenized_samples),
                    "completed_languages": completed_languages,
                }
            )
            batch_idx += 1

            all_tokenized.extend(tokenized_samples)
            stats["languages"][lang] = len(tokenized_samples)

        if _shutdown_requested:
            print("Checkpoint saved. Resume by re-running.")
            return {"status": "interrupted", "completed_languages": completed_languages}

        # Merge and save final output
        print(f"\nMerging {len(all_tokenized)} samples...")

        # Shuffle
        rng = np.random.default_rng(self.mixer.config.seed)
        indices = rng.permutation(len(all_tokenized))

        # Save tokenized data
        output_path = self.output_dir / "tokenized_data.parquet"
        output_df = pd.DataFrame({
            "input_ids": [all_tokenized[i]["input_ids"] for i in indices],
            "labels": [all_tokenized[i]["labels"] for i in indices],
            "language": [all_tokenized[i]["language"] for i in indices],
        })
        output_df.to_parquet(output_path)
        print(f"Saved to: {output_path}")

        # Save tokenizer
        tokenizer_dir = self.output_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_dir)

        # Save stats
        stats["total_samples"] = len(all_tokenized)
        stats["output_path"] = str(output_path)

        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Cleanup checkpoints
        self.checkpoint_mgr.cleanup()
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize datasets with language mixing"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to language mixing config YAML"
    )
    parser.add_argument(
        "--translated-dir",
        type=str,
        required=True,
        help="Directory containing translated parquet files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for tokenized data"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="HuggingFace tokenizer name"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32768,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=1000000,
        help="Total samples in output dataset"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Sample fraction for testing (e.g., 0.01 for 1%%)"
    )

    args = parser.parse_args()

    pipeline = TokenizationPipeline(
        config_path=Path(args.config),
        translated_dir=Path(args.translated_dir),
        output_dir=Path(args.output_dir),
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        sample_fraction=args.sample,
    )

    stats = pipeline.run(total_samples=args.total_samples)
    print(f"\nFinal statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
