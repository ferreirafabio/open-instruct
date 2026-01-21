"""Integration tests for tokenization pipeline resumability.

Tests that tokenization can be interrupted and resumed correctly,
and that language mixing produces correct output ratios.
"""

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from oellm.utils.checkpoint import CheckpointManager
from oellm.utils.language_mixer import LanguageMixer, LanguageMixConfig, EU_LANGUAGES


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_messages_by_lang() -> dict[str, list[list[dict[str, str]]]]:
    """Create sample messages for multiple languages."""
    languages = ["en", "de", "fr", "es", "it", "nl"]
    messages_by_lang = {}

    for lang in languages:
        messages_by_lang[lang] = [
            [
                {"role": "user", "content": f"[{lang}] Question {i}"},
                {"role": "assistant", "content": f"[{lang}] Answer {i}"},
            ]
            for i in range(100)  # 100 samples per language
        ]

    return messages_by_lang


@pytest.fixture
def language_data_dir(sample_messages_by_lang, tmp_path) -> Path:
    """Create a directory structure with language-specific parquet files."""
    data_dir = tmp_path / "translated_data"

    for lang, messages in sample_messages_by_lang.items():
        lang_dir = data_dir / lang
        lang_dir.mkdir(parents=True)

        df = pd.DataFrame({"messages": messages})
        df.to_parquet(lang_dir / f"{lang}_data.parquet")

    return data_dir


@pytest.fixture
def sample_mix_config() -> LanguageMixConfig:
    """Create a sample language mixing configuration."""
    return LanguageMixConfig(
        english_ratio=0.6,
        explicit_languages={"de": 0.15, "fr": 0.10},
        others_ratio=0.15,  # Distributed among es, it, nl
        datasets=["test_dataset.parquet"],
        seed=42,
    )


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        # Return a simple string representation
        text_parts = []
        for msg in messages:
            text_parts.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(text_parts)

    def tokenize(text, truncation=True, max_length=32768, return_tensors="np"):
        # Simple mock tokenization: each word becomes a token
        words = text.split()
        input_ids = np.array([hash(w) % 50000 for w in words], dtype=np.int64)
        attention_mask = np.ones(len(input_ids), dtype=np.int64)
        return {
            "input_ids": np.array([input_ids]),
            "attention_mask": np.array([attention_mask]),
        }

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer.__call__ = tokenize
    tokenizer.side_effect = tokenize

    return tokenizer


# ============================================================================
# Helper Functions for Testing
# ============================================================================


def mock_tokenize_messages(messages: list[dict], tokenizer) -> dict[str, Any]:
    """Mock tokenization of messages."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    words = text.split()
    input_ids = np.array([hash(w) % 50000 for w in words], dtype=np.int64)
    attention_mask = np.ones(len(input_ids), dtype=np.int64)
    labels = input_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def simulate_tokenization_with_checkpoint(
    lang_data_paths: dict[str, list[Path]],
    output_dir: Path,
    samples_per_lang: dict[str, int],
    mock_tokenizer,
    stop_after_langs: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[dict]]:
    """Simulate tokenization with checkpointing.

    Args:
        lang_data_paths: Dict mapping language code to list of parquet file paths
        output_dir: Directory for checkpoints and output
        samples_per_lang: Dict mapping language code to number of samples
        mock_tokenizer: Mock tokenizer instance
        stop_after_langs: If set, stop after processing this many languages
        seed: Random seed

    Returns:
        Tuple of (list of completed languages, list of tokenized samples)
    """
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix="tokenization")

    # Check for resume point
    resume_batch_idx, last_metadata = checkpoint_mgr.get_resume_point()
    completed_languages = []
    if resume_batch_idx > 0 and last_metadata:
        completed_languages = last_metadata.get("completed_languages", [])

    all_tokenized = []
    batch_idx = resume_batch_idx

    languages_processed = 0

    for lang, paths in lang_data_paths.items():
        if stop_after_langs is not None and languages_processed >= stop_after_langs:
            break

        if lang in completed_languages:
            # Load previously tokenized data
            continue

        n_samples = samples_per_lang.get(lang, 0)
        if n_samples == 0:
            continue

        # Load data
        dfs = [pd.read_parquet(p) for p in paths]
        combined = pd.concat(dfs, ignore_index=True)

        # Sample
        rng = np.random.default_rng(seed + hash(lang) % 1000)
        if len(combined) > n_samples:
            indices = rng.choice(len(combined), size=n_samples, replace=False)
            sampled = combined.iloc[indices]
        else:
            sampled = combined

        # Tokenize
        tokenized_samples = []
        for _, row in sampled.iterrows():
            messages = row["messages"]
            if hasattr(messages, "tolist"):
                messages = messages.tolist()

            tokenized = mock_tokenize_messages(messages, mock_tokenizer)
            tokenized["language"] = lang
            tokenized_samples.append(tokenized)

        # Save checkpoint
        checkpoint_df = pd.DataFrame(
            {
                "input_ids": [t["input_ids"] for t in tokenized_samples],
                "labels": [t["labels"] for t in tokenized_samples],
                "language": [t["language"] for t in tokenized_samples],
            }
        )

        completed_languages.append(lang)
        checkpoint_mgr.save(
            checkpoint_df,
            batch_idx=batch_idx,
            metadata={
                "language": lang,
                "samples": len(tokenized_samples),
                "completed_languages": completed_languages.copy(),
            },
        )
        batch_idx += 1

        all_tokenized.extend(tokenized_samples)
        languages_processed += 1

    return completed_languages, all_tokenized


def merge_tokenization_checkpoints(output_dir: Path) -> pd.DataFrame:
    """Merge tokenization checkpoints into final output."""
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix="tokenization")

    final_path = output_dir / "tokenized_data.parquet"
    checkpoint_mgr.merge_checkpoints(final_path)

    return pd.read_parquet(final_path)


# ============================================================================
# Integration Tests: Resumability
# ============================================================================


class TestTokenizationResume:
    """Integration tests for tokenization resumability."""

    def test_complete_tokenization_without_interruption(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Test complete tokenization produces expected output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Set up language paths
        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        # Define samples per language
        samples_per_lang = {
            "en": 50,
            "de": 30,
            "fr": 20,
            "es": 10,
            "it": 10,
            "nl": 10,
        }

        completed, tokenized = simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Verify all languages completed
        assert len(completed) == 6

        # Verify total samples
        assert len(tokenized) == sum(samples_per_lang.values())

    def test_resume_after_interruption(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Test that interrupted tokenization can be resumed correctly."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        samples_per_lang = {
            "en": 50,
            "de": 30,
            "fr": 20,
            "es": 10,
            "it": 10,
            "nl": 10,
        }

        # Run 1: Stop after 2 languages
        completed1, _ = simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
            stop_after_langs=2,
        )
        assert len(completed1) == 2

        # Run 2: Resume and complete
        completed2, tokenized2 = simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Should complete all languages
        assert len(completed2) == 6

    def test_resume_produces_same_total_samples(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Verify resumed run produces same total sample count as uninterrupted."""
        samples_per_lang = {
            "en": 50,
            "de": 30,
            "fr": 20,
            "es": 10,
            "it": 10,
            "nl": 10,
        }
        total_expected = sum(samples_per_lang.values())

        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        # Run 1: Uninterrupted
        unintr_dir = tmp_path / "uninterrupted"
        unintr_dir.mkdir()

        _, unintr_samples = simulate_tokenization_with_checkpoint(
            lang_paths,
            unintr_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Run 2: With interruption
        intr_dir = tmp_path / "interrupted"
        intr_dir.mkdir()

        simulate_tokenization_with_checkpoint(
            lang_paths,
            intr_dir,
            samples_per_lang,
            mock_tokenizer,
            stop_after_langs=3,
        )

        simulate_tokenization_with_checkpoint(
            lang_paths,
            intr_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Merge checkpoints to get total - this is how the real pipeline works
        resumed_df = merge_tokenization_checkpoints(intr_dir)

        assert len(unintr_samples) == total_expected
        assert len(resumed_df) == total_expected

    def test_checkpoint_metadata_tracks_progress(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Test that checkpoint metadata correctly tracks completed languages."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        samples_per_lang = {"en": 50, "de": 30, "fr": 20}

        # Process all 3 languages
        simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Check metadata
        checkpoint_dir = output_dir / ".checkpoints"
        checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix="tokenization")
        resume_idx, metadata = checkpoint_mgr.get_resume_point()

        assert resume_idx == 3  # 3 languages processed
        assert metadata is not None
        assert "completed_languages" in metadata
        assert len(metadata["completed_languages"]) == 3


# ============================================================================
# Integration Tests: Language Mixing
# ============================================================================


class TestLanguageMixing:
    """Integration tests for language mixing functionality."""

    def test_language_ratios_respected(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Test that language mixing produces correct ratios."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        total_samples = 100
        samples_per_lang = {
            "en": 60,   # 60%
            "de": 15,   # 15%
            "fr": 10,   # 10%
            "es": 5,    # 5%
            "it": 5,    # 5%
            "nl": 5,    # 5%
        }

        _, tokenized = simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Count actual distribution
        lang_counts = Counter(t["language"] for t in tokenized)

        # Verify ratios (with some tolerance for rounding)
        assert lang_counts["en"] == 60
        assert lang_counts["de"] == 15
        assert lang_counts["fr"] == 10

    def test_language_mixer_integration(self, language_data_dir, tmp_path):
        """Test LanguageMixer computes correct sample counts."""
        config = LanguageMixConfig(
            english_ratio=0.5,
            explicit_languages={"de": 0.2, "fr": 0.1},
            others_ratio=0.2,  # Will be split among remaining languages
            datasets=["test.parquet"],
            seed=42,
        )

        mixer = LanguageMixer(config)
        samples = mixer.compute_samples_per_language(1000)

        assert samples["en"] == 500
        assert samples["de"] == 200
        assert samples["fr"] == 100

        # Others should be evenly distributed
        others_langs = [l for l in EU_LANGUAGES if l not in ["de", "fr"]]
        per_other = 200 // len(others_langs)

        for lang in others_langs:
            assert samples[lang] == per_other

    def test_config_from_yaml(self, tmp_path):
        """Test loading language mixing config from YAML."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            """
english_ratio: 0.7
languages:
  de: 0.1
  fr: 0.05
  others: 0.15
datasets:
  - dataset1.parquet
  - dataset2.parquet
seed: 123
"""
        )

        mixer = LanguageMixer.from_yaml(config_path)

        assert mixer.config.english_ratio == 0.7
        assert mixer.config.explicit_languages["de"] == 0.1
        assert mixer.config.explicit_languages["fr"] == 0.05
        assert mixer.config.others_ratio == 0.15

    def test_ratios_sum_validation(self):
        """Test that invalid ratios are rejected."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            # Validation happens in LanguageMixer.__init__, not LanguageMixConfig
            config = LanguageMixConfig(
                english_ratio=0.5,
                explicit_languages={"de": 0.3},
                others_ratio=0.3,  # Total = 1.1
                datasets=[],
                seed=42,
            )
            LanguageMixer(config)  # This triggers validation

    def test_empty_others_pool(self):
        """Test config with no 'others' pool."""
        config = LanguageMixConfig(
            english_ratio=0.8,
            explicit_languages={"de": 0.1, "fr": 0.1},
            others_ratio=0.0,
            datasets=["test.parquet"],
            seed=42,
        )

        mixer = LanguageMixer(config)
        samples = mixer.compute_samples_per_language(100)

        assert samples["en"] == 80
        assert samples["de"] == 10
        assert samples["fr"] == 10

        # Other languages should not be in the result (or have 0)
        for lang in EU_LANGUAGES:
            if lang not in ["de", "fr"]:
                assert samples.get(lang, 0) == 0

    def test_all_explicit_languages(self, tmp_path):
        """Test config where all EU languages are explicitly specified."""
        # Create config with all languages explicit
        explicit = {
            lang: 0.04
            for lang in EU_LANGUAGES[:5]  # 5 languages at 4%
        }  # Total: 20%

        config = LanguageMixConfig(
            english_ratio=0.8,
            explicit_languages=explicit,
            others_ratio=0.0,
            datasets=["test.parquet"],
            seed=42,
        )

        mixer = LanguageMixer(config)
        samples = mixer.compute_samples_per_language(1000)

        assert samples["en"] == 800
        for lang, ratio in explicit.items():
            assert samples[lang] == int(1000 * ratio)


# ============================================================================
# Integration Tests: Edge Cases
# ============================================================================


class TestTokenizationEdgeCases:
    """Tests for edge cases in tokenization pipeline."""

    def test_missing_language_data(self, tmp_path, mock_tokenizer):
        """Test handling when some languages have no data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Only create data for English
        en_dir = data_dir / "en"
        en_dir.mkdir()
        df = pd.DataFrame(
            {
                "messages": [
                    [{"role": "user", "content": f"Q{i}"}]
                    for i in range(50)
                ]
            }
        )
        df.to_parquet(en_dir / "en_data.parquet")

        lang_paths = {"en": list(en_dir.glob("*.parquet"))}

        # Request samples from languages that don't exist
        samples_per_lang = {"en": 50, "de": 30, "fr": 20}

        _, tokenized = simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Should only have English samples
        assert len(tokenized) == 50
        assert all(t["language"] == "en" for t in tokenized)

    def test_more_samples_requested_than_available(
        self, tmp_path, mock_tokenizer
    ):
        """Test handling when requesting more samples than available."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create 30 samples
        en_dir = data_dir / "en"
        en_dir.mkdir()
        df = pd.DataFrame(
            {
                "messages": [
                    [{"role": "user", "content": f"Q{i}"}]
                    for i in range(30)
                ]
            }
        )
        df.to_parquet(en_dir / "en_data.parquet")

        lang_paths = {"en": list(en_dir.glob("*.parquet"))}

        # Request 100 samples
        samples_per_lang = {"en": 100}

        _, tokenized = simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        # Should use all available (30) since we can't get 100
        assert len(tokenized) == 30

    def test_empty_language_directory(self, tmp_path, mock_tokenizer):
        """Test handling of empty language directories."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Empty language paths
        lang_paths = {}

        samples_per_lang = {"en": 50}

        completed, tokenized = simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        assert len(completed) == 0
        assert len(tokenized) == 0

    def test_reproducibility_with_seed(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Test that same seed produces same sampling results."""
        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        samples_per_lang = {"en": 50, "de": 30}

        # Run 1
        output1 = tmp_path / "output1"
        output1.mkdir()
        _, tokenized1 = simulate_tokenization_with_checkpoint(
            lang_paths,
            output1,
            samples_per_lang,
            mock_tokenizer,
            seed=42,
        )

        # Run 2 with same seed
        output2 = tmp_path / "output2"
        output2.mkdir()
        _, tokenized2 = simulate_tokenization_with_checkpoint(
            lang_paths,
            output2,
            samples_per_lang,
            mock_tokenizer,
            seed=42,
        )

        # Should produce same results
        assert len(tokenized1) == len(tokenized2)

        # Compare content hashes
        for t1, t2 in zip(tokenized1, tokenized2):
            assert t1["language"] == t2["language"]
            assert np.array_equal(t1["input_ids"], t2["input_ids"])


class TestMergeAndCleanup:
    """Tests for checkpoint merging and cleanup."""

    def test_merge_checkpoints_creates_valid_parquet(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Test that merged checkpoints create valid parquet file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        samples_per_lang = {"en": 50, "de": 30, "fr": 20}

        simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        final_df = merge_tokenization_checkpoints(output_dir)

        # Verify structure
        assert "input_ids" in final_df.columns
        assert "labels" in final_df.columns
        assert "language" in final_df.columns

        # Verify counts
        assert len(final_df) == sum(samples_per_lang.values())

    def test_cleanup_removes_checkpoints(
        self, language_data_dir, mock_tokenizer, tmp_path
    ):
        """Test that cleanup removes checkpoint files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        lang_paths = {}
        for lang_dir in language_data_dir.iterdir():
            if lang_dir.is_dir():
                lang = lang_dir.name
                lang_paths[lang] = list(lang_dir.glob("*.parquet"))

        samples_per_lang = {"en": 50, "de": 30}

        simulate_tokenization_with_checkpoint(
            lang_paths,
            output_dir,
            samples_per_lang,
            mock_tokenizer,
        )

        checkpoint_dir = output_dir / ".checkpoints"
        assert checkpoint_dir.exists()

        # Merge and cleanup
        merge_tokenization_checkpoints(output_dir)

        checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix="tokenization")
        checkpoint_mgr.cleanup()

        # Checkpoint files should be removed
        assert len(list(checkpoint_dir.glob("*.parquet"))) == 0
        assert len(list(checkpoint_dir.glob("*.meta.json"))) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
