"""Integration tests for translation pipeline resumability.

Tests that translation can be interrupted and resumed correctly,
producing the same output as an uninterrupted run.
"""

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oellm.utils.checkpoint import CheckpointManager


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_messages() -> list[list[dict[str, str]]]:
    """Create sample conversation messages for testing."""
    return [
        [
            {"role": "user", "content": f"Hello, how are you? Question {i}"},
            {"role": "assistant", "content": f"I'm doing well! Answer {i}"},
        ]
        for i in range(100)
    ]


@pytest.fixture
def sample_dataframe(sample_messages) -> pd.DataFrame:
    """Create a sample DataFrame with messages."""
    return pd.DataFrame({"messages": sample_messages})


@pytest.fixture
def mock_translator():
    """Create a mock translator that 'translates' by prepending [TRANSLATED]."""

    def translate(text: str, src_lang: str, tgt_lang: str) -> str:
        return f"[{tgt_lang.upper()}] {text}"

    mock = MagicMock()
    mock.translate.side_effect = translate
    return mock


# ============================================================================
# Helper Functions for Testing
# ============================================================================


def translate_messages_mock(
    messages: list[dict[str, str]], target_lang: str
) -> list[dict[str, str]]:
    """Mock translation that prepends language code."""
    return [
        {
            "role": msg["role"],
            "content": f"[{target_lang.upper()}] {msg['content']}",
        }
        for msg in messages
    ]


def simulate_translation_with_checkpoint(
    df: pd.DataFrame,
    output_dir: Path,
    target_lang: str,
    batch_size: int = 10,
    stop_after_batches: int | None = None,
) -> tuple[int, list[pd.DataFrame]]:
    """Simulate translation with checkpointing.

    Args:
        df: Input DataFrame with 'messages' column
        output_dir: Directory for checkpoints and output
        target_lang: Target language code
        batch_size: Number of samples per checkpoint
        stop_after_batches: If set, stop after this many batches (simulate interrupt)

    Returns:
        Tuple of (batches_completed, list of checkpoint dataframes)
    """
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix=f"translate_{target_lang}")

    # Check for resume point
    resume_batch_idx, _ = checkpoint_mgr.get_resume_point()
    start_idx = resume_batch_idx * batch_size if resume_batch_idx > 0 else 0

    batches_completed = resume_batch_idx
    checkpoint_dfs = []

    for batch_start in range(start_idx, len(df), batch_size):
        if stop_after_batches is not None and batches_completed >= stop_after_batches:
            break

        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end].copy()

        # "Translate" the messages
        translated_messages = []
        for messages in batch_df["messages"]:
            translated = translate_messages_mock(messages, target_lang)
            translated_messages.append(translated)

        batch_df["messages"] = translated_messages
        batch_df["language"] = target_lang

        # Save checkpoint
        checkpoint_mgr.save(
            batch_df,
            batch_idx=batches_completed,
            metadata={
                "batch_start": batch_start,
                "batch_end": batch_end,
                "language": target_lang,
            },
        )

        checkpoint_dfs.append(batch_df)
        batches_completed += 1

    return batches_completed, checkpoint_dfs


def merge_and_finalize(output_dir: Path, target_lang: str) -> pd.DataFrame:
    """Merge checkpoints and create final output."""
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix=f"translate_{target_lang}")

    final_path = output_dir / f"{target_lang}_translated.parquet"
    checkpoint_mgr.merge_checkpoints(final_path)

    return pd.read_parquet(final_path)


# ============================================================================
# Integration Tests
# ============================================================================


class TestTranslationResume:
    """Integration tests for translation resumability."""

    def test_complete_translation_without_interruption(
        self, sample_dataframe, tmp_path
    ):
        """Test complete translation produces expected output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        batches, _ = simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="de",
            batch_size=10,
            stop_after_batches=None,  # Complete all
        )

        # Should complete 10 batches (100 samples / 10 per batch)
        assert batches == 10

        # Merge and verify
        final_df = merge_and_finalize(output_dir, "de")
        assert len(final_df) == 100

        # Verify all translations have the language tag
        for messages in final_df["messages"]:
            for msg in messages:
                assert msg["content"].startswith("[DE]")

    def test_resume_after_interruption(self, sample_dataframe, tmp_path):
        """Test that interrupted translation can be resumed correctly."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run 1: Stop after 3 batches
        batches_run1, _ = simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="de",
            batch_size=10,
            stop_after_batches=3,
        )
        assert batches_run1 == 3

        # Run 2: Resume and complete
        batches_run2, _ = simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="de",
            batch_size=10,
            stop_after_batches=None,
        )
        assert batches_run2 == 10  # Completed all 10

        # Merge and verify
        final_df = merge_and_finalize(output_dir, "de")
        assert len(final_df) == 100

    def test_resume_produces_same_output_as_uninterrupted(
        self, sample_dataframe, tmp_path
    ):
        """Verify resumed run produces identical output to uninterrupted run."""
        # Run 1: Uninterrupted
        uninterrupted_dir = tmp_path / "uninterrupted"
        uninterrupted_dir.mkdir()

        simulate_translation_with_checkpoint(
            sample_dataframe,
            uninterrupted_dir,
            target_lang="fr",
            batch_size=10,
        )
        uninterrupted_df = merge_and_finalize(uninterrupted_dir, "fr")

        # Run 2: With interruption and resume
        interrupted_dir = tmp_path / "interrupted"
        interrupted_dir.mkdir()

        # First part - stop after 5 batches
        simulate_translation_with_checkpoint(
            sample_dataframe,
            interrupted_dir,
            target_lang="fr",
            batch_size=10,
            stop_after_batches=5,
        )

        # Resume and complete
        simulate_translation_with_checkpoint(
            sample_dataframe,
            interrupted_dir,
            target_lang="fr",
            batch_size=10,
        )
        resumed_df = merge_and_finalize(interrupted_dir, "fr")

        # Compare outputs
        assert len(uninterrupted_df) == len(resumed_df)

        # Compare all messages (handle both list and numpy array)
        for idx in range(len(uninterrupted_df)):
            unintr_msgs = uninterrupted_df.iloc[idx]["messages"]
            resumed_msgs = resumed_df.iloc[idx]["messages"]
            # Convert to list if numpy array
            if hasattr(unintr_msgs, "tolist"):
                unintr_msgs = unintr_msgs.tolist()
            if hasattr(resumed_msgs, "tolist"):
                resumed_msgs = resumed_msgs.tolist()
            assert unintr_msgs == resumed_msgs

    def test_multiple_interruptions_and_resumes(self, sample_dataframe, tmp_path):
        """Test multiple interrupt/resume cycles work correctly."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run 1: Stop after 2 batches
        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="es",
            batch_size=10,
            stop_after_batches=2,
        )

        # Run 2: Resume, stop after 2 more
        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="es",
            batch_size=10,
            stop_after_batches=4,
        )

        # Run 3: Resume, stop after 3 more
        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="es",
            batch_size=10,
            stop_after_batches=7,
        )

        # Run 4: Complete
        batches = simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="es",
            batch_size=10,
        )[0]

        assert batches == 10
        final_df = merge_and_finalize(output_dir, "es")
        assert len(final_df) == 100

    def test_checkpoint_metadata_preserved(self, sample_dataframe, tmp_path):
        """Test that checkpoint metadata is correctly preserved across resumes."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create some checkpoints
        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="it",
            batch_size=10,
            stop_after_batches=3,
        )

        # Check checkpoint directory
        checkpoint_dir = output_dir / ".checkpoints"
        checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix="translate_it")

        resume_idx, metadata = checkpoint_mgr.get_resume_point()
        assert resume_idx == 3
        assert metadata is not None
        assert metadata["language"] == "it"
        assert metadata["batch_end"] == 30  # 3 batches * 10 samples

    def test_empty_dataframe_handling(self, tmp_path):
        """Test handling of empty input DataFrame."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        empty_df = pd.DataFrame({"messages": []})

        batches, _ = simulate_translation_with_checkpoint(
            empty_df,
            output_dir,
            target_lang="de",
            batch_size=10,
        )

        assert batches == 0

    def test_single_sample_handling(self, tmp_path):
        """Test handling of single sample DataFrame."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        single_df = pd.DataFrame(
            {
                "messages": [
                    [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there"},
                    ]
                ]
            }
        )

        simulate_translation_with_checkpoint(
            single_df,
            output_dir,
            target_lang="nl",
            batch_size=10,
        )

        final_df = merge_and_finalize(output_dir, "nl")
        assert len(final_df) == 1
        assert final_df.iloc[0]["messages"][0]["content"] == "[NL] Hello"

    def test_partial_batch_at_end(self, tmp_path):
        """Test handling when last batch is smaller than batch_size."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # 35 samples with batch size 10 = 3 full batches + 1 partial (5 samples)
        messages = [
            [
                {"role": "user", "content": f"Q{i}"},
                {"role": "assistant", "content": f"A{i}"},
            ]
            for i in range(35)
        ]
        df = pd.DataFrame({"messages": messages})

        batches, _ = simulate_translation_with_checkpoint(
            df,
            output_dir,
            target_lang="pt",
            batch_size=10,
        )

        assert batches == 4  # 3 full + 1 partial
        final_df = merge_and_finalize(output_dir, "pt")
        assert len(final_df) == 35


class TestTranslationCheckpointCorruption:
    """Tests for checkpoint corruption handling."""

    def test_corrupted_checkpoint_detected(self, sample_dataframe, tmp_path):
        """Test that corrupted checkpoints cause merge to fail."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create some valid checkpoints
        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="de",
            batch_size=10,
            stop_after_batches=3,
        )

        # Corrupt one checkpoint file
        checkpoint_dir = output_dir / ".checkpoints"
        checkpoint_files = sorted(checkpoint_dir.glob("*.parquet"))
        assert len(checkpoint_files) == 3

        if checkpoint_files:
            # Write garbage to the last checkpoint
            last_checkpoint = checkpoint_files[-1]
            last_checkpoint.write_bytes(b"corrupted data")

        # Note: get_resume_point doesn't validate file content (by design for speed)
        # Corruption is detected during merge when files are actually read
        checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix="translate_de")

        # Merge will fail on corrupted file
        final_path = output_dir / "merged.parquet"
        with pytest.raises(Exception):  # ArrowInvalid or similar
            checkpoint_mgr.merge_checkpoints(final_path)

    def test_missing_metadata_handled(self, sample_dataframe, tmp_path):
        """Test handling when metadata file is missing."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="de",
            batch_size=10,
            stop_after_batches=3,
        )

        # Delete metadata for last checkpoint
        checkpoint_dir = output_dir / ".checkpoints"
        meta_files = sorted(checkpoint_dir.glob("*.meta.json"))
        if meta_files:
            meta_files[-1].unlink()

        checkpoint_mgr = CheckpointManager(checkpoint_dir, prefix="translate_de")
        resume_idx, metadata = checkpoint_mgr.get_resume_point()

        # Should find the previous valid checkpoint
        assert resume_idx < 3


class TestMultiLanguageTranslation:
    """Tests for multiple language translations."""

    def test_independent_language_checkpoints(self, sample_dataframe, tmp_path):
        """Test that different languages have independent checkpoints."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Translate to German - stop at 5
        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="de",
            batch_size=10,
            stop_after_batches=5,
        )

        # Translate to French - complete
        simulate_translation_with_checkpoint(
            sample_dataframe,
            output_dir,
            target_lang="fr",
            batch_size=10,
        )

        # Resume German - should resume from 5, not from French's completion
        checkpoint_dir = output_dir / ".checkpoints"
        de_mgr = CheckpointManager(checkpoint_dir, prefix="translate_de")
        fr_mgr = CheckpointManager(checkpoint_dir, prefix="translate_fr")

        de_resume, _ = de_mgr.get_resume_point()
        fr_resume, _ = fr_mgr.get_resume_point()

        assert de_resume == 5
        assert fr_resume == 10

    def test_parallel_language_progress(self, sample_dataframe, tmp_path):
        """Test tracking progress of multiple languages simultaneously."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        languages = ["de", "fr", "es", "it"]
        stop_points = [3, 5, 7, 10]

        for lang, stop in zip(languages, stop_points):
            simulate_translation_with_checkpoint(
                sample_dataframe,
                output_dir,
                target_lang=lang,
                batch_size=10,
                stop_after_batches=stop,
            )

        # Verify each language has correct progress
        checkpoint_dir = output_dir / ".checkpoints"
        for lang, expected_batches in zip(languages, stop_points):
            mgr = CheckpointManager(checkpoint_dir, prefix=f"translate_{lang}")
            resume_idx, _ = mgr.get_resume_point()
            assert resume_idx == expected_batches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
