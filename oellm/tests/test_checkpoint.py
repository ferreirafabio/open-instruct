"""Unit tests for the checkpoint manager.

These tests verify:
1. Atomic writes survive interruption (no partial/corrupt files)
2. Resume finds the latest valid checkpoint
3. Merging checkpoints preserves data order
4. Cleanup removes all checkpoint files
"""

import json
import os
import signal
import tempfile
import threading
import time
from multiprocessing import Process
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from oellm.utils.checkpoint import CheckpointManager


class TestCheckpointManagerBasics:
    """Basic functionality tests for CheckpointManager."""

    def test_init_creates_checkpoint_dir(self, tmp_path: Path):
        """Checkpoint directory is created if it doesn't exist."""
        checkpoint_dir = tmp_path / "checkpoints"
        assert not checkpoint_dir.exists()

        manager = CheckpointManager(checkpoint_dir, prefix="test")
        assert checkpoint_dir.exists()

    def test_save_creates_parquet_file(self, tmp_path: Path):
        """Save creates a parquet file with correct name."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        metadata = {"batch_start": 0, "batch_end": 3}

        path = manager.save(df, batch_idx=0, metadata=metadata)

        assert path.exists()
        assert path.name == "test.0000.parquet"

    def test_save_creates_metadata_file(self, tmp_path: Path):
        """Save creates a JSON metadata file."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        metadata = {"batch_start": 0, "batch_end": 3, "custom": "value"}

        manager.save(df, batch_idx=0, metadata=metadata)

        meta_path = tmp_path / "test.0000.meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            saved_meta = json.load(f)
        assert saved_meta["batch_start"] == 0
        assert saved_meta["batch_end"] == 3
        assert saved_meta["custom"] == "value"

    def test_save_data_is_correct(self, tmp_path: Path):
        """Saved parquet file contains correct data."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        path = manager.save(df, batch_idx=0, metadata={})

        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_multiple_saves_increment_batch_idx(self, tmp_path: Path):
        """Multiple saves create separate files with incrementing batch index."""
        manager = CheckpointManager(tmp_path, prefix="test")

        for i in range(3):
            df = pd.DataFrame({"idx": [i]})
            manager.save(df, batch_idx=i, metadata={"batch": i})

        assert (tmp_path / "test.0000.parquet").exists()
        assert (tmp_path / "test.0001.parquet").exists()
        assert (tmp_path / "test.0002.parquet").exists()


class TestAtomicWrites:
    """Tests for atomic write behavior - crucial for crash recovery."""

    def test_no_temp_files_after_successful_save(self, tmp_path: Path):
        """No .tmp files remain after successful save."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        manager.save(df, batch_idx=0, metadata={})

        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0

    def test_interrupted_save_leaves_no_corrupt_file(self, tmp_path: Path):
        """If save is interrupted during write, no corrupt final file exists."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": range(10000)})  # Larger data

        # Simulate interruption by mocking rename to fail
        original_rename = Path.rename

        def failing_rename(self, target):
            if ".tmp" in str(self):
                raise OSError("Simulated interruption during rename")
            return original_rename(self, target)

        with patch.object(Path, "rename", failing_rename):
            with pytest.raises(OSError):
                manager.save(df, batch_idx=0, metadata={})

        # The final file should NOT exist (only temp file might exist)
        final_path = tmp_path / "test.0000.parquet"
        assert not final_path.exists()

    def test_temp_file_cleanup_on_failure(self, tmp_path: Path):
        """Temp files are cleaned up even if save fails."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Make parquet write fail
        with patch("pandas.DataFrame.to_parquet", side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                manager.save(df, batch_idx=0, metadata={})

        # No temp files should remain
        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0


class TestResumePoint:
    """Tests for finding the correct resume point after restart."""

    def test_get_resume_point_empty_dir(self, tmp_path: Path):
        """Empty directory returns (0, None) as resume point."""
        manager = CheckpointManager(tmp_path, prefix="test")

        batch_idx, metadata = manager.get_resume_point()

        assert batch_idx == 0
        assert metadata is None

    def test_get_resume_point_single_checkpoint(self, tmp_path: Path):
        """Single checkpoint returns next batch index."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": [1, 2, 3]})
        manager.save(df, batch_idx=0, metadata={"processed": 100})

        batch_idx, metadata = manager.get_resume_point()

        assert batch_idx == 1
        assert metadata["processed"] == 100

    def test_get_resume_point_multiple_checkpoints(self, tmp_path: Path):
        """Multiple checkpoints returns the one after the latest."""
        manager = CheckpointManager(tmp_path, prefix="test")

        for i in range(5):
            df = pd.DataFrame({"batch": [i]})
            manager.save(df, batch_idx=i, metadata={"processed": (i + 1) * 100})

        batch_idx, metadata = manager.get_resume_point()

        assert batch_idx == 5
        assert metadata["processed"] == 500

    def test_get_resume_point_ignores_incomplete_checkpoints(self, tmp_path: Path):
        """Checkpoints without metadata files are ignored."""
        manager = CheckpointManager(tmp_path, prefix="test")

        # Create valid checkpoint
        df = pd.DataFrame({"a": [1]})
        manager.save(df, batch_idx=0, metadata={"complete": True})

        # Create incomplete checkpoint (no metadata)
        incomplete_path = tmp_path / "test.0001.parquet"
        df.to_parquet(incomplete_path)
        # Don't create metadata file

        batch_idx, metadata = manager.get_resume_point()

        # Should resume from batch 1, not 2
        assert batch_idx == 1
        assert metadata["complete"] is True

    def test_get_resume_point_handles_gaps(self, tmp_path: Path):
        """Resume finds latest even if there are gaps in batch indices."""
        manager = CheckpointManager(tmp_path, prefix="test")

        # Create checkpoints with gaps
        for i in [0, 1, 5, 10]:
            df = pd.DataFrame({"idx": [i]})
            manager.save(df, batch_idx=i, metadata={"batch": i})

        batch_idx, metadata = manager.get_resume_point()

        assert batch_idx == 11  # After the highest (10)
        assert metadata["batch"] == 10


class TestMergeCheckpoints:
    """Tests for merging checkpoints into final output."""

    def test_merge_single_checkpoint(self, tmp_path: Path):
        """Single checkpoint merges correctly."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        manager.save(df, batch_idx=0, metadata={})

        output_path = tmp_path / "merged.parquet"
        manager.merge_checkpoints(output_path)

        merged = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df, merged)

    def test_merge_multiple_checkpoints_preserves_order(self, tmp_path: Path):
        """Multiple checkpoints are merged in correct order."""
        manager = CheckpointManager(tmp_path, prefix="test")

        # Create checkpoints in order
        for i in range(3):
            df = pd.DataFrame({"batch": [i], "data": [f"data_{i}"]})
            manager.save(df, batch_idx=i, metadata={})

        output_path = tmp_path / "merged.parquet"
        manager.merge_checkpoints(output_path)

        merged = pd.read_parquet(output_path)
        assert len(merged) == 3
        assert list(merged["batch"]) == [0, 1, 2]
        assert list(merged["data"]) == ["data_0", "data_1", "data_2"]

    def test_merge_handles_large_data(self, tmp_path: Path):
        """Merge handles checkpoints with many rows efficiently."""
        manager = CheckpointManager(tmp_path, prefix="test")

        total_rows = 0
        for i in range(5):
            rows = 1000 * (i + 1)
            df = pd.DataFrame({"idx": range(total_rows, total_rows + rows)})
            manager.save(df, batch_idx=i, metadata={})
            total_rows += rows

        output_path = tmp_path / "merged.parquet"
        manager.merge_checkpoints(output_path)

        merged = pd.read_parquet(output_path)
        assert len(merged) == total_rows
        assert list(merged["idx"]) == list(range(total_rows))

    def test_merge_empty_checkpoints(self, tmp_path: Path):
        """Empty checkpoint directory raises appropriate error."""
        manager = CheckpointManager(tmp_path, prefix="test")

        output_path = tmp_path / "merged.parquet"
        with pytest.raises(ValueError, match="No checkpoints found"):
            manager.merge_checkpoints(output_path)


class TestCleanup:
    """Tests for checkpoint cleanup after successful completion."""

    def test_cleanup_removes_all_checkpoint_files(self, tmp_path: Path):
        """Cleanup removes all parquet and metadata files."""
        manager = CheckpointManager(tmp_path, prefix="test")

        # Create several checkpoints
        for i in range(3):
            df = pd.DataFrame({"idx": [i]})
            manager.save(df, batch_idx=i, metadata={"batch": i})

        # Verify files exist
        assert len(list(tmp_path.glob("test.*.parquet"))) == 3
        assert len(list(tmp_path.glob("test.*.meta.json"))) == 3

        manager.cleanup()

        # All checkpoint files should be gone
        assert len(list(tmp_path.glob("test.*.parquet"))) == 0
        assert len(list(tmp_path.glob("test.*.meta.json"))) == 0

    def test_cleanup_leaves_other_files(self, tmp_path: Path):
        """Cleanup only removes checkpoint files, not other files."""
        manager = CheckpointManager(tmp_path, prefix="test")

        # Create a checkpoint
        df = pd.DataFrame({"a": [1]})
        manager.save(df, batch_idx=0, metadata={})

        # Create an unrelated file
        other_file = tmp_path / "other_data.parquet"
        df.to_parquet(other_file)

        manager.cleanup()

        # Other file should still exist
        assert other_file.exists()

    def test_cleanup_different_prefixes(self, tmp_path: Path):
        """Cleanup only removes files with matching prefix."""
        manager1 = CheckpointManager(tmp_path, prefix="dataset1")
        manager2 = CheckpointManager(tmp_path, prefix="dataset2")

        df = pd.DataFrame({"a": [1]})
        manager1.save(df, batch_idx=0, metadata={})
        manager2.save(df, batch_idx=0, metadata={})

        manager1.cleanup()

        # Only dataset1 files should be removed
        assert not (tmp_path / "dataset1.0000.parquet").exists()
        assert (tmp_path / "dataset2.0000.parquet").exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_batch_idx_zero_padding(self, tmp_path: Path):
        """Batch indices are zero-padded for correct sorting."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": [1]})

        # Create checkpoints at various indices
        for i in [0, 9, 10, 99, 100, 999, 1000]:
            manager.save(df, batch_idx=i, metadata={"idx": i})

        # Verify filenames are sortable
        files = sorted(tmp_path.glob("test.*.parquet"))
        indices = [int(f.stem.split(".")[1]) for f in files]
        assert indices == sorted(indices)

    def test_special_characters_in_prefix(self, tmp_path: Path):
        """Prefix with special characters is handled safely."""
        # Underscores and hyphens should work
        manager = CheckpointManager(tmp_path, prefix="my-dataset_v2")
        df = pd.DataFrame({"a": [1]})
        path = manager.save(df, batch_idx=0, metadata={})
        assert path.exists()

    def test_unicode_data(self, tmp_path: Path):
        """Unicode data is preserved through checkpoint cycle."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({
            "text": ["Hello", "Привет", "你好", "مرحبا", "🎉"],
            "lang": ["en", "ru", "zh", "ar", "emoji"]
        })
        manager.save(df, batch_idx=0, metadata={})

        output_path = tmp_path / "merged.parquet"
        manager.merge_checkpoints(output_path)

        merged = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df, merged)

    def test_empty_dataframe(self, tmp_path: Path):
        """Empty dataframe is handled correctly."""
        manager = CheckpointManager(tmp_path, prefix="test")
        df = pd.DataFrame({"a": []})

        path = manager.save(df, batch_idx=0, metadata={"empty": True})
        assert path.exists()

        loaded = pd.read_parquet(path)
        assert len(loaded) == 0

    def test_concurrent_access_different_prefixes(self, tmp_path: Path):
        """Multiple managers with different prefixes can work concurrently."""
        results = {}

        def worker(prefix: str, data: list):
            manager = CheckpointManager(tmp_path, prefix=prefix)
            df = pd.DataFrame({"data": data})
            manager.save(df, batch_idx=0, metadata={})
            results[prefix] = True

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(f"worker_{i}", [i] * 100))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(results.values())


class TestProgressTracking:
    """Tests for progress tracking via metadata."""

    def test_metadata_tracks_progress(self, tmp_path: Path):
        """Metadata correctly tracks processing progress."""
        manager = CheckpointManager(tmp_path, prefix="test")

        total_processed = 0
        for i in range(3):
            batch_size = 100
            df = pd.DataFrame({"idx": range(batch_size)})
            total_processed += batch_size
            manager.save(df, batch_idx=i, metadata={
                "batch_start": i * batch_size,
                "batch_end": (i + 1) * batch_size,
                "total_processed": total_processed
            })

        batch_idx, metadata = manager.get_resume_point()
        assert metadata["total_processed"] == 300
        assert metadata["batch_end"] == 300

    def test_get_all_metadata(self, tmp_path: Path):
        """Can retrieve metadata for all checkpoints."""
        manager = CheckpointManager(tmp_path, prefix="test")

        for i in range(3):
            df = pd.DataFrame({"idx": [i]})
            manager.save(df, batch_idx=i, metadata={"batch": i, "info": f"batch_{i}"})

        all_metadata = manager.get_all_metadata()

        assert len(all_metadata) == 3
        assert all_metadata[0]["batch"] == 0
        assert all_metadata[1]["info"] == "batch_1"
        assert all_metadata[2]["batch"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
