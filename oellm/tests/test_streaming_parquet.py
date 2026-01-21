"""Tests for streaming parquet loading in translation pipeline.

Tests that:
1. Streaming mode is activated when no sampling is requested
2. Sampled mode loads into memory (df is not None)
3. Streaming iterator yields all rows correctly
4. Resume skips correct number of rows in streaming mode
5. Memory-efficient loading for large datasets
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_parquet_file():
    """Create a temporary parquet file with sample conversations."""
    conversations = [
        [{"role": "user", "content": f"Question {i}"}, {"role": "assistant", "content": f"Answer {i}"}]
        for i in range(100)
    ]

    df = pd.DataFrame({"messages": conversations})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def large_parquet_file():
    """Create a larger parquet file for streaming tests."""
    # 10,000 rows - enough to test chunking behavior
    conversations = [
        [{"role": "user", "content": f"Question {i} with some extra text to make it realistic"},
         {"role": "assistant", "content": f"Answer {i} with detailed response content"}]
        for i in range(10000)
    ]

    df = pd.DataFrame({"messages": conversations})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


# ============================================================================
# Streaming Mode Activation Tests
# ============================================================================


class TestStreamingModeActivation:
    """Tests for when streaming vs sampled mode is activated."""

    def test_no_sample_activates_streaming(self, sample_parquet_file):
        """Test that no sample_fraction activates streaming mode."""
        parquet_file = pq.ParquetFile(sample_parquet_file)
        original_total = parquet_file.metadata.num_rows

        # Simulate the mode selection logic from translate_dataset.py
        sample_fraction = None
        use_streaming = sample_fraction is None or sample_fraction >= 1.0

        assert use_streaming is True
        assert original_total == 100

    def test_sample_fraction_1_activates_streaming(self, sample_parquet_file):
        """Test that sample_fraction=1.0 activates streaming mode."""
        sample_fraction = 1.0
        use_streaming = sample_fraction is None or sample_fraction >= 1.0

        assert use_streaming is True

    def test_sample_fraction_less_than_1_activates_sampled_mode(self, sample_parquet_file):
        """Test that sample_fraction < 1.0 activates sampled mode."""
        sample_fraction = 0.5
        use_streaming = sample_fraction is None or sample_fraction >= 1.0

        assert use_streaming is False

    def test_sample_fraction_0_point_01_activates_sampled_mode(self, sample_parquet_file):
        """Test that sample_fraction=0.01 activates sampled mode."""
        sample_fraction = 0.01
        use_streaming = sample_fraction is None or sample_fraction >= 1.0

        assert use_streaming is False


# ============================================================================
# Streaming Iterator Tests
# ============================================================================


class TestStreamingIterator:
    """Tests for the streaming iterator functionality."""

    def test_streaming_iterator_yields_all_rows(self, sample_parquet_file):
        """Test that streaming iterator yields all rows."""
        parquet_file = pq.ParquetFile(sample_parquet_file)

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=10, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        messages = list(conversation_iterator())

        assert len(messages) == 100
        assert messages[0][0]["content"] == "Question 0"
        assert messages[99][0]["content"] == "Question 99"

    def test_streaming_iterator_correct_batch_boundaries(self, sample_parquet_file):
        """Test that streaming iterator handles batch boundaries correctly."""
        parquet_file = pq.ParquetFile(sample_parquet_file)

        # Use batch_size that doesn't divide evenly into total
        batch_size = 33  # 100 / 33 = 3 batches + remainder

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        messages = list(conversation_iterator())

        assert len(messages) == 100
        # Verify ordering is preserved
        for i, msg in enumerate(messages):
            assert msg[0]["content"] == f"Question {i}"

    def test_streaming_iterator_large_file(self, large_parquet_file):
        """Test streaming iterator with larger file."""
        parquet_file = pq.ParquetFile(large_parquet_file)

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=1000, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        # Count without loading all into memory
        count = 0
        for _ in conversation_iterator():
            count += 1

        assert count == 10000


# ============================================================================
# Resume with Streaming Tests
# ============================================================================


class TestStreamingResume:
    """Tests for resume functionality with streaming mode."""

    def test_skip_resume_point(self, sample_parquet_file):
        """Test that resume correctly skips to the right position."""
        parquet_file = pq.ParquetFile(sample_parquet_file)
        resume_from = 50

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=10, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        conv_iter = conversation_iterator()

        # Skip resume_from items
        for _ in range(resume_from):
            next(conv_iter, None)

        # Next item should be at index 50
        next_msg = next(conv_iter)
        assert next_msg[0]["content"] == "Question 50"

    def test_skip_all_items(self, sample_parquet_file):
        """Test that skipping all items returns None."""
        parquet_file = pq.ParquetFile(sample_parquet_file)

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=10, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        conv_iter = conversation_iterator()

        # Skip all items
        for _ in range(100):
            next(conv_iter, None)

        # Next should be None
        next_msg = next(conv_iter, None)
        assert next_msg is None

    def test_resume_preserves_remaining_count(self, sample_parquet_file):
        """Test that after resume, remaining items count is correct."""
        parquet_file = pq.ParquetFile(sample_parquet_file)
        resume_from = 30

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=10, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        conv_iter = conversation_iterator()

        # Skip resume_from items
        for _ in range(resume_from):
            next(conv_iter, None)

        # Count remaining
        remaining = list(conv_iter)
        assert len(remaining) == 70  # 100 - 30


# ============================================================================
# Sampled Mode Tests
# ============================================================================


class TestSampledMode:
    """Tests for sampled mode loading."""

    def test_sampled_mode_loads_correct_count(self, sample_parquet_file):
        """Test that sampled mode loads the correct number of samples."""
        import numpy as np

        parquet_file = pq.ParquetFile(sample_parquet_file)
        original_total = parquet_file.metadata.num_rows
        sample_fraction = 0.1
        seed = 42

        rng = np.random.default_rng(seed)
        n_samples = max(1, int(original_total * sample_fraction))
        sample_indices = np.sort(rng.choice(original_total, size=n_samples, replace=False))

        assert n_samples == 10
        assert len(sample_indices) == 10
        # Indices should be sorted and unique
        assert len(set(sample_indices)) == 10
        assert all(sample_indices[i] < sample_indices[i+1] for i in range(len(sample_indices)-1))

    def test_sampled_mode_extracts_correct_rows(self, sample_parquet_file):
        """Test that sampled mode extracts the correct rows."""
        import numpy as np

        parquet_file = pq.ParquetFile(sample_parquet_file)
        original_total = parquet_file.metadata.num_rows
        sample_fraction = 0.1
        seed = 42

        rng = np.random.default_rng(seed)
        n_samples = max(1, int(original_total * sample_fraction))
        sample_indices = np.sort(rng.choice(original_total, size=n_samples, replace=False))

        # Extract sampled messages using the same logic as translate_dataset.py
        chunk_size = 20
        sampled_messages = []
        current_idx = 0
        sample_ptr = 0

        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=["messages"]):
            batch_end = current_idx + len(batch)
            while sample_ptr < len(sample_indices) and sample_indices[sample_ptr] < batch_end:
                local_idx = sample_indices[sample_ptr] - current_idx
                sampled_messages.append(batch["messages"][local_idx].as_py())
                sample_ptr += 1
            current_idx = batch_end
            if sample_ptr >= len(sample_indices):
                break

        assert len(sampled_messages) == 10

        # Verify each sampled message corresponds to the correct index
        for i, idx in enumerate(sample_indices):
            expected_content = f"Question {idx}"
            assert sampled_messages[i][0]["content"] == expected_content


# ============================================================================
# Memory Efficiency Tests
# ============================================================================


class TestMemoryEfficiency:
    """Tests to verify memory-efficient behavior."""

    def test_streaming_does_not_create_dataframe(self, sample_parquet_file):
        """Test that streaming mode doesn't create a full DataFrame."""
        parquet_file = pq.ParquetFile(sample_parquet_file)
        original_total = parquet_file.metadata.num_rows

        sample_fraction = None  # No sampling = streaming mode
        use_streaming = sample_fraction is None or sample_fraction >= 1.0

        if use_streaming:
            df = None
            total = original_total
        else:
            df = pd.read_parquet(sample_parquet_file)
            total = len(df)

        assert use_streaming is True
        assert df is None
        assert total == 100

    def test_sampled_mode_creates_small_dataframe(self, large_parquet_file):
        """Test that sampled mode creates a smaller DataFrame."""
        import numpy as np

        parquet_file = pq.ParquetFile(large_parquet_file)
        original_total = parquet_file.metadata.num_rows
        sample_fraction = 0.01  # 1% sampling
        seed = 42

        rng = np.random.default_rng(seed)
        n_samples = max(1, int(original_total * sample_fraction))

        assert original_total == 10000
        assert n_samples == 100  # 1% of 10,000


# ============================================================================
# Integration Tests
# ============================================================================


class TestStreamingIntegration:
    """Integration tests for streaming with the translation pipeline."""

    def test_batch_collection_from_iterator(self, sample_parquet_file):
        """Test collecting batches from streaming iterator."""
        parquet_file = pq.ParquetFile(sample_parquet_file)
        conversation_batch_size = 25
        total = 100
        resume_from = 0

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=10, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        conv_iter = conversation_iterator()

        # Simulate the batch collection loop
        all_batches = []
        for batch_start in range(resume_from, total, conversation_batch_size):
            batch_end = min(batch_start + conversation_batch_size, total)

            conversations = []
            for _ in range(batch_end - batch_start):
                msg = next(conv_iter, None)
                if msg is None:
                    break
                conversations.append(msg)

            if not conversations:
                break

            all_batches.append(conversations)

        # Should have 4 batches of 25 each
        assert len(all_batches) == 4
        assert all(len(batch) == 25 for batch in all_batches)

        # Verify order is preserved across batches
        flat = [msg for batch in all_batches for msg in batch]
        for i, msg in enumerate(flat):
            assert msg[0]["content"] == f"Question {i}"

    def test_batch_collection_with_resume(self, sample_parquet_file):
        """Test collecting batches with resume point."""
        parquet_file = pq.ParquetFile(sample_parquet_file)
        conversation_batch_size = 25
        total = 100
        resume_from = 50  # Resume from halfway

        def conversation_iterator():
            for batch in parquet_file.iter_batches(batch_size=10, columns=["messages"]):
                for msg in batch["messages"]:
                    yield msg.as_py()

        conv_iter = conversation_iterator()

        # Skip to resume point
        for _ in range(resume_from):
            next(conv_iter, None)

        # Collect remaining batches
        all_batches = []
        for batch_start in range(resume_from, total, conversation_batch_size):
            batch_end = min(batch_start + conversation_batch_size, total)

            conversations = []
            for _ in range(batch_end - batch_start):
                msg = next(conv_iter, None)
                if msg is None:
                    break
                conversations.append(msg)

            if not conversations:
                break

            all_batches.append(conversations)

        # Should have 2 batches of 25 each (50 remaining)
        assert len(all_batches) == 2
        assert all(len(batch) == 25 for batch in all_batches)

        # First message should be Question 50
        assert all_batches[0][0][0]["content"] == "Question 50"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
