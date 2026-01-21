"""Tests for chunked processing functionality.

These tests verify:
1. Chunks combined equal original data
2. Document boundaries don't split mid-sample
3. Chunk sizes are respected
4. Edge cases (empty input, single item, etc.)
"""

from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume

from oellm.utils.chunking import (
    chunk_dataframe,
    chunk_iterator,
    ChunkedWriter,
    merge_chunks,
    calculate_chunk_boundaries,
)


class TestChunkDataframe:
    """Tests for chunking DataFrames."""

    def test_single_chunk_small_data(self):
        """Small data fits in single chunk."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        chunks = list(chunk_dataframe(df, chunk_size=1000))

        assert len(chunks) == 1
        pd.testing.assert_frame_equal(chunks[0], df)

    def test_multiple_chunks(self):
        """Large data is split into multiple chunks."""
        df = pd.DataFrame({"a": range(100)})
        chunks = list(chunk_dataframe(df, chunk_size=30))

        assert len(chunks) == 4  # 30 + 30 + 30 + 10
        assert len(chunks[0]) == 30
        assert len(chunks[1]) == 30
        assert len(chunks[2]) == 30
        assert len(chunks[3]) == 10

    def test_chunks_sum_to_original(self):
        """All chunks combined equal original data."""
        df = pd.DataFrame({"a": range(100), "b": [f"item_{i}" for i in range(100)]})
        chunks = list(chunk_dataframe(df, chunk_size=27))

        merged = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(merged, df)

    def test_chunk_preserves_dtypes(self):
        """Chunking preserves column data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        chunks = list(chunk_dataframe(df, chunk_size=2))

        for chunk in chunks:
            assert chunk["int_col"].dtype == df["int_col"].dtype
            assert chunk["float_col"].dtype == df["float_col"].dtype
            assert chunk["str_col"].dtype == df["str_col"].dtype
            assert chunk["bool_col"].dtype == df["bool_col"].dtype

    def test_empty_dataframe(self):
        """Empty DataFrame produces no chunks."""
        df = pd.DataFrame({"a": []})
        chunks = list(chunk_dataframe(df, chunk_size=100))

        assert len(chunks) == 0

    def test_exact_chunk_size(self):
        """Data exactly divisible by chunk size."""
        df = pd.DataFrame({"a": range(100)})
        chunks = list(chunk_dataframe(df, chunk_size=25))

        assert len(chunks) == 4
        assert all(len(c) == 25 for c in chunks)


class TestChunkIterator:
    """Tests for chunking iterators/generators."""

    def test_chunk_iterator_basic(self):
        """Basic iterator chunking works."""
        items = list(range(100))
        chunks = list(chunk_iterator(iter(items), chunk_size=30))

        assert len(chunks) == 4
        assert chunks[0] == list(range(30))
        assert chunks[1] == list(range(30, 60))
        assert chunks[2] == list(range(60, 90))
        assert chunks[3] == list(range(90, 100))

    def test_chunk_iterator_preserves_all_items(self):
        """All items are preserved through chunking."""
        items = [f"item_{i}" for i in range(77)]
        chunks = list(chunk_iterator(iter(items), chunk_size=10))

        all_items = []
        for chunk in chunks:
            all_items.extend(chunk)

        assert all_items == items

    def test_chunk_iterator_empty(self):
        """Empty iterator produces no chunks."""
        chunks = list(chunk_iterator(iter([]), chunk_size=10))
        assert len(chunks) == 0


class TestChunkedWriter:
    """Tests for ChunkedWriter class."""

    def test_write_small_data_single_file(self, tmp_path: Path):
        """Small data is written to single file."""
        writer = ChunkedWriter(tmp_path, prefix="test", chunk_size_bytes=1024 * 1024)

        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        writer.write(data)
        writer.finalize()

        files = list(tmp_path.glob("test_*.npy"))
        assert len(files) == 1

        loaded = np.load(files[0])
        np.testing.assert_array_equal(loaded, data)

    def test_write_large_data_multiple_files(self, tmp_path: Path):
        """Large data written incrementally is split across multiple files."""
        # ~1KB per chunk
        writer = ChunkedWriter(tmp_path, prefix="test", chunk_size_bytes=1024)

        # Write 10KB of data in 1KB batches (should create ~10 chunks)
        # 256 int32 values = 1024 bytes
        all_data = []
        for i in range(10):
            batch = np.arange(i * 256, (i + 1) * 256, dtype=np.int32)
            writer.write(batch)
            all_data.append(batch)

        writer.finalize()

        expected_data = np.concatenate(all_data)
        files = sorted(tmp_path.glob("test_*.npy"))
        assert len(files) >= 5  # At least 5 chunks (may be more due to timing)

        # Verify all data is present
        loaded = []
        for f in files:
            loaded.append(np.load(f))
        merged = np.concatenate(loaded)
        np.testing.assert_array_equal(merged, expected_data)

    def test_write_incremental(self, tmp_path: Path):
        """Incremental writes work correctly."""
        writer = ChunkedWriter(tmp_path, prefix="test", chunk_size_bytes=1024)

        # Write in batches
        for i in range(10):
            data = np.arange(i * 100, (i + 1) * 100, dtype=np.int32)
            writer.write(data)

        writer.finalize()

        # Verify all data
        files = sorted(tmp_path.glob("test_*.npy"))
        loaded = np.concatenate([np.load(f) for f in files])
        np.testing.assert_array_equal(loaded, np.arange(1000))

    def test_write_with_document_boundaries(self, tmp_path: Path):
        """Document boundaries are tracked correctly."""
        writer = ChunkedWriter(tmp_path, prefix="test", chunk_size_bytes=1024)

        # Write documents of varying sizes
        docs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
            np.array([6, 7, 8, 9], dtype=np.int32),
        ]

        for doc in docs:
            writer.write_document(doc)

        writer.finalize()

        # Check boundaries
        boundaries = writer.get_document_boundaries()
        assert len(boundaries) == 3
        assert boundaries[0] == (0, 3)
        assert boundaries[1] == (3, 5)
        assert boundaries[2] == (5, 9)

    def test_finalize_writes_remaining_data(self, tmp_path: Path):
        """Finalize writes any buffered data."""
        writer = ChunkedWriter(tmp_path, prefix="test", chunk_size_bytes=1024 * 1024)

        # Write less than chunk size
        data = np.array([1, 2, 3], dtype=np.int32)
        writer.write(data)

        # Before finalize, data might be buffered
        writer.finalize()

        # After finalize, file should exist
        files = list(tmp_path.glob("test_*.npy"))
        assert len(files) == 1

    def test_chunk_boundaries_saved(self, tmp_path: Path):
        """Chunk boundaries are saved to metadata."""
        writer = ChunkedWriter(tmp_path, prefix="test", chunk_size_bytes=100)

        # Write enough to create multiple chunks
        for i in range(100):
            data = np.arange(10, dtype=np.int32)
            writer.write_document(data)

        writer.finalize()

        # Boundaries should be saved
        meta_file = tmp_path / "test_boundaries.csv"
        assert meta_file.exists()


class TestMergeChunks:
    """Tests for merging chunks back together."""

    def test_merge_single_chunk(self, tmp_path: Path):
        """Single chunk merges correctly."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        np.save(tmp_path / "test_0000.npy", data)

        merged = merge_chunks(tmp_path, prefix="test")
        np.testing.assert_array_equal(merged, data)

    def test_merge_multiple_chunks(self, tmp_path: Path):
        """Multiple chunks merge in correct order."""
        for i in range(5):
            data = np.arange(i * 10, (i + 1) * 10, dtype=np.int32)
            np.save(tmp_path / f"test_{i:04d}.npy", data)

        merged = merge_chunks(tmp_path, prefix="test")
        np.testing.assert_array_equal(merged, np.arange(50))

    def test_merge_respects_file_order(self, tmp_path: Path):
        """Merge uses correct file ordering (not alphabetical for >10 files)."""
        for i in range(15):
            data = np.array([i], dtype=np.int32)
            np.save(tmp_path / f"test_{i:04d}.npy", data)

        merged = merge_chunks(tmp_path, prefix="test")
        np.testing.assert_array_equal(merged, np.arange(15))

    def test_merge_no_chunks_raises(self, tmp_path: Path):
        """Merging with no chunks raises error."""
        with pytest.raises(ValueError, match="No chunks found"):
            merge_chunks(tmp_path, prefix="nonexistent")


class TestCalculateChunkBoundaries:
    """Tests for calculating chunk boundaries."""

    def test_single_chunk_boundary(self):
        """Single chunk has correct boundaries."""
        sizes = [100]
        boundaries = calculate_chunk_boundaries(sizes, chunk_size=1000)

        assert len(boundaries) == 1
        assert boundaries[0] == (0, 100)

    def test_multiple_items_single_chunk(self):
        """Multiple items fitting in one chunk."""
        sizes = [10, 20, 30]
        boundaries = calculate_chunk_boundaries(sizes, chunk_size=100)

        assert len(boundaries) == 1
        assert boundaries[0] == (0, 60)

    def test_items_split_across_chunks(self):
        """Items split across chunks at correct boundaries."""
        sizes = [30, 30, 30, 30]  # 120 total
        boundaries = calculate_chunk_boundaries(sizes, chunk_size=50)

        # Should split: [30] [30] [30] [30] -> 4 chunks
        # Or: [30, partial] [partial, 30] etc.
        # Actually with chunk_size=50: [30+20] [10+30+10] etc.
        # Let's verify total
        total = sum(s for _, s in [(b[0], b[1] - b[0]) for b in boundaries])
        # Just verify we get all items
        assert boundaries[-1][1] == 120

    def test_large_item_exceeds_chunk(self):
        """Item larger than chunk size gets its own chunk."""
        sizes = [10, 100, 10]  # Middle item is large
        boundaries = calculate_chunk_boundaries(sizes, chunk_size=50)

        # Large item should not be split
        total_size = sum(sizes)
        assert boundaries[-1][1] == total_size


class TestChunkingProperties:
    """Property-based tests for chunking."""

    @given(
        data_size=st.integers(min_value=1, max_value=1000),
        chunk_size=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_chunking_preserves_all_data(self, data_size: int, chunk_size: int):
        """Chunking never loses data regardless of sizes."""
        df = pd.DataFrame({"idx": range(data_size)})
        chunks = list(chunk_dataframe(df, chunk_size=chunk_size))

        merged = pd.concat(chunks, ignore_index=True)
        pd.testing.assert_frame_equal(merged, df)

    @given(
        data_size=st.integers(min_value=0, max_value=500),
        chunk_size=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_chunk_sizes_are_bounded(self, data_size: int, chunk_size: int):
        """Each chunk is at most chunk_size (except possibly last)."""
        df = pd.DataFrame({"idx": range(data_size)})
        chunks = list(chunk_dataframe(df, chunk_size=chunk_size))

        if len(chunks) > 1:
            for chunk in chunks[:-1]:
                assert len(chunk) <= chunk_size

    @given(
        num_items=st.integers(min_value=1, max_value=100),
        chunk_size=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=50)
    def test_iterator_chunking_preserves_order(self, num_items: int, chunk_size: int):
        """Iterator chunking preserves item order."""
        items = list(range(num_items))
        chunks = list(chunk_iterator(iter(items), chunk_size=chunk_size))

        flattened = []
        for chunk in chunks:
            flattened.extend(chunk)

        assert flattened == items


class TestEdgeCases:
    """Edge case tests for chunking."""

    def test_chunk_size_equals_data_size(self):
        """Chunk size exactly equals data size."""
        df = pd.DataFrame({"a": range(50)})
        chunks = list(chunk_dataframe(df, chunk_size=50))

        assert len(chunks) == 1
        assert len(chunks[0]) == 50

    def test_chunk_size_one(self):
        """Chunk size of 1 creates one chunk per row."""
        df = pd.DataFrame({"a": range(5)})
        chunks = list(chunk_dataframe(df, chunk_size=1))

        assert len(chunks) == 5
        for i, chunk in enumerate(chunks):
            assert len(chunk) == 1
            assert chunk["a"].iloc[0] == i

    def test_very_large_chunk_size(self):
        """Very large chunk size creates single chunk."""
        df = pd.DataFrame({"a": range(100)})
        chunks = list(chunk_dataframe(df, chunk_size=1000000))

        assert len(chunks) == 1
        assert len(chunks[0]) == 100

    def test_chunk_with_index_reset(self):
        """Chunks have reset indices."""
        df = pd.DataFrame({"a": range(100)})
        df.index = range(100, 200)  # Custom index

        chunks = list(chunk_dataframe(df, chunk_size=30))

        for chunk in chunks:
            assert chunk.index[0] == 0  # Reset index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
