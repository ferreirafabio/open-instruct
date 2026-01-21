"""Chunking utilities for processing large datasets.

Provides functions and classes for splitting data into manageable chunks
for memory-efficient processing and incremental checkpointing.
"""

import csv
import gzip
from pathlib import Path
from typing import Any, Generator, Iterator, List, Tuple

import numpy as np
import pandas as pd


def chunk_dataframe(
    df: pd.DataFrame,
    chunk_size: int
) -> Generator[pd.DataFrame, None, None]:
    """Split a DataFrame into chunks of specified size.

    Args:
        df: DataFrame to chunk
        chunk_size: Maximum number of rows per chunk

    Yields:
        DataFrame chunks with reset indices
    """
    if len(df) == 0:
        return

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end].reset_index(drop=True)
        yield chunk


def chunk_iterator(
    items: Iterator[Any],
    chunk_size: int
) -> Generator[List[Any], None, None]:
    """Split an iterator into chunks.

    Args:
        items: Iterator to chunk
        chunk_size: Number of items per chunk

    Yields:
        Lists of items
    """
    chunk: List[Any] = []

    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


class ChunkedWriter:
    """Writes data in chunks to multiple files.

    Automatically splits data across multiple files when the size
    exceeds the configured chunk size. Tracks document boundaries
    for later reconstruction.

    Example:
        >>> writer = ChunkedWriter(Path("/tmp/output"), "tokens", chunk_size_bytes=1024**3)
        >>> for doc in documents:
        ...     writer.write_document(doc.tokens)
        >>> writer.finalize()
    """

    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        chunk_size_bytes: int = 1024 * 1024 * 1024,  # 1GB default
    ):
        """Initialize chunked writer.

        Args:
            output_dir: Directory to write chunk files
            prefix: Prefix for chunk filenames
            chunk_size_bytes: Target size per chunk file in bytes
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.chunk_size_bytes = chunk_size_bytes

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._buffer: List[np.ndarray] = []
        self._buffer_bytes = 0
        self._chunk_idx = 0
        self._global_offset = 0
        self._document_boundaries: List[Tuple[int, int]] = []

    def write(self, data: np.ndarray) -> None:
        """Write data to the buffer, flushing to disk if needed.

        Args:
            data: Numpy array to write
        """
        self._buffer.append(data)
        self._buffer_bytes += data.nbytes

        if self._buffer_bytes >= self.chunk_size_bytes:
            self._flush_chunk()

    def write_document(self, data: np.ndarray) -> None:
        """Write a document and track its boundaries.

        Args:
            data: Numpy array representing a single document
        """
        start = self._global_offset
        end = start + len(data)
        self._document_boundaries.append((start, end))
        self._global_offset = end

        self.write(data)

    def _flush_chunk(self) -> None:
        """Write buffered data to a chunk file."""
        if not self._buffer:
            return

        # Concatenate buffer
        chunk_data = np.concatenate(self._buffer)

        # Write to file
        chunk_path = self.output_dir / f"{self.prefix}_{self._chunk_idx:04d}.npy"
        np.save(chunk_path, chunk_data)

        # Reset buffer
        self._buffer = []
        self._buffer_bytes = 0
        self._chunk_idx += 1

    def finalize(self) -> None:
        """Flush remaining data and write metadata."""
        # Write any remaining buffered data
        self._flush_chunk()

        # Write document boundaries
        boundaries_path = self.output_dir / f"{self.prefix}_boundaries.csv"
        with open(boundaries_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start", "end"])
            writer.writerows(self._document_boundaries)

    def get_document_boundaries(self) -> List[Tuple[int, int]]:
        """Get document boundaries tracked so far.

        Returns:
            List of (start, end) tuples
        """
        return self._document_boundaries.copy()

    def get_chunk_count(self) -> int:
        """Get number of chunks written so far.

        Returns:
            Number of chunk files
        """
        return self._chunk_idx


def merge_chunks(
    chunk_dir: Path,
    prefix: str,
    output_path: Path = None
) -> np.ndarray:
    """Merge chunk files into a single array.

    Args:
        chunk_dir: Directory containing chunk files
        prefix: Prefix used when writing chunks
        output_path: Optional path to write merged array

    Returns:
        Merged numpy array

    Raises:
        ValueError: If no chunks are found
    """
    chunk_files = sorted(chunk_dir.glob(f"{prefix}_*.npy"))

    if not chunk_files:
        raise ValueError(f"No chunks found for prefix '{prefix}' in {chunk_dir}")

    # Load and concatenate
    arrays = [np.load(f) for f in chunk_files]
    merged = np.concatenate(arrays)

    if output_path:
        np.save(output_path, merged)

    return merged


def calculate_chunk_boundaries(
    item_sizes: List[int],
    chunk_size: int
) -> List[Tuple[int, int]]:
    """Calculate boundaries for chunking items of varying sizes.

    Ensures items are not split across chunks.

    Args:
        item_sizes: List of item sizes
        chunk_size: Maximum chunk size

    Returns:
        List of (start, end) boundaries for each chunk
    """
    if not item_sizes:
        return []

    boundaries = []
    current_start = 0
    current_size = 0

    for i, size in enumerate(item_sizes):
        if current_size + size > chunk_size and current_size > 0:
            # Start new chunk
            current_end = sum(item_sizes[:i])
            boundaries.append((current_start, current_end))
            current_start = current_end
            current_size = size
        else:
            current_size += size

    # Final chunk
    total_size = sum(item_sizes)
    if current_start < total_size:
        boundaries.append((current_start, total_size))

    return boundaries


class ChunkedReader:
    """Reads chunked data files."""

    def __init__(self, chunk_dir: Path, prefix: str):
        """Initialize reader.

        Args:
            chunk_dir: Directory containing chunk files
            prefix: Prefix used when writing chunks
        """
        self.chunk_dir = Path(chunk_dir)
        self.prefix = prefix
        self._chunk_files = sorted(self.chunk_dir.glob(f"{prefix}_*.npy"))

    def __len__(self) -> int:
        """Get number of chunks."""
        return len(self._chunk_files)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterate over chunks."""
        for chunk_file in self._chunk_files:
            yield np.load(chunk_file)

    def get_chunk(self, idx: int) -> np.ndarray:
        """Get a specific chunk by index.

        Args:
            idx: Chunk index

        Returns:
            Chunk data
        """
        return np.load(self._chunk_files[idx])

    def get_document_boundaries(self) -> List[Tuple[int, int]]:
        """Load document boundaries if available.

        Returns:
            List of (start, end) tuples, or empty list if not available
        """
        boundaries_path = self.chunk_dir / f"{self.prefix}_boundaries.csv"

        if not boundaries_path.exists():
            return []

        boundaries = []
        with open(boundaries_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                boundaries.append((int(row["start"]), int(row["end"])))

        return boundaries
