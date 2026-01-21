"""Checkpoint manager for resumable processing.

Provides atomic checkpoint writes with resume support for long-running jobs.
Used by both translation and tokenization pipelines.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class CheckpointManager:
    """Handles atomic checkpoint writes with resume support.

    Features:
    - Atomic writes: temp file + rename to prevent corruption
    - Automatic resume: finds latest valid checkpoint
    - Metadata tracking: stores processing state with each checkpoint
    - Cleanup: removes checkpoint files after successful completion

    Example:
        >>> manager = CheckpointManager(Path("/tmp/checkpoints"), prefix="translation")
        >>> batch_idx, metadata = manager.get_resume_point()
        >>> for i, batch in enumerate(data_batches, start=batch_idx):
        ...     result = process_batch(batch)
        ...     manager.save(result, batch_idx=i, metadata={"processed": (i+1) * 100})
        >>> manager.merge_checkpoints(Path("/tmp/output.parquet"))
        >>> manager.cleanup()
    """

    def __init__(self, checkpoint_dir: Path, prefix: str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            prefix: Prefix for checkpoint filenames (e.g., dataset name)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prefix = prefix

        # Create directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: pd.DataFrame, batch_idx: int, metadata: dict) -> Path:
        """Save a checkpoint atomically.

        Writes to a temp file first, then renames to prevent corruption
        if the process is killed mid-write.

        Args:
            data: DataFrame to save
            batch_idx: Batch index for ordering checkpoints
            metadata: Dictionary of metadata (e.g., progress info)

        Returns:
            Path to the saved checkpoint file
        """
        # Format batch index with zero-padding for correct sorting
        batch_str = f"{batch_idx:04d}"

        temp_data_path = self.checkpoint_dir / f"{self.prefix}.{batch_str}.tmp.parquet"
        temp_meta_path = self.checkpoint_dir / f"{self.prefix}.{batch_str}.tmp.meta.json"
        final_data_path = self.checkpoint_dir / f"{self.prefix}.{batch_str}.parquet"
        final_meta_path = self.checkpoint_dir / f"{self.prefix}.{batch_str}.meta.json"

        try:
            # Write data to temp file
            data.to_parquet(temp_data_path, index=False)

            # Add timestamp to metadata
            metadata_with_timestamp = {
                **metadata,
                "timestamp": datetime.now().isoformat(),
                "batch_idx": batch_idx,
            }

            # Write metadata to temp file
            with open(temp_meta_path, "w") as f:
                json.dump(metadata_with_timestamp, f, indent=2)

            # Atomic rename - this is the commit point
            temp_data_path.rename(final_data_path)
            temp_meta_path.rename(final_meta_path)

            return final_data_path

        except Exception:
            # Clean up temp files on failure
            if temp_data_path.exists():
                temp_data_path.unlink()
            if temp_meta_path.exists():
                temp_meta_path.unlink()
            raise

    def get_resume_point(self) -> tuple[int, Optional[dict]]:
        """Find the latest valid checkpoint and return resume point.

        A valid checkpoint has both a parquet file and a metadata file.

        Returns:
            Tuple of (next_batch_idx, last_metadata)
            - next_batch_idx: The batch index to resume from
            - last_metadata: Metadata from the last checkpoint, or None if no checkpoints
        """
        # Find all valid checkpoints (have both data and metadata)
        checkpoints = []

        for parquet_file in self.checkpoint_dir.glob(f"{self.prefix}.*.parquet"):
            # Skip temp files
            if ".tmp." in parquet_file.name:
                continue

            # Extract batch index from filename
            try:
                batch_str = parquet_file.stem.split(".")[-1]
                batch_idx = int(batch_str)
            except (IndexError, ValueError):
                continue

            # Check for corresponding metadata file
            # Can't use with_suffix() because it only handles single extension
            meta_file = parquet_file.parent / parquet_file.name.replace(".parquet", ".meta.json")
            if not meta_file.exists():
                continue

            checkpoints.append((batch_idx, meta_file))

        if not checkpoints:
            return 0, None

        # Find the latest checkpoint
        checkpoints.sort(key=lambda x: x[0])
        latest_batch_idx, latest_meta_file = checkpoints[-1]

        # Load metadata
        with open(latest_meta_file) as f:
            metadata = json.load(f)

        return latest_batch_idx + 1, metadata

    def merge_checkpoints(self, output_path: Path) -> None:
        """Merge all checkpoints into a single output file.

        Checkpoints are merged in batch index order.

        Args:
            output_path: Path to write the merged output

        Raises:
            ValueError: If no checkpoints are found
        """
        # Find all valid checkpoint files
        checkpoint_files = []

        for parquet_file in sorted(self.checkpoint_dir.glob(f"{self.prefix}.*.parquet")):
            # Skip temp files
            if ".tmp." in parquet_file.name:
                continue

            # Extract batch index for sorting
            try:
                batch_str = parquet_file.stem.split(".")[-1]
                batch_idx = int(batch_str)
                checkpoint_files.append((batch_idx, parquet_file))
            except (IndexError, ValueError):
                continue

        if not checkpoint_files:
            raise ValueError(f"No checkpoints found for prefix '{self.prefix}'")

        # Sort by batch index
        checkpoint_files.sort(key=lambda x: x[0])

        # Merge all checkpoints
        dfs = []
        for _, parquet_file in checkpoint_files:
            df = pd.read_parquet(parquet_file)
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        merged.to_parquet(output_path, index=False)

    def cleanup(self) -> None:
        """Remove all checkpoint files for this prefix.

        Should be called after successful completion and merge.
        """
        # Remove parquet files
        for f in self.checkpoint_dir.glob(f"{self.prefix}.*.parquet"):
            if ".tmp." not in f.name:
                f.unlink()

        # Remove metadata files
        for f in self.checkpoint_dir.glob(f"{self.prefix}.*.meta.json"):
            if ".tmp." not in f.name:
                f.unlink()

        # Remove temp files (shouldn't exist, but just in case)
        for f in self.checkpoint_dir.glob(f"{self.prefix}.*.tmp.*"):
            f.unlink()

    def get_all_metadata(self) -> list[dict]:
        """Get metadata from all checkpoints in order.

        Returns:
            List of metadata dictionaries, sorted by batch index
        """
        metadata_list = []

        for meta_file in self.checkpoint_dir.glob(f"{self.prefix}.*.meta.json"):
            # Skip temp files
            if ".tmp." in meta_file.name:
                continue

            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
                    batch_idx = metadata.get("batch_idx", 0)
                    metadata_list.append((batch_idx, metadata))
            except (json.JSONDecodeError, OSError):
                continue

        # Sort by batch index
        metadata_list.sort(key=lambda x: x[0])

        return [m for _, m in metadata_list]

    def get_checkpoint_count(self) -> int:
        """Get the number of valid checkpoints.

        Returns:
            Number of checkpoint files
        """
        count = 0
        for f in self.checkpoint_dir.glob(f"{self.prefix}.*.parquet"):
            if ".tmp." not in f.name:
                count += 1
        return count

    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist.

        Returns:
            True if at least one valid checkpoint exists
        """
        return self.get_checkpoint_count() > 0
