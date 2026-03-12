"""Checkpoint chunk writer for SLURM array jobs.

Aggregates unchunked per-image Parquet files into dashboard chunks,
rebuilds the combined analysis Parquet, and updates the master CSV
so users can download partial results mid-run.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
import polars as pl

from ._cli_file_locking import file_lock
from ._cli_output_manager import _atomic_write

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def aggregate_chunks(output_dir: Path) -> None:
    """Aggregate unchunked per-image Parquets into a dashboard chunk."""
    progress_dir = output_dir / "progress"
    chunks_dir = progress_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    state_path = progress_dir / "chunk_state.json"
    state = _read_json_locked(
        state_path, default={"chunked_files": [], "next_chunk_id": 0}
    )
    chunked_files: set[str] = set(state.get("chunked_files", []))
    next_chunk_id: int = state.get("next_chunk_id", 0)

    new_files = _scan_unchunked_parquets(output_dir / "results", chunked_files)
    if not new_files:
        logger.info("No new Parquet files to chunk")
        return

    chunk_df = _read_and_concat(new_files)
    if chunk_df is None:
        return

    logger.info(
        "Chunk data: %d rows x %d cols, %.0f MB estimated",
        chunk_df.height,
        chunk_df.width,
        chunk_df.estimated_size("mb"),
    )

    chunk_name = f"chunk_{next_chunk_id:03d}.parquet"
    _atomic_write(
        chunks_dir / chunk_name,
        lambda p: chunk_df.write_parquet(p, compression="zstd", compression_level=3),
    )

    state["chunked_files"] = sorted(chunked_files)
    state["next_chunk_id"] = next_chunk_id + 1
    _write_json_locked(state, state_path)

    manifest_path = progress_dir / "chunk_manifest.json"
    manifest = _read_json_locked(
        manifest_path, default={"chunks": [], "total_rows": 0}
    )
    datasets_in_chunk = chunk_df["Metadata_Dataset"].unique().to_list()
    manifest["chunks"].append(
        {
            "name": chunk_name,
            "rows": chunk_df.height,
            "datasets": sorted(str(d) for d in datasets_in_chunk),
        }
    )
    manifest["total_rows"] = sum(c["rows"] for c in manifest["chunks"])
    _write_json_locked(manifest, manifest_path)

    combined = _incremental_combined(chunk_df, progress_dir / "analysis_full.parquet")
    if combined is not None:
        _atomic_write(
            progress_dir / "analysis_full.parquet",
            lambda p: combined.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )
        _atomic_write(
            output_dir / "master_measurements.csv",
            lambda p: combined.write_csv(p),
        )

    logger.info(
        "Chunk %s written: %d new files, %d rows (total: %d rows across %d chunks)",
        chunk_name,
        len(new_files),
        chunk_df.height,
        manifest["total_rows"],
        len(manifest["chunks"]),
    )


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------


def _scan_unchunked_parquets(
    results_dir: Path, chunked_files: set[str]
) -> list[Path]:
    """Find new per-image Parquet files not yet included in any chunk.

    Args:
        results_dir: The ``results/`` directory containing dataset subdirs.
        chunked_files: Set of relative keys already chunked (mutated in place
            to include newly discovered files).

    Returns:
        Sorted list of new Parquet file paths.
    """
    new_files: list[Path] = []
    if not results_dir.is_dir():
        return new_files

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        meas_dir = dataset_dir / "measurements"
        if not meas_dir.is_dir():
            continue
        for pq_file in sorted(meas_dir.glob("*.parquet")):
            rel_key = f"{dataset_dir.name}/{pq_file.name}"
            if rel_key not in chunked_files:
                new_files.append(pq_file)
                chunked_files.add(rel_key)

    return new_files


def _read_and_concat(parquet_files: list[Path]) -> pl.DataFrame | None:
    """Read per-image Parquet files, ensure ``Metadata_Dataset``, and concat.

    Args:
        parquet_files: Paths to per-image Parquet files.

    Returns:
        Concatenated DataFrame, or ``None`` if no files could be read.
    """
    frames: list[pl.DataFrame] = []
    for pq_file in parquet_files:
        try:
            df = pl.read_parquet(pq_file)
            if "Metadata_Dataset" not in df.columns:
                dataset_name = pq_file.parent.parent.name
                df = df.insert_column(
                    0, pl.lit(dataset_name).alias("Metadata_Dataset")
                )
            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pq_file, exc)

    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def _incremental_combined(
    new_chunk: pl.DataFrame, existing_path: Path
) -> pl.DataFrame:
    """Append *new_chunk* to the existing combined Parquet, or return *new_chunk* if none exists.

    Falls back to `_rebuild_combined()` if the existing file is corrupt.

    Args:
        new_chunk: Newly created chunk DataFrame.
        existing_path: Path to the existing ``analysis_full.parquet``.

    Returns:
        Combined DataFrame.
    """
    if not existing_path.exists():
        return new_chunk
    try:
        prev = pl.read_parquet(existing_path)
        return pl.concat([prev, new_chunk], how="diagonal_relaxed")
    except Exception as exc:
        logger.warning("Failed to read existing combined Parquet, rebuilding: %s", exc)
        return _rebuild_combined(existing_path.parent / "chunks") or new_chunk


def _rebuild_combined(chunks_dir: Path) -> pl.DataFrame | None:
    """Rebuild a single DataFrame from all existing chunk files.

    Args:
        chunks_dir: Directory containing ``chunk_*.parquet`` files.

    Returns:
        Concatenated DataFrame, or ``None`` if no chunks could be read.
    """
    frames: list[pl.DataFrame] = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.parquet")):
        try:
            frames.append(pl.read_parquet(chunk_file))
        except Exception as exc:
            logger.warning("Failed to read chunk %s: %s", chunk_file, exc)

    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_json_locked(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    """Read JSON with a shared file lock, returning *default* on failure.

    Args:
        path: JSON file to read.
        default: Value returned when *path* is missing or corrupt.

    Returns:
        Parsed dict or a copy of *default*.
    """
    if not path.exists():
        return dict(default)
    try:
        with open(path, "r") as f:
            with file_lock(f, shared=True):
                f.seek(0)
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(default)


def _write_json_locked(data: dict[str, Any], path: Path) -> None:
    """Write JSON with an exclusive file lock and fsync for durability.

    Args:
        data: Dict to serialize.
        path: Destination file (parent dirs created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        with file_lock(f, shared=False):
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())


if __name__ == "__main__":
    aggregate_chunks()
