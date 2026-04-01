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
from typing import Any, Dict, List, Optional

import click
import polars as pl

from ._cli_file_locking import file_lock
from ._cli_output_manager import _atomic_write, join_metadata
from ._cli_utils import load_job_metadata, scan_parquets

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def aggregate_chunks(output_dir: Path) -> None:
    """Aggregate unchunked per-image Parquets into a dashboard chunk.

    The entire read-scan-write cycle is serialised via an exclusive
    file lock on ``progress/.chunk_lock`` so that concurrent checkpoint
    tasks (SLURM may schedule multiple sentinels near-simultaneously)
    do not race on the shared state files or duplicate Parquet data.
    """
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    lock_path = progress_dir / ".chunk_lock"
    lock_path.touch(exist_ok=True)

    with open(lock_path, "r") as lock_fh:
        with file_lock(lock_fh, shared=False, timeout=120.0):
            _aggregate_chunks_locked(output_dir, progress_dir)


def _aggregate_chunks_locked(output_dir: Path, progress_dir: Path) -> None:
    """Inner body of chunk aggregation, called under exclusive lock."""
    chunks_dir = progress_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    state_path = progress_dir / "chunk_state.json"
    state = _read_json(state_path, default={"chunked_files": [], "next_chunk_id": 0})
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
    _write_json(state, state_path)

    manifest_path = progress_dir / "chunk_manifest.json"
    manifest = _read_json(manifest_path, default={"chunks": [], "total_rows": 0})
    datasets_in_chunk = chunk_df["Metadata_Dataset"].unique().to_list()
    manifest["chunks"].append(
        {
            "name": chunk_name,
            "rows": chunk_df.height,
            "datasets": sorted(str(d) for d in datasets_in_chunk),
        }
    )
    manifest["total_rows"] = sum(c["rows"] for c in manifest["chunks"])
    _write_json(manifest, manifest_path)

    combined = _incremental_combined(chunk_df, progress_dir / "analysis_full.parquet")
    combined_with_metadata: Optional[pl.DataFrame] = None
    if combined is not None:
        _atomic_write(
            progress_dir / "analysis_full.parquet",
            lambda p: combined.write_parquet(
                p, compression="zstd", compression_level=3
            ),
        )

        job_meta = load_job_metadata(progress_dir)
        csv_str = job_meta.get("metadata_csv") if job_meta else None
        metadata_csv = Path(csv_str) if csv_str else None
        combined_with_metadata = combined
        if metadata_csv is not None:
            try:
                combined_with_metadata = join_metadata(combined, metadata_csv)
            except Exception:
                logger.warning(
                    "Metadata join failed, writing CSV without metadata",
                    exc_info=True,
                )
                combined_with_metadata = combined

        _atomic_write(
            output_dir / "master_measurements.csv",
            lambda p: combined_with_metadata.write_csv(p),  # type: ignore[union-attr]
        )

        _run_analysis_plugins(output_dir, progress_dir, combined_with_metadata)

    for ds_name, ds_df in chunk_df.group_by("Metadata_Dataset"):
        _update_dataset_parquet(output_dir, str(ds_name[0]), ds_df)

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
            if pq_file.name.startswith("_"):
                continue
            rel_key = f"{dataset_dir.name}/{pq_file.name}"
            if rel_key not in chunked_files:
                new_files.append(pq_file)
                chunked_files.add(rel_key)

    return new_files


def _read_and_concat(parquet_files: list[Path]) -> pl.DataFrame | None:
    """Read per-image Parquet files, ensure ``Metadata_Dataset``, and concat.

    Uses :func:`scan_parquets` for lazy scans locally and ``tar``
    streaming on HPC shared filesystems to reduce per-file metadata
    overhead.

    Args:
        parquet_files: Paths to per-image Parquet files.

    Returns:
        Concatenated DataFrame, or ``None`` if no files could be read.
    """
    lazy_frames = scan_parquets(parquet_files)
    if not lazy_frames:
        return None

    frames: list[pl.DataFrame] = []
    for pq_path, lf in lazy_frames.items():
        try:
            df = lf.collect()
            df = df.with_columns(pl.lit(pq_path.stem).alias("Metadata_ImageFile"))
            if "Metadata_Dataset" not in df.columns:
                dataset_name = pq_path.parent.parent.name
                df = df.insert_column(
                    0, pl.lit(dataset_name).alias("Metadata_Dataset")
                )
            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pq_path, exc)

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
# Metadata / analysis helpers
# ---------------------------------------------------------------------------


def _run_analysis_plugins(
    output_dir: Path, progress_dir: Path, merged_df: Optional["pl.DataFrame"]
) -> None:
    """Dispatch to analysis plugins with the combined DataFrame.

    Args:
        output_dir: Root output directory.
        progress_dir: Progress directory for sidecar files.
        merged_df: Merged measurement DataFrame, or ``None``.
    """
    try:
        from ._dashboard._analysis._prepare_context import AnalysisPrepareContext
        from phenotypic.tools_.register import AnalysisPluginRegistry
        # Trigger plugin registration
        from ._dashboard._analysis import (  # noqa: F401
            _image_viewer,
            _raw_table,
            _scatter_plot,
            _summary_stats,
        )
    except ImportError:
        logger.debug("Analysis plugins not available")
        return

    if merged_df is None:
        return

    ctx = AnalysisPrepareContext(
        output_dir=output_dir, progress_dir=progress_dir, merged_df=merged_df
    )
    for name in AnalysisPluginRegistry.available():
        plugin = AnalysisPluginRegistry.get(name)()
        try:
            plugin.prepare_data(ctx)
        except Exception:
            logger.exception("Plugin %r failed during prepare_data", name)


def _update_dataset_parquet(
    output_dir: Path, dataset_name: str, new_df: pl.DataFrame
) -> None:
    """Append new measurements to the dataset-level aggregated parquet.

    Args:
        output_dir: Root output directory.
        dataset_name: Name of the dataset.
        new_df: DataFrame of newly chunked measurements for this dataset.
    """
    agg_path = (
        output_dir / "results" / dataset_name / "measurements" / "_dataset_aggregated.parquet"
    )
    if agg_path.exists():
        try:
            prev = pl.read_parquet(agg_path)
            new_df = pl.concat([prev, new_df], how="diagonal_relaxed")
        except Exception:
            logger.warning("Corrupt %s, rebuilding from new data", agg_path)
    _atomic_write(
        agg_path,
        lambda p: new_df.write_parquet(p, compression="zstd", compression_level=3),
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    """Read JSON, returning *default* on failure.

    Caller must hold the outer chunk lock.

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
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(default)


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON with fsync for durability.

    Caller must hold the outer chunk lock.

    Args:
        data: Dict to serialize.
        path: Destination file (parent dirs created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    aggregate_chunks()
