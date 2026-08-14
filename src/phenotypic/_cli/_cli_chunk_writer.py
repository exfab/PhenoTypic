"""Checkpoint chunk writer for SLURM array jobs.

Aggregates unchunked per-image Parquet files into dashboard chunks,
rebuilds the rolling combined measurement state, and updates the master CSV
(:data:`~phenotypic.sdk_.MASTER_MEASUREMENTS_CSV`) so users can download
partial results mid-run.

Note: this writer intentionally bypasses
:func:`~phenotypic._cli._cli_output_manager.finalize_post_master_outputs`
(chunks are mid-run intermediate publications; the full post / per-feature
split / analysis chain runs once at the end via
:func:`~phenotypic._cli._cli_output_manager.aggregate_measurements`). See
the project CLAUDE.md "Master vs. mirror outputs" gotcha for the full
contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import polars as pl

from ._cli_file_locking import file_lock
from ._cli_utils import scan_parquets
from phenotypic.schema import EXPERIMENT_METADATA, METADATA
from phenotypic.sdk_ import (
    DIR_CHUNKS,
    CHUNK_STATE_JSON,
    CHUNK_MANIFEST_JSON,
    DATASET_AGGREGATED_PARQUET,
    DIR_RESULTS,
    DIR_MEASUREMENTS,
    ChunkStateKey,
    ChunkManifestKey,
    chunk_state_path,
    PARQUET_WRITE_OPTIONS,
    atomic_write_json,
    atomic_write_with_writer,
    chunk_parquet_filename,
    chunk_lock_path,
    analysis_full_parquet_path,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    progress_dir as progress_dir_helper,
)

logger = logging.getLogger(__name__)


def flush_unchunked_measurements(output_dir: Path) -> None:
    """Chunk any per-image measurement files not yet consumed.

    The plain-function form of :func:`aggregate_chunks`, so in-process callers
    can flush without going through click. The SLURM finalize path
    (``_cli_checkpoint_handler._run_finalize``) uses it: checkpoint sentinels
    fire only every ``checkpoint_interval`` images and no terminal sentinel is
    emitted, so a run whose image count is not a multiple of the interval
    leaves its last partial group unchunked.

    Idempotent — :func:`_scan_unchunked_parquets` skips files already recorded
    in the chunk state, so a run that divides evenly flushes nothing.

    The entire read-scan-write cycle is serialised via an exclusive file lock
    on ``progress/.chunk_lock`` so that concurrent checkpoint tasks (SLURM may
    schedule multiple sentinels near-simultaneously) do not race on the shared
    state files or duplicate data.
    """
    progress_dir = progress_dir_helper(output_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)

    lock_path = chunk_lock_path(progress_dir)
    lock_path.touch(exist_ok=True)

    with open(lock_path, "r") as lock_fh:
        with file_lock(lock_fh, shared=False, timeout=120.0):
            _aggregate_chunks_locked(output_dir, progress_dir)


def flush_trailing_measurements_if_chunked(output_dir: Path) -> None:
    """Flush trailing per-image parquets, but only for runs that were chunked.

    Guards :func:`flush_unchunked_measurements` on the presence of chunk state,
    which is what distinguishes the two aggregation regimes:

    * A **chunked** run (SLURM) publishes its master from
      ``_dataset_aggregated.parquet``, because
      ``discover_measurement_sources`` prefers the aggregate and skips the
      individual per-image parquets. Any image not yet chunked is therefore
      invisible to aggregation — and checkpoint sentinels fire only every
      ``checkpoint_interval`` images with no terminal sentinel, so the last
      partial group is always in that state. Flushing first is what makes the
      master complete.
    * An **unchunked** run (local, staged-local) has no aggregate, so
      aggregation already reads every per-image parquet directly. Flushing
      would only manufacture chunk artifacts the run never had, so it is
      skipped.

    Without the guard this would change local runs' output layout to fix a bug
    they do not have.
    """
    if not chunk_state_path(output_dir).is_file():
        return
    flush_unchunked_measurements(output_dir)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def aggregate_chunks(output_dir: Path) -> None:
    """Aggregate unchunked per-image measurement files into a dashboard chunk.

    The checkpoint-sentinel entry point. See
    :func:`flush_unchunked_measurements` for the behaviour and its locking.
    """
    flush_unchunked_measurements(output_dir)


def _aggregate_chunks_locked(output_dir: Path, progress_dir: Path) -> None:
    """Inner body of chunk aggregation, called under exclusive lock."""
    chunks_dir = progress_dir / DIR_CHUNKS
    chunks_dir.mkdir(parents=True, exist_ok=True)

    state_path = progress_dir / CHUNK_STATE_JSON
    state = _read_json(
        state_path,
        default={
            ChunkStateKey.CHUNKED_FILES: [],
            ChunkStateKey.NEXT_CHUNK_ID: 0,
        },
    )
    chunked_files: set[str] = set(state.get(ChunkStateKey.CHUNKED_FILES, []))
    next_chunk_id: int = state.get(ChunkStateKey.NEXT_CHUNK_ID, 0)

    new_files = _scan_unchunked_parquets(
        output_dir / DIR_RESULTS, chunked_files
    )
    if not new_files:
        logger.info("No new measurement files to chunk")
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

    chunk_name = chunk_parquet_filename(next_chunk_id)
    atomic_write_with_writer(
        chunks_dir / chunk_name,
        lambda p: chunk_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
    )

    state[ChunkStateKey.CHUNKED_FILES] = sorted(chunked_files)
    state[ChunkStateKey.NEXT_CHUNK_ID] = next_chunk_id + 1
    _write_json(state, state_path)

    manifest_path = progress_dir / CHUNK_MANIFEST_JSON
    manifest = _read_json(
        manifest_path,
        default={ChunkManifestKey.CHUNKS: [], ChunkManifestKey.TOTAL_ROWS: 0},
    )
    datasets_in_chunk = (
        chunk_df[str(EXPERIMENT_METADATA.DATASET)].unique().to_list()
    )
    manifest[ChunkManifestKey.CHUNKS].append(
        {
            ChunkManifestKey.NAME: chunk_name,
            ChunkManifestKey.ROWS: chunk_df.height,
            ChunkManifestKey.DATASETS: sorted(
                str(d) for d in datasets_in_chunk
            ),
        }
    )
    manifest[ChunkManifestKey.TOTAL_ROWS] = sum(
        c[ChunkManifestKey.ROWS] for c in manifest[ChunkManifestKey.CHUNKS]
    )
    _write_json(manifest, manifest_path)

    combined = _incremental_combined(
        chunk_df, analysis_full_parquet_path(progress_dir)
    )
    if combined is not None:
        atomic_write_with_writer(
            analysis_full_parquet_path(progress_dir),
            lambda p: combined.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )

        atomic_write_with_writer(
            master_measurements_csv_path(output_dir),
            lambda p: combined.write_csv(p),
        )
        atomic_write_with_writer(
            master_measurements_parquet_path(output_dir),
            lambda p: combined.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )

    for ds_name, ds_df in chunk_df.group_by(str(EXPERIMENT_METADATA.DATASET)):
        _update_dataset_parquet(output_dir, str(ds_name[0]), ds_df)

    logger.info(
        "Chunk %s written: %d new files, %d rows (total: %d rows across %d chunks)",
        chunk_name,
        len(new_files),
        chunk_df.height,
        manifest[ChunkManifestKey.TOTAL_ROWS],
        len(manifest[ChunkManifestKey.CHUNKS]),
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
        Sorted list of new measurement file paths.
    """
    new_files: list[Path] = []
    if not results_dir.is_dir():
        return new_files

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        meas_dir = dataset_dir / DIR_MEASUREMENTS
        if not meas_dir.is_dir():
            continue
        for meas_file in sorted(meas_dir.glob("*.parquet")):
            if meas_file.name.startswith("_"):
                continue
            rel_key = f"{dataset_dir.name}/{meas_file.name}"
            if rel_key not in chunked_files:
                new_files.append(meas_file)
                chunked_files.add(rel_key)

    return new_files


def _attach_image_identity(
    df: pl.DataFrame, stem: str, suffix: str = ""
) -> pl.DataFrame:
    """Attach the canonical image-identity columns to *df*.

    Adds :data:`~phenotypic.schema.METADATA.IMAGE_NAME` (the image stem) and,
    when not already present, :data:`~phenotypic.schema.METADATA.SUFFIX` (the
    file extension). Non-clobbering: an existing ``Metadata_FileSuffix`` emitted
    upstream by ``insert_metadata`` is preserved rather than overwritten with a
    caller-supplied fallback. Replaces the retired ad-hoc per-image-file
    stem column, which duplicated the canonical image name.

    Args:
        df: Per-image measurement frame to annotate.
        stem: Image stem, written to ``Metadata_ImageName``.
        suffix: File extension (e.g. ``".tif"``), written to
            ``Metadata_FileSuffix`` only when the frame lacks it.

    Returns:
        The frame with the identity columns attached.
    """
    exprs = [pl.lit(stem).alias(str(METADATA.IMAGE_NAME))]
    if str(METADATA.SUFFIX) not in df.columns:
        exprs.append(pl.lit(suffix).alias(str(METADATA.SUFFIX)))
    return df.with_columns(exprs)


def _read_and_concat(parquet_files: list[Path]) -> pl.DataFrame | None:
    """Read per-image Parquet files, ensure ``Metadata_Dataset``, and concat.

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
            df = _attach_image_identity(df, pq_path.stem)
            if str(EXPERIMENT_METADATA.DATASET) not in df.columns:
                dataset_name = pq_path.parent.parent.name
                df = df.insert_column(
                    0,
                    pl.lit(dataset_name).alias(
                        str(EXPERIMENT_METADATA.DATASET)
                    ),
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
    """Append *new_chunk* to the existing combined file, or return *new_chunk* if none exists.

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
        logger.warning(
            "Failed to read existing combined file, rebuilding: %s", exc
        )
        return _rebuild_combined(existing_path.parent / "chunks") or new_chunk


def _rebuild_combined(chunks_dir: Path) -> pl.DataFrame | None:
    """Rebuild a single DataFrame from all existing chunk files.

    Args:
        chunks_dir: Directory containing ``chunk_*.parquet`` files.

    Returns:
        Concatenated DataFrame, or ``None`` if no chunks could be read.
    """
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    frames: list[pl.DataFrame] = []
    for chunk_file in chunk_files:
        try:
            frames.append(pl.read_parquet(chunk_file))
        except Exception as exc:
            logger.warning("Failed to read chunk %s: %s", chunk_file, exc)

    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
def _update_dataset_parquet(
    output_dir: Path, dataset_name: str, new_df: pl.DataFrame
) -> None:
    """Append new measurements to the dataset-level aggregated Parquet file.

    Args:
        output_dir: Root output directory.
        dataset_name: Name of the dataset.
        new_df: DataFrame of newly chunked measurements for this dataset.
    """
    agg_path = (
        output_dir
        / DIR_RESULTS
        / dataset_name
        / DIR_MEASUREMENTS
        / DATASET_AGGREGATED_PARQUET
    )
    if agg_path.exists():
        try:
            prev = pl.read_parquet(agg_path)
            new_df = pl.concat([prev, new_df], how="diagonal_relaxed")
        except Exception:
            logger.warning("Corrupt %s, rebuilding from new data", agg_path)
    atomic_write_with_writer(
        agg_path,
        lambda p: new_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),
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
    """Write JSON atomically with fsync for durability.

    Caller must hold the outer chunk lock.

    Args:
        data: Dict to serialize.
        path: Destination file (parent dirs created if needed).
    """
    atomic_write_json(path, data, sort_keys=False)


if __name__ == "__main__":
    aggregate_chunks()
