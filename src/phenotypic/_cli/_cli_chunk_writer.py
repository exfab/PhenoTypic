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
import shutil
from pathlib import Path
from typing import Any, Final, NamedTuple

import click
import polars as pl

from ._cli_file_locking import file_lock
from ._metadata_join import normalize_measurement_metadata_columns
from ._cli_utils import scan_parquets
from phenotypic.schema import EXPERIMENT, IMAGE, OBJECT
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

    from ._cli_completion import authorized_measurement_sources

    authorized_sources = authorized_measurement_sources(output_dir)
    embedded_authority = authorized_sources is not None
    if authorized_sources is not None:
        new_files = [
            path
            for path in sorted(authorized_sources)
            if _chunk_state_key(path) not in chunked_files
        ]
    else:
        new_files = _scan_unchunked_parquets(
            output_dir / DIR_RESULTS, chunked_files
        )
    if not new_files:
        logger.info("No new measurement files to chunk")
        return

    read = _read_and_concat(new_files)
    if read.unreadable:
        # ERROR and by name: these are user measurements. They are deliberately
        # left out of `chunked_files` below, so the next checkpoint -- or the
        # finalize-time flush -- retries them.
        logger.error(
            "Could not read %d per-image parquet(s); leaving them unchunked "
            "for a later retry: %s",
            len(read.unreadable),
            ", ".join(read.unreadable),
        )
    chunk_df = read.frame
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

    manifest_path = progress_dir / CHUNK_MANIFEST_JSON
    manifest = _read_json(
        manifest_path,
        default={ChunkManifestKey.CHUNKS: [], ChunkManifestKey.TOTAL_ROWS: 0},
    )
    datasets_in_chunk = chunk_df[str(EXPERIMENT.DATASET)].unique().to_list()
    entry = {
        ChunkManifestKey.NAME: chunk_name,
        ChunkManifestKey.ROWS: chunk_df.height,
        ChunkManifestKey.DATASETS: sorted(str(d) for d in datasets_in_chunk),
    }
    # Replace any entry for this chunk name rather than appending. Chunk state
    # (including `next_chunk_id`) is committed last, so a killed task's retry
    # regenerates the *same* chunk name — and an append would record it twice
    # and double `total_rows`, which the dashboard reports as run progress.
    existing = manifest[ChunkManifestKey.CHUNKS]
    for index, chunk_entry in enumerate(existing):
        if chunk_entry.get(ChunkManifestKey.NAME) == chunk_name:
            existing[index] = entry
            break
    else:
        existing.append(entry)
    manifest[ChunkManifestKey.TOTAL_ROWS] = sum(
        c[ChunkManifestKey.ROWS] for c in manifest[ChunkManifestKey.CHUNKS]
    )
    _write_json(manifest, manifest_path)

    combined = _incremental_combined(
        chunk_df, analysis_full_parquet_path(progress_dir)
    )
    if combined is not None:
        # Same idempotency requirement as the aggregate below: chunk state is
        # committed last, so a killed task re-chunks images whose rows already
        # reached these files. `_incremental_combined` is a bare concat, so
        # without this the retry doubles every colony in the rolling master —
        # the very artifact this module exists to publish mid-run.
        combined = _dedupe_on_colony_key(combined, context="rolling master")
        atomic_write_with_writer(
            analysis_full_parquet_path(progress_dir),
            lambda p: combined.write_parquet(p, **PARQUET_WRITE_OPTIONS),
        )

        if not embedded_authority:
            atomic_write_with_writer(
                master_measurements_csv_path(output_dir),
                lambda p: combined.write_csv(p),
            )
            atomic_write_with_writer(
                master_measurements_parquet_path(output_dir),
                lambda p: combined.write_parquet(p, **PARQUET_WRITE_OPTIONS),
            )

    if not embedded_authority:
        for ds_name, ds_df in chunk_df.group_by(str(EXPERIMENT.DATASET)):
            _update_dataset_parquet(output_dir, str(ds_name[0]), ds_df)

    # Commit the chunk state LAST, once every artifact it describes is durable.
    # SLURM kills checkpoint tasks routinely -- walltime, preemption, node
    # failure -- and this state is what tells later checkpoints (and the
    # finalize-time flush) that a per-image parquet has been consumed. Written
    # before `_update_dataset_parquet`, a kill in between would mark images
    # consumed whose rows never reached the aggregate; since final aggregation
    # prefers the aggregate they would never reach the master, and no flush
    # could recover them because the state says there is nothing to flush.
    #
    # Committing last inverts the failure into a safe one: a kill now leaves
    # images *unmarked* whose rows may already be in the aggregate, so the next
    # pass re-chunks them -- harmless, because `_update_dataset_parquet`
    # deduplicates on the colony key.
    #
    # Only files that actually produced rows are recorded. `_scan_unchunked_
    # parquets` deliberately does not mark on discovery.
    chunked_files.update(_chunk_state_key(path) for path in read.consumed)
    state[ChunkStateKey.CHUNKED_FILES] = sorted(chunked_files)
    state[ChunkStateKey.NEXT_CHUNK_ID] = next_chunk_id + 1
    _write_json(state, state_path)

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
        chunked_files: Relative keys already chunked. **Read only** — a file is
            recorded as consumed by the caller once it has actually been read,
            not on discovery. Marking here instead would consume a file the
            reader then failed on, and since final aggregation prefers the
            aggregate, a transient NFS/GPFS read error would silently drop
            that image from the master for good.

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
            # `_` skips this writer's own `_dataset_aggregated.parquet`.
            # `.` skips macOS AppleDouble sidecars, which exist beside every
            # file on an exFAT/FAT volume and are binary, not parquet — the
            # reader would raise on them.
            if meas_file.name.startswith(("_", ".")):
                continue
            if _chunk_state_key(meas_file) not in chunked_files:
                new_files.append(meas_file)

    return new_files


def _chunk_state_key(parquet_path: Path) -> str:
    """Return a stable dataset/image key for legacy or embedded authority."""
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, STORE_SUFFIX

    path = Path(parquet_path)
    suffix = MEASUREMENT_TABLE_RELATIVE_PATH.parts
    if tuple(path.parts[-len(suffix) :]) == suffix:
        store = path.parents[2]
        dataset = path.parents[4].name
        stem = store.name.removesuffix(STORE_SUFFIX)
        return f"{dataset}/{stem}.parquet"
    return f"{path.parent.parent.name}/{path.name}"


def _attach_image_identity(
    df: pl.DataFrame, stem: str, suffix: str = ""
) -> pl.DataFrame:
    """Attach the canonical image-identity columns to *df*.

    Adds :data:`~phenotypic.schema.IMAGE.IMAGE_NAME` (the image stem) and,
    when not already present, :data:`~phenotypic.schema.IMAGE.SUFFIX` (the
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
    exprs = [pl.lit(stem).alias(str(IMAGE.IMAGE_NAME))]
    if str(IMAGE.SUFFIX) not in df.columns:
        exprs.append(pl.lit(suffix).alias(str(IMAGE.SUFFIX)))
    return df.with_columns(exprs)


class _ConcatResult(NamedTuple):
    """What :func:`_read_and_concat` actually managed to read.

    Attributes:
        frame: The concatenated measurements, or ``None`` if nothing read.
        consumed: The paths that contributed rows. **Only these may be
            recorded in chunk state** — marking a file consumed that was never
            read is what makes a read error permanent.
        unreadable: ``"<name> (<reason>)"`` per file that could not be read.
            Reported rather than skipped in silence, which is how a short
            master goes unnoticed.
    """

    frame: pl.DataFrame | None
    consumed: list[Path]
    unreadable: list[str]


def _read_and_concat(parquet_files: list[Path]) -> _ConcatResult:
    """Read per-image Parquet files, normalize identity, and concat.

    Each frame is normalized *before* concatenation: ``Metadata_ImageName`` is
    set from the file stem, ``Metadata_FileSuffix`` is added when absent, and
    ``Metadata_Dataset`` is injected from the directory when absent (a
    ``--no-dataset-column`` run omits it).

    Two of those three are colony-key columns, which is why every reader of
    these files must go through here. Concatenating them raw and then
    deduplicating on the colony key collapses unrelated colonies: without
    ``Metadata_ImageName`` normalization every image shares one key value, so
    ``unique`` keeps one row per ``Object_Label`` across the whole dataset.

    Args:
        parquet_files: Paths to per-image Parquet files.

    Returns:
        A :class:`_ConcatResult`; ``frame`` is ``None`` when no file read.
    """
    lazy_frames = scan_parquets(parquet_files)
    # `scan_parquets` omits what it could not scan rather than raising.
    unreadable = [
        f"{path.name} (could not be scanned)"
        for path in parquet_files
        if path not in lazy_frames
    ]
    frames: list[pl.DataFrame] = []
    consumed: list[Path] = []
    for pq_path, lf in lazy_frames.items():
        try:
            df = normalize_measurement_metadata_columns(lf.collect())
            df = _attach_image_identity(df, pq_path.stem)
            if str(EXPERIMENT.DATASET) not in df.columns:
                dataset_name = pq_path.parent.parent.name
                df = df.insert_column(
                    0,
                    pl.lit(dataset_name).alias(str(EXPERIMENT.DATASET)),
                )
            frames.append(df)
            consumed.append(pq_path)
        except Exception as exc:
            unreadable.append(f"{pq_path.name} ({exc})")
    if not frames:
        return _ConcatResult(None, consumed, unreadable)
    return _ConcatResult(
        pl.concat(frames, how="diagonal_relaxed"), consumed, unreadable
    )


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
    new_chunk = normalize_measurement_metadata_columns(new_chunk)
    if not existing_path.exists():
        return new_chunk
    try:
        prev = normalize_measurement_metadata_columns(
            pl.read_parquet(existing_path)
        )
        return pl.concat([prev, new_chunk], how="diagonal_relaxed")
    except Exception as exc:
        logger.warning(
            "Failed to read existing combined file, rebuilding: %s", exc
        )
        rebuilt = _rebuild_combined(existing_path.parent / "chunks")
        return rebuilt if rebuilt is not None else new_chunk


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
            frames.append(
                normalize_measurement_metadata_columns(
                    pl.read_parquet(chunk_file)
                )
            )
        except Exception as exc:
            logger.warning("Failed to read chunk %s: %s", chunk_file, exc)

    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
#: Colony primary key -- one row per detected object per image per dataset,
#: the grain of every per-image measurement parquet.
_AGGREGATE_KEY: Final[tuple[str, ...]] = (
    str(EXPERIMENT.DATASET),
    str(IMAGE.IMAGE_NAME),
    str(OBJECT.LABEL),
)


def _dedupe_on_colony_key(df: pl.DataFrame, *, context: str) -> pl.DataFrame:
    """Collapse repeated colonies, keeping the later row.

    This is what makes every retry-exposed write in this module idempotent, so
    :func:`_aggregate_chunks_locked` can commit its chunk state *after* the
    data: a task killed mid-write re-chunks images whose rows already landed,
    and the second copy collapses into the first. It also absorbs the
    duplicates a ``--restart`` produces, since ``results/`` survives a restart
    while its images are re-measured and appended over the old rows.

    **The full key is required.** An earlier revision filtered
    :data:`_AGGREGATE_KEY` down to whichever columns happened to be present,
    which looks defensive and is the opposite: without ``Object_Label`` the key
    degrades to ``(dataset, image)`` and ``unique`` keeps exactly **one row per
    image**, silently deleting every other colony in it. Nothing guarantees
    that column — ``_read_and_concat`` only guarantees the dataset and image
    columns — so a partial key is refused rather than applied. Skipping leaves
    duplicate rows, which is recoverable and visible; dropping colonies is
    neither.
    """
    missing = [c for c in _AGGREGATE_KEY if c not in df.columns]
    if missing:
        logger.warning(
            "Skipping dedup of %s: frame lacks colony-key column(s) %s. "
            "Repeated rows may survive; this is preferred to collapsing on a "
            "partial key, which would drop real colonies.",
            context,
            missing,
        )
        return df
    return df.unique(
        subset=list(_AGGREGATE_KEY), keep="last", maintain_order=True
    )


def _quarantine_corrupt_aggregate(agg_path: Path) -> Path | None:
    """Move an unreadable aggregate aside, returning where it went.

    Preserves the bytes for diagnosis instead of overwriting them: a file that
    cannot be read is evidence, and this is the only place that would notice.
    The quarantined name keeps the ``_`` prefix so neither
    :func:`_scan_unchunked_parquets` nor ``discover_measurement_sources`` can
    re-ingest it as a per-image measurement source. Numbered rather than fixed
    so a second corruption cannot clobber the first.
    """
    for index in range(1, 1000):
        candidate = agg_path.with_name(
            f"{agg_path.stem}.corrupt{index}{agg_path.suffix}"
        )
        if not candidate.exists():
            try:
                agg_path.replace(candidate)
            except OSError as exc:
                # A rename within one directory cannot cross devices, but some
                # network/FUSE mounts refuse rename-over regardless. Copy
                # instead; the caller overwrites `agg_path` either way, so a
                # copy preserves the same evidence.
                try:
                    shutil.copy2(agg_path, candidate)
                except OSError as copy_exc:
                    logger.error(
                        "Could not preserve %s (rename: %s; copy: %s). The "
                        "rebuilt aggregate will overwrite it and the "
                        "unreadable bytes will be lost.",
                        agg_path,
                        exc,
                        copy_exc,
                    )
                    return None
            return candidate
    logger.error(
        "Too many quarantined copies beside %s; leaving it in place", agg_path
    )
    return None


def _rebuild_dataset_aggregate(
    agg_path: Path, read_error: Exception
) -> pl.DataFrame | None:
    """Re-derive an unreadable aggregate from the per-image parquets.

    The aggregate is a *cache* of ``results/<ds>/measurements/*.parquet``, so
    the source of truth for rebuilding it sits in the same directory, and the
    rebuild goes through :func:`_read_and_concat` — the same reader that built
    the aggregate originally. Reading those files raw would *not* reproduce
    it: the identity normalization lives in that reader, and two of the three
    columns it fixes up are colony-key columns, so a raw re-read followed by
    :func:`_dedupe_on_colony_key` collapses unrelated colonies rather than
    restoring them.

    Nothing deletes the per-image parquets. Two staged-GPU paths relocate some
    of them — ``reconcile_stage3_publications`` (Stage 3 never completed) and
    ``quarantine_unchanged_restart_parquets`` (stale epoch) — but neither can
    co-occur with chunking: both are staged-GPU-only, and the staged
    strategies never write the ``chunk_state.json`` that gates every entry
    into this module. The two path sets being disjoint is what makes the
    rebuild sound. Note that it is *not* sound in general — a healthy
    aggregate is append-only and never drops a row when a per-image parquet
    goes away, so do not read this as "the aggregate tracks the directory".

    An earlier revision logged "rebuilding from new data" here and then wrote
    only the incoming chunk, destroying every previously aggregated colony
    while chunk state still listed their sources as consumed. Nothing could
    recover them, and final aggregation publishes the master from this file.

    Returns the rebuilt frame, or ``None`` when nothing could be rebuilt, in
    which case the caller writes the incoming chunk alone. That branch is
    unreachable from the chunk path — the sources include the very files that
    produced the incoming chunk, which were read moments earlier — so it is a
    guard, not a designed behaviour.
    """
    logger.error(
        "Cannot read %s (%s); rebuilding it from the per-image parquets",
        agg_path,
        read_error,
    )
    quarantined = _quarantine_corrupt_aggregate(agg_path)
    if quarantined is not None:
        logger.error("Preserved the unreadable file at %s", quarantined)

    sources = [
        path
        for path in sorted(agg_path.parent.glob("*.parquet"))
        if not path.name.startswith(("_", "."))
    ]
    if not sources:
        logger.error(
            "No per-image parquets under %s to rebuild from; the aggregate "
            "will contain only the incoming chunk",
            agg_path.parent,
        )
        return None

    rebuilt = _read_and_concat(sources)
    if rebuilt.unreadable:
        logger.error(
            "Rebuild of %s could not read %d per-image parquet(s): %s",
            agg_path,
            len(rebuilt.unreadable),
            ", ".join(rebuilt.unreadable),
        )
    if rebuilt.frame is None:
        return None

    logger.info(
        "Rebuilt %s from %d per-image parquet(s): %d rows before dedup",
        agg_path,
        len(rebuilt.consumed),
        rebuilt.frame.height,
    )
    return rebuilt.frame


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
    new_df = normalize_measurement_metadata_columns(new_df)
    if agg_path.exists():
        try:
            prev = normalize_measurement_metadata_columns(
                pl.read_parquet(agg_path)
            )
            new_df = pl.concat([prev, new_df], how="diagonal_relaxed")
        except Exception as exc:
            rebuilt = _rebuild_dataset_aggregate(agg_path, exc)
            if rebuilt is not None:
                new_df = pl.concat([rebuilt, new_df], how="diagonal_relaxed")

    new_df = _dedupe_on_colony_key(
        new_df, context=f"aggregate for {dataset_name}"
    )

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
