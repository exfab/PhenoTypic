"""Shared discovery helpers for CLI measurement parquet sources."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from phenotypic.schema import IMAGE
from phenotypic.sdk_ import metadata_member_for_header
from phenotypic.sdk_ import (
    DATASET_AGGREGATED_PARQUET,
    ChunkStateKey,
    DIR_MEASUREMENTS,
    DIR_RESULTS,
    chunk_state_path,
)

logger = logging.getLogger(__name__)

_UUID_PATTERN = (
    r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_DATASET_AGGREGATED_STEM = Path(DATASET_AGGREGATED_PARQUET).stem


@dataclass(frozen=True)
class MeasurementSource:
    """One measurement parquet source and its dataset label."""

    path: Path
    dataset: str


def _image_name_column(columns: Iterable[str]) -> str | None:
    """Return the column semantically owned by ``IMAGE.IMAGE_NAME``."""
    return next(
        (
            column
            for column in columns
            if metadata_member_for_header(column) is IMAGE.IMAGE_NAME
        ),
        None,
    )


def _aggregate_needs_image_name_recovery(path: Path) -> bool:
    """Return whether an aggregate cannot supply valid per-image identities."""
    try:
        aggregate = pl.scan_parquet(path)
        schema = aggregate.collect_schema()
        image_name_col = _image_name_column(schema.names())
        if image_name_col is None:
            return True
        if schema[image_name_col] != pl.String:
            return True
        invalid_identity_found = (
            aggregate.select(
                (
                    pl.col(image_name_col).is_null()
                    | (
                        pl.col(image_name_col)
                        .str.strip_chars()
                        .eq("")
                        .fill_null(True)
                    )
                    | pl.col(image_name_col)
                    .str.contains(_UUID_PATTERN)
                    .fill_null(True)
                ).any()
            )
            .collect()
            .item()
        )
    except Exception:
        logger.debug(
            "Could not inspect aggregate image identities in %s",
            path,
            exc_info=True,
        )
        # Recompile may combine this aggregate with individual Parquets.  If
        # inspection fails, membership cannot be proved, so conservatively
        # recover from the individual files instead of risking duplicate rows.
        return True
    return bool(invalid_identity_found)


def _aggregate_image_names(path: Path) -> set[str] | None:
    """Return canonical image identities represented by one aggregate."""
    try:
        aggregate = pl.scan_parquet(path)
        image_name_col = _image_name_column(
            aggregate.collect_schema().names()
        )
        if image_name_col is None:
            return None
        values = (
            aggregate.select(
                pl.col(image_name_col).cast(pl.String, strict=False).unique()
            )
            .collect()
            .get_column(image_name_col)
            .drop_nulls()
            .to_list()
        )
    except Exception:
        logger.warning(
            "Could not prove aggregate image membership in %s",
            path,
            exc_info=True,
        )
        return None
    return {str(value) for value in values}


def discover_measurement_sources(
    output_dir: Path,
    dataset_names: Iterable[str] | None = None,
) -> list[MeasurementSource]:
    """Return deterministic measurement parquet sources under an output root.

    Args:
        output_dir: CLI output directory containing ``results/``.
        dataset_names: Optional dataset names to scan. When omitted, dataset
            directories are discovered from ``results/``.

    Returns:
        Ordered measurement sources. Per dataset, ``_dataset_aggregated.parquet``
        is preferred over individual non-internal parquet files unless its
        image identities need recovery from those individual filenames.
    """
    results_dir = output_dir / DIR_RESULTS
    if not results_dir.is_dir():
        return []

    names = (
        sorted(d.name for d in results_dir.iterdir() if d.is_dir())
        if dataset_names is None
        else list(dataset_names)
    )

    sources: list[MeasurementSource] = []
    for dataset_name in names:
        meas_dir = results_dir / dataset_name / DIR_MEASUREMENTS
        if not meas_dir.is_dir():
            continue

        individual_paths = [
            path
            for path in sorted(meas_dir.glob("*.parquet"))
            # `_` skips `_dataset_aggregated.parquet`; `.` skips macOS
            # AppleDouble sidecars, which are binary and would force
            # `aggregate_parquet_files` off its fast multi-path read onto the
            # per-file fallback for the whole dataset.
            if not path.name.startswith(("_", "."))
        ]
        aggregated = meas_dir / DATASET_AGGREGATED_PARQUET
        aggregate_needs_recovery = (
            aggregated.exists()
            and _aggregate_needs_image_name_recovery(aggregated)
        )
        if aggregated.exists():
            if aggregate_needs_recovery and individual_paths:
                logger.warning(
                    "Aggregate %s has missing or UUID image identities; "
                    "using individual measurement parquets for recovery",
                    aggregated,
                )
            else:
                if aggregate_needs_recovery:
                    logger.warning(
                        "Aggregate %s has missing or UUID image identities, "
                        "but no individual measurement parquets remain for "
                        "recovery",
                        aggregated,
                    )
                sources.append(MeasurementSource(aggregated, dataset_name))
                continue

        sources.extend(
            MeasurementSource(path, dataset_name) for path in individual_paths
        )
    return sources


def discover_recompile_measurement_sources(
    output_dir: Path,
    dataset_names: Iterable[str] | None = None,
) -> list[MeasurementSource]:
    """Return complete recompile sources without changing chunk state.

    A checkpointed run can have both a dataset aggregate and a trailing group
    of per-image Parquets that has not yet been recorded in chunk state.  The
    ordinary source resolver intentionally prefers the aggregate, but an
    asynchronous recompile cannot flush the trailing group before metadata
    migration finishes.  This resolver therefore combines the existing
    aggregate with only the individual files absent from the immutable chunk
    state snapshot.

    If chunk state is unreadable, individual files are preferred over the
    aggregate when they exist.  That avoids double-counting rows whose
    membership can no longer be established.

    Args:
        output_dir: CLI output directory containing ``results/``.
        dataset_names: Optional dataset names to scan.

    Returns:
        Deterministic aggregate-plus-trailing sources for recompile.
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / DIR_RESULTS
    if not results_dir.is_dir():
        return []

    names = (
        sorted(d.name for d in results_dir.iterdir() if d.is_dir())
        if dataset_names is None
        else list(dataset_names)
    )
    has_chunk_state, chunked_files = _read_chunked_file_keys(output_dir)

    sources: list[MeasurementSource] = []
    for dataset_name in names:
        meas_dir = results_dir / dataset_name / DIR_MEASUREMENTS
        if not meas_dir.is_dir():
            continue
        individual_paths = [
            path
            for path in sorted(meas_dir.glob("*.parquet"))
            if not path.name.startswith(("_", "."))
        ]
        aggregate = meas_dir / DATASET_AGGREGATED_PARQUET
        aggregate_needs_recovery = (
            aggregate.exists()
            and _aggregate_needs_image_name_recovery(aggregate)
        )

        if aggregate.exists() and not aggregate_needs_recovery:
            if has_chunk_state and chunked_files is None and individual_paths:
                logger.warning(
                    "Chunk state is unreadable; using individual measurement "
                    "Parquets instead of %s to avoid duplicate rows",
                    aggregate,
                )
                sources.extend(
                    MeasurementSource(path, dataset_name)
                    for path in individual_paths
                )
                continue
            sources.append(MeasurementSource(aggregate, dataset_name))
            if has_chunk_state and chunked_files is not None:
                aggregate_names = _aggregate_image_names(aggregate)
                if aggregate_names is None and individual_paths:
                    sources.pop()
                    sources.extend(
                        MeasurementSource(path, dataset_name)
                        for path in individual_paths
                    )
                    continue
                known_aggregate_names = aggregate_names or set()
                sources.extend(
                    MeasurementSource(path, dataset_name)
                    for path in individual_paths
                    if _chunk_state_key(path) not in chunked_files
                    and path.stem not in known_aggregate_names
                )
            continue

        if aggregate_needs_recovery:
            logger.warning(
                "Aggregate %s has missing or UUID image identities; using "
                "individual measurement parquets for recovery",
                aggregate,
            )
            if not individual_paths:
                logger.warning(
                    "No individual measurement parquets remain; retaining "
                    "%s as the only recompile source",
                    aggregate,
                )
                sources.append(MeasurementSource(aggregate, dataset_name))
                continue
        sources.extend(
            MeasurementSource(path, dataset_name) for path in individual_paths
        )
    return sources


def _read_chunked_file_keys(
    output_dir: Path,
) -> tuple[bool, set[str] | None]:
    """Read chunk membership without creating or updating any artifact.

    The first item records whether chunk state exists. The second is its
    membership, or ``None`` when an existing state file could not be trusted.
    """
    state_path = chunk_state_path(output_dir)
    if not state_path.is_file():
        return False, set()
    try:
        import json

        payload: Any = json.loads(state_path.read_text(encoding="utf-8"))
        raw_keys = payload.get(ChunkStateKey.CHUNKED_FILES, [])
        if not isinstance(raw_keys, list) or not all(
            isinstance(key, str) for key in raw_keys
        ):
            raise ValueError("chunked_files must be a list of strings")
    except (OSError, ValueError, TypeError):
        logger.warning("Could not read chunk state %s", state_path, exc_info=True)
        return True, None
    return True, set(raw_keys)


def _chunk_state_key(parquet_path: Path) -> str:
    """Return the stable ``dataset/file.parquet`` chunk-state key."""
    return f"{parquet_path.parent.parent.name}/{parquet_path.name}"


def measurement_sources_by_path(
    sources: Iterable[MeasurementSource],
) -> dict[Path, str]:
    """Return the path-to-dataset mapping used by parquet aggregation."""
    return {source.path: source.dataset for source in sources}


def add_metadata_image_name_from_filename(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive or repair ``Metadata_ImageName`` from ``filename``.

    Args:
        frame: Aggregated measurement frame. ``filename`` is produced by
            ``aggregate_parquet_files(..., keep_filename=True)``.

    Returns:
        Frame with no ``filename`` column. Existing non-UUID image names are
        preserved; UUID-shaped values from pre-fix staged HDF reloads are
        replaced with an individual parquet's filename stem.
    """
    from ._metadata_join import normalize_measurement_metadata_columns

    frame = normalize_measurement_metadata_columns(frame)
    image_name_col = str(IMAGE.IMAGE_NAME)
    if "filename" in frame.columns:
        filename_stem = pl.col("filename").str.extract(r"([^/\\]+)\.[^.]+$", 1)
        is_individual_source = filename_stem != _DATASET_AGGREGATED_STEM
        if image_name_col not in frame.columns:
            frame = frame.with_columns(
                pl.when(is_individual_source)
                .then(filename_stem)
                .otherwise(None)
                .alias(image_name_col)
            )
        else:
            frame = frame.with_columns(
                pl.when(
                    is_individual_source
                    & pl.col(image_name_col)
                    .cast(pl.String, strict=False)
                    .str.contains(_UUID_PATTERN)
                )
                .then(filename_stem)
                .otherwise(pl.col(image_name_col))
                .alias(image_name_col)
            )
    if "filename" in frame.columns:
        frame = frame.drop("filename")
    return frame
