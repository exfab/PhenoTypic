"""Shared discovery helpers for CLI measurement parquet sources."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import polars as pl

from phenotypic.schema import METADATA
from phenotypic.sdk_ import (
    DATASET_AGGREGATED_PARQUET,
    DIR_MEASUREMENTS,
    DIR_RESULTS,
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


def _aggregate_needs_image_name_recovery(path: Path) -> bool:
    """Return whether an aggregate cannot supply valid per-image identities."""
    image_name_col = str(METADATA.IMAGE_NAME)
    try:
        aggregate = pl.scan_parquet(path)
        if image_name_col not in aggregate.collect_schema():
            return True
        uuid_found = (
            aggregate.select(
                pl.col(image_name_col)
                .cast(pl.String, strict=False)
                .str.contains(_UUID_PATTERN)
                .any()
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
        return False
    return bool(uuid_found)


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
            MeasurementSource(path, dataset_name)
            for path in individual_paths
        )
    return sources


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
    image_name_col = str(METADATA.IMAGE_NAME)
    if "filename" in frame.columns:
        filename_stem = pl.col("filename").str.extract(
            r"([^/\\]+)\.[^.]+$", 1
        )
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
