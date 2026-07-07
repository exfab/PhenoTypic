"""Shared discovery helpers for CLI measurement parquet sources."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from phenotypic.schema import METADATA
from phenotypic.sdk_ import (
    DATASET_AGGREGATED_PARQUET,
    DIR_MEASUREMENTS,
    DIR_RESULTS,
)


@dataclass(frozen=True)
class MeasurementSource:
    """One measurement parquet source and its dataset label."""

    path: Path
    dataset: str


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
        is preferred over individual non-internal parquet files.
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

        aggregated = meas_dir / DATASET_AGGREGATED_PARQUET
        if aggregated.exists():
            sources.append(MeasurementSource(aggregated, dataset_name))
            continue

        sources.extend(
            MeasurementSource(path, dataset_name)
            for path in sorted(meas_dir.glob("*.parquet"))
            if not path.name.startswith("_")
        )
    return sources


def measurement_sources_by_path(
    sources: Iterable[MeasurementSource],
) -> dict[Path, str]:
    """Return the path-to-dataset mapping used by parquet aggregation."""
    return {source.path: source.dataset for source in sources}


def add_metadata_image_name_from_filename(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive ``Metadata_ImageName`` from ``filename`` and drop ``filename``.

    Args:
        frame: Aggregated measurement frame. ``filename`` is produced by
            ``aggregate_parquet_files(..., keep_filename=True)``.

    Returns:
        Frame with no ``filename`` column. If ``Metadata_ImageName`` already
        exists, it is preserved.
    """
    image_name_col = str(METADATA.IMAGE_NAME)
    if image_name_col not in frame.columns and "filename" in frame.columns:
        frame = frame.with_columns(
            pl.col("filename")
            .str.extract(r"([^/\\]+)\.[^.]+$", 1)
            .alias(image_name_col)
        )
    if "filename" in frame.columns:
        frame = frame.drop("filename")
    return frame
