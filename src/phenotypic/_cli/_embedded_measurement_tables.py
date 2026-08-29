"""Prepare authoritative per-image measurement tables for storage."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import polars as pl

from phenotypic.sdk_ import PreparedEmbeddedMeasurementTable

from ._metadata_join import prepare_metadata_join_keys

logger = logging.getLogger(__name__)


def _restore_join_key_dtypes(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    join_keys: tuple[str, ...],
) -> pd.DataFrame:
    """Restore measurement-side join-key dtypes after string-safe matching."""
    restored = frame.copy()
    for column in join_keys:
        dtype = baseline[column].dtype
        try:
            restored[column] = restored[column].astype(dtype)
        except (TypeError, ValueError):
            logger.warning(
                "Joined key %s could not be restored to measurement dtype %s",
                column,
                dtype,
            )
    return restored


def prepare_embedded_measurement_table(
    measurements: pd.DataFrame,
    metadata_csv: Path | None,
) -> PreparedEmbeddedMeasurementTable:
    """Right-join effective metadata onto one image's baseline measurements.

    Metadata is the left frame and measurements are the right frame. This
    preserves every measured row, excludes metadata-only rows, and retains
    duplicate metadata-key fan-out.
    """
    baseline = measurements.copy()
    measurement_columns = tuple(str(column) for column in baseline.columns)
    if metadata_csv is None:
        return PreparedEmbeddedMeasurementTable(
            frame=baseline,
            measurement_columns=measurement_columns,
            join_status="not_requested",
            join_keys=(),
            metadata_snapshot_sha256="",
        )

    metadata_csv = Path(metadata_csv)
    digest = hashlib.sha256(metadata_csv.read_bytes()).hexdigest()
    metadata = pl.read_csv(metadata_csv)
    prepared = prepare_metadata_join_keys(pl.from_pandas(baseline), metadata)
    common = prepared.analysis.columns
    if not common:
        logger.warning(
            "Metadata CSV has no columns in common with measurements — "
            "embedding unchanged measurements"
        )
        return PreparedEmbeddedMeasurementTable(
            frame=baseline,
            measurement_columns=measurement_columns,
            join_status="no_common_keys",
            join_keys=(),
            metadata_snapshot_sha256=digest,
        )

    if prepared.analysis.duplicate_metadata_key_count:
        logger.warning(
            "Metadata CSV has duplicate keys on columns %s — preserving "
            "duplicate-key fan-out",
            list(common),
        )

    joined = prepared.metadata.join(
        prepared.measurements,
        on=list(common),
        how="right",
        maintain_order="right",
    ).to_pandas()
    joined = _restore_join_key_dtypes(joined, baseline, common)
    return PreparedEmbeddedMeasurementTable(
        frame=joined,
        measurement_columns=measurement_columns,
        join_status="joined",
        join_keys=common,
        metadata_snapshot_sha256=digest,
    )
