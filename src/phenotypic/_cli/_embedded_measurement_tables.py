"""Prepare authoritative per-image measurement tables for storage."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl

from phenotypic.sdk_ import (
    PreparedEmbeddedMeasurementTable,
    PreparedImageTables,
)

from ._metadata_join import (
    PreparedMetadataJoin,
    normalize_measurement_metadata_columns,
    prepare_metadata_join_keys,
)

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


@dataclass(frozen=True)
class _NormalizedTableInputs:
    """One normalization of the two frames, shared by both producers.

    ``prepare_image_tables`` and the joined producer it replaces must agree
    exactly on the baseline, the digest, and the common keys -- the migrator
    compares a store's bytes against a freshly prepared table, so any drift
    between the two would read as a corrupt store rather than as a code
    difference. Computing them once is what makes that agreement structural.
    """

    baseline: pd.DataFrame
    measurement_columns: tuple[str, ...]
    digest: str
    prepared: PreparedMetadataJoin | None
    common: tuple[str, ...]


def _normalize_table_inputs(
    measurements: pd.DataFrame,
    metadata_csv: Path | None,
) -> _NormalizedTableInputs:
    """Normalize the baseline and, when supplied, the metadata snapshot."""
    baseline = normalize_measurement_metadata_columns(
        pl.from_pandas(measurements)
    ).to_pandas()
    measurement_columns = tuple(str(column) for column in baseline.columns)
    if metadata_csv is None:
        return _NormalizedTableInputs(
            baseline=baseline,
            measurement_columns=measurement_columns,
            digest="",
            prepared=None,
            common=(),
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
    elif prepared.analysis.duplicate_metadata_key_count:
        logger.warning(
            "Metadata CSV has duplicate keys on columns %s — preserving "
            "duplicate-key fan-out",
            list(common),
        )
    return _NormalizedTableInputs(
        baseline=baseline,
        measurement_columns=measurement_columns,
        digest=digest,
        prepared=prepared,
        common=common,
    )


def prepare_image_tables(
    measurements: pd.DataFrame,
    metadata_csv: Path | None,
) -> PreparedImageTables:
    """Split one image's payload into measurements and its own metadata rows.

    Spec §7.1-7.2. This is **subtraction, not invention**: the boundary is
    already named by ``measurement_columns``, which the joined producer
    computes from the baseline *before* joining. The measurements table is
    exactly that projection; the metadata table is the run's metadata
    snapshot projected onto the join keys this image carries.

    Duplicate metadata keys keep their fan-out, because the metadata rows are
    selected by a semi join rather than deduplicated -- losing it silently
    changes row counts in the finalized mirror.

    Args:
        measurements: One image's baseline per-object measurements.
        metadata_csv: The run's ``deliverables/metadata.csv`` snapshot, or
            ``None`` when the run was given no ``--metadata``.

    Returns:
        The split payload. ``metadata`` is ``None`` unless a snapshot was
        supplied *and* it shares at least one column with the measurements.
    """
    inputs = _normalize_table_inputs(measurements, metadata_csv)
    if inputs.prepared is None:
        return PreparedImageTables(
            measurements=inputs.baseline,
            metadata=None,
            measurement_columns=inputs.measurement_columns,
            join_status="not_requested",
            join_keys=(),
            metadata_snapshot_sha256="",
        )
    if not inputs.common:
        return PreparedImageTables(
            measurements=inputs.baseline,
            metadata=None,
            measurement_columns=inputs.measurement_columns,
            join_status="no_common_keys",
            join_keys=(),
            metadata_snapshot_sha256=inputs.digest,
        )

    common = list(inputs.common)
    # A SEMI join, not a distinct-key inner join: it keeps every metadata row
    # whose key this image carries, in the snapshot's own order, which is what
    # preserves duplicate-key fan-out. `maintain_order="left"` makes that
    # order a guarantee rather than an implementation detail, so two identical
    # runs write byte-identical metadata tables.
    projected = inputs.prepared.metadata.join(
        inputs.prepared.measurements.select(common).unique(),
        on=common,
        how="semi",
        maintain_order="left",
    ).to_pandas()
    projected = _restore_join_key_dtypes(
        projected, inputs.baseline, inputs.common
    )
    return PreparedImageTables(
        measurements=inputs.baseline,
        metadata=projected,
        measurement_columns=inputs.measurement_columns,
        join_status="joined",
        join_keys=inputs.common,
        metadata_snapshot_sha256=inputs.digest,
    )


def prepare_embedded_measurement_table(
    measurements: pd.DataFrame,
    metadata_csv: Path | None,
) -> PreparedEmbeddedMeasurementTable:
    """Right-join effective metadata onto one image's baseline measurements.

    Metadata is the left frame and measurements are the right frame. This
    preserves every measured row, excludes metadata-only rows, and retains
    duplicate metadata-key fan-out.

    **Superseded by** :func:`prepare_image_tables` on every forward path.
    It survives only for the consumers that still read and rewrite
    *pre-inversion* stores byte-exactly -- ``--mode migrate``
    (``_cli_migrate.py``, ``_cli_migrate_image.py``) and ``--mode recompile``
    (``_cli_recompile_tables.py``). **Retire it** with the last of those call
    sites; nothing else may grow a new one.
    """
    inputs = _normalize_table_inputs(measurements, metadata_csv)
    if inputs.prepared is None:
        return PreparedEmbeddedMeasurementTable(
            frame=inputs.baseline,
            measurement_columns=inputs.measurement_columns,
            join_status="not_requested",
            join_keys=(),
            metadata_snapshot_sha256="",
        )
    if not inputs.common:
        return PreparedEmbeddedMeasurementTable(
            frame=inputs.baseline,
            measurement_columns=inputs.measurement_columns,
            join_status="no_common_keys",
            join_keys=(),
            metadata_snapshot_sha256=inputs.digest,
        )

    joined = inputs.prepared.metadata.join(
        inputs.prepared.measurements,
        on=list(inputs.common),
        how="right",
        maintain_order="right",
    ).to_pandas()
    joined = _restore_join_key_dtypes(joined, inputs.baseline, inputs.common)
    return PreparedEmbeddedMeasurementTable(
        frame=joined,
        measurement_columns=inputs.measurement_columns,
        join_status="joined",
        join_keys=inputs.common,
        metadata_snapshot_sha256=inputs.digest,
    )


def embedded_measurement_table_matches(
    store_path: Path,
    prepared: PreparedEmbeddedMeasurementTable,
) -> bool:
    """Return whether the embedded Arrow table exactly equals *prepared*.

    Equality includes column order and types, schema metadata, row order,
    values, null placement, and row count. The latter is what makes duplicate
    metadata-key fan-out part of reclaim authority rather than an incidental
    property of a readable Parquet payload.
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH

    try:
        expected = pa.Table.from_pandas(
            prepared.frame, preserve_index=False
        ).replace_schema_metadata(prepared.parquet_metadata())
        actual = pq.read_table(
            Path(store_path) / MEASUREMENT_TABLE_RELATIVE_PATH
        )
    except Exception:  # noqa: BLE001 - inability to compare refuses deletion
        return False
    return actual.equals(expected, check_metadata=True)
