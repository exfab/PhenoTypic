"""Pure preparation and diagnostics for CLI metadata joins."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class MetadataJoinKeyAnalysis:
    """Key-grain facts computed with the production metadata join semantics."""

    columns: tuple[str, ...]
    measurement_row_count: int
    metadata_row_count: int
    matched_measurement_count: int
    unmatched_measurement_count: int
    unmatched_metadata_count: int
    duplicate_metadata_key_count: int


@dataclass(frozen=True)
class PreparedMetadataJoin:
    """String-normalized frames plus their common-key analysis."""

    measurements: pl.DataFrame
    metadata: pl.DataFrame
    analysis: MetadataJoinKeyAnalysis


def prepare_metadata_join_keys(
    measurements: pl.DataFrame,
    metadata: pl.DataFrame,
) -> PreparedMetadataJoin:
    """Normalize and analyze the exact common keys used by ``join_metadata``.

    Common columns are sorted for deterministic behavior, then cast to Polars
    ``String`` on both sides. Anti joins use the same default null semantics as
    the production join.

    Args:
        measurements: Measurement rows, or a source-derived key projection for
            a preflight.
        metadata: Parsed external metadata rows.

    Returns:
        Normalized frames and key-grain match, orphan, and duplicate counts.
    """
    common = tuple(sorted(set(measurements.columns) & set(metadata.columns)))
    if not common:
        return PreparedMetadataJoin(
            measurements=measurements,
            metadata=metadata,
            analysis=MetadataJoinKeyAnalysis(
                columns=(),
                measurement_row_count=measurements.height,
                metadata_row_count=metadata.height,
                matched_measurement_count=0,
                unmatched_measurement_count=measurements.height,
                unmatched_metadata_count=metadata.height,
                duplicate_metadata_key_count=0,
            ),
        )

    normalized_measurements = measurements.with_columns(
        pl.col(column).cast(pl.String) for column in common
    )
    normalized_metadata = metadata.with_columns(
        pl.col(column).cast(pl.String) for column in common
    )
    metadata_keys = normalized_metadata.select(common).unique()
    measurement_keys = normalized_measurements.select(common).unique()
    unmatched_measurements = normalized_measurements.join(
        metadata_keys,
        on=common,
        how="anti",
    ).height
    unmatched_metadata = normalized_metadata.join(
        measurement_keys,
        on=common,
        how="anti",
    ).height
    unique_metadata_keys = normalized_metadata.n_unique(subset=common)
    return PreparedMetadataJoin(
        measurements=normalized_measurements,
        metadata=normalized_metadata,
        analysis=MetadataJoinKeyAnalysis(
            columns=common,
            measurement_row_count=normalized_measurements.height,
            metadata_row_count=normalized_metadata.height,
            matched_measurement_count=(
                normalized_measurements.height - unmatched_measurements
            ),
            unmatched_measurement_count=unmatched_measurements,
            unmatched_metadata_count=unmatched_metadata,
            duplicate_metadata_key_count=(
                normalized_metadata.height - unique_metadata_keys
            ),
        ),
    )
