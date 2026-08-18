"""Pure preparation and diagnostics for CLI metadata joins."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from phenotypic.schema import header_to_module
from phenotypic.sdk_ import (
    is_metadata_header,
    metadata_member_for_header,
    metadata_member_for_label,
    normalize_metadata_columns,
)


# The SDK leaves unknown already-prefixed metadata names unchanged. That makes
# this a stable temporary spelling while bare names are shielded from its
# external-frame normalization rule.
_SHIELDED_COLUMN_PREFIX = "Metadata___phenotypic_non_metadata_column_"


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


def _normalize_selected_metadata_columns(
    frame: pl.DataFrame,
    *,
    preserve: set[str],
) -> pl.DataFrame:
    """Normalize metadata columns while preserving explicitly non-metadata names.

    ``normalize_metadata_columns`` intentionally treats every bare column in an
    external metadata frame as metadata. Measurement frames are mixed-schema,
    however, and an external CSV may also contain raw shared keys such as
    ``plate``. Temporarily shielding those names lets the SDK own all metadata
    canonicalization and duplicate-conflict handling without reclassifying the
    mixed-schema columns.
    """
    if not frame.columns:
        return frame.clone()

    taken = set(frame.columns)
    shield: dict[str, str] = {}
    for index, column in enumerate(frame.columns):
        if column not in preserve:
            continue
        base_candidate = f"{_SHIELDED_COLUMN_PREFIX}{index}"
        candidate = base_candidate
        suffix = 1
        while candidate in taken:
            candidate = f"{base_candidate}_{suffix}"
            suffix += 1
        shield[column] = candidate
        taken.add(candidate)

    protected = frame.rename(shield) if shield else frame.clone()
    normalized = normalize_metadata_columns(protected)
    if not shield:
        return normalized
    return normalized.rename(
        {temporary: original for original, temporary in shield.items()}
    )


def normalize_measurement_metadata_columns(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Return a copy with only metadata-family columns canonicalized.

    Known bare metadata labels, current per-topic headers, future flat headers,
    and unknown already-prefixed metadata are normalized through the SDK.
    Measurement, locator, source-path, and arbitrary custom columns retain their
    names. Duplicate metadata spellings coalesce or raise according to the
    central normalization contract.
    """
    preserve = {
        column
        for column in frame.columns
        if metadata_member_for_header(column) is None
        and metadata_member_for_label(column) is None
        and not is_metadata_header(column)
    }
    return _normalize_selected_metadata_columns(frame, preserve=preserve)


def normalize_external_metadata_columns(
    measurements: pl.DataFrame,
    metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize an external metadata frame without mutating caller state.

    Raw columns already shared with the measurements frame are join keys and
    remain raw unless the metadata registry recognizes them. Non-metadata schema
    headers such as ``Grid_RowNum`` also remain unchanged. Every other bare
    external column is an attribute and receives the live metadata spelling.

    Args:
        measurements: Mixed-schema measurement frame used to identify raw join
            keys that must retain their names.
        metadata: External metadata frame to canonicalize in memory.

    Returns:
        A normalized copy of ``metadata``.
    """
    raw_common = set(measurements.columns) & set(metadata.columns)
    known_schema_headers = set(header_to_module())
    preserve = {
        column
        for column in metadata.columns
        if (
            metadata_member_for_header(column) is None
            and metadata_member_for_label(column) is None
            and not is_metadata_header(column)
            and (column in raw_common or column in known_schema_headers)
        )
    }
    return _normalize_selected_metadata_columns(metadata, preserve=preserve)


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
    normalized_measurements = normalize_measurement_metadata_columns(
        measurements
    )
    normalized_metadata = normalize_external_metadata_columns(
        measurements,
        metadata,
    )
    common = tuple(
        sorted(
            set(normalized_measurements.columns)
            & set(normalized_metadata.columns)
        )
    )
    if not common:
        return PreparedMetadataJoin(
            measurements=normalized_measurements,
            metadata=normalized_metadata,
            analysis=MetadataJoinKeyAnalysis(
                columns=(),
                measurement_row_count=normalized_measurements.height,
                metadata_row_count=normalized_metadata.height,
                matched_measurement_count=0,
                unmatched_measurement_count=normalized_measurements.height,
                unmatched_metadata_count=normalized_metadata.height,
                duplicate_metadata_key_count=0,
            ),
        )

    normalized_measurements = normalized_measurements.with_columns(
        pl.col(column).cast(pl.String) for column in common
    )
    normalized_metadata = normalized_metadata.with_columns(
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
