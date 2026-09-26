"""Metadata-join analysis shared by the CLI run preflight and the GUI run console.

Spec ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §9 (F21, F22;
review R1, R38). Before this module the analysis lived only in the GUI
(``_gui/run_console/_request_safety.py:build_metadata_preflight``), bound to
GUI types, so the CLI could not ask whether a ``--metadata`` CSV would join.

The source key frame holds what the CLI's aggregation can know about an image
before measuring it: ``ImageName`` (its stem), ``FileSuffix`` and ``Dataset``.
The production join runs later against the measurement frame, which also
carries measurement headers such as ``Grid_RowNum``; a CSV keyed on those is
legitimate but only partly verifiable here. That is what
:attr:`MetadataJoinAnalysis.unverified_join_columns` reports, and why the
run preflight softens its key findings when it is non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class MetadataJoinAnalysis:
    """How a metadata frame would join onto the scanned source images.

    Attributes:
        join_columns: Columns shared with the source key frame, sorted.
        unverified_join_columns: Qualified metadata columns that only a
            measurement can supply (e.g. ``Grid_RowNum``); the production join
            uses any of them the measurements emit.
        source_count: Scanned images.
        metadata_row_count: Rows in the metadata frame.
        matched_source_count: Images with a metadata row on all join columns.
        unmatched_images: Images with no such row, as ``dataset/path``
            strings; empty when there are no join columns (the join is then
            skipped, not lossy).
        metadata_only_count: Metadata rows that match no image.
        duplicate_key_count: Metadata rows repeating a join key.
    """

    join_columns: tuple[str, ...]
    unverified_join_columns: tuple[str, ...]
    source_count: int
    metadata_row_count: int
    matched_source_count: int
    unmatched_images: tuple[str, ...]
    metadata_only_count: int
    duplicate_key_count: int


def source_join_key_frame(images: Sequence[tuple[str, Path]]) -> "pl.DataFrame":
    """Project a source inventory into the keys CLI aggregation emits.

    Args:
        images: ``(dataset, path)`` pairs, as scanned.

    Returns:
        One row per image: ``IMAGE.IMAGE_NAME``, ``IMAGE.SUFFIX``,
        ``EXPERIMENT.DATASET``.
    """
    import polars as pl

    from phenotypic.schema import EXPERIMENT, IMAGE
    from phenotypic.sdk_ import source_image_stem, source_image_suffix

    return pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [source_image_stem(image) for _, image in images],
            str(IMAGE.SUFFIX): [source_image_suffix(image) for _, image in images],
            str(EXPERIMENT.DATASET): [dataset for dataset, _ in images],
        }
    )


def unverified_measurement_join_columns(
    metadata_columns: Sequence[str],
    source_columns: Sequence[str],
) -> tuple[str, ...]:
    """Non-source columns that may remain production join keys.

    Every qualified name (``Prefix_Label``) is conservative join-key territory
    because a built-in or external operation may emit that exact column. This
    deliberately does not depend on the registered schema. Unqualified bare
    labels are excluded because, when they are not common, ``join_metadata``
    prefixes them as metadata attributes.

    The ``"_" in column`` test asks whether a name is *qualified*; it is not a
    ``Metadata_`` prefix test, and it does not decide metadata ownership --
    ``metadata_member_for_header`` does. Do not "fix" it into a prefix check
    (``CLAUDE.md``: metadata semantics by schema ownership).
    """
    from phenotypic.sdk_ import metadata_member_for_header

    source = set(source_columns)
    return tuple(
        sorted(
            column
            for column in metadata_columns
            if column not in source
            and "_" in column
            and metadata_member_for_header(column) is None
        )
    )


def analyze_metadata_join(
    images: Sequence[tuple[str, Path]],
    metadata: "pl.DataFrame",
) -> MetadataJoinAnalysis:
    """Analyze how *metadata* would join onto *images*.

    Args:
        images: ``(dataset, path)`` pairs, as scanned.
        metadata: The metadata frame, parsed (and, for the GUI, already
            normalized by its own input rules).

    Returns:
        The analysis.

    Raises:
        ValueError: Header normalization found conflicting legacy and
            canonical aliases; the production join would fail the same way.
    """
    from ._metadata_join import prepare_metadata_join_keys

    source = source_join_key_frame(images)
    prepared = prepare_metadata_join_keys(source, metadata)
    analysis = prepared.analysis
    unverified = unverified_measurement_join_columns(metadata.columns, source.columns)
    unmatched: tuple[str, ...] = ()
    if analysis.columns and analysis.unmatched_measurement_count:
        keys = list(analysis.columns)
        missing = (
            prepared.measurements.with_row_index("__row")
            .join(prepared.metadata.select(keys).unique(), on=keys, how="anti")
            .get_column("__row")
            .to_list()
        )
        unmatched = tuple(f"{images[i][0]}/{images[i][1]}" for i in missing)
    # With no join columns the production join is skipped and no image is
    # dropped, so nothing is "unmatched"; the key finding reports that case.
    return MetadataJoinAnalysis(
        join_columns=analysis.columns,
        unverified_join_columns=unverified,
        source_count=len(images),
        metadata_row_count=metadata.height,
        matched_source_count=analysis.matched_measurement_count,
        unmatched_images=unmatched,
        metadata_only_count=analysis.unmatched_metadata_count,
        duplicate_key_count=analysis.duplicate_metadata_key_count,
    )
