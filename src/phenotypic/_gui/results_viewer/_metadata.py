"""Read-only metadata compatibility helpers for Results Viewer inputs.

Viewer frames mix metadata, measurements, locators, and arbitrary post-operation
columns.  The public SDK normalizer intentionally treats bare columns in an
external *metadata* frame as metadata, so this module shields unrelated columns
while delegating metadata spelling, duplicate coalescing, and conflict detection
to that single source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, overload

from phenotypic.sdk_ import (
    is_metadata_header,
    metadata_member_for_header,
    metadata_member_for_label,
    normalize_metadata_columns,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

_FrameT = TypeVar("_FrameT", "pd.DataFrame", "pl.DataFrame")
_SHIELD_PREFIX = "Metadata___phenotypic_viewer_nonmetadata_"


def normalize_metadata_reference(column: str) -> str:
    """Return the canonical spelling for one metadata column reference.

    Nonmetadata references are returned unchanged.  Known bare labels are
    accepted for compatibility with old filter and QC recipe payloads.
    """
    value = str(column)
    member = metadata_member_for_header(value)
    if member is None:
        member = metadata_member_for_label(value)
    return member.value if member is not None else value


def normalize_metadata_references(columns: Sequence[str]) -> list[str]:
    """Canonicalize and stable-deduplicate metadata column references."""
    result: list[str] = []
    for column in columns:
        normalized = normalize_metadata_reference(column)
        if normalized not in result:
            result.append(normalized)
    return result


@overload
def normalize_viewer_frame(frame: "pd.DataFrame") -> "pd.DataFrame": ...


@overload
def normalize_viewer_frame(frame: "pl.DataFrame") -> "pl.DataFrame": ...


def normalize_viewer_frame(frame: _FrameT) -> _FrameT:
    """Return a normalized copy of a mixed measurement/metadata frame.

    Metadata columns use the canonical flat namespace.  Measurement, locator,
    QC, and arbitrary post-operation columns retain their original names.
    Duplicate metadata spellings coalesce only when lossless; conflicts raise
    the SDK normalizer's descriptive :class:`ValueError`.
    """
    import pandas as pd
    import polars as pl

    columns = [str(column) for column in frame.columns]
    preserved = {
        column
        for column in columns
        if metadata_member_for_header(column) is None
        and metadata_member_for_label(column) is None
        and not is_metadata_header(column)
    }
    taken = set(columns)
    target_columns: list[str] = []
    reverse: dict[str, str] = {}
    for position, column in enumerate(columns):
        if column not in preserved:
            target_columns.append(column)
            continue
        candidate = f"{_SHIELD_PREFIX}{position}"
        suffix = 1
        while candidate in taken:
            candidate = f"{_SHIELD_PREFIX}{position}_{suffix}"
            suffix += 1
        target_columns.append(candidate)
        reverse[candidate] = column
        taken.add(candidate)

    if isinstance(frame, pd.DataFrame):
        protected = frame.copy(deep=True)
        protected.columns = target_columns
        normalized = normalize_metadata_columns(protected)
        return normalized.rename(columns=reverse)  # type: ignore[return-value]
    if isinstance(frame, pl.DataFrame):
        shield = {
            original: temporary
            for original, temporary in zip(columns, target_columns, strict=True)
            if original != temporary
        }
        protected = frame.rename(shield)
        normalized = normalize_metadata_columns(protected)
        return normalized.rename(reverse)  # type: ignore[return-value]
    raise TypeError(
        "normalize_viewer_frame requires a pandas or Polars DataFrame; "
        f"got {type(frame).__name__}"
    )


def normalize_column_value_sets(
    value_sets: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    """Return value sets keyed by canonical metadata references.

    When old and canonical keys coexist, values are unioned rather than one
    entry silently overwriting the other.
    """
    normalized: dict[str, list[str]] = {}
    for column, values in value_sets.items():
        target = normalize_metadata_reference(column)
        bucket = normalized.setdefault(target, [])
        bucket.extend(value for value in values if value not in bucket)
    return normalized


__all__ = [
    "normalize_column_value_sets",
    "normalize_metadata_reference",
    "normalize_metadata_references",
    "normalize_viewer_frame",
]
