"""Compatibility adapters for metadata-bearing plotting inputs."""

from __future__ import annotations

import pandas as pd

from phenotypic.sdk_ import (
    ensure_metadata_prefix,
    is_metadata_header,
    metadata_member_for_header,
    metadata_member_for_label,
    normalize_metadata_columns,
)


def normalize_metadata_column_reference(column: str) -> str:
    """Resolve a metadata reference while leaving measurement names untouched."""
    name = str(column)
    if (
        is_metadata_header(name)
        or metadata_member_for_header(name) is not None
        or metadata_member_for_label(name) is not None
    ):
        return ensure_metadata_prefix(name)
    return name


def normalize_metadata_column_references(value: object) -> list[str]:
    """Normalize a plotting column list without accepting malformed inputs."""
    if isinstance(value, str):
        return [normalize_metadata_column_reference(value)]
    if not isinstance(value, (list, tuple)):
        raise ValueError("column references must be a string, list, or tuple")
    if not all(isinstance(column, str) for column in value):
        raise ValueError("every column reference must be a string")
    return [normalize_metadata_column_reference(column) for column in value]


def normalize_measurement_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copied measurement frame with metadata columns normalized only."""
    names = [str(column) for column in frame.columns]
    positions = [
        position
        for position, name in enumerate(names)
        if (
            is_metadata_header(name)
            or metadata_member_for_header(name) is not None
            or metadata_member_for_label(name) is not None
        )
    ]
    if not positions:
        return frame.copy(deep=True)
    metadata = normalize_metadata_columns(frame.iloc[:, positions])
    groups: dict[str, list[int]] = {}
    for position in positions:
        groups.setdefault(ensure_metadata_prefix(names[position]), []).append(position)
    anchors = {
        target: next(
            (position for position in grouped if names[position] == target),
            grouped[0],
        )
        for target, grouped in groups.items()
    }
    normalized_by_anchor = {
        anchor: metadata.iloc[:, index]
        for index, anchor in enumerate(sorted(anchors.values()))
    }
    consumed = {position for grouped in groups.values() for position in grouped}
    columns = [
        normalized_by_anchor[position]
        if position in normalized_by_anchor
        else frame.iloc[:, position].copy(deep=True)
        for position in range(len(names))
        if position not in consumed or position in normalized_by_anchor
    ]
    return pd.concat(columns, axis=1)
