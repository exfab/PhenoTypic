from __future__ import annotations

import pandas as pd

# Schema-aware metadata prefixing lives in ``phenotypic.sdk_`` so the post
# package does not pull in the core image stack. Re-exported here for the post
# ops (and any legacy callers) that import it from this module.
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from phenotypic.sdk_ import (
    ensure_metadata_prefix,
    metadata_member_for_header,
    metadata_member_for_label,
    normalize_metadata_columns,
)

if TYPE_CHECKING:
    from phenotypic.schema import MetadataInfo

__all__ = [
    "affix_preserving_na",
    "ensure_metadata_prefix",
    "resolve_metadata_column",
]


def resolve_metadata_column(columns: Iterable[object], requested: str) -> str:
    """Return the existing column matching a metadata request without renaming it.

    Post operations retain the live output spelling selected by
    :func:`ensure_metadata_prefix`, but their input frames may come from a
    historical per-topic export or a future flat-namespace export. Known schema
    members bridge those equivalent spellings by identity; generic metadata
    remains an exact-name match.

    Args:
        columns: Existing frame column names.
        requested: Bare label, live header, or compatible metadata header.

    Returns:
        The matching existing column name.

    Raises:
        KeyError: If no existing column names the requested metadata field.
    """
    available = [str(column) for column in columns]
    request = str(requested)
    if request in available:
        return request

    normalized = ensure_metadata_prefix(request)
    if normalized in available:
        return normalized

    member = metadata_member_for_header(request)
    if member is None:
        member = metadata_member_for_label(request)
    if member is not None:
        for column in available:
            if metadata_member_for_header(column) is member:
                return column

    raise KeyError(request)


def coalesce_metadata_aliases(
    frame: pd.DataFrame, requested: Iterable[str]
) -> pd.DataFrame:
    """Return a copy with requested known metadata aliases coalesced.

    Only aliases of requested, schema-known metadata fields are considered. This
    deliberately leaves unrelated, nonmetadata columns untouched, including
    their order. The SDK normalizer supplies dtype validation plus lossless
    equal/complementary-null coalescing before aliases are dropped.

    Args:
        frame: Source measurement frame.
        requested: Metadata names whose aliases should be reconciled.

    Returns:
        A frame copy with each requested alias set represented once.

    Raises:
        ValueError: If equivalent aliases have conflicting values.
    """
    result = frame.copy()
    members: list[MetadataInfo] = []
    for name in requested:
        member = metadata_member_for_header(name) or metadata_member_for_label(name)
        if member is not None and member not in members:
            members.append(member)

    for member in members:
        aliases = [
            column
            for column in result.columns
            if metadata_member_for_header(str(column)) is member
        ]
        if not aliases:
            continue

        canonical_alias = next(
            (alias for alias in aliases if str(alias) == str(member.value)), aliases[0]
        )
        canonical_position = cast(int, result.columns.get_loc(canonical_alias))
        aliases_before_canonical = sum(
            cast(int, result.columns.get_loc(alias)) < canonical_position
            for alias in aliases
        )
        # SDK normalization emits the live current spelling. Reinsert it where
        # that spelling originally appeared after accounting for any alias
        # columns removed ahead of it. If it was absent, ``aliases[0]`` is the
        # stable first-alias fallback.
        position = canonical_position - aliases_before_canonical
        normalized = normalize_metadata_columns(result.loc[:, aliases])
        header = normalized.columns[0]
        result = result.drop(columns=aliases)
        result.insert(position, header, normalized.iloc[:, 0])
    return result


def affix_preserving_na(
    col: "pd.Series", *, prefix: str = "", suffix: str = ""
) -> "pd.Series":
    """Wrap each value of *col* in *prefix*/*suffix*, leaving NA cells as NA.

    Shared by :class:`~phenotypic.post.PrependString` and
    :class:`~phenotypic.post.AppendString`, which differ only in which side
    they affix.

    ``.astype(str)`` renders NA as the literal string ``"nan"``, so a naive
    concatenation turns a missing value into a present-looking ``"WT-nan"``.
    That matters for an undetected colony carried by the CLI's ``--metadata``
    left join: every measurement column on such a row is null, and a
    stringified ``"nan"`` would read as real data. Masking the NA cells back
    out is a no-op when nothing is NA, so frames without missing values are
    unaffected.

    Args:
        col: The column to affix. Any dtype; cells are stringified.
        prefix: String concatenated before each present value.
        suffix: String concatenated after each present value.

    Returns:
        A new Series of affixed strings, NA-preserving.
    """
    return (prefix + col.astype(str) + suffix).mask(col.isna())
