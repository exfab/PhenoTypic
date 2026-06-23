"""Pure matrix model for the timeline view.

Turns flat ``records`` (each a mapping with a row value, a time value, and an
opaque cell reference) into an ordered ``(row × time)`` matrix. Both axes sort
via :func:`_natural_sort_key`, which coerces values at sort time (numeric →
datetime → lexical) because the stored dtype is unreliable — ``join_metadata``
casts join-key columns to ``pl.String``, so a conceptually-numeric
``Metadata_Time`` arrives as strings (spec §15.3).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone


def _natural_sort_key(value: object) -> tuple[int, object]:
    """Return a sort key that orders numerics, then datetimes, then strings.

    Coercion is attempted on ``str(value)`` so String-dtype numerics sort
    numerically. The leading rank int keeps the three families segregated and
    comparable (Python compares the rank first, the coerced value only within
    a rank, where the types match).

    Datetimes are returned as a **posix-timestamp float**, not a ``datetime``
    object, so that an axis mixing tz-aware and tz-naive ISO datetimes (now
    reachable since the Results X axis is a user-pickable time column) stays
    sortable — comparing a tz-aware ``datetime`` against a naive one raises
    ``TypeError``, but two floats always compare. Naive datetimes are treated
    as UTC before taking the timestamp (F4).

    Args:
        value: Any axis value (typically a ``str``).

    Returns:
        ``(0, float)`` for numerics, ``(1, float)`` for ISO datetimes (a
        UTC posix timestamp), ``(2, str)`` otherwise.
    """
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        # Reject non-finite floats (nan/inf): nan breaks sort determinism
        # (all comparisons False) and inf has no meaningful axis position.
        # Fall through to datetime/lexical instead.
        if math.isfinite(number):
            return (0, number)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return (2, text)
    # Naive datetimes are treated as UTC so a mixed tz-aware/naive axis yields
    # comparable floats (never a tz-aware-vs-naive TypeError).
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (1, parsed.timestamp())


@dataclass(frozen=True)
class TimelineCell:
    """One ``(row, time)`` cell of the matrix.

    Attributes:
        row_value: The cell's row (group) value, stringified.
        time_value: The cell's time (column) value, stringified.
        representative: The opaque cell-ref rendered as the tile (smallest
            ``str(cell_ref)`` among ``members`` — deterministic).
        members: Every cell-ref that maps to this cell (length == ``count``).
        count: Number of members (drives the ``N=k`` badge).
    """

    row_value: str
    time_value: str
    representative: object
    members: tuple[object, ...]
    count: int


@dataclass(frozen=True)
class TimelineMatrix:
    """An ordered ``(row × time)`` matrix of cells.

    Attributes:
        columns: Time values, naturally ordered (the X axis).
        rows: Row/group values, naturally ordered (the Y axis).
        cells: ``(row_value, time_value) -> TimelineCell``. Missing pairs are
            absent (empty cells render as placeholders downstream).
    """

    columns: list[str]
    rows: list[str]
    cells: dict[tuple[str, str], TimelineCell]


def build_matrix(
    records: Iterable[Mapping[str, object]],
    *,
    row_key: str = "row_value",
    time_key: str = "time_value",
    ref_key: str = "cell_ref",
) -> TimelineMatrix:
    """Build a :class:`TimelineMatrix` from flat records.

    Args:
        records: Iterable of mappings, each carrying a row value, a time
            value, and an opaque cell reference under the given keys.
        row_key: Mapping key for the row (group) value.
        time_key: Mapping key for the time (column) value.
        ref_key: Mapping key for the opaque per-image cell reference.

    Returns:
        A matrix with naturally-ordered ``columns``/``rows`` and a
        ``cells`` map whose representative is the smallest ``str(cell_ref)``
        in each cell.
    """
    grouped: dict[tuple[str, str], list[object]] = {}
    row_set: set[str] = set()
    col_set: set[str] = set()
    for record in records:
        rv = str(record[row_key])
        tv = str(record[time_key])
        row_set.add(rv)
        col_set.add(tv)
        grouped.setdefault((rv, tv), []).append(record[ref_key])

    cells: dict[tuple[str, str], TimelineCell] = {}
    for (rv, tv), refs in grouped.items():
        ordered = tuple(sorted(refs, key=lambda r: str(r)))
        cells[(rv, tv)] = TimelineCell(
            row_value=rv,
            time_value=tv,
            representative=ordered[0],
            members=ordered,
            count=len(ordered),
        )

    return TimelineMatrix(
        columns=sorted(col_set, key=_natural_sort_key),
        rows=sorted(row_set, key=_natural_sort_key),
        cells=cells,
    )
