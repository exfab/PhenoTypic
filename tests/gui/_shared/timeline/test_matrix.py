"""Pure matrix-model helpers for the timeline engine."""
from __future__ import annotations

from phenotypic.gui._shared.timeline._matrix import _natural_sort_key


def test_numeric_strings_sort_numerically_not_lexically() -> None:
    values = ["10", "2", "1"]
    assert sorted(values, key=_natural_sort_key) == ["1", "2", "10"]


def test_numerics_sort_before_plain_strings() -> None:
    values = ["b", "10", "a", "2"]
    assert sorted(values, key=_natural_sort_key) == ["2", "10", "a", "b"]


def test_iso_datetimes_sort_chronologically() -> None:
    values = ["2024-01-10", "2024-01-02", "2024-01-01"]
    assert sorted(values, key=_natural_sort_key) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-10",
    ]


def test_non_finite_floats_fall_through_to_lexical() -> None:
    # nan/inf must NOT enter the numeric bucket (rank 0): nan breaks sort
    # determinism and inf has no axis position. They fall to lexical (rank 2).
    assert _natural_sort_key("nan")[0] == 2
    assert _natural_sort_key("inf")[0] == 2
    assert _natural_sort_key("-inf")[0] == 2


def test_mixed_tzaware_and_naive_datetimes_do_not_raise_and_sort_chronologically() -> (
    None
):
    # F4: once the Results X axis is a user-pickable time column, an axis can
    # mix tz-aware and tz-naive ISO datetimes. Both land in rank 1; comparing a
    # tz-aware datetime against a naive one raises TypeError. The fix returns a
    # posix-timestamp float (naive treated as UTC) so rank-1 values are always
    # comparable floats. Chronological order: 09:00Z < 10:00 naive (=10:00Z)
    # < 11:00+02:00 (=09:00Z) -- wait, recompute below in UTC to be explicit.
    #
    #   2024-01-01T09:00:00+00:00  -> 09:00 UTC
    #   2024-01-01T10:00:00        -> 10:00 UTC (naive treated as UTC)
    #   2024-01-01T13:00:00+02:00  -> 11:00 UTC
    values = [
        "2024-01-01T13:00:00+02:00",  # 11:00 UTC (latest)
        "2024-01-01T10:00:00",        # 10:00 UTC (naive)
        "2024-01-01T09:00:00+00:00",  # 09:00 UTC (earliest)
    ]
    # Must not raise TypeError when ranks-1 datetimes are mixed offsets.
    ordered = sorted(values, key=_natural_sort_key)
    assert ordered == [
        "2024-01-01T09:00:00+00:00",
        "2024-01-01T10:00:00",
        "2024-01-01T13:00:00+02:00",
    ]
    # All three are rank-1 (datetime family) and now carry comparable floats.
    keys = [_natural_sort_key(v) for v in values]
    assert all(rank == 1 for rank, _ in keys)
    assert all(isinstance(coerced, float) for _, coerced in keys)


def test_build_matrix_with_mixed_offset_time_axis_does_not_raise() -> None:
    # F4 reached through the public surface: build_matrix sorts the columns
    # via _natural_sort_key, so a mixed-offset time axis must not blow up.
    records = [
        {"row_value": "p", "time_value": "2024-01-01T13:00:00+02:00", "cell_ref": "c3"},
        {"row_value": "p", "time_value": "2024-01-01T10:00:00", "cell_ref": "c2"},
        {"row_value": "p", "time_value": "2024-01-01T09:00:00+00:00", "cell_ref": "c1"},
    ]
    m = build_matrix(records)
    assert m.columns == [
        "2024-01-01T09:00:00+00:00",
        "2024-01-01T10:00:00",
        "2024-01-01T13:00:00+02:00",
    ]


from phenotypic.gui._shared.timeline._matrix import TimelineMatrix, build_matrix


def _records() -> list[dict[str, object]]:
    return [
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1"},
        {"row_value": "plateA", "time_value": "10", "cell_ref": "a10"},
        {"row_value": "plateA", "time_value": "2", "cell_ref": "a2"},
        {"row_value": "plateB", "time_value": "1", "cell_ref": "b1"},
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1b"},  # collide
    ]


def test_build_matrix_orders_axes_numerically() -> None:
    m = build_matrix(_records())
    assert isinstance(m, TimelineMatrix)
    assert m.columns == ["1", "2", "10"]
    assert m.rows == ["plateA", "plateB"]


def test_build_matrix_aggregates_collisions_with_deterministic_representative() -> None:
    m = build_matrix(_records())
    cell = m.cells[("plateA", "1")]
    assert cell.count == 2
    assert set(cell.members) == {"a1", "a1b"}
    assert cell.representative == "a1"  # smallest str(cell_ref)


def test_build_matrix_omits_empty_cells() -> None:
    m = build_matrix(_records())
    assert ("plateB", "2") not in m.cells
    assert ("plateB", "1") in m.cells
