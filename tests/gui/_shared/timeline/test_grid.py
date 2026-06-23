"""build_timeline_grid: row-major key order + per-cell URL generation."""
from __future__ import annotations

from dash import html

from phenotypic.gui._shared.timeline._grid import build_timeline_grid
from phenotypic.gui._shared.timeline._matrix import build_matrix


def _matrix():
    records = [
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1"},
        {"row_value": "plateA", "time_value": "2", "cell_ref": "a2"},
        {"row_value": "plateA", "time_value": "10", "cell_ref": "a10"},
        {"row_value": "plateB", "time_value": "1", "cell_ref": "b1"},
        {"row_value": "plateA", "time_value": "1", "cell_ref": "a1b"},  # count=2
    ]
    return build_matrix(records)


def test_grid_order_is_row_major_over_nonempty_cells() -> None:
    calls: list[tuple[object, int]] = []

    def url_builder(ref: object, fetch: int) -> str:
        calls.append((ref, fetch))
        return f"/thumb/{ref}?size={fetch}"

    component, order = build_timeline_grid(
        _matrix(), url_builder=url_builder, display_size=120, fetch_size=128
    )

    assert isinstance(component, html.Div)
    # Row-major (Y outer, X inner); ("plateB","2") is empty and excluded.
    assert order == [
        ("plateA", "1"),
        ("plateA", "2"),
        ("plateA", "10"),
        ("plateB", "1"),
    ]


def test_url_builder_called_once_per_nonempty_cell_with_fetch_size() -> None:
    calls: list[tuple[object, int]] = []

    def url_builder(ref: object, fetch: int) -> str:
        calls.append((ref, fetch))
        return "x"

    build_timeline_grid(
        _matrix(), url_builder=url_builder, display_size=120, fetch_size=128
    )

    assert len(calls) == 4  # one per non-empty cell
    assert all(fetch == 128 for _ref, fetch in calls)
    # The (plateA,1) cell aggregates 2 members; its representative is "a1".
    assert ("a1", 128) in calls


def test_ref_builder_called_once_per_nonempty_cell() -> None:
    refs: list[object] = []

    def ref_builder(ref: object) -> str:
        refs.append(ref)
        return f"TOKEN::{ref}"

    build_timeline_grid(
        _matrix(),
        url_builder=lambda ref, fetch: "x",
        display_size=120,
        fetch_size=128,
        ref_builder=ref_builder,
    )
    assert len(refs) == 4  # one per non-empty cell
    assert "a1" in refs  # representative of the aggregated (plateA,1) cell


def test_cells_carry_grid_coordinate_indices() -> None:
    # The focus-navigate controller addresses cells by 0-based grid coordinate
    # (spec §16.8); every cell — empty or not — must expose both indices.
    # In _matrix(), (plateB, "2") and (plateB, "10") are EMPTY cells — the test
    # must prove THOSE carry coordinates (the new §16.8 requirement), not just
    # the always-attributed non-empty cells.
    component, _ = build_timeline_grid(
        _matrix(), url_builder=lambda ref, fetch: "x", display_size=120, fetch_size=128
    )

    def _walk(node):
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)

    empties = [
        n for n in _walk(component)
        if "timeline-cell--empty" in (getattr(n, "className", "") or "")
    ]
    assert empties, "expected at least one empty placeholder cell"
    for cell in empties:
        props = cell.to_plotly_json().get("props", {})
        assert "data-row-index" in props and "data-col-index" in props
    # plateB row is index 1; its empty time-columns "2"/"10" are col-index 1/2.
    empty_coords = {
        (cell.to_plotly_json()["props"]["data-row-index"],
         cell.to_plotly_json()["props"]["data-col-index"])
        for cell in empties
    }
    assert ("1", "1") in empty_coords and ("1", "2") in empty_coords
