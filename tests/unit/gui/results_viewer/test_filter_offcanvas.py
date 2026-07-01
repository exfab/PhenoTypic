"""Unit tests for the filter-offcanvas pure helpers + callback wiring."""

from __future__ import annotations

import dash

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filter_offcanvas import (
    active_filter_count,
    badge_children,
    badge_style,
    next_offcanvas_state,
    register_filter_offcanvas_callbacks,
)
from phenotypic.schema import METADATA


class TestNextOffcanvasState:
    def test_falsy_clicks_leave_state_unchanged(self) -> None:
        assert next_offcanvas_state(None, False) is False
        assert next_offcanvas_state(None, True) is True
        assert next_offcanvas_state(0, True) is True

    def test_click_toggles_state(self) -> None:
        assert next_offcanvas_state(1, False) is True
        assert next_offcanvas_state(2, True) is False

    def test_none_is_open_treated_as_closed(self) -> None:
        assert next_offcanvas_state(1, None) is True


class TestActiveFilterCount:
    def test_empty_or_none_is_zero(self) -> None:
        assert active_filter_count([]) == 0
        assert active_filter_count(None) == 0

    def test_counts_only_rows_with_a_usable_constraint(self) -> None:
        spec = [
            {"id": "a", "column": "Metadata_Dataset", "values": ["WT"]},
            {"id": "b", "column": "", "values": []},  # unconfigured row
            {"id": "c", "column": "Grid_RowNum", "values": []},  # column, no values
        ]
        assert active_filter_count(spec) == 1

    def test_ignores_malformed_entries(self) -> None:
        assert active_filter_count(["junk", {"values": []}, {"column": "  "}]) == 0

    def test_counts_configured_range_and_contains_rows(self) -> None:
        spec = [
            {"id": "a", "column": "Size_Area", "method": "range",
             "range_min": 100, "range_max": None},
            {"id": "b", "column": str(METADATA.IMAGE_NAME), "method": "contains",
             "text_pattern": "plate"},
            {"id": "c", "column": "Size_Area", "method": "range"},  # unset → 0
            {"id": "d", "column": "", "method": "is_any_of", "values": ["x"]},
        ]
        assert active_filter_count(spec) == 2


class TestRowIsActive:
    def test_list_methods_need_values(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active({"column": "a", "method": "is_any_of", "values": ["x"]})
        assert not row_is_active({"column": "a", "method": "is_any_of", "values": []})
        assert row_is_active({"column": "a", "method": "is_none_of", "values": ["x"]})

    def test_range_needs_a_bound(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active({"column": "a", "method": "range", "range_min": 1})
        assert row_is_active({"column": "a", "method": "range", "range_max": 9})
        assert not row_is_active({"column": "a", "method": "range"})

    def test_compare_needs_op_and_value(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active(
            {"column": "a", "method": "compare", "compare_op": ">", "compare_value": 1}
        )
        assert not row_is_active(
            {"column": "a", "method": "compare", "compare_op": "~", "compare_value": 1}
        )
        assert not row_is_active(
            {"column": "a", "method": "compare", "compare_op": ">"}
        )

    def test_contains_needs_nonblank_pattern(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active({"column": "a", "method": "contains", "text_pattern": "x"})
        assert not row_is_active(
            {"column": "a", "method": "contains", "text_pattern": "  "}
        )

    def test_no_column_is_inactive(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert not row_is_active({"column": "", "method": "is_any_of", "values": ["x"]})


class TestBadge:
    def test_children_blank_at_zero(self) -> None:
        assert badge_children(0) == ""
        assert badge_children(3) == "3"

    def test_style_hides_at_zero(self) -> None:
        assert badge_style(0) == {"display": "none"}
        assert badge_style(2) == {"display": "inline-block"}


def test_register_adds_toggle_and_badge_callbacks() -> None:
    app = dash.Dash(__name__)
    register_filter_offcanvas_callbacks(app)
    outputs = set(app.callback_map.keys())
    assert any(ids.OFFCANVAS_FILTER_ID in key and "is_open" in key for key in outputs)
    assert any(
        ids.FILTER_TOGGLE_BADGE_ID in key and "children" in key for key in outputs
    )


def _iter_components(component):
    """Yield a component and all descendants (local helper)."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _iter_components(child)


def test_bulk_paste_popover_opens_left() -> None:
    """The per-row bulk-paste popover opens leftward so it stays on-screen
    inside the right-docked offcanvas."""
    from phenotypic.gui.results_viewer._filter_panel import (
        _normalise_spec,
        _render_filter_row,
    )

    row = _normalise_spec([{"id": "idx1", "column": "Metadata_Dataset",
                            "values": ["WT"]}])[0]
    node = _render_filter_row("idx1", row, [], is_numeric=False)
    popovers = [
        n for n in _iter_components(node) if getattr(n, "_type", None) == "Popover"
    ]
    assert popovers, "expected a bulk-paste popover in the rendered row"
    assert all(getattr(p, "placement", None) == "left" for p in popovers)
