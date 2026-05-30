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

    def test_counts_only_rows_with_a_column(self) -> None:
        spec = [
            {"id": "a", "column": "Metadata_Dataset", "values": ["WT"]},
            {"id": "b", "column": "", "values": []},  # unconfigured row
            {"id": "c", "column": "Grid_RowNum", "values": []},  # column, no values
        ]
        assert active_filter_count(spec) == 2

    def test_ignores_malformed_entries(self) -> None:
        assert active_filter_count(["junk", {"values": []}, {"column": "  "}]) == 0


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
