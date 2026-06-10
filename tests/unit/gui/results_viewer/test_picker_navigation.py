"""Unit tests for Results Viewer picker navigation helpers."""

from __future__ import annotations

from phenotypic.gui.results_viewer._picker_navigation import (
    enabled_picker_values,
    picker_button_disabled_states,
    step_picker_value,
)


def test_enabled_picker_values_skips_disabled_options() -> None:
    options = [
        {"label": "a", "value": "a"},
        {"label": "b", "value": "b", "disabled": True},
        {"label": "c", "value": "c"},
    ]
    assert enabled_picker_values(options) == ["a", "c"]


def test_step_picker_value_clamps_at_edges() -> None:
    options = [{"label": v, "value": v} for v in ["a", "b", "c"]]
    assert step_picker_value("a", options, "previous") == "a"
    assert step_picker_value("a", options, "next") == "b"
    assert step_picker_value("c", options, "next") == "c"
    assert step_picker_value("c", options, "previous") == "b"


def test_step_picker_value_missing_current_uses_directional_edge() -> None:
    options = [{"label": v, "value": v} for v in ["a", "b", "c"]]
    assert step_picker_value(None, options, "next") == "a"
    assert step_picker_value("missing", options, "previous") == "c"


def test_picker_button_disabled_states() -> None:
    options = [{"label": v, "value": v} for v in ["a", "b", "c"]]
    assert picker_button_disabled_states("a", options) == (True, False)
    assert picker_button_disabled_states("b", options) == (False, False)
    assert picker_button_disabled_states("c", options) == (False, True)
    assert picker_button_disabled_states(None, options) == (False, False)
    assert picker_button_disabled_states(None, []) == (True, True)
