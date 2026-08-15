from phenotypic.gui._shared._picker_navigation import (
    enabled_picker_values,
    offset_picker_value,
    picker_button_disabled_states,
    picker_position,
    step_picker_value,
)
from phenotypic.gui.results_viewer import _picker_navigation as rv_nav


def _opts(*values):
    return [{"label": v, "value": v} for v in values]


def test_step_next_and_prev():
    opts = _opts("a", "b", "c")
    assert step_picker_value("a", opts, "next") == "b"
    assert step_picker_value("c", opts, "previous") == "b"
    assert step_picker_value("c", opts, "next") == "c"  # clamp at end


def test_bounds_disabled_states():
    opts = _opts("a", "b", "c")
    assert picker_button_disabled_states("a", opts) == (True, False)
    assert picker_button_disabled_states("c", opts) == (False, True)


def test_offset_clamps_and_skips_disabled_options():
    opts = [
        {"label": "a", "value": "a"},
        {"label": "b", "value": "b", "disabled": True},
        {"label": "c", "value": "c"},
        {"label": "d", "value": "d"},
    ]
    assert offset_picker_value("a", opts, 10) == "d"
    assert offset_picker_value("d", opts, -10) == "a"
    assert offset_picker_value("a", opts, 1) == "c"
    assert offset_picker_value(None, opts, -1) == "d"


def test_picker_position_counts_enabled_options_only():
    opts = [
        {"label": "a", "value": "a"},
        {"label": "b", "value": "b", "disabled": True},
        {"label": "c", "value": "c"},
    ]
    assert picker_position("c", opts) == (2, 2)
    assert picker_position("missing", opts) == (0, 2)


def test_results_viewer_reexports_same_callables():
    assert rv_nav.step_picker_value is step_picker_value
    assert rv_nav.picker_button_disabled_states is picker_button_disabled_states
    assert rv_nav.enabled_picker_values is enabled_picker_values
