"""Unit tests for the filter-panel pure helpers (no Dash runtime)."""

from __future__ import annotations

from phenotypic.gui.results_viewer._filter_panel import (
    _blank_row,
    _normalise_spec,
    _render_filter_row,
    _value_options_for_mounted_values,
    set_row_compare,
    set_row_method,
    set_row_range,
    set_row_text,
)
from phenotypic.gui.results_viewer._filter_state import (
    METHOD_COMPARE,
    METHOD_CONTAINS,
    METHOD_IS_ANY_OF,
    METHOD_RANGE,
)


def test_blank_row_has_all_keys_and_defaults() -> None:
    row = _blank_row()
    for key in (
        "id", "column", "method", "values", "range_min", "range_max",
        "compare_op", "compare_value", "text_pattern", "text_regex",
        "text_case_sensitive",
    ):
        assert key in row
    assert row["method"] == METHOD_IS_ANY_OF
    assert row["values"] == []


def test_normalise_spec_backfills_legacy_rows() -> None:
    rows = _normalise_spec([{"column": "a", "values": ["x"]}])
    assert rows[0]["method"] == METHOD_IS_ANY_OF
    assert rows[0]["range_min"] is None
    assert isinstance(rows[0]["id"], str) and rows[0]["id"]


def test_set_row_method_resets_payload() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    rows[0]["values"] = ["keep-me?"]
    rows[0]["range_min"] = 5.0
    out = set_row_method(rows, idx, METHOD_RANGE)
    assert out[0]["method"] == METHOD_RANGE
    assert out[0]["values"] == []          # payload reset
    assert out[0]["range_min"] is None     # payload reset


def test_set_row_range_writes_bounds() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_range(rows, idx, 1.0, 9.0)
    assert out[0]["range_min"] == 1.0 and out[0]["range_max"] == 9.0


def test_set_row_compare_writes_op_and_value() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_compare(rows, idx, ">=", 0.5)
    assert out[0]["method"] == METHOD_COMPARE or out[0]["compare_op"] == ">="
    assert out[0]["compare_op"] == ">=" and out[0]["compare_value"] == 0.5


def test_set_row_text_writes_pattern_and_flags() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_text(rows, idx, "plate", regex=True, case=False)
    assert out[0]["text_pattern"] == "plate"
    assert out[0]["text_regex"] is True
    assert out[0]["text_case_sensitive"] is False


def test_setters_ignore_unknown_idx() -> None:
    rows = [_blank_row()]
    out = set_row_range(rows, "nope", 1.0, 2.0)
    assert out[0]["range_min"] is None


def _iter(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _iter(child)


def _type_ids(node):
    return {
        c.id["type"]
        for c in _iter(node)
        if isinstance(getattr(c, "id", None), dict) and "type" in c.id
    }


def test_render_row_has_method_dropdown() -> None:
    row = _normalise_spec([{"id": "r1", "column": "Metadata_Strain"}])[0]
    node = _render_filter_row("r1", row, [], is_numeric=False)
    assert "filter-row-method" in _type_ids(node)


def test_range_method_renders_min_max_inputs() -> None:
    row = _normalise_spec(
        [{"id": "r1", "column": "Size_Area", "method": METHOD_RANGE}]
    )[0]
    node = _render_filter_row("r1", row, [], is_numeric=True)
    ids_present = _type_ids(node)
    assert "filter-row-range-min" in ids_present
    assert "filter-row-range-max" in ids_present
    assert "filter-row-values" not in ids_present


def test_contains_method_renders_text_controls() -> None:
    row = _normalise_spec(
        [{"id": "r1", "column": "Metadata_ImageFile", "method": METHOD_CONTAINS}]
    )[0]
    node = _render_filter_row("r1", row, [], is_numeric=False)
    ids_present = _type_ids(node)
    assert "filter-row-text-pattern" in ids_present
    assert "filter-row-text-regex" in ids_present
    assert "filter-row-text-case" in ids_present


def test_method_dropdown_disables_range_compare_for_text_column() -> None:
    row = _normalise_spec([{"id": "r1", "column": "Metadata_Strain"}])[0]
    node = _render_filter_row("r1", row, [], is_numeric=False)
    dropdown = next(
        c for c in _iter(node)
        if isinstance(getattr(c, "id", None), dict)
        and c.id.get("type") == "filter-row-method"
    )
    disabled = {o["value"] for o in dropdown.options if o.get("disabled")}
    assert {"range", "compare"} <= disabled


def test_method_dropdown_enables_range_compare_for_numeric_column() -> None:
    row = _normalise_spec([{"id": "r1", "column": "Size_Area"}])[0]
    node = _render_filter_row("r1", row, [], is_numeric=True)
    dropdown = next(
        c for c in _iter(node)
        if isinstance(getattr(c, "id", None), dict)
        and c.id.get("type") == "filter-row-method"
    )
    disabled = {o["value"] for o in dropdown.options if o.get("disabled")}
    assert "range" not in disabled and "compare" not in disabled


def test_value_options_only_target_mounted_list_value_controls() -> None:
    options = _value_options_for_mounted_values(
        ["Size_Area", "Metadata_Strain"],
        [
            {"type": "filter-row-column", "index": "range-row"},
            {"type": "filter-row-column", "index": "list-row"},
        ],
        [{"type": "filter-row-values", "index": "list-row"}],
        {
            "Size_Area": ["1", "2"],
            "Metadata_Strain": ["BY4741", "BY4742"],
        },
    )

    assert options == [
        [
            {"label": "BY4741", "value": "BY4741"},
            {"label": "BY4742", "value": "BY4742"},
        ]
    ]


def test_register_callbacks_wires_method_controls(tmp_path) -> None:
    """register_callbacks adds the per-method sync callbacks."""
    import dash
    import polars as pl

    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.gui.results_viewer._output_root import OutputRoot
    from phenotypic.gui.results_viewer import _filter_panel

    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1"],
            "Metadata_ImageFile": ["a", "b"],
            "Size_Area": [1.0, 2.0],
        }
    )
    from phenotypic.sdk_ import master_measurements_parquet_path

    target = master_measurements_parquet_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)
    for stem in ("a", "b"):
        (tmp_path / "results" / "d1" / "overlays" / f"{stem}.png").touch()

    output_root = OutputRoot.discover(tmp_path)
    state = CurationLabels.load(output_root.root, output_root.clean_master_df)
    app = dash.Dash(__name__)
    _filter_panel.register_callbacks(app, output_root, state)

    # The per-method callbacks all write the (allow_duplicate) spec store, so
    # their distinguishing surface is the *input* id, not the output-keyed
    # callback_map key. Collect every registered input id-string.
    registered_input_ids = {
        spec["id"]
        for entry in app.callback_map.values()
        for spec in entry["inputs"]
    }
    blob = " ".join(registered_input_ids)
    assert "filter-row-method" in blob
    assert "filter-row-range-min" in blob
    assert "filter-row-compare-op" in blob
    assert "filter-row-text-pattern" in blob
