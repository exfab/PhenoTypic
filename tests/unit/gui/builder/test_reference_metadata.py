"""Builder reference metadata: state field, helpers, dropdown, preview."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from phenotypic import ReferenceContext
from phenotypic._gui.builder import _reference_metadata as rm
from phenotypic._gui.builder._state import (
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    state_from_json,
    state_to_json,
)

_COLUMN_SCALAR = "param-column-scalar"


def _table(tmp_path):
    path = tmp_path / "blank_map.csv"
    pd.DataFrame({"ImageName": ["t01"], "BlankImage": ["t00"]}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def registry():
    from phenotypic._gui._operation_registry import OperationRegistry

    reg = OperationRegistry()
    reg.discover()
    return reg


def _state_with_selected_block(class_name: str, params: dict | None = None) -> _DagBuilderState:
    """One op wired after the root InputImage, and selected."""
    scope = _DagBuilderScope()  # __post_init__ seeds InputImage at index 0
    block = BlockNode(block_id=_new_block_id(), class_name=class_name, params=dict(params or {}))
    scope.blocks.append(block)
    scope.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=scope.blocks[0].block_id,
            source_port="out",
            target_block_id=block.block_id,
            target_port="in",
            kind="image",
        )
    )
    return _DagBuilderState(root=scope, selected_block_id=block.block_id)


def _components_with_id_type(component, id_type: str) -> list:
    found = []
    stack = [component]
    while stack:
        node = stack.pop()
        node_id = getattr(node, "id", None)
        if isinstance(node_id, dict) and node_id.get("type") == id_type:
            found.append(node)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None and not isinstance(children, str):
            stack.append(children)
    return found


def _column_widget_for(component, name: str):
    widgets = [w for w in _components_with_id_type(component, _COLUMN_SCALAR) if w.id["name"] == name]
    return widgets[0] if widgets else None


# --------------------------------------------------------------------- state


def test_state_round_trips_the_reference_path(tmp_path):
    state = _DagBuilderState()
    state.reference_metadata_path = str(_table(tmp_path))
    assert state_from_json(state_to_json(state)).reference_metadata_path == state.reference_metadata_path


def test_older_state_json_without_the_key_loads_as_none():
    data = state_to_json(_DagBuilderState())
    data.pop("reference_metadata_path", None)
    assert state_from_json(data).reference_metadata_path is None


def test_the_path_is_never_written_into_the_pipeline(tmp_path):
    from phenotypic._gui.builder._conversion_dag import to_pipeline_dag

    state = _state_with_selected_block("SubtractBlank")
    plain = to_pipeline_dag(state).to_json()
    state.reference_metadata_path = str(_table(tmp_path))
    assert to_pipeline_dag(state).to_json() == plain
    assert "blank_map" not in plain


# ------------------------------------------------------------------- helpers


def test_describe_empty_missing_and_valid(tmp_path):
    assert rm.describe_reference_table("") == ("", "")
    value, message = rm.describe_reference_table(str(tmp_path / "nope.csv"))
    assert value == "" and "not found" in message.lower()
    value, message = rm.describe_reference_table(str(_table(tmp_path)))
    assert value.endswith("blank_map.csv")
    assert "1 rows" in message


def test_describe_reports_an_invalid_table(tmp_path):
    bad = tmp_path / "no_names.csv"
    bad.write_text("Plate,BlankImage\np1,t00\n", encoding="utf-8")
    value, message = rm.describe_reference_table(str(bad))
    assert value == ""
    assert "Metadata_ImageName" in message


def test_columns_provider_serves_only_reference_metadata(tmp_path):
    provide = rm.reference_columns_provider(str(_table(tmp_path)))
    assert provide is not None
    assert "Metadata_BlankImage" in provide("reference_metadata")
    assert provide("measurements") == []
    assert rm.reference_columns_provider("") is None
    assert rm.reference_columns_provider(str(tmp_path / "nope.csv")) is None


def test_preview_context_activates_with_the_image_directory_as_root(tmp_path):
    image = tmp_path / "plates" / "t01.tif"
    with rm.preview_reference_context(str(_table(tmp_path)), str(image)) as ctx:
        assert ReferenceContext.current() is ctx
        assert ctx.image_root == image.parent
    assert ReferenceContext.current() is None
    with rm.preview_reference_context(None, str(image)) as ctx:
        assert ctx is None and ReferenceContext.current() is None


def test_reference_identity_follows_file_content(tmp_path):
    path = _table(tmp_path)
    first = rm.reference_identity(str(path))
    assert rm.reference_identity(None) == ""
    assert first != ""
    path.write_text(path.read_text() + "t02,t00\n", encoding="utf-8")
    assert rm.reference_identity(str(path)) != first


def test_reference_error_message_reaches_through_pipeline_wrappers():
    from phenotypic._core._reference_context import ReferenceLookupError

    inner = ReferenceLookupError("t01 has no row", reason="unmatched", image_name="t01")
    try:
        try:
            raise inner
        except ReferenceLookupError as exc:
            raise RuntimeError("[SubtractBlank] (step 1/1, key='sb'): t01 has no row") from exc
    except RuntimeError as wrapped:
        try:
            raise RuntimeError("outer") from wrapped
        except RuntimeError as outer:
            message = rm.reference_error_message(outer)
    assert message == "ReferenceLookupError: t01 has no row"
    assert rm.reference_error_message(ValueError("plain")) is None


def test_reference_error_message_points_a_missing_table_at_the_picker():
    from phenotypic._core._reference_context import RefMetadataUnavailableError

    message = rm.reference_error_message(RefMetadataUnavailableError("SubtractBlank reads x"))
    assert message is not None
    assert "Reference metadata" in message


# ------------------------------------------------------------------ dropdown


def test_inspector_renders_a_dropdown_of_the_tables_columns(tmp_path, registry):
    """The live dropdown: a selected SubtractBlank block + a picked table."""
    from phenotypic._gui.builder._layout import build_inspector

    state = _state_with_selected_block("SubtractBlank")
    assert _column_widget_for(build_inspector(state, registry), "blank_column") is None

    state.reference_metadata_path = str(_table(tmp_path))
    widget = _column_widget_for(build_inspector(state, registry), "blank_column")
    assert widget is not None
    values = {o["value"] for o in widget.options}
    assert "Metadata_BlankImage" in values


def test_side_loader_renders_the_dropdown_with_the_default_selected(tmp_path, registry):
    """The side loader is the inspector the linear builder actually mounts."""
    from phenotypic._gui.builder._linear_layout import build_linear_side_loader

    state = _state_with_selected_block("SubtractBlank")
    assert _column_widget_for(build_linear_side_loader(state, registry), "blank_column") is None

    state.reference_metadata_path = str(_table(tmp_path))
    widget = _column_widget_for(build_linear_side_loader(state, registry), "blank_column")
    assert widget is not None
    assert "Metadata_BlankImage" in {o["value"] for o in widget.options}
    # Unset in params, so the operation's default is what runs -- and shows.
    assert widget.value == "Metadata_BlankImage"


def test_a_dropdown_choice_is_written_into_the_block(tmp_path, monkeypatch):
    from phenotypic._gui.builder import _callbacks

    state = _state_with_selected_block("SubtractBlank")
    block_id = state.selected_block_id
    data = state_to_json(state)
    triggered = {"type": _COLUMN_SCALAR, "prefix": block_id, "name": "blank_column"}

    def edit(value):
        monkeypatch.setattr(_callbacks, "ctx", SimpleNamespace(triggered=[{"value": value}]))
        return _callbacks._handle_param_edit(
            data,
            triggered=triggered,
            bool_vals=[],
            enum_vals=[],
            num_values=[],
            num_ids=[],
            str_values=[],
            str_ids=[],
            list_values=[],
            list_ids=[],
            tuple_values=[],
            tuple_ids=[],
        )

    picked = state_from_json(edit("Metadata_Blank2"))
    block = next(b for b in picked.root.blocks if b.block_id == block_id)
    assert block.params["blank_column"] == "Metadata_Blank2"
    # A dropdown rendered with no selection reports None: never write it.
    assert edit(None) == data


# --------------------------------------------------------------- the picker


def test_picking_a_table_updates_state_inspector_and_status(tmp_path, monkeypatch, registry):
    from phenotypic._gui.builder import _callbacks

    monkeypatch.setattr(_callbacks, "_registry", lambda: registry)
    data = state_to_json(_state_with_selected_block("SubtractBlank"))

    new_data, inspector, message = _callbacks._reference_metadata_pick(str(_table(tmp_path)), data)
    assert state_from_json(new_data).reference_metadata_path.endswith("blank_map.csv")
    assert "1 rows" in message
    assert _column_widget_for(SimpleNamespace(children=inspector), "blank_column") is not None

    cleared, inspector, message = _callbacks._reference_metadata_pick("", new_data)
    assert state_from_json(cleared).reference_metadata_path is None
    assert message == ""
    assert _column_widget_for(SimpleNamespace(children=inspector), "blank_column") is None

    missing, _inspector, message = _callbacks._reference_metadata_pick(str(tmp_path / "nope.csv"), new_data)
    assert state_from_json(missing).reference_metadata_path is None
    assert "not found" in message.lower()


def test_loading_a_pipeline_keeps_the_picked_table(tmp_path, monkeypatch):
    from phenotypic import ImagePipeline
    from phenotypic._gui.builder import _callbacks
    from phenotypic.enhance import SubtractBlank

    monkeypatch.setattr(_callbacks, "_render_views", lambda state: ([], [], []))
    path = str(_table(tmp_path))
    state_dict, *_ = _callbacks._state_replacement_payload(
        ImagePipeline(ops={"sb": SubtractBlank()}), reference_metadata_path=path
    )
    assert state_dict["reference_metadata_path"] == path
    state_dict, *_ = _callbacks._state_replacement_payload(ImagePipeline(ops={"sb": SubtractBlank()}))
    assert state_dict["reference_metadata_path"] is None


# ------------------------------------------------------------------ preview


def test_pipeline_revision_changes_only_when_a_table_is_set(tmp_path):
    from phenotypic._gui.builder._callbacks import _pipeline_revision

    data = state_to_json(_DagBuilderState())
    plain = _pipeline_revision(data)
    assert _pipeline_revision({**data, "reference_metadata_path": None}) == plain
    assert _pipeline_revision({**data, "reference_metadata_path": str(_table(tmp_path))}) != plain


def _plate_and_blank(tmp_path):
    plates = tmp_path / "plates"
    plates.mkdir()
    target = np.full((64, 64), 120, dtype=np.uint8)
    target[24:40, 24:40] = 220
    tifffile.imwrite(plates / "t01.tif", target)
    tifffile.imwrite(plates / "t00.tif", np.full((64, 64), 120, dtype=np.uint8))
    return plates / "t01.tif"


def test_node_preview_runs_subtract_blank_against_the_picked_table(tmp_path, monkeypatch):
    from phenotypic._gui.builder import _preview_cache as pc

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    image = _plate_and_blank(tmp_path)
    state = _state_with_selected_block("SubtractBlank")

    without = pc.compute_scope("s", state, [], str(image), None, None)
    assert without["error"] is not None
    assert "Reference metadata" in without["error"]

    state.reference_metadata_path = str(_table(tmp_path))
    with_table = pc.compute_scope("s", state, [], str(image), None, None)
    assert with_table["error"] is None, with_table["error"]
    assert with_table["fingerprint"] != without["fingerprint"]
    assert ReferenceContext.current() is None
