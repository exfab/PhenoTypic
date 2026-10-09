"""Builder reference metadata: state field, helpers, dropdown, preview."""

from __future__ import annotations

import json
import os
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


@pytest.fixture(autouse=True)
def image_root(tmp_path):
    """The builder's ``--image-root`` is ``tmp_path``, as a request would see it.

    Every table path is confined to it, so each test runs inside an app
    context carrying it; a table this file writes under ``tmp_path`` is
    inside the root.
    """
    import flask

    from phenotypic._gui._config import CFG_IMAGE_ROOT

    app = flask.Flask("reference-metadata-tests")
    app.config[CFG_IMAGE_ROOT] = tmp_path
    with app.app_context():
        yield tmp_path


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


@pytest.mark.parametrize("stored", [123, ["/x/blank_map.csv"], {"path": "x"}, ""])
def test_a_non_string_reference_path_loads_as_none(stored):
    """Store data is client-writable; a non-string must not reach ``Path(...)``."""
    data = state_to_json(_DagBuilderState())
    data["reference_metadata_path"] = stored
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


# ------------------------------------------------------------- confinement


@pytest.fixture
def outside(tmp_path_factory):
    """A directory outside the image root, holding tables a user must not reach.

    Their headers are distinctive, so a status line that leaks any of a
    refused file's contents (header names, a parse error listing them) fails.
    """
    directory = tmp_path_factory.mktemp("outside")
    pd.DataFrame({"ImageName": ["t01"], "SecretHeader": ["x"]}).to_csv(
        directory / "secret.csv", index=False
    )
    (directory / "no_names.csv").write_text("SecretHeader,Other\nx,y\n", encoding="utf-8")
    return directory


def _refused(value: str, message: str) -> None:
    assert value == ""
    assert "image root" in message
    assert "SecretHeader" not in message


@pytest.mark.parametrize("name", ["secret.csv", "no_names.csv"])
def test_picker_refuses_a_table_outside_the_image_root(outside, name):
    _refused(*rm.describe_reference_table(str(outside / name)))


def test_picker_refusal_stores_nothing(outside, monkeypatch, registry):
    from phenotypic._gui.builder import _callbacks

    monkeypatch.setattr(_callbacks, "_registry", lambda: registry)
    data = state_to_json(_state_with_selected_block("SubtractBlank"))

    new_data, inspector, message = _callbacks._reference_metadata_pick(
        str(outside / "secret.csv"), data
    )

    assert state_from_json(new_data).reference_metadata_path is None
    _refused("", message)
    assert _column_widget_for(SimpleNamespace(children=inspector), "blank_column") is None


def test_picker_refuses_traversal_out_of_the_image_root(tmp_path, outside):
    _refused(*rm.describe_reference_table(str(tmp_path / ".." / outside.name / "secret.csv")))
    _refused(*rm.describe_reference_table(f"../{outside.name}/secret.csv"))


def test_picker_refuses_a_symlink_that_escapes_the_image_root(tmp_path, outside):
    """``SandboxRoot`` follows symlinks and checks where they land."""
    link = tmp_path / "linked.csv"
    link.symlink_to(outside / "secret.csv")
    _refused(*rm.describe_reference_table(str(link)))


def test_picker_accepts_a_symlink_that_stays_inside_the_image_root(tmp_path):
    """Control for the test above: a symlink is refused for where it lands."""
    target = _table(tmp_path)
    link = tmp_path / "alias.csv"
    link.symlink_to(target)
    value, message = rm.describe_reference_table(str(link))
    assert value == str(target.resolve())
    assert "1 rows" in message


@pytest.mark.parametrize("name", ["blank_map.txt", "blank_map.json", "blank_map"])
def test_picker_refuses_a_file_that_is_not_csv_or_parquet(tmp_path, name):
    path = tmp_path / name
    path.write_text("ImageName,SecretHeader\nt01,t00\n", encoding="utf-8")
    _refused(*rm.describe_reference_table(str(path)))


def test_picker_accepts_a_parquet_table_under_the_image_root(tmp_path):
    import polars as pl

    path = tmp_path / "blank_map.parquet"
    pl.DataFrame({"ImageName": ["t01"], "BlankImage": ["t00"]}).write_parquet(path)
    value, message = rm.describe_reference_table(str(path))
    assert value == str(path.resolve())
    assert "1 rows" in message


def test_a_relative_path_resolves_against_the_image_root(tmp_path):
    """Not against the server's working directory."""
    nested = tmp_path / "maps"
    nested.mkdir()
    table = _table(nested)
    value, message = rm.describe_reference_table("maps/blank_map.csv")
    assert value == str(table.resolve())
    assert "1 rows" in message


def test_no_configured_image_root_refuses_every_table(tmp_path):
    """The debug launcher without ``--image-root``: fail closed, not open."""
    import flask

    from phenotypic._gui._config import CFG_IMAGE_ROOT

    flask.current_app.config[CFG_IMAGE_ROOT] = None
    _refused(*rm.describe_reference_table(str(_table(tmp_path))))
    assert rm.reference_columns_provider(str(_table(tmp_path))) is None


def test_confined_reference_path_is_the_single_rule(tmp_path, outside):
    assert rm.confined_reference_path(str(_table(tmp_path)), tmp_path) == (
        tmp_path / "blank_map.csv"
    ).resolve()
    assert rm.confined_reference_path(str(outside / "secret.csv"), tmp_path) is None
    assert rm.confined_reference_path(str(_table(tmp_path)), None) is None
    assert rm.confined_reference_path("", tmp_path) is None
    assert rm.confined_reference_path(123, tmp_path) is None


def test_a_tampered_state_path_outside_the_root_is_treated_as_no_table(
    tmp_path, outside, monkeypatch, registry
):
    """``store-builder-state`` is client-writable; it must not bypass the picker."""
    from phenotypic._gui.builder import _preview_cache as pc
    from phenotypic._gui.builder._linear_layout import build_linear_side_loader

    secret = str(outside / "secret.csv")
    assert rm.reference_columns_provider(secret) is None

    def _never_read(*args):
        raise AssertionError(f"read a refused table: {args}")

    monkeypatch.setattr(rm, "_file_sha256", _never_read)
    monkeypatch.setattr(rm, "_columns_for", _never_read)
    assert rm.reference_identity(secret) == rm.reference_identity(None)
    assert rm.reference_columns_provider(secret) is None
    with rm.preview_reference_context(secret, str(tmp_path / "t01.tif")) as ctx:
        assert ctx is None and ReferenceContext.current() is None

    state = _state_with_selected_block("SubtractBlank")
    state.reference_metadata_path = secret
    assert _column_widget_for(build_linear_side_loader(state, registry), "blank_column") is None

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    image = _plate_and_blank(tmp_path)
    tampered = pc.compute_scope("s", state, [], str(image), None, None)
    state.reference_metadata_path = None
    untouched = pc.compute_scope("s", state, [], str(image), None, None)
    assert tampered["error"] is not None
    assert tampered["error"] == untouched["error"]
    assert "Reference metadata" in tampered["error"]


def test_columns_provider_serves_only_reference_metadata(tmp_path):
    provide = rm.reference_columns_provider(str(_table(tmp_path)))
    assert provide is not None
    assert "Metadata_BlankImage" in provide("reference_metadata")
    assert provide("measurements") == []
    assert rm.reference_columns_provider("") is None
    assert rm.reference_columns_provider(str(tmp_path / "nope.csv")) is None


def test_columns_provider_follows_a_rewrite_that_keeps_the_mtime(tmp_path):
    """``cp -p`` / coarse mtimes: the same mtime must not serve stale columns."""
    path = _table(tmp_path)
    provide = rm.reference_columns_provider(str(path))
    assert "Metadata_Blank2" not in provide("reference_metadata")

    before = path.stat()
    pd.DataFrame(
        {"ImageName": ["t01"], "BlankImage": ["t00"], "Blank2": ["t00b"]}
    ).to_csv(path, index=False)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_size != before.st_size

    provide = rm.reference_columns_provider(str(path))
    assert "Metadata_Blank2" in provide("reference_metadata")


def test_preview_context_activates_with_the_image_directory_as_root(tmp_path):
    image = tmp_path / "plates" / "t01.tif"
    with rm.preview_reference_context(str(_table(tmp_path)), str(image)) as ctx:
        assert ReferenceContext.current() is ctx
        assert ctx.image_root == image.parent
    assert ReferenceContext.current() is None
    with rm.preview_reference_context(None, str(image)) as ctx:
        assert ctx is None and ReferenceContext.current() is None


@pytest.mark.parametrize("image_path", [None, ""])
def test_preview_context_without_an_image_has_no_root(tmp_path, image_path):
    with rm.preview_reference_context(str(_table(tmp_path)), image_path) as ctx:
        assert ctx.image_root is None


def test_the_synthetic_sentinel_does_not_root_at_the_working_directory(tmp_path):
    """``Path("<synthetic>").parent`` is ``.``: blanks would resolve in the cwd."""
    from phenotypic._gui.builder._directory_browser import SYNTHETIC_SENTINEL

    with rm.preview_reference_context(str(_table(tmp_path)), SYNTHETIC_SENTINEL) as ctx:
        assert ctx.image_root is None


def _two_dataset_table(tmp_path, *, with_dataset: bool = True):
    """``plateA/t01 -> t00`` and ``plateB/t01 -> t00b``, as the CLI plans them."""
    from phenotypic.schema import EXPERIMENT

    frame = pd.DataFrame({"ImageName": ["t01", "t01"], "BlankImage": ["t00", "t00b"]})
    if with_dataset:
        frame.insert(0, str(EXPERIMENT.DATASET), ["plateA", "plateB"])
    path = tmp_path / "two_datasets.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("dataset,blank", [("plateA", "t00"), ("plateB", "t00b")])
def test_preview_context_narrows_to_the_image_directorys_dataset(tmp_path, dataset, blank):
    """The CLI narrows each dataset by its input directory name; so must the preview."""
    image = tmp_path / dataset / "t01.tif"
    with rm.preview_reference_context(str(_two_dataset_table(tmp_path)), str(image)) as ctx:
        assert ctx.dataset == dataset
        assert ctx.lookup("t01", ["BlankImage"]) == {"BlankImage": blank}


def test_preview_context_does_not_narrow_without_a_dataset_column(tmp_path):
    table = _two_dataset_table(tmp_path, with_dataset=False)
    with rm.preview_reference_context(str(table), str(tmp_path / "plateA" / "t01.tif")) as ctx:
        assert ctx.dataset is None


def test_preview_context_does_not_narrow_to_a_directory_the_table_does_not_name(tmp_path):
    """Narrowing to an unknown dataset would make every lookup "unmatched"."""
    from phenotypic.schema import EXPERIMENT

    table = tmp_path / "one_dataset.csv"
    pd.DataFrame(
        {str(EXPERIMENT.DATASET): ["plateA"], "ImageName": ["t01"], "BlankImage": ["t00"]}
    ).to_csv(table, index=False)
    with rm.preview_reference_context(str(table), str(tmp_path / "scans" / "t01.tif")) as ctx:
        assert ctx.dataset is None
        assert ctx.lookup("t01", ["BlankImage"]) == {"BlankImage": "t00"}


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
    # "outer" carries no step prefix, so the wrapper below it supplies one.
    assert message == "[SubtractBlank] (step 1/1, key='sb'): ReferenceLookupError: t01 has no row"
    assert rm.reference_error_message(inner) == "ReferenceLookupError: t01 has no row"
    assert rm.reference_error_message(ValueError("plain")) is None


def test_reference_error_message_names_the_failing_op_through_nested_pipelines():
    """With two reference ops, the bare lookup message does not say which failed."""
    from phenotypic import Image, ImagePipeline
    from phenotypic.enhance import SubtractBlank

    target = Image(arr=np.full((8, 8), 0.5, dtype=np.float32), name="t04")
    layout = pd.DataFrame({"ImageName": ["t99"], "BlankImage": ["t00"]})
    pipeline = ImagePipeline(ops={"branch": ImagePipeline(ops={"sb": SubtractBlank()})})
    with ReferenceContext(layout), pytest.raises(RuntimeError) as info:
        pipeline.apply(target)

    assert rm.reference_error_message(info.value) == (
        "[ImagePipeline] (step 1/1, key='branch'): "
        "[SubtractBlank] (step 1/1, key='sb'): "
        "ReferenceLookupError: No row in the reference metadata for image 't04'"
    )


def test_reference_error_message_ignores_a_suppressed_context():
    """``raise ... from None`` hides the handled error, as the traceback module does."""
    from phenotypic._core._reference_context import ReferenceLookupError

    try:
        try:
            raise ReferenceLookupError("handled", reason="unmatched", image_name="t01")
        except ReferenceLookupError:
            raise KeyError("unrelated") from None
    except KeyError as exc:
        assert rm.reference_error_message(exc) is None


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


def _components_with_class(component, class_name: str) -> list:
    found = []
    stack = [component]
    while stack:
        node = stack.pop()
        if class_name in (getattr(node, "className", None) or "").split():
            found.append(node)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None and not isinstance(children, str):
            stack.append(children)
    return found


def test_a_bare_header_value_selects_its_prefixed_column(registry):
    """``ReferenceContext`` resolves ``BlankImage`` to ``Metadata_BlankImage``,
    so the dropdown must show it selected rather than as stale."""
    from phenotypic._gui.builder._param_form import param_form
    from phenotypic.sdk_ import ensure_metadata_prefix

    prefixed = ensure_metadata_prefix("BlankImage")
    columns = [ensure_metadata_prefix("ImageName"), prefixed]
    form = param_form(
        registry.get("SubtractBlank"),
        {"blank_column": "BlankImage"},
        form_id_prefix="b",
        columns_provider=lambda source: columns if source == rm.REFERENCE_SOURCE else [],
    )
    widget = _column_widget_for(form, "blank_column")
    assert widget is not None
    assert widget.value == prefixed
    assert _components_with_class(form, "param-column-stale") == []


def test_an_unresolvable_value_is_still_shown_as_stale(registry):
    """Control for the test above: the mapping is not a blanket "accept"."""
    from phenotypic._gui.builder._param_form import param_form
    from phenotypic.sdk_ import ensure_metadata_prefix

    columns = [ensure_metadata_prefix("ImageName"), ensure_metadata_prefix("BlankImage")]
    form = param_form(
        registry.get("SubtractBlank"),
        {"blank_column": "Blank2"},
        form_id_prefix="b",
        columns_provider=lambda source: columns if source == rm.REFERENCE_SOURCE else [],
    )
    assert _column_widget_for(form, "blank_column").value is None
    assert len(_components_with_class(form, "param-column-stale")) == 1


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


# ------------------------------------------------------------- Dash wiring


def _builder_app(tmp_path, monkeypatch, registry):
    from phenotypic._gui.builder import _preview_cache as pc
    from phenotypic._gui.builder._app import create_app

    # create_app wipes the preview cache root; keep it inside tmp_path.
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "preview-cache")
    return create_app(image_root=tmp_path, registry=registry)


def _spec(app, name: str):
    return next(
        (key, spec)
        for key, spec in app.callback_map.items()
        if getattr(spec.get("callback"), "__wrapped__", None) is not None
        and spec["callback"].__wrapped__.__name__ == name
    )


def _with_table(state: _DagBuilderState, tmp_path) -> dict:
    return {**state_to_json(state), "reference_metadata_path": str(_table(tmp_path))}


def test_picker_is_wired_to_value_and_enter(tmp_path, monkeypatch, registry):
    """``n_submit``: Enter re-validates an unchanged path (an edited/deleted table)."""
    from phenotypic._gui.builder import _ids as ids

    app = _builder_app(tmp_path, monkeypatch, registry)
    key, spec = _spec(app, "set_reference_metadata")
    assert spec["inputs"] == [
        {"id": ids.INPUT_REFERENCE_METADATA, "property": "value"},
        {"id": ids.INPUT_REFERENCE_METADATA, "property": "n_submit"},
    ]
    assert spec["state"] == [{"id": ids.STORE_BUILDER_STATE, "property": "data"}]
    for output in (
        f"{ids.STORE_BUILDER_STATE}.data",
        f"{ids.INSPECTOR_CONTENT}.children",
        f"{ids.REFERENCE_METADATA_STATUS}.children",
    ):
        assert output in key


def test_a_column_choice_reaches_the_block_through_the_fan_in(tmp_path, monkeypatch, registry):
    """Pins both the subscription and the dispatch-set membership."""
    from phenotypic._gui.builder import _callbacks

    app = _builder_app(tmp_path, monkeypatch, registry)
    _, fan_in = _spec(app, "fan_in_state_mutation")
    # callback_map holds a pattern id as Dash's stringified JSON, ALL as ["ALL"].
    assert {
        "id": json.dumps(
            {"type": _COLUMN_SCALAR, "prefix": ["ALL"], "name": ["ALL"]},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "property": "value",
    } in fan_in["inputs"]

    monkeypatch.setattr(_callbacks, "_render_views", lambda state: ([], [], []))
    state = _state_with_selected_block("SubtractBlank")
    block_id = state.selected_block_id
    triggered = {"type": _COLUMN_SCALAR, "prefix": block_id, "name": "blank_column"}
    monkeypatch.setattr(
        _callbacks,
        "ctx",
        SimpleNamespace(triggered_id=triggered, triggered=[{"value": "Metadata_Blank2"}]),
    )
    n_args = len(fan_in["inputs"]) + len(fan_in["state"])
    args = [[] for _ in range(n_args - 1)] + [state_to_json(state)]

    result = fan_in["callback"].__wrapped__(*args)

    block = next(b for b in state_from_json(result[0]).root.blocks if b.block_id == block_id)
    assert block.params["blank_column"] == "Metadata_Blank2"


def test_starting_a_new_state_keeps_the_picked_table(tmp_path, monkeypatch, registry):
    """The picker still shows the table, so the reset state must still hold it."""
    from phenotypic._gui.builder import _callbacks
    from phenotypic._gui.builder import _ids as ids

    app = _builder_app(tmp_path, monkeypatch, registry)
    monkeypatch.setattr(_callbacks, "_render_views", lambda state: ([], [], []))
    _, spec = _spec(app, "start_new_builder_state")
    assert {"id": ids.STORE_BUILDER_STATE, "property": "data"} in spec["state"]
    data = _with_table(_state_with_selected_block("SubtractBlank"), tmp_path)

    result = spec["callback"].__wrapped__([1], data)

    fresh = state_from_json(result[0])
    assert fresh.reference_metadata_path == data["reference_metadata_path"]
    assert all(b.class_name != "SubtractBlank" for b in fresh.root.blocks)


def test_loading_a_prefab_keeps_the_picked_table(tmp_path, monkeypatch, registry):
    from phenotypic._gui.builder import _callbacks

    app = _builder_app(tmp_path, monkeypatch, registry)
    monkeypatch.setattr(_callbacks, "_render_views", lambda state: ([], [], []))
    monkeypatch.setattr(
        _callbacks,
        "ctx",
        SimpleNamespace(
            triggered_id={"type": "prefab-card", "class_name": "HeavyOtsuPipeline"},
            triggered=[{"value": 1}],
        ),
    )
    data = _with_table(_DagBuilderState(), tmp_path)

    result = _spec(app, "click_prefab_card")[1]["callback"].__wrapped__([1], data)

    assert result[4] is False, result  # the modal closed: the load succeeded
    assert state_from_json(result[0]).reference_metadata_path == data["reference_metadata_path"]


def test_loading_a_pipeline_json_keeps_the_picked_table(tmp_path, monkeypatch, registry):
    from phenotypic import ImagePipeline
    from phenotypic._gui.builder import _callbacks
    from phenotypic._gui.builder import _ids as ids
    from phenotypic.enhance import SubtractBlank

    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(ImagePipeline(ops={"sb": SubtractBlank()}).to_json(), encoding="utf-8")
    app = _builder_app(tmp_path, monkeypatch, registry)
    monkeypatch.setattr(_callbacks, "_render_views", lambda state: ([], [], []))
    monkeypatch.setattr(
        _callbacks,
        "ctx",
        SimpleNamespace(
            triggered_id={"type": ids.DIR_ENTRY_TYPE_JSON, "kind": "file", "path": str(pipeline)},
            triggered=[{"value": 1}],
        ),
    )
    data = _with_table(_DagBuilderState(), tmp_path)

    result = _spec(app, "click_json_entry")[1]["callback"].__wrapped__([1], data)

    assert result[5] is False, result  # the modal closed: the load succeeded
    loaded = state_from_json(result[1])
    assert loaded.reference_metadata_path == data["reference_metadata_path"]
    assert any(b.class_name == "SubtractBlank" for b in loaded.root.blocks)


# ------------------------------------------------------------------ preview


def test_pipeline_revision_changes_only_when_a_table_is_set(tmp_path):
    from phenotypic._gui.builder._callbacks import _pipeline_revision

    data = state_to_json(_DagBuilderState())
    plain = _pipeline_revision(data)
    assert _pipeline_revision({**data, "reference_metadata_path": None}) == plain
    assert _pipeline_revision({**data, "reference_metadata_path": str(_table(tmp_path))}) != plain


def test_pipeline_revision_follows_the_tables_content(tmp_path):
    """Same path, new content: the preview it computed is no longer current."""
    from phenotypic._gui.builder._callbacks import _pipeline_revision

    path = _table(tmp_path)
    data = {**state_to_json(_DagBuilderState()), "reference_metadata_path": str(path)}
    before = _pipeline_revision(data)
    stat = path.stat()
    path.write_text(path.read_text() + "t02,t00\n", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert _pipeline_revision(data) != before


def _plate_and_blank(tmp_path, directory: str = "plates"):
    plates = tmp_path / directory
    plates.mkdir()
    target = np.full((64, 64), 120, dtype=np.uint8)
    target[24:40, 24:40] = 220
    tifffile.imwrite(plates / "t01.tif", target)
    tifffile.imwrite(plates / "t00.tif", np.full((64, 64), 120, dtype=np.uint8))
    return plates / "t01.tif"


def _node_detect_mat(pc, manifest, session_id: str, block_id: str) -> np.ndarray:
    from phenotypic.sdk_ import load_image_from_store

    store = pc.scope_dir(session_id, []) / manifest["nodes"][block_id]["store"]
    return load_image_from_store(store).detect_mat[:]


#: The colony is 220 over a 120 background, both uint8, so subtracting the
#: 120 blank leaves 100/255 on the colony and 0 everywhere else.
_COLONY_OVER_BLANK = 100 / 255


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

    detect_mat = _node_detect_mat(pc, with_table, "s", state.selected_block_id)
    assert float(detect_mat.max()) == pytest.approx(_COLONY_OVER_BLANK, abs=1e-6)
    assert float(detect_mat.min()) == pytest.approx(0.0, abs=1e-6)


def test_node_preview_subtracts_the_blank_of_the_images_own_dataset(tmp_path, monkeypatch):
    """plateA/t01 -> t00 and plateB/t01 -> t00b: un-narrowed, t01 is ambiguous."""
    from phenotypic._gui.builder import _preview_cache as pc

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    image = _plate_and_blank(tmp_path, "plateA")
    # plateB's blank, present beside plateA's images so a wrong pick would resolve.
    tifffile.imwrite(image.parent / "t00b.tif", np.full((64, 64), 60, dtype=np.uint8))
    state = _state_with_selected_block("SubtractBlank")
    state.reference_metadata_path = str(_two_dataset_table(tmp_path))

    manifest = pc.compute_scope("s", state, [], str(image), None, None)

    assert manifest["error"] is None, manifest["error"]
    detect_mat = _node_detect_mat(pc, manifest, "s", state.selected_block_id)
    assert float(detect_mat.max()) == pytest.approx(_COLONY_OVER_BLANK, abs=1e-6)
    assert float(detect_mat.min()) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------- preview: the blank's identity


def _rewrite_blank(blank, value: int) -> None:
    """Overwrite the blank's pixels and move its mtime, as a re-export would."""
    before = blank.stat()
    tifffile.imwrite(blank, np.full((64, 64), value, dtype=np.uint8))
    os.utime(blank, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


def test_overwriting_the_blank_invalidates_the_cached_preview(tmp_path, monkeypatch):
    from phenotypic._gui.builder import _preview_cache as pc

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    image = _plate_and_blank(tmp_path)
    state = _state_with_selected_block("SubtractBlank")
    state.reference_metadata_path = str(_table(tmp_path))
    first = pc.compute_scope("s", state, [], str(image), None, None)
    assert pc.compute_scope("s", state, [], str(image), None, None)["fingerprint"] == (
        first["fingerprint"]
    )

    _rewrite_blank(image.parent / "t00.tif", 100)
    second = pc.compute_scope("s", state, [], str(image), None, None)

    assert second["fingerprint"] != first["fingerprint"]
    assert second["error"] is None, second["error"]
    # Recomputed against the new blank: 220 - 100 on the colony, not 220 - 120.
    detect_mat = _node_detect_mat(pc, second, "s", state.selected_block_id)
    assert float(detect_mat.max()) == pytest.approx(120 / 255, abs=1e-6)


def test_a_missing_blank_changes_the_fingerprint_without_failing_it(tmp_path, monkeypatch):
    from phenotypic._gui.builder import _preview_cache as pc

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    image = _plate_and_blank(tmp_path)
    state = _state_with_selected_block("SubtractBlank")
    state.reference_metadata_path = str(_table(tmp_path))
    present = pc.compute_scope("s", state, [], str(image), None, None)

    (image.parent / "t00.tif").unlink()
    missing = pc.compute_scope("s", state, [], str(image), None, None)

    assert missing["fingerprint"] != present["fingerprint"]
    assert missing["error"] is not None and "t00" in missing["error"]


def test_the_blank_does_not_enter_the_fingerprint_of_a_pipeline_that_reads_none(
    tmp_path, monkeypatch
):
    """Only reference-reading pipelines key on the blank.

    For one that reads none, the root fingerprint is built from exactly the
    inputs it had before the blank was keyed: the table-less inputs plus the
    table's identity, and nothing more.
    """
    from phenotypic._gui.builder import _preview_cache as pc

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    image = _plate_and_blank(tmp_path)
    state = _state_with_selected_block("BlurGauss")
    without_table = pc.compute_scope("s", state, [], str(image), None, None)
    table = str(_table(tmp_path))
    state.reference_metadata_path = table
    first = pc.compute_scope("s", state, [], str(image), None, None)

    assert first["fingerprint_inputs"] == [
        *without_table["fingerprint_inputs"],
        rm.reference_identity(table),
    ]

    _rewrite_blank(image.parent / "t00.tif", 100)

    assert pc.compute_scope("s", state, [], str(image), None, None)["fingerprint"] == (
        first["fingerprint"]
    )


# ------------------------------------------------- preview: one parse per table


@pytest.fixture
def table_reads(monkeypatch):
    """Count how often a reference table is parsed (``ReferenceContext``'s one read)."""
    from phenotypic._core import _reference_context as reference_context

    reads: list[str] = []
    real = reference_context._read_table

    def counting(source):
        reads.append(str(source))
        return real(source)

    monkeypatch.setattr(reference_context, "_read_table", counting)
    return reads


def test_two_previews_of_an_unchanged_table_parse_it_once(tmp_path, table_reads):
    table = str(_table(tmp_path))
    image = str(tmp_path / "plates" / "t01.tif")
    for _ in range(2):
        with rm.preview_reference_context(table, image) as ctx:
            assert ctx.lookup("t01", ["BlankImage"]) == {"BlankImage": "t00"}
    assert len(table_reads) == 1


def test_previews_from_different_directories_share_one_parse(tmp_path, table_reads):
    """Each preview narrows its own root and dataset from the shared parse."""
    table = str(_two_dataset_table(tmp_path))
    with rm.preview_reference_context(table, str(tmp_path / "plateA" / "t01.tif")) as a:
        assert a.image_root == tmp_path / "plateA"
        assert a.lookup("t01", ["BlankImage"]) == {"BlankImage": "t00"}
    with rm.preview_reference_context(table, str(tmp_path / "plateB" / "t01.tif")) as b:
        assert b.image_root == tmp_path / "plateB"
        assert b.lookup("t01", ["BlankImage"]) == {"BlankImage": "t00b"}
    assert len(table_reads) == 1


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_a_changed_table_is_parsed_again(tmp_path, table_reads, change):
    """Unchanged, the parse is reused (1, 1); changed, it is redone (2)."""
    path = _table(tmp_path)
    image = str(tmp_path / "plates" / "t01.tif")
    with rm.preview_reference_context(str(path), image):
        pass
    assert len(table_reads) == 1
    with rm.preview_reference_context(str(path), image):
        pass
    assert len(table_reads) == 1
    before = path.stat()
    if change == "size":
        pd.DataFrame({"ImageName": ["t01"], "BlankImage": ["t00x"]}).to_csv(path, index=False)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    with rm.preview_reference_context(str(path), image) as ctx:
        expected = "t00x" if change == "size" else "t00"
        assert ctx.lookup("t01", ["BlankImage"]) == {"BlankImage": expected}
    assert len(table_reads) == 2
