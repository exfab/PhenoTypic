"""Phase 2 linear-builder dispatcher tests."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from phenotypic.abc_ import ImageOperation
from phenotypic.gui._operation_registry import OperationInfo, ParamInfo
from phenotypic.gui.builder._callbacks import (
    _dispatch_state_update,
    _linear_prefix_state_for_preview,
    _linear_state_with_preview_selection,
    _state_with_issue_focus,
)
from phenotypic.gui.builder._linear_model import (
    ROOT_SCOPE_KEY,
    LinearTarget,
    scope_key,
)
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    BuilderScope,
    BuilderState,
    Edge,
    _new_block_id,
    state_from_json,
    state_to_json,
)

from .conftest import _make_op_info, _make_param


class _LinearFakeImageOperation(ImageOperation):
    """Concrete ImageOperation shell for dispatcher compatibility tests."""

    def _operate(self, image):
        return image


class _LinearSpecificImageOperation(_LinearFakeImageOperation):
    """Specific operation base used for narrow parameter accept tests."""


def _make_linear_op_info(
    cls_name: str,
    parameters: Dict[str, Any] | None = None,
    *,
    category: str = "Enhancer",
    base_cls: type[ImageOperation] = _LinearFakeImageOperation,
) -> OperationInfo:
    cls = type(
        cls_name,
        (base_cls,),
        {
            "__module__": "tests.fake",
            "_operate": _LinearFakeImageOperation._operate,
        },
    )
    return OperationInfo(
        cls=cls,
        name=cls_name,
        category=category,
        module="tests.fake",
        docstring="",
        parameters=parameters or {},
    )


@pytest.fixture
def linear_registry(empty_registry, monkeypatch):
    """Seed a tiny registry for linear-dispatch compatibility checks."""

    empty_registry.ops.update(
        {
            "SourceOp": _make_linear_op_info("SourceOp"),
            "OtherOp": _make_linear_op_info("OtherOp"),
            "ConsumerOp": _make_linear_op_info(
                "ConsumerOp",
                parameters={
                    "detector": _make_param(
                        "detector", has_default=False, is_operation=True
                    ),
                    "ops": _make_param(
                        "ops",
                        has_default=True,
                        is_operation=True,
                        is_list=True,
                    ),
                    "pipe": _make_param(
                        "pipe", has_default=True, is_pipeline=True
                    ),
                    "specific_detector": ParamInfo(
                        name="specific_detector",
                        type_hint=_LinearSpecificImageOperation,
                        default=None,
                        has_default=True,
                        is_operation=True,
                        is_pipeline=False,
                        is_optional=False,
                        is_list=False,
                    ),
                },
            ),
            "SpecificSourceOp": _make_linear_op_info(
                "SpecificSourceOp",
                base_cls=_LinearSpecificImageOperation,
            ),
            "AnalysisThing": _make_op_info("AnalysisThing", category="Filter"),
        }
    )
    monkeypatch.setattr(
        "phenotypic.gui._operation_registry.get_registry",
        lambda: empty_registry,
    )
    return empty_registry


def _block(class_name: str, block_id: str | None = None) -> BlockNode:
    return BlockNode(
        block_id=block_id or _new_block_id(),
        class_name=class_name,
        params={},
    )


def _state_with_chain(*class_names: str) -> Dict[str, Any]:
    scope = BuilderScope()
    previous = scope.blocks[0]
    for class_name in class_names:
        block = _block(class_name)
        scope.blocks.append(block)
        scope.edges.append(
            Edge(
                edge_id=_new_block_id(),
                source_block_id=previous.block_id,
                target_block_id=block.block_id,
                target_port="in",
                kind="image",
            )
        )
        previous = block
    return state_to_json(BuilderState(root=scope))


def _input_block(state: Dict[str, Any]) -> Dict[str, Any]:
    return next(
        block
        for block in state["root"]["blocks"]
        if block["class_name"] == INPUT_IMAGE_CLASS_NAME
    )


def _block_by_class(state: Dict[str, Any], class_name: str) -> Dict[str, Any]:
    return next(
        block
        for block in state["root"]["blocks"]
        if block["class_name"] == class_name
    )


def _edges(state: Dict[str, Any], *, kind: str | None = None) -> List[Dict[str, Any]]:
    edges = list(state["root"]["edges"])
    if kind is None:
        return edges
    return [edge for edge in edges if edge["kind"] == kind]


def _image_pairs(state: Dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (edge["source_block_id"], edge["target_block_id"])
        for edge in _edges(state, kind="image")
    }


def _target(
    kind: str,
    *,
    block_id: str | None = None,
    param: str | None = None,
    slot: int | None = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "scope_path": [],
        "block_id": block_id,
        "param": param,
        "slot": slot,
    }


def _block_dict(class_name: str) -> Dict[str, Any]:
    return {
        "block_id": _new_block_id(),
        "class_name": class_name,
        "params": {},
        "label": None,
        "nested": None,
        "collapsed": False,
        "list_slot_counts": {},
    }


def _aux_edge_dict(
    source_id: str,
    target_id: str,
    param: str,
    *,
    slot: int | None = None,
) -> Dict[str, Any]:
    return {
        "edge_id": _new_block_id(),
        "source_block_id": source_id,
        "source_port": "out",
        "target_block_id": target_id,
        "target_port": param,
        "target_slot": slot,
        "kind": "aux",
    }


def _image_edge_dict(source_id: str, target_id: str) -> Dict[str, Any]:
    return {
        "edge_id": _new_block_id(),
        "source_block_id": source_id,
        "source_port": "out",
        "target_block_id": target_id,
        "target_port": "in",
        "target_slot": None,
        "kind": "image",
    }


def test_target_select_stores_green_target_and_open_menu(linear_registry):
    state = _state_with_chain()
    input_id = _input_block(state)["block_id"]

    out = _dispatch_state_update(
        state,
        "target_select",
        {"target": _target("image_output", block_id=input_id)},
    )

    assert out["selected_targets_by_scope"][ROOT_SCOPE_KEY] == _target(
        "image_output", block_id=input_id
    )
    assert out["open_port_menu"] == _target("image_output", block_id=input_id)


def test_target_menu_close_preserves_selected_target(linear_registry):
    state = _state_with_chain()
    input_id = _input_block(state)["block_id"]
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target("image_output", block_id=input_id)
    }
    state["open_port_menu"] = _target("image_output", block_id=input_id)

    out = _dispatch_state_update(state, "target_menu_close", {})

    assert out["open_port_menu"] is None
    assert out["selected_targets_by_scope"][ROOT_SCOPE_KEY] == _target(
        "image_output", block_id=input_id
    )


def test_linear_palette_add_continuation_appends_after_terminal(linear_registry):
    state = _state_with_chain()
    input_id = _input_block(state)["block_id"]

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "SourceOp"},
    )

    source = _block_by_class(out, "SourceOp")
    assert (input_id, source["block_id"]) in _image_pairs(out)
    assert out["selected_block_id"] == source["block_id"]
    assert out["selected_edge_id"] is None
    assert out["selected_targets_by_scope"][ROOT_SCOPE_KEY] == _target("continuation")
    assert out["open_port_menu"] is None


def test_linear_palette_add_image_output_inserts_between_nodes(linear_registry):
    state = _state_with_chain("SourceOp", "OtherOp")
    source = _block_by_class(state, "SourceOp")
    other = _block_by_class(state, "OtherOp")
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target("image_output", block_id=source["block_id"])
    }

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "ConsumerOp"},
    )

    inserted = _block_by_class(out, "ConsumerOp")
    assert (source["block_id"], other["block_id"]) not in _image_pairs(out)
    assert (source["block_id"], inserted["block_id"]) in _image_pairs(out)
    assert (inserted["block_id"], other["block_id"]) in _image_pairs(out)


def test_linear_palette_add_image_input_inserts_before_node(linear_registry):
    state = _state_with_chain("SourceOp", "OtherOp")
    source = _block_by_class(state, "SourceOp")
    other = _block_by_class(state, "OtherOp")
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target("image_input", block_id=other["block_id"])
    }

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "ConsumerOp"},
    )

    inserted = _block_by_class(out, "ConsumerOp")
    assert (source["block_id"], other["block_id"]) not in _image_pairs(out)
    assert (source["block_id"], inserted["block_id"]) in _image_pairs(out)
    assert (inserted["block_id"], other["block_id"]) in _image_pairs(out)


def test_linear_palette_add_parameter_fills_operation_target(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target(
            "parameter", block_id=consumer["block_id"], param="detector"
        )
    }

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "SourceOp"},
    )

    source = _block_by_class(out, "SourceOp")
    assert {
        "source_block_id": source["block_id"],
        "target_block_id": consumer["block_id"],
        "target_port": "detector",
        "target_slot": None,
        "kind": "aux",
    }.items() <= _edges(out, kind="aux")[0].items()
    assert out["selected_block_id"] == consumer["block_id"]
    assert out["selected_targets_by_scope"][ROOT_SCOPE_KEY] == _target("continuation")


def test_linear_palette_add_rejects_incompatible_parameter_fill(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    selected_target = _target(
        "parameter", block_id=consumer["block_id"], param="detector"
    )
    state["selected_targets_by_scope"] = {ROOT_SCOPE_KEY: selected_target}

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": PIPELINE_CLASS_NAME},
    )

    assert len(_edges(out, kind="aux")) == 0
    assert out["selected_targets_by_scope"][ROOT_SCOPE_KEY] == selected_target
    assert out["toast_queue"][-1]["kind"] == "warning"


def test_linear_palette_add_rejects_non_operation_source(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    selected_target = _target(
        "parameter", block_id=consumer["block_id"], param="detector"
    )
    state["selected_targets_by_scope"] = {ROOT_SCOPE_KEY: selected_target}

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "AnalysisThing"},
    )

    assert len(_edges(out, kind="aux")) == 0
    assert all(block["class_name"] != "AnalysisThing" for block in out["root"]["blocks"])
    assert out["toast_queue"][-1]["kind"] == "warning"


def test_linear_palette_add_rejects_operation_outside_specific_param_type(
    linear_registry,
):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    selected_target = _target(
        "parameter", block_id=consumer["block_id"], param="specific_detector"
    )
    state["selected_targets_by_scope"] = {ROOT_SCOPE_KEY: selected_target}

    rejected = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "OtherOp"},
    )

    assert len(_edges(rejected, kind="aux")) == 0
    assert all(
        block["class_name"] != "OtherOp"
        for block in rejected["root"]["blocks"]
        if block["block_id"] != consumer["block_id"]
    )
    assert rejected["toast_queue"][-1]["kind"] == "warning"

    accepted = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "SpecificSourceOp"},
    )
    assert _block_by_class(accepted, "SpecificSourceOp")
    assert len(_edges(accepted, kind="aux")) == 1


def test_linear_palette_add_rejects_unknown_parameter_source(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target(
            "parameter", block_id=consumer["block_id"], param="detector"
        )
    }

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "UnknownOp"},
    )

    assert len(_edges(out, kind="aux")) == 0
    assert all(block["class_name"] != "UnknownOp" for block in out["root"]["blocks"])
    assert out["toast_queue"][-1]["kind"] == "warning"


def test_linear_palette_add_blocks_unsupported_non_linear_scope(linear_registry):
    state = _state_with_chain("SourceOp")
    input_id = _input_block(state)["block_id"]
    fork = _block_dict("OtherOp")
    state["root"]["blocks"].append(fork)
    state["root"]["edges"].append(_image_edge_dict(input_id, fork["block_id"]))
    before_blocks = [block["block_id"] for block in state["root"]["blocks"]]

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "ConsumerOp"},
    )

    assert [block["block_id"] for block in out["root"]["blocks"]] == before_blocks
    assert out["toast_queue"][-1]["kind"] == "warning"
    assert "Linear editing is paused" in out["toast_queue"][-1]["text"]


def test_linear_palette_add_blocks_when_nested_scope_is_unsupported(
    linear_registry,
):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    pipeline = _block_dict(PIPELINE_CLASS_NAME)
    nested_input = _block_dict(INPUT_IMAGE_CLASS_NAME)
    nested_a = _block_dict("SourceOp")
    nested_b = _block_dict("OtherOp")
    pipeline["nested"] = {
        "blocks": [nested_input, nested_a, nested_b],
        "edges": [
            _image_edge_dict(nested_input["block_id"], nested_a["block_id"]),
            _image_edge_dict(nested_input["block_id"], nested_b["block_id"]),
        ],
        "name": "Pipeline",
        "desc": "",
        "nrows": None,
        "ncols": None,
    }
    state["root"]["blocks"].append(pipeline)
    state["root"]["edges"].append(
        _aux_edge_dict(pipeline["block_id"], consumer["block_id"], "pipe")
    )
    before_root_blocks = [block["block_id"] for block in state["root"]["blocks"]]

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "SourceOp"},
    )

    assert [block["block_id"] for block in out["root"]["blocks"]] == before_root_blocks
    assert out["toast_queue"][-1]["kind"] == "warning"
    assert "Linear editing is paused" in out["toast_queue"][-1]["text"]


def test_linear_palette_add_replaces_scalar_param_and_deletes_old_value(
    linear_registry,
):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    old_source = _block_dict("OtherOp")
    state["root"]["blocks"].append(old_source)
    state["root"]["edges"].append(
        _aux_edge_dict(old_source["block_id"], consumer["block_id"], "detector")
    )
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target(
            "parameter", block_id=consumer["block_id"], param="detector"
        )
    }

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "SourceOp"},
    )

    block_ids = {block["block_id"] for block in out["root"]["blocks"]}
    assert old_source["block_id"] not in block_ids
    aux_edges = [
        edge
        for edge in _edges(out, kind="aux")
        if edge["target_block_id"] == consumer["block_id"]
        and edge["target_port"] == "detector"
    ]
    assert len(aux_edges) == 1
    new_source = next(
        block
        for block in out["root"]["blocks"]
        if block["block_id"] == aux_edges[0]["source_block_id"]
    )
    assert new_source["class_name"] == "SourceOp"


def test_linear_clear_scalar_param_deletes_value(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    source = _block_dict("SourceOp")
    state["root"]["blocks"].append(source)
    state["root"]["edges"].append(
        _aux_edge_dict(source["block_id"], consumer["block_id"], "detector")
    )

    out = _dispatch_state_update(
        state,
        "linear_clear_param",
        {
            "target": _target(
                "parameter", block_id=consumer["block_id"], param="detector"
            )
        },
    )

    assert source["block_id"] not in {block["block_id"] for block in out["root"]["blocks"]}
    assert len(_edges(out, kind="aux")) == 0


def test_linear_palette_add_replaces_list_slot_without_gaps(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    old_source = _block_dict("OtherOp")
    keep_source = _block_dict("SourceOp")
    state["root"]["blocks"].extend([old_source, keep_source])
    consumer["list_slot_counts"]["ops"] = 2
    state["root"]["edges"].extend(
        [
            _aux_edge_dict(
                old_source["block_id"],
                consumer["block_id"],
                "ops",
                slot=0,
            ),
            _aux_edge_dict(
                keep_source["block_id"],
                consumer["block_id"],
                "ops",
                slot=1,
            ),
        ]
    )
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target(
            "parameter_slot",
            block_id=consumer["block_id"],
            param="ops",
            slot=0,
        )
    }

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": "SourceOp"},
    )

    assert old_source["block_id"] not in {
        block["block_id"] for block in out["root"]["blocks"]
    }
    list_edges = [
        edge
        for edge in _edges(out, kind="aux")
        if edge["target_block_id"] == consumer["block_id"]
        and edge["target_port"] == "ops"
    ]
    assert sorted(edge["target_slot"] for edge in list_edges) == [0, 1]


def test_linear_palette_add_pipeline_parameter_drills_into_nested_scope(
    linear_registry,
):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    state["selected_targets_by_scope"] = {
        ROOT_SCOPE_KEY: _target(
            "parameter", block_id=consumer["block_id"], param="pipe"
        )
    }

    out = _dispatch_state_update(
        state,
        "linear_palette_add",
        {"class_name": PIPELINE_CLASS_NAME},
    )

    pipeline = _block_by_class(out, PIPELINE_CLASS_NAME)
    assert pipeline["nested"]["blocks"][0]["class_name"] == INPUT_IMAGE_CLASS_NAME
    assert out["breadcrumb"] == [pipeline["block_id"]]
    assert out["selected_targets_by_scope"][scope_key(out["breadcrumb"])] == _target(
        "continuation"
    ) | {"scope_path": [pipeline["block_id"]]}


def test_linear_drill_param_pipeline_opens_existing_aux_pipeline(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    pipeline = _block_dict(PIPELINE_CLASS_NAME)
    pipeline["nested"] = {
        "blocks": [],
        "edges": [],
        "name": "Pipeline",
        "desc": "",
        "nrows": None,
        "ncols": None,
    }
    state["root"]["blocks"].append(pipeline)
    state["root"]["edges"].append(
        _aux_edge_dict(pipeline["block_id"], consumer["block_id"], "pipe")
    )

    out = _dispatch_state_update(
        state,
        "linear_drill_param_pipeline",
        {"source_block_id": pipeline["block_id"]},
    )

    assert out["breadcrumb"] == [pipeline["block_id"]]
    assert out["selected_targets_by_scope"][scope_key(out["breadcrumb"])] == _target(
        "continuation"
    ) | {"scope_path": [pipeline["block_id"]]}


def test_issue_focus_drills_to_nested_scope_and_selects_block(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    pipeline = _block_dict(PIPELINE_CLASS_NAME)
    nested_input = _block_dict(INPUT_IMAGE_CLASS_NAME)
    nested_source = _block_dict("SourceOp")
    pipeline["nested"] = {
        "blocks": [nested_input, nested_source],
        "edges": [
            _image_edge_dict(
                nested_input["block_id"],
                nested_source["block_id"],
            )
        ],
        "name": "Pipeline",
        "desc": "",
        "nrows": None,
        "ncols": None,
    }
    state["root"]["blocks"].append(pipeline)
    state["root"]["edges"].append(
        _aux_edge_dict(pipeline["block_id"], consumer["block_id"], "pipe")
    )

    out = _state_with_issue_focus(
        state,
        {
            "kind": "issue_focus",
            "block_id": nested_source["block_id"],
            "target_breadcrumb": [pipeline["block_id"]],
        },
    )

    assert out["breadcrumb"] == [pipeline["block_id"]]
    assert out["selected_block_id"] == nested_source["block_id"]
    assert out["selected_edge_id"] is None


def test_linear_select_aux_value_selects_source_block(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    source = _block_dict("SourceOp")
    state["root"]["blocks"].append(source)
    state["root"]["edges"].append(
        _aux_edge_dict(source["block_id"], consumer["block_id"], "detector")
    )

    out = _dispatch_state_update(
        state,
        "linear_select_aux_value",
        {"source_block_id": source["block_id"]},
    )

    assert out["selected_block_id"] == source["block_id"]
    assert out["selected_edge_id"] is None


def test_linear_clear_list_param_deletes_aux_subtree_and_compacts(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    source_a = _block_dict("SourceOp")
    source_b = _block_dict("OtherOp")
    nested_dep = _block_dict("SourceOp")
    state["root"]["blocks"].extend([source_a, source_b, nested_dep])
    consumer["list_slot_counts"]["ops"] = 2
    state["root"]["edges"].extend(
        [
            _aux_edge_dict(
                source_a["block_id"],
                consumer["block_id"],
                "ops",
                slot=0,
            ),
            _aux_edge_dict(
                source_b["block_id"],
                consumer["block_id"],
                "ops",
                slot=1,
            ),
            _aux_edge_dict(nested_dep["block_id"], source_b["block_id"], "detector"),
        ]
    )

    out = _dispatch_state_update(
        state,
        "linear_clear_param",
        {
            "target": _target(
                "parameter_slot",
                block_id=consumer["block_id"],
                param="ops",
                slot=1,
            )
        },
    )

    block_ids = {block["block_id"] for block in out["root"]["blocks"]}
    assert source_b["block_id"] not in block_ids
    assert nested_dep["block_id"] not in block_ids
    remaining = [
        edge
        for edge in _edges(out, kind="aux")
        if edge["target_block_id"] == consumer["block_id"]
        and edge["target_port"] == "ops"
    ]
    assert [edge["target_slot"] for edge in remaining] == [0]
    assert _block_by_class(out, "ConsumerOp")["list_slot_counts"]["ops"] == 1


def test_linear_clear_malformed_list_slot_is_noop(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    source_a = _block_dict("SourceOp")
    source_b = _block_dict("OtherOp")
    state["root"]["blocks"].extend([source_a, source_b])
    consumer["list_slot_counts"]["ops"] = 2
    state["root"]["edges"].extend(
        [
            _aux_edge_dict(
                source_a["block_id"],
                consumer["block_id"],
                "ops",
                slot=0,
            ),
            _aux_edge_dict(
                source_b["block_id"],
                consumer["block_id"],
                "ops",
                slot=1,
            ),
        ]
    )

    out = _dispatch_state_update(
        state,
        "linear_clear_param",
        {
            "target": {
                "kind": "parameter_slot",
                "scope_path": [],
                "block_id": consumer["block_id"],
                "param": "ops",
            }
        },
    )

    assert {source_a["block_id"], source_b["block_id"]} <= {
        block["block_id"] for block in out["root"]["blocks"]
    }
    assert len(_edges(out, kind="aux")) == 2


def test_linear_clear_shared_aux_source_is_paused_as_unsupported(
    linear_registry,
):
    state = _state_with_chain("ConsumerOp", "OtherOp")
    consumer = _block_by_class(state, "ConsumerOp")
    other = _block_by_class(state, "OtherOp")
    shared = _block_dict("SourceOp")
    state["root"]["blocks"].append(shared)
    state["root"]["edges"].extend(
        [
            _aux_edge_dict(shared["block_id"], consumer["block_id"], "detector"),
            _aux_edge_dict(shared["block_id"], other["block_id"], "detector"),
        ]
    )

    out = _dispatch_state_update(
        state,
        "linear_clear_param",
        {
            "target": _target(
                "parameter", block_id=consumer["block_id"], param="detector"
            )
        },
    )

    assert shared["block_id"] in {block["block_id"] for block in out["root"]["blocks"]}
    remaining_aux = _edges(out, kind="aux")
    assert len(remaining_aux) == 2
    assert out["toast_queue"][-1]["kind"] == "warning"


def test_linear_node_move_right_swaps_adjacent_spine_nodes(linear_registry):
    state = _state_with_chain("SourceOp", "ConsumerOp", "OtherOp")
    source = _block_by_class(state, "SourceOp")
    consumer = _block_by_class(state, "ConsumerOp")
    other = _block_by_class(state, "OtherOp")

    out = _dispatch_state_update(
        state,
        "linear_node_move",
        {"block_id": consumer["block_id"], "direction": "right"},
    )

    assert (source["block_id"], other["block_id"]) in _image_pairs(out)
    assert (other["block_id"], consumer["block_id"]) in _image_pairs(out)


def test_linear_node_move_left_swaps_adjacent_spine_nodes(linear_registry):
    state = _state_with_chain("SourceOp", "ConsumerOp", "OtherOp")
    source = _block_by_class(state, "SourceOp")
    consumer = _block_by_class(state, "ConsumerOp")
    other = _block_by_class(state, "OtherOp")

    out = _dispatch_state_update(
        state,
        "linear_node_move",
        {"block_id": consumer["block_id"], "direction": "left"},
    )

    input_id = _input_block(state)["block_id"]
    assert (input_id, consumer["block_id"]) in _image_pairs(out)
    assert (consumer["block_id"], source["block_id"]) in _image_pairs(out)
    assert (source["block_id"], other["block_id"]) in _image_pairs(out)


def test_linear_delete_node_confirm_reconnects_spine_and_deletes_side_values(
    linear_registry,
):
    state = _state_with_chain("SourceOp", "ConsumerOp", "OtherOp")
    source = _block_by_class(state, "SourceOp")
    consumer = _block_by_class(state, "ConsumerOp")
    other = _block_by_class(state, "OtherOp")
    aux = _block_dict("SourceOp")
    state["root"]["blocks"].append(aux)
    state["root"]["edges"].append(
        _aux_edge_dict(aux["block_id"], consumer["block_id"], "detector")
    )
    state["selected_block_id"] = consumer["block_id"]

    out = _dispatch_state_update(
        state,
        "linear_delete_node_confirm",
        {"block_id": consumer["block_id"]},
    )

    block_ids = {block["block_id"] for block in out["root"]["blocks"]}
    assert consumer["block_id"] not in block_ids
    assert aux["block_id"] not in block_ids
    assert (source["block_id"], other["block_id"]) in _image_pairs(out)
    assert out["selected_block_id"] is None


def test_linear_delete_node_request_sets_pending_confirmation(linear_registry):
    state = _state_with_chain("SourceOp", "ConsumerOp")
    source = _block_by_class(state, "SourceOp")
    consumer = _block_by_class(state, "ConsumerOp")

    out = _dispatch_state_update(
        state,
        "linear_delete_node_request",
        {"block_id": source["block_id"]},
    )

    assert source["block_id"] in {
        block["block_id"] for block in out["root"]["blocks"]
    }
    assert out["pending_delete_block_id"] == f"linear_node:{source['block_id']}"
    assert (_input_block(state)["block_id"], source["block_id"]) in _image_pairs(out)
    assert (source["block_id"], consumer["block_id"]) in _image_pairs(out)


def test_linear_clear_pipeline_parameter_requires_confirm(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    pipeline = _block_dict(PIPELINE_CLASS_NAME)
    pipeline["nested"] = {
        "blocks": [],
        "edges": [],
        "name": "Pipeline",
        "desc": "",
        "nrows": None,
        "ncols": None,
    }
    state["root"]["blocks"].append(pipeline)
    state["root"]["edges"].append(
        _aux_edge_dict(pipeline["block_id"], consumer["block_id"], "pipe")
    )
    target = _target("parameter", block_id=consumer["block_id"], param="pipe")

    pending = _dispatch_state_update(
        state,
        "linear_clear_param",
        {"target": target},
    )

    assert pipeline["block_id"] in {
        block["block_id"] for block in pending["root"]["blocks"]
    }
    assert pending["pending_delete_block_id"].startswith("linear_clear:")

    confirmed = _dispatch_state_update(
        pending,
        "linear_clear_param_confirm",
        {"target": target},
    )

    assert pipeline["block_id"] not in {
        block["block_id"] for block in confirmed["root"]["blocks"]
    }
    assert len(_edges(confirmed, kind="aux")) == 0
    assert confirmed["pending_delete_block_id"] is None


def test_linear_prefix_preview_state_keeps_prefix_and_aux_dependency(
    linear_registry,
):
    state_dict = _state_with_chain("ConsumerOp", "OtherOp")
    consumer = _block_by_class(state_dict, "ConsumerOp")
    other = _block_by_class(state_dict, "OtherOp")
    aux = _block_dict("SourceOp")
    state_dict["root"]["blocks"].append(aux)
    state_dict["root"]["edges"].append(
        _aux_edge_dict(aux["block_id"], consumer["block_id"], "detector")
    )
    state = state_from_json(state_dict)

    prefix = _linear_prefix_state_for_preview(
        state,
        LinearTarget(
            kind="image_output",
            scope_path=[],
            block_id=consumer["block_id"],
        ),
    )

    block_ids = {block.block_id for block in prefix.root.blocks}
    assert consumer["block_id"] in block_ids
    assert aux["block_id"] in block_ids
    assert other["block_id"] not in block_ids
    assert all(
        edge.source_block_id in block_ids and edge.target_block_id in block_ids
        for edge in prefix.root.edges
    )


def test_linear_prefix_preview_state_continuation_uses_terminal(linear_registry):
    state_dict = _state_with_chain("SourceOp", "ConsumerOp")
    terminal = _block_by_class(state_dict, "ConsumerOp")
    state = state_from_json(state_dict)

    prefix = _linear_prefix_state_for_preview(
        state,
        LinearTarget(kind="continuation", scope_path=[]),
    )

    assert prefix.selected_block_id == terminal["block_id"]
    assert {block.class_name for block in prefix.root.blocks} == {
        INPUT_IMAGE_CLASS_NAME,
        "SourceOp",
        "ConsumerOp",
    }


def test_linear_preview_selection_updates_visible_selected_block(linear_registry):
    state_dict = _state_with_chain("SourceOp", "ConsumerOp")
    source = _block_by_class(state_dict, "SourceOp")
    state_dict["selected_block_id"] = _block_by_class(state_dict, "ConsumerOp")[
        "block_id"
    ]
    state_dict["open_port_menu"] = _target(
        "image_output", block_id=source["block_id"]
    )

    out = _linear_state_with_preview_selection(
        state_dict,
        _target("image_output", block_id=source["block_id"]),
    )

    assert out["selected_block_id"] == source["block_id"]
    assert out["selected_edge_id"] is None
    assert out["open_port_menu"] is None


def test_linear_preview_selection_continuation_selects_terminal(linear_registry):
    state_dict = _state_with_chain("SourceOp", "ConsumerOp")
    terminal = _block_by_class(state_dict, "ConsumerOp")
    state_dict["selected_block_id"] = _input_block(state_dict)["block_id"]
    state_dict["open_port_menu"] = _target("continuation")

    out = _linear_state_with_preview_selection(
        state_dict,
        _target("continuation"),
    )

    assert out["selected_block_id"] == terminal["block_id"]
    assert out["selected_edge_id"] is None
    assert out["open_port_menu"] is None
