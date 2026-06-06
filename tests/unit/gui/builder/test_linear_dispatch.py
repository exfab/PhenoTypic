"""Phase 2 linear-builder dispatcher tests."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from phenotypic.gui.builder._callbacks import _dispatch_state_update
from phenotypic.gui.builder._linear_model import ROOT_SCOPE_KEY, scope_key
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    BuilderScope,
    BuilderState,
    Edge,
    _new_block_id,
    state_to_json,
)

from .conftest import _make_op_info, _make_param


@pytest.fixture
def linear_registry(empty_registry, monkeypatch):
    """Seed a tiny registry for linear-dispatch compatibility checks."""

    empty_registry.ops.update(
        {
            "SourceOp": _make_op_info("SourceOp"),
            "OtherOp": _make_op_info("OtherOp"),
            "ConsumerOp": _make_op_info(
                "ConsumerOp",
                parameters={
                    "detector": _make_param(
                        "detector", has_default=False, is_operation=True
                    ),
                    "detectors": _make_param(
                        "detectors",
                        has_default=True,
                        is_operation=True,
                        is_list=True,
                    ),
                    "pipe": _make_param(
                        "pipe", has_default=True, is_pipeline=True
                    ),
                },
            ),
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


def test_linear_clear_list_param_deletes_aux_subtree_and_compacts(linear_registry):
    state = _state_with_chain("ConsumerOp")
    consumer = _block_by_class(state, "ConsumerOp")
    source_a = {
        "block_id": _new_block_id(),
        "class_name": "SourceOp",
        "params": {},
        "label": None,
        "nested": None,
        "collapsed": False,
        "list_slot_counts": {},
    }
    source_b = {
        "block_id": _new_block_id(),
        "class_name": "OtherOp",
        "params": {},
        "label": None,
        "nested": None,
        "collapsed": False,
        "list_slot_counts": {},
    }
    nested_dep = {
        "block_id": _new_block_id(),
        "class_name": "SourceOp",
        "params": {},
        "label": None,
        "nested": None,
        "collapsed": False,
        "list_slot_counts": {},
    }
    state["root"]["blocks"].extend([source_a, source_b, nested_dep])
    consumer["list_slot_counts"]["detectors"] = 2
    state["root"]["edges"].extend(
        [
            {
                "edge_id": _new_block_id(),
                "source_block_id": source_a["block_id"],
                "source_port": "out",
                "target_block_id": consumer["block_id"],
                "target_port": "detectors",
                "target_slot": 0,
                "kind": "aux",
            },
            {
                "edge_id": _new_block_id(),
                "source_block_id": source_b["block_id"],
                "source_port": "out",
                "target_block_id": consumer["block_id"],
                "target_port": "detectors",
                "target_slot": 1,
                "kind": "aux",
            },
            {
                "edge_id": _new_block_id(),
                "source_block_id": nested_dep["block_id"],
                "source_port": "out",
                "target_block_id": source_b["block_id"],
                "target_port": "detector",
                "target_slot": None,
                "kind": "aux",
            },
        ]
    )

    out = _dispatch_state_update(
        state,
        "linear_clear_param",
        {
            "target": _target(
                "parameter_slot",
                block_id=consumer["block_id"],
                param="detectors",
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
        and edge["target_port"] == "detectors"
    ]
    assert [edge["target_slot"] for edge in remaining] == [0]
    assert _block_by_class(out, "ConsumerOp")["list_slot_counts"]["detectors"] == 1


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


def test_linear_delete_node_confirm_reconnects_spine_and_deletes_side_values(
    linear_registry,
):
    state = _state_with_chain("SourceOp", "ConsumerOp", "OtherOp")
    source = _block_by_class(state, "SourceOp")
    consumer = _block_by_class(state, "ConsumerOp")
    other = _block_by_class(state, "OtherOp")
    aux = {
        "block_id": _new_block_id(),
        "class_name": "SourceOp",
        "params": {},
        "label": None,
        "nested": None,
        "collapsed": False,
        "list_slot_counts": {},
    }
    state["root"]["blocks"].append(aux)
    state["root"]["edges"].append(
        {
            "edge_id": _new_block_id(),
            "source_block_id": aux["block_id"],
            "source_port": "out",
            "target_block_id": consumer["block_id"],
            "target_port": "detector",
            "target_slot": None,
            "kind": "aux",
        }
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
