"""Unit tests for the fixed linear builder port-map model."""

from __future__ import annotations

from typing import Any

from phenotypic.gui.builder._state import (
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    state_from_json,
    state_to_json,
)


def _block(class_name: str) -> BlockNode:
    return BlockNode(
        block_id=_new_block_id(),
        class_name=class_name,
        params={},
    )


def _image_edge(source: BlockNode, target: BlockNode) -> Edge:
    return Edge(
        edge_id=_new_block_id(),
        source_block_id=source.block_id,
        target_block_id=target.block_id,
        target_port="in",
        kind="image",
    )


def _aux_edge(source: BlockNode, target: BlockNode, param: str, slot: int | None = None) -> Edge:
    return Edge(
        edge_id=_new_block_id(),
        source_block_id=source.block_id,
        target_block_id=target.block_id,
        target_port=param,
        target_slot=slot,
        kind="aux",
    )


def test_dag_state_round_trips_linear_target_fields() -> None:
    """Linear target state is builder-only metadata and survives store JSON."""

    from phenotypic.gui.builder._linear_model import (
        default_continuation_target,
        scope_key,
        target_to_dict,
    )

    key = scope_key([])
    target = default_continuation_target([])
    state = _DagBuilderState(
        selected_targets_by_scope={key: target_to_dict(target)},
        open_port_menu=target_to_dict(target),
    )

    round_trip = state_from_json(state_to_json(state))

    assert round_trip.selected_targets_by_scope == {
        key: {
            "kind": "continuation",
            "scope_path": [],
            "block_id": None,
            "param": None,
            "slot": None,
        }
    }
    assert round_trip.open_port_menu == round_trip.selected_targets_by_scope[key]


def test_invalid_target_falls_back_to_current_scope_continuation() -> None:
    """A stale selected target resolves to the current scope continuation."""

    from phenotypic.gui.builder._linear_model import (
        resolve_selected_target,
        scope_key,
    )

    state = _DagBuilderState(
        selected_targets_by_scope={
            scope_key([]): {
                "kind": "image_output",
                "scope_path": [],
                "block_id": "missing",
                "param": None,
                "slot": None,
            }
        }
    )

    resolved = resolve_selected_target(state)

    assert resolved.kind == "continuation"
    assert resolved.scope_path == []


def test_linear_scope_derives_unique_spine_and_ignores_owned_aux_blocks() -> None:
    """A clean DAG with side-loaded aux blocks derives one visible spine."""

    from phenotypic.gui.builder._linear_model import derive_linear_scope

    scope = _DagBuilderScope()
    input_block = scope.blocks[0]
    blur = _block("GaussianBlur")
    consumer = _block("FilamentousFungiDetector")
    aux = _block("OtsuDetector")
    scope.blocks.extend([blur, consumer, aux])
    scope.edges.extend(
        [
            _image_edge(input_block, blur),
            _image_edge(blur, consumer),
            _aux_edge(aux, consumer, "inoculum_detector"),
        ]
    )

    model = derive_linear_scope(scope, scope_path=[])

    assert model.unsupported is None
    assert [block.class_name for block in model.spine_blocks] == [
        "InputImage",
        "GaussianBlur",
        "FilamentousFungiDetector",
    ]
    assert model.terminal_block.block_id == consumer.block_id


def test_linear_scope_classifies_image_fork_as_unsupported() -> None:
    """Forked image flow is rejected before rendering the linear editor."""

    from phenotypic.gui.builder._linear_model import derive_linear_scope

    scope = _DagBuilderScope()
    input_block = scope.blocks[0]
    left = _block("GaussianBlur")
    right = _block("OtsuDetector")
    scope.blocks.extend([left, right])
    scope.edges.extend([_image_edge(input_block, left), _image_edge(input_block, right)])

    model = derive_linear_scope(scope, scope_path=[])

    assert model.unsupported is not None
    assert model.unsupported.reason == "image_fork"


def test_compact_list_aux_slots_removes_empty_gaps() -> None:
    """List aux slots are renumbered contiguously after a removal."""

    from phenotypic.gui.builder._linear_model import compact_list_aux_slots

    scope = _DagBuilderScope()
    consumer = _block("ConsumerOp")
    source_a = _block("SourceA")
    source_b = _block("SourceB")
    scope.blocks.extend([consumer, source_a, source_b])
    first = _aux_edge(source_a, consumer, "steps", 0)
    third = _aux_edge(source_b, consumer, "steps", 2)
    scope.edges.extend([first, third])
    consumer.list_slot_counts["steps"] = 3

    compact_list_aux_slots(scope, consumer.block_id, "steps")

    slots = [
        edge.target_slot
        for edge in scope.edges
        if edge.target_block_id == consumer.block_id and edge.target_port == "steps"
    ]
    assert slots == [0, 1]
    assert consumer.list_slot_counts["steps"] == 2


def test_state_replacement_payload_uses_dag_conversion(monkeypatch: Any) -> None:
    """Loading JSON/prefabs replaces the builder with a DAG state."""

    import phenotypic.gui.builder._callbacks as callbacks

    pipeline = object()
    dag_state = _DagBuilderState()

    def fake_from_pipeline_dag(seen_pipeline: object) -> _DagBuilderState:
        assert seen_pipeline is pipeline
        return dag_state

    def legacy_from_pipeline_must_not_run(seen_pipeline: object) -> None:
        raise AssertionError("legacy from_pipeline should not handle loaded pipelines")

    def fake_render_views(seen_state: _DagBuilderState) -> tuple[list[Any], str, str]:
        assert seen_state is dag_state
        return [], "[]", "inspector"

    monkeypatch.setattr(callbacks, "from_pipeline_dag", fake_from_pipeline_dag, raising=False)
    monkeypatch.setattr(callbacks, "from_pipeline", legacy_from_pipeline_must_not_run)
    monkeypatch.setattr(callbacks, "_render_views", fake_render_views)

    state_dict, breadcrumb, canvas_elements, inspector = callbacks._state_replacement_payload(
        pipeline
    )

    assert state_dict["_schema"] == "dag"
    assert breadcrumb == []
    assert canvas_elements == "[]"
    assert inspector == "inspector"
