"""Unit tests for the fixed linear builder port-map model."""

from __future__ import annotations

from typing import Any

from phenotypic.gui.builder._state import (
    PIPELINE_CLASS_NAME,
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


def test_nested_scope_target_resolves_without_legacy_current_scope() -> None:
    """Target fallback works inside DAG breadcrumbs."""

    from phenotypic.gui.builder._linear_model import (
        resolve_selected_target,
        scope_key,
    )

    nested = _DagBuilderScope()
    nested_input = nested.blocks[0]
    nested_blur = _block("BlurGauss")
    nested.blocks.append(nested_blur)
    nested.edges.append(_image_edge(nested_input, nested_blur))
    container = BlockNode(
        block_id=_new_block_id(),
        class_name=PIPELINE_CLASS_NAME,
        params={},
        nested=nested,
    )
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container]),
        breadcrumb=[container.block_id],
        selected_targets_by_scope={
            scope_key([container.block_id]): {
                "kind": "image_output",
                "scope_path": [container.block_id],
                "block_id": nested_blur.block_id,
                "param": None,
                "slot": None,
            }
        },
    )

    resolved = resolve_selected_target(state)

    assert resolved.kind == "image_output"
    assert resolved.block_id == nested_blur.block_id
    assert resolved.scope_path == [container.block_id]


def test_linear_scope_derives_unique_spine_and_ignores_owned_aux_blocks() -> None:
    """A clean DAG with side-loaded aux blocks derives one visible spine."""

    from phenotypic.gui.builder._linear_model import derive_linear_scope

    scope = _DagBuilderScope()
    input_block = scope.blocks[0]
    blur = _block("BlurGauss")
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
        "BlurGauss",
        "FilamentousFungiDetector",
    ]
    assert model.terminal_block.block_id == consumer.block_id


def test_linear_scope_classifies_image_fork_as_unsupported() -> None:
    """Forked image flow is rejected before rendering the linear editor."""

    from phenotypic.gui.builder._linear_model import derive_linear_scope

    scope = _DagBuilderScope()
    input_block = scope.blocks[0]
    left = _block("BlurGauss")
    right = _block("OtsuDetector")
    scope.blocks.extend([left, right])
    scope.edges.extend([_image_edge(input_block, left), _image_edge(input_block, right)])

    model = derive_linear_scope(scope, scope_path=[])

    assert model.unsupported is not None
    assert model.unsupported.reason == "image_fork"


def test_linear_scope_classifies_aux_edge_into_input_as_unsupported() -> None:
    """InputImage may not be the consumer for an aux side value."""

    from phenotypic.gui.builder._linear_model import derive_linear_scope

    scope = _DagBuilderScope()
    input_block = scope.blocks[0]
    aux = _block("OtsuDetector")
    scope.blocks.append(aux)
    scope.edges.append(_aux_edge(aux, input_block, "inoculum_detector"))

    model = derive_linear_scope(scope, scope_path=[])

    assert model.unsupported is not None
    assert model.unsupported.reason == "input_as_aux_target"


def test_linear_scope_tracks_unknown_classes_without_dropping_renderable_nodes(
    empty_registry: Any,
    monkeypatch: Any,
) -> None:
    """Unknown classes stay renderable but are tracked for limited editing."""

    from phenotypic.gui.builder._linear_model import derive_linear_scope

    monkeypatch.setattr(
        "phenotypic.gui._operation_registry.get_registry",
        lambda: empty_registry,
    )
    scope = _DagBuilderScope()
    input_block = scope.blocks[0]
    unknown = _block("RemovedOperation")
    scope.blocks.append(unknown)
    scope.edges.append(_image_edge(input_block, unknown))

    model = derive_linear_scope(scope, scope_path=[])

    assert model.unsupported is None
    assert model.unknown_block_ids == {unknown.block_id}
    assert [block.class_name for block in model.spine_blocks] == [
        "InputImage",
        "RemovedOperation",
    ]


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

    def fake_render_views(seen_state: _DagBuilderState) -> tuple[list[Any], str, str]:
        assert seen_state is dag_state
        return [], "[]", "inspector"

    monkeypatch.setattr(callbacks, "from_pipeline_dag", fake_from_pipeline_dag, raising=False)
    monkeypatch.setattr(callbacks, "_render_views", fake_render_views)

    state_dict, breadcrumb, canvas_elements, inspector = callbacks._state_replacement_payload(
        pipeline
    )

    assert state_dict["_schema"] == "dag"
    assert breadcrumb == []
    assert canvas_elements == "[]"
    assert inspector == "inspector"
