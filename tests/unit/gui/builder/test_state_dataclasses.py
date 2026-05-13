"""Round-trip + idempotency tests for the DAG state dataclasses.

These tests exercise the new schema introduced in Phase 1 of the
Pipeline Builder DAG redesign (see
``docs/superpowers/specs/2026-05-12-builder-dag-redesign-design.md``).
The DAG types are imported under their stable underscore-prefixed
names (``_DagBuilderScope`` / ``_DagBuilderState``) so the tests work
regardless of whether the ``PHENOTYPIC_GUI_DAG`` feature flag is on
during the test run.
"""

from __future__ import annotations

import pytest

from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _LegacyBuilderScope,
    _LegacyBuilderState,
    _LegacyStepNode,
    _new_block_id,
    _seed_input_image,
    state_from_json,
    state_to_json,
)


# ---------------------------------------------------------------------------
# _seed_input_image idempotency + auto-seed
# ---------------------------------------------------------------------------


def test_seed_input_image_idempotent() -> None:
    """Calling :func:`_seed_input_image` repeatedly leaves exactly one block."""

    scope = _DagBuilderScope()
    for _ in range(3):
        _seed_input_image(scope)
    input_blocks = [
        b for b in scope.blocks if b.class_name == INPUT_IMAGE_CLASS_NAME
    ]
    assert len(input_blocks) == 1


def test_seed_input_image_on_loaded_state() -> None:
    """Seeding a fresh empty-blocks scope inserts an ``InputImage`` at index 0."""

    # Construct without going through __post_init__'s seed (use object.__new__).
    # Simpler: construct the scope and then forcibly wipe its blocks, then
    # call _seed_input_image directly.
    scope = _DagBuilderScope()
    scope.blocks.clear()
    assert scope.blocks == []
    _seed_input_image(scope)
    assert len(scope.blocks) == 1
    assert scope.blocks[0].class_name == INPUT_IMAGE_CLASS_NAME


def test_block_id_is_32_char_hex() -> None:
    """``_new_block_id`` returns a 32-char lowercase hex string."""

    block_id = _new_block_id()
    assert len(block_id) == 32
    assert block_id == block_id.lower()
    int(block_id, 16)  # raises ValueError if not hex


def test_builderscope_postinit_seeds_input_image() -> None:
    """:meth:`_DagBuilderScope.__post_init__` auto-seeds an ``InputImage``."""

    scope = _DagBuilderScope()
    assert len(scope.blocks) == 1
    assert scope.blocks[0].class_name == INPUT_IMAGE_CLASS_NAME
    assert scope.blocks[0].params == {}


def test_builderscope_postinit_does_not_double_seed() -> None:
    """An existing ``InputImage`` block survives ``__post_init__`` unchanged."""

    existing_id = _new_block_id()
    scope = _DagBuilderScope(
        blocks=[
            BlockNode(
                block_id=existing_id,
                class_name=INPUT_IMAGE_CLASS_NAME,
                params={},
                label=None,
            )
        ]
    )
    input_blocks = [
        b for b in scope.blocks if b.class_name == INPUT_IMAGE_CLASS_NAME
    ]
    assert len(input_blocks) == 1
    assert input_blocks[0].block_id == existing_id


# ---------------------------------------------------------------------------
# JSON round-trip — DAG
# ---------------------------------------------------------------------------


def _make_two_block_dag_state() -> _DagBuilderState:
    """Build a DAG state with 2 main-flow blocks + 1 edge + a nested container.

    Returns:
        A :class:`_DagBuilderState` whose root scope has:

        * The auto-seeded ``InputImage`` block.
        * A regular op block (``GaussianBlur``) wired from ``InputImage``.
        * A pipeline container with its own auto-seeded inner ``InputImage``.
    """

    state = _DagBuilderState()
    input_block = state.root.blocks[0]  # auto-seeded.

    blur_id = _new_block_id()
    state.root.blocks.append(
        BlockNode(
            block_id=blur_id,
            class_name="GaussianBlur",
            params={"sigma": 1.5},
            label="Blur",
        )
    )

    state.root.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=input_block.block_id,
            target_block_id=blur_id,
            target_port="in",
            kind="image",
        )
    )

    container_id = _new_block_id()
    state.root.blocks.append(
        BlockNode(
            block_id=container_id,
            class_name=PIPELINE_CLASS_NAME,
            params={},
            label="my-container",
            nested=_DagBuilderScope(name="inner"),
        )
    )

    state.selected_block_id = blur_id
    return state


def test_state_to_json_dag_round_trip() -> None:
    """A DAG state survives ``state_to_json`` → ``state_from_json``."""

    original = _make_two_block_dag_state()
    payload = state_to_json(original)
    assert payload["_schema"] == "dag"

    restored = state_from_json(payload)
    assert isinstance(restored, _DagBuilderState)

    # Top-level fields.
    assert restored.selected_block_id == original.selected_block_id
    assert restored.selected_edge_id == original.selected_edge_id

    # Root scope shape.
    orig_root = original.root
    rest_root = restored.root
    assert rest_root.name == orig_root.name
    assert rest_root.desc == orig_root.desc

    # Same block IDs in same order.
    assert [b.block_id for b in rest_root.blocks] == [
        b.block_id for b in orig_root.blocks
    ]
    assert [b.class_name for b in rest_root.blocks] == [
        b.class_name for b in orig_root.blocks
    ]
    assert [b.label for b in rest_root.blocks] == [
        b.label for b in orig_root.blocks
    ]

    # Same edges.
    assert [e.edge_id for e in rest_root.edges] == [
        e.edge_id for e in orig_root.edges
    ]
    assert [e.kind for e in rest_root.edges] == [
        e.kind for e in orig_root.edges
    ]

    # Container nested scope round-tripped + re-seeded.
    container = next(
        b for b in rest_root.blocks if b.class_name == PIPELINE_CLASS_NAME
    )
    assert container.nested is not None
    assert container.nested.name == "inner"
    inner_input_blocks = [
        b
        for b in container.nested.blocks
        if b.class_name == INPUT_IMAGE_CLASS_NAME
    ]
    assert len(inner_input_blocks) == 1


def test_state_to_json_legacy_round_trip() -> None:
    """A pre-existing legacy state still round-trips correctly."""

    scope = _LegacyBuilderScope(
        nodes=[
            _LegacyStepNode(
                node_id="abc01234",
                class_name="GaussianBlur",
                params={"sigma": 2.0},
                label="Smooth",
            )
        ],
        name="legacy-demo",
        desc="legacy description",
    )
    original = _LegacyBuilderState(root=scope, selected_node_id="abc01234")

    payload = state_to_json(original)
    assert payload["_schema"] == "legacy"

    restored = state_from_json(payload)
    assert isinstance(restored, _LegacyBuilderState)
    assert restored.selected_node_id == "abc01234"
    assert restored.root.name == "legacy-demo"
    assert restored.root.desc == "legacy description"
    assert len(restored.root.nodes) == 1
    assert restored.root.nodes[0].class_name == "GaussianBlur"
    assert restored.root.nodes[0].params == {"sigma": 2.0}


def test_state_from_json_seeds_missing_input_image() -> None:
    """Loading a DAG scope missing its ``InputImage`` auto-recovers Rule 6."""

    payload = {
        "_schema": "dag",
        "root": {
            "blocks": [
                {
                    "block_id": "deadbeef" * 4,
                    "class_name": "GaussianBlur",
                    "params": {"sigma": 1.0},
                    "label": None,
                    "nested": None,
                    "collapsed": False,
                    "list_slot_counts": {},
                }
            ],
            "edges": [],
            "name": "Pipeline",
            "desc": "",
            "nrows": None,
            "ncols": None,
        },
        "breadcrumb": [],
        "selected_block_id": None,
        "selected_edge_id": None,
        "pending_delete_block_id": None,
        "toast_queue": [],
    }

    restored = state_from_json(payload)
    assert isinstance(restored, _DagBuilderState)

    input_blocks = [
        b
        for b in restored.root.blocks
        if b.class_name == INPUT_IMAGE_CLASS_NAME
    ]
    assert len(input_blocks) == 1
    # The InputImage is at index 0 (inserted at head by `_seed_input_image`).
    assert restored.root.blocks[0].class_name == INPUT_IMAGE_CLASS_NAME


def test_state_from_json_seeds_missing_input_image_in_nested_scope() -> None:
    """Nested container scopes with no ``InputImage`` heal recursively."""

    payload = {
        "_schema": "dag",
        "root": {
            "blocks": [
                {
                    # Root scope has its InputImage already (good).
                    "block_id": "a" * 32,
                    "class_name": INPUT_IMAGE_CLASS_NAME,
                    "params": {},
                    "label": None,
                    "nested": None,
                    "collapsed": False,
                    "list_slot_counts": {},
                },
                {
                    # Container's nested scope is empty (bad).
                    "block_id": "b" * 32,
                    "class_name": PIPELINE_CLASS_NAME,
                    "params": {},
                    "label": None,
                    "nested": {
                        "blocks": [],
                        "edges": [],
                        "name": "inner",
                        "desc": "",
                        "nrows": None,
                        "ncols": None,
                    },
                    "collapsed": False,
                    "list_slot_counts": {},
                },
            ],
            "edges": [],
            "name": "Pipeline",
            "desc": "",
            "nrows": None,
            "ncols": None,
        },
    }

    restored = state_from_json(payload)
    assert isinstance(restored, _DagBuilderState)
    container = restored.root.blocks[1]
    assert container.nested is not None
    inner_inputs = [
        b
        for b in container.nested.blocks
        if b.class_name == INPUT_IMAGE_CLASS_NAME
    ]
    assert len(inner_inputs) == 1


def test_list_slot_counts_default_is_empty_dict() -> None:
    """``BlockNode.list_slot_counts`` defaults to a fresh dict per instance."""

    a = BlockNode(block_id=_new_block_id(), class_name="OtsuDetector", params={})
    b = BlockNode(block_id=_new_block_id(), class_name="OtsuDetector", params={})

    assert a.list_slot_counts == {}
    assert b.list_slot_counts == {}

    a.list_slot_counts["detectors"] = 3
    assert b.list_slot_counts == {}  # mutating one doesn't leak.
    assert a.list_slot_counts == {"detectors": 3}
    assert a.list_slot_counts is not b.list_slot_counts


# ---------------------------------------------------------------------------
# Edge invariants
# ---------------------------------------------------------------------------


def test_edge_target_block_id_required() -> None:
    """:class:`Edge` raises if ``target_block_id`` is left as the default."""

    with pytest.raises(AssertionError):
        Edge(edge_id=_new_block_id(), source_block_id=_new_block_id())


def test_edge_kind_round_trip_aux() -> None:
    """An ``aux`` edge survives JSON round-trip with kind/slot preserved."""

    state = _DagBuilderState()
    input_block = state.root.blocks[0]
    consumer_id = _new_block_id()
    state.root.blocks.append(
        BlockNode(
            block_id=consumer_id,
            class_name="FilamentousFungiDetector",
            params={},
        )
    )
    state.root.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=input_block.block_id,
            target_block_id=consumer_id,
            target_port="inoculum_detector",
            target_slot=None,
            kind="aux",
        )
    )

    restored = state_from_json(state_to_json(state))
    assert isinstance(restored, _DagBuilderState)
    aux_edges = [e for e in restored.root.edges if e.kind == "aux"]
    assert len(aux_edges) == 1
    assert aux_edges[0].target_port == "inoculum_detector"
    assert aux_edges[0].target_slot is None
