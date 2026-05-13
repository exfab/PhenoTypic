"""Unit tests for the ``block_create`` dispatch kind (Phase 3, spec §5.6).

Exercises :func:`phenotypic.gui.builder._callbacks._dispatch_state_update`
with the new ``block_create`` dispatch payload introduced for the DAG
palette drag-and-drop redesign.  Each test runs the dispatcher against
a JSON-shaped DAG state (``state_to_json`` output) and asserts on the
returned dict — no Dash boot required.

Test surface covered:

* Empty-canvas drop appends to root.
* ``class_name == "InputImage"`` rejected with an info toast.
* ``container_block_id`` resolved by DFS adopts the new block into the
  container's nested scope (innermost-wins when nested containers
  share an id).
* Stale ``container_block_id`` short-circuits with a warning + toast.
* :func:`validate` is callable on the post-dispatch state and the
  validation pipeline reports clean / dirty against expected scopes.
* New blocks inherit the registry-derived default params.

All tests use the ``empty_registry`` fixture from ``conftest.py`` so
registry-driven helpers (``_default_params_for``, ``validate``) reach
a stable, isolated registry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pytest

from phenotypic.gui._operation_registry import ParamInfo
from phenotypic.gui.builder._callbacks import _dispatch_state_update
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
    state_to_json,
)
from phenotypic.gui.builder._validation import validate

from .conftest import _make_op_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_dag_state_dict() -> Dict[str, Any]:
    """Return a freshly serialised :class:`_DagBuilderState` dict.

    The state's root scope auto-seeds an ``InputImage`` block via
    :meth:`_DagBuilderScope.__post_init__`, so the dict starts with
    exactly one block.
    """

    state = _DagBuilderState(root=_DagBuilderScope())
    return state_to_json(state)


def _make_container_block(
    *, block_id: Optional[str] = None
) -> BlockNode:
    """Return a fresh :class:`BlockNode` configured as a Pipeline container.

    The container's nested scope auto-seeds an ``InputImage`` via
    :meth:`_DagBuilderScope.__post_init__`, mirroring the Rule 6
    invariant.
    """

    return BlockNode(
        block_id=block_id or _new_block_id(),
        class_name=PIPELINE_CLASS_NAME,
        params={},
        nested=_DagBuilderScope(),
    )


def _make_palette_drop_payload(
    class_name: str,
    *,
    x: float = 100.0,
    y: float = 100.0,
    container_block_id: Optional[str] = None,
    ts: int = 0,
) -> Dict[str, Any]:
    """Construct the JS-side ``STORE_PALETTE_DROP`` payload shape."""

    return {
        "kind": "block_create",
        "class_name": class_name,
        "x": x,
        "y": y,
        "container_block_id": container_block_id,
        "ts": ts,
    }


def _seed_registry(empty_registry: Any) -> None:
    """Stash one fresh GaussianBlur op + one Pipeline sentinel.

    Mirrored from ``test_validation.py``'s use of ``empty_registry``.
    ``_dispatch_state_update`` uses the **registry returned by
    ``phenotypic.gui._operation_registry.get_registry``**, not the one
    monkeypatched into the validation module, so the dispatcher tests
    take a parallel monkeypatch via the same hook to avoid registry
    drift between the dispatcher and the validator.
    """

    empty_registry.ops["GaussianBlur"] = _make_op_info("GaussianBlur")
    empty_registry.ops["DummyOp"] = _make_op_info("DummyOp")


@pytest.fixture
def patched_registry(empty_registry: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Repoint both the dispatcher and validator at a shared fake registry.

    The dispatcher reads
    ``phenotypic.gui._operation_registry.get_registry()`` via the
    ``_default_params_for`` helper; the validator reads the same symbol
    re-exported under ``phenotypic.gui.builder._validation.get_registry``.
    Patching both keeps the tests deterministic across re-imports.
    """

    _seed_registry(empty_registry)
    monkeypatch.setattr(
        "phenotypic.gui._operation_registry.get_registry",
        lambda: empty_registry,
    )
    return empty_registry


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_block_create_appends_to_root_scope(
    patched_registry: Any,
) -> None:
    """Fresh state + empty-canvas drop → new BlockNode in ``state.root.blocks``."""

    state_dict = _empty_dag_state_dict()
    initial_block_count = len(state_dict["root"]["blocks"])

    new_state = _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload("GaussianBlur"),
    )

    blocks = new_state["root"]["blocks"]
    assert len(blocks) == initial_block_count + 1, (
        f"Expected 1 new block; got {len(blocks)} (was {initial_block_count})"
    )
    new_blocks = [b for b in blocks if b["class_name"] == "GaussianBlur"]
    assert len(new_blocks) == 1
    new_block = new_blocks[0]
    # The dispatcher should select the new block so the inspector
    # surfaces it without an extra click.
    assert new_state["selected_block_id"] == new_block["block_id"]
    # Selecting a block clears any wire-selection focus.
    assert new_state["selected_edge_id"] is None
    # block_id should be a fresh 32-char hex string per ``_new_block_id``.
    assert len(new_block["block_id"]) == 32
    # Drop coords are NOT persisted (spec §4.7 — dagre re-lays).
    assert "x" not in new_block
    assert "y" not in new_block


def test_block_create_with_container_block_id_adopts(
    patched_registry: Any,
) -> None:
    """``container_block_id`` resolves → new block goes into nested scope."""

    container = _make_container_block()
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container]),
    )
    state_dict = state_to_json(state)
    # The container's nested scope already carries an auto-seeded
    # ``InputImage`` block; the new block lands after it.
    initial_nested_count = sum(
        len(b.get("nested", {}).get("blocks", []) or [])
        for b in state_dict["root"]["blocks"]
        if b.get("class_name") == PIPELINE_CLASS_NAME
    )

    new_state = _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload(
            "GaussianBlur",
            container_block_id=container.block_id,
        ),
    )

    # Root scope still only carries the auto-seeded InputImage + the
    # container; the new block is NOT in the root.
    root_class_names = [
        b["class_name"] for b in new_state["root"]["blocks"]
    ]
    assert "GaussianBlur" not in root_class_names

    # The container's nested scope grew by one.
    container_blocks = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == container.block_id
    )["nested"]["blocks"]
    assert len(container_blocks) == initial_nested_count + 1
    nested_class_names = [b["class_name"] for b in container_blocks]
    assert "GaussianBlur" in nested_class_names


def test_block_create_nested_container_innermost_wins(
    patched_registry: Any,
) -> None:
    """When the JS sends the innermost container's id, the new block lands there.

    The JS does the innermost-wins hit-test in cytoscape; the
    dispatcher just resolves the *given* ``container_block_id`` to the
    matching nested scope.  This test confirms the DFS in
    :func:`_find_dag_container_scope` finds the inner container's
    ``nested`` and not the outer's even when both are present.
    """

    inner = _make_container_block()
    outer = BlockNode(
        block_id=_new_block_id(),
        class_name=PIPELINE_CLASS_NAME,
        params={},
        nested=_DagBuilderScope(blocks=[inner]),
    )
    # Re-seed the outer's nested scope so it carries the inner +
    # InputImage; dataclass field default doesn't run __post_init__
    # against a pre-built list.  This is fine — Phase 1 already exposes
    # ``_seed_input_image`` for paths that hand-build a scope.
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[outer]),
    )
    state_dict = state_to_json(state)

    new_state = _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload(
            "GaussianBlur",
            container_block_id=inner.block_id,
        ),
    )

    # Walk the tree: the new block sits in INNER's nested scope, not
    # OUTER's nested scope (which only has inner + InputImage).
    outer_in_new = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == outer.block_id
    )
    outer_nested_classes = [
        b["class_name"] for b in outer_in_new["nested"]["blocks"]
    ]
    # Outer scope contains: InputImage + inner container — no GaussianBlur.
    assert outer_nested_classes.count("GaussianBlur") == 0

    inner_in_new = next(
        b for b in outer_in_new["nested"]["blocks"]
        if b["block_id"] == inner.block_id
    )
    inner_nested_classes = [
        b["class_name"] for b in inner_in_new["nested"]["blocks"]
    ]
    assert inner_nested_classes.count("GaussianBlur") == 1


def test_block_create_uses_default_params(
    empty_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New BlockNode's ``params`` matches ``_default_params_for(class_name)``.

    Builds an ``OperationInfo`` with one scalar default and one no-default
    parameter; the new block should carry only the param with a default.
    """

    from phenotypic.gui.builder._callbacks import _default_params_for

    info = _make_op_info(
        "WithDefaults",
        parameters={
            "sigma": ParamInfo(
                name="sigma",
                type_hint=float,
                default=1.5,
                has_default=True,
                is_operation=False,
                is_pipeline=False,
                is_optional=False,
                is_list=False,
            ),
            "no_default": ParamInfo(
                name="no_default",
                type_hint=Any,
                default=None,
                has_default=False,
                is_operation=False,
                is_pipeline=False,
                is_optional=False,
                is_list=False,
            ),
        },
    )
    empty_registry.ops["WithDefaults"] = info
    monkeypatch.setattr(
        "phenotypic.gui._operation_registry.get_registry",
        lambda: empty_registry,
    )

    state_dict = _empty_dag_state_dict()
    new_state = _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload("WithDefaults"),
    )

    new_block = next(
        b for b in new_state["root"]["blocks"]
        if b["class_name"] == "WithDefaults"
    )
    expected_params = _default_params_for("WithDefaults")
    assert new_block["params"] == expected_params
    assert new_block["params"] == {"sigma": 1.5}


# ---------------------------------------------------------------------------
# Guard cases
# ---------------------------------------------------------------------------


def test_block_create_rejects_input_image(
    patched_registry: Any,
) -> None:
    """Dispatch with ``class_name == "InputImage"`` is rejected with a toast."""

    state_dict = _empty_dag_state_dict()
    initial_blocks = list(state_dict["root"]["blocks"])
    initial_toast_count = len(state_dict.get("toast_queue") or [])

    new_state = _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload(INPUT_IMAGE_CLASS_NAME),
    )

    # State unchanged — same block list (length + class names).
    assert new_state["root"]["blocks"] == initial_blocks
    # A toast was queued with the "Input Image" wording from the spec.
    toast_queue = new_state.get("toast_queue") or []
    assert len(toast_queue) == initial_toast_count + 1
    new_toast = toast_queue[-1]
    assert "Input Image" in new_toast["text"]
    assert new_toast.get("kind") in {"info", "warning"}


def test_block_create_stale_container_id_short_circuits(
    patched_registry: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unresolvable ``container_block_id`` → state unchanged + warning + toast."""

    state_dict = _empty_dag_state_dict()
    initial_blocks = list(state_dict["root"]["blocks"])
    stale_id = "0" * 32  # 32-char hex but doesn't match any container

    with caplog.at_level(logging.WARNING, logger="phenotypic.gui.builder._callbacks"):
        new_state = _dispatch_state_update(
            state_dict,
            "block_create",
            _make_palette_drop_payload(
                "GaussianBlur",
                container_block_id=stale_id,
            ),
        )

    # State unchanged.
    assert new_state["root"]["blocks"] == initial_blocks
    # A toast was queued.
    toast_queue = new_state.get("toast_queue") or []
    assert len(toast_queue) >= 1
    last_toast = toast_queue[-1]
    assert "container" in last_toast["text"].lower()
    # A warning was logged.
    assert any(
        "stale container_block_id" in record.getMessage()
        for record in caplog.records
    ), f"Expected warning log; got {[r.getMessage() for r in caplog.records]}"


def test_block_create_missing_class_name_short_circuits(
    patched_registry: Any,
) -> None:
    """Payload missing ``class_name`` short-circuits cleanly (no crash)."""

    state_dict = _empty_dag_state_dict()
    initial_blocks = list(state_dict["root"]["blocks"])

    # Payload-missing case: ``class_name`` empty / non-string.
    new_state = _dispatch_state_update(
        state_dict,
        "block_create",
        {
            "kind": "block_create",
            "class_name": "",
            "x": 0.0,
            "y": 0.0,
            "container_block_id": None,
            "ts": 0,
        },
    )
    assert new_state["root"]["blocks"] == initial_blocks


# ---------------------------------------------------------------------------
# Validation pipeline integration
# ---------------------------------------------------------------------------


def test_block_create_validates_after_mutation(
    patched_registry: Any,
) -> None:
    """The post-dispatch state passes through :func:`validate` cleanly.

    Spec §5.6: ``block_create`` updates ``STORE_ISSUES`` via the
    validation pipeline.  This test confirms the dispatcher leaves the
    state in a shape :func:`validate` can ingest and that the resulting
    issue list reflects the expected post-mutation invariants.
    """

    from phenotypic.gui.builder._state import state_from_json

    state_dict = _empty_dag_state_dict()
    new_state_dict = _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload("GaussianBlur"),
    )

    # The new block is added but not yet wired — Rule 2 (stub) fires
    # for the unwired GaussianBlur.  Existence of the issue confirms
    # validation ran AND the new block participates in the state.
    state = state_from_json(new_state_dict)
    issues = validate(state)
    stub_issues = [i for i in issues if i.kind == "stub"]
    # At least one stub issue points at the GaussianBlur block.
    gb_block = next(
        b for b in new_state_dict["root"]["blocks"]
        if b["class_name"] == "GaussianBlur"
    )
    assert any(i.block_id == gb_block["block_id"] for i in stub_issues)


def test_block_create_does_not_double_seed_input_image(
    patched_registry: Any,
) -> None:
    """Defense-in-depth seed pass is idempotent.

    The dispatcher calls the seed helper before appending the new
    block — that should not add a second ``InputImage`` if one is
    already present (Rule 6 invariant).
    """

    state_dict = _empty_dag_state_dict()
    # Initially exactly one InputImage.
    input_count_before = sum(
        1 for b in state_dict["root"]["blocks"]
        if b["class_name"] == INPUT_IMAGE_CLASS_NAME
    )
    assert input_count_before == 1

    new_state = _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload("GaussianBlur"),
    )

    input_count_after = sum(
        1 for b in new_state["root"]["blocks"]
        if b["class_name"] == INPUT_IMAGE_CLASS_NAME
    )
    assert input_count_after == 1, (
        f"InputImage block count changed: {input_count_before} → "
        f"{input_count_after}"
    )


def test_block_create_does_not_mutate_input_dict(
    patched_registry: Any,
) -> None:
    """The dispatcher returns a NEW dict; the caller's payload survives intact.

    Pure-function discipline: ``_dispatch_state_update`` is documented
    to ``deepcopy`` the input state.  This guards against future
    refactors that might accidentally mutate the input in place.
    """

    state_dict = _empty_dag_state_dict()
    snapshot = {
        "block_count": len(state_dict["root"]["blocks"]),
        "block_ids": [b["block_id"] for b in state_dict["root"]["blocks"]],
        "selected_block_id": state_dict.get("selected_block_id"),
    }

    _dispatch_state_update(
        state_dict,
        "block_create",
        _make_palette_drop_payload("GaussianBlur"),
    )

    assert len(state_dict["root"]["blocks"]) == snapshot["block_count"]
    assert [b["block_id"] for b in state_dict["root"]["blocks"]] == snapshot["block_ids"]
    assert state_dict.get("selected_block_id") == snapshot["selected_block_id"]


# ---------------------------------------------------------------------------
# Edge-dispatch helpers + fixtures
# ---------------------------------------------------------------------------


def _make_edge_create_payload(
    *,
    source_block_id: str,
    target_block_id: str,
    target_port: str,
    edge_kind: str = "aux",
    ts: int = 0,
) -> Dict[str, Any]:
    """Construct an ``edge_create`` payload matching the JS contract.

    Note: clients send ``edge_kind`` (not ``kind`` — ``kind`` is the
    dispatch discriminator at the top level).
    """

    return {
        "kind": "edge_create",
        "source_block_id": source_block_id,
        "target_block_id": target_block_id,
        "target_port": target_port,
        "edge_kind": edge_kind,
        "ts": ts,
    }


def _seed_aux_consumer_registry(empty_registry: Any) -> Dict[str, str]:
    """Seed two ops in the registry: a Source op + a Consumer op.

    The Consumer carries one **scalar aux** param ``scalar_aux`` and one
    **list aux** param ``list_aux``.  Returns a dict mapping logical
    name to registered class name for clarity.
    """

    from .conftest import _make_param

    source_info = _make_op_info("SourceOp")
    consumer_info = _make_op_info(
        "ConsumerOp",
        parameters={
            "scalar_aux": _make_param(
                "scalar_aux",
                has_default=True,
                is_operation=True,
                is_list=False,
            ),
            "list_aux": _make_param(
                "list_aux",
                has_default=True,
                is_operation=True,
                is_list=True,
            ),
        },
    )
    empty_registry.ops["SourceOp"] = source_info
    empty_registry.ops["ConsumerOp"] = consumer_info
    return {"source": "SourceOp", "consumer": "ConsumerOp"}


def _build_source_consumer_state(
    patched_registry: Any,
) -> Tuple[Dict[str, Any], str, str]:
    """Return (state_dict, source_block_id, consumer_block_id).

    Boots a DAG state with two blocks via the ``block_create`` dispatch
    so the source + consumer block ids are deterministic and the
    registry-derived params dict matches a real palette drop.
    """

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("SourceOp"),
    )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    blocks = state_dict["root"]["blocks"]
    source_block = next(b for b in blocks if b["class_name"] == "SourceOp")
    consumer_block = next(b for b in blocks if b["class_name"] == "ConsumerOp")
    return state_dict, source_block["block_id"], consumer_block["block_id"]


# ---------------------------------------------------------------------------
# edge_create
# ---------------------------------------------------------------------------


def test_edge_create_image_flow_succeeds(patched_registry: Any) -> None:
    """Image-flow ``edge_create`` adds one edge to the shared scope."""

    state_dict, src_id, tgt_id = _build_source_consumer_state(patched_registry)
    initial_edges = list(state_dict["root"].get("edges", []) or [])

    new_state = _dispatch_state_update(
        state_dict,
        "edge_create",
        _make_edge_create_payload(
            source_block_id=src_id,
            target_block_id=tgt_id,
            target_port="in",
            edge_kind="image",
        ),
    )

    edges = new_state["root"]["edges"]
    assert len(edges) == len(initial_edges) + 1
    new_edge = next(
        e for e in edges
        if e["source_block_id"] == src_id and e["target_block_id"] == tgt_id
    )
    assert new_edge["target_port"] == "in"
    assert new_edge["kind"] == "image"
    assert new_edge["target_slot"] is None
    assert len(new_edge["edge_id"]) == 32  # uuid hex


def test_edge_create_scalar_aux_replaces_existing(
    patched_registry: Any,
) -> None:
    """Second edge to same scalar aux port → first edge gone, second present."""

    _seed_aux_consumer_registry(patched_registry)
    # Three blocks: source_a, source_b, consumer.  Wire source_a → consumer.scalar_aux,
    # then wire source_b → consumer.scalar_aux.  Result: only the source_b edge.
    state_dict = _empty_dag_state_dict()
    for cls in ("SourceOp", "SourceOp", "ConsumerOp"):
        state_dict = _dispatch_state_update(
            state_dict, "block_create",
            _make_palette_drop_payload(cls),
        )
    source_blocks = [
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    ]
    src_a, src_b = source_blocks[0]["block_id"], source_blocks[1]["block_id"]
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]

    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src_a,
            target_block_id=tgt,
            target_port="scalar_aux",
            edge_kind="aux",
        ),
    )
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src_b,
            target_block_id=tgt,
            target_port="scalar_aux",
            edge_kind="aux",
        ),
    )

    scalar_edges = [
        e for e in state_dict["root"]["edges"]
        if e["target_block_id"] == tgt and e["target_port"] == "scalar_aux"
    ]
    assert len(scalar_edges) == 1
    assert scalar_edges[0]["source_block_id"] == src_b


def test_edge_create_list_aux_appends_to_next_slot(
    patched_registry: Any,
) -> None:
    """Three ``edge_create`` to the same list aux port → slots 0, 1, 2."""

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    for _ in range(3):
        state_dict = _dispatch_state_update(
            state_dict, "block_create",
            _make_palette_drop_payload("SourceOp"),
        )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )

    source_ids = [
        b["block_id"]
        for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    ]
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]

    # Wire all three sources into the consumer's list_aux port.
    for sid in source_ids:
        state_dict = _dispatch_state_update(
            state_dict, "edge_create",
            _make_edge_create_payload(
                source_block_id=sid,
                target_block_id=tgt,
                target_port="list_aux",
                edge_kind="aux",
            ),
        )

    list_edges = sorted(
        [
            e for e in state_dict["root"]["edges"]
            if e["target_block_id"] == tgt
            and e["target_port"] == "list_aux"
        ],
        key=lambda e: e["target_slot"],
    )
    assert [e["target_slot"] for e in list_edges] == [0, 1, 2]
    # Server-side slot resolution: ``list_slot_counts`` reaches 3.
    tgt_block = next(
        b for b in state_dict["root"]["blocks"] if b["block_id"] == tgt
    )
    assert tgt_block["list_slot_counts"]["list_aux"] == 3


def test_edge_create_list_aux_slot_no_collision_after_delete(
    patched_registry: Any,
) -> None:
    """Wire slots 0,1,2; delete slot 1; wire 4th → slot 3 (no collision)."""

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    for _ in range(4):
        state_dict = _dispatch_state_update(
            state_dict, "block_create",
            _make_palette_drop_payload("SourceOp"),
        )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    source_ids = [
        b["block_id"]
        for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    ]
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]

    # Wire first three sources → slots 0, 1, 2.
    for sid in source_ids[:3]:
        state_dict = _dispatch_state_update(
            state_dict, "edge_create",
            _make_edge_create_payload(
                source_block_id=sid,
                target_block_id=tgt,
                target_port="list_aux",
                edge_kind="aux",
            ),
        )
    # Delete the slot-1 edge.
    slot1_edge = next(
        e for e in state_dict["root"]["edges"]
        if e["target_block_id"] == tgt
        and e["target_port"] == "list_aux"
        and e["target_slot"] == 1
    )
    state_dict = _dispatch_state_update(
        state_dict, "edge_delete",
        {"kind": "edge_delete", "edge_id": slot1_edge["edge_id"], "ts": 1},
    )
    # Slot count stays at 3 (no renumbering on delete).
    tgt_block = next(
        b for b in state_dict["root"]["blocks"] if b["block_id"] == tgt
    )
    assert tgt_block["list_slot_counts"]["list_aux"] == 3

    # Wire the 4th source → slot 3 (server-side resolution; never reuses
    # the freed slot 1).
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=source_ids[3],
            target_block_id=tgt,
            target_port="list_aux",
            edge_kind="aux",
            ts=2,
        ),
    )
    new_edge = next(
        e for e in state_dict["root"]["edges"]
        if e["source_block_id"] == source_ids[3]
    )
    assert new_edge["target_slot"] == 3
    tgt_block = next(
        b for b in state_dict["root"]["blocks"] if b["block_id"] == tgt
    )
    assert tgt_block["list_slot_counts"]["list_aux"] == 4


def test_edge_create_cross_scope_rejected(patched_registry: Any) -> None:
    """Source in root, target in container → reject + toast."""

    _seed_aux_consumer_registry(patched_registry)
    # Build a container + put a ConsumerOp inside it.
    container = _make_container_block()
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container]),
    )
    state_dict = state_to_json(state)
    # Drop a source into the root, a consumer inside the container.
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("SourceOp"),
    )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload(
            "ConsumerOp",
            container_block_id=container.block_id,
        ),
    )

    src = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    )["block_id"]
    container_dict = next(
        b for b in state_dict["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    consumer = next(
        b for b in container_dict["nested"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]

    initial_root_edges = list(state_dict["root"].get("edges", []) or [])
    initial_nested_edges = list(
        container_dict["nested"].get("edges", []) or []
    )

    new_state = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src,
            target_block_id=consumer,
            target_port="scalar_aux",
            edge_kind="aux",
        ),
    )

    # No edges added anywhere.
    assert (
        list(new_state["root"].get("edges", []) or []) == initial_root_edges
    )
    new_container_dict = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    assert (
        list(new_container_dict["nested"].get("edges", []) or [])
        == initial_nested_edges
    )
    # Toast surfaced.
    toasts = new_state.get("toast_queue") or []
    assert any("Cross-scope" in t["text"] for t in toasts)


def test_edge_create_concurrent_drag_determinism(
    patched_registry: Any,
) -> None:
    """Two ``edge_create`` payloads same-tick → deterministic slot index.

    The fan-in callback serialises by ``ts`` (spec §5.5).  Per-dispatch
    each call sees the incremented ``list_slot_counts`` so two
    independent wires resolve to slots 0 and 1 deterministically.
    """

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    for _ in range(2):
        state_dict = _dispatch_state_update(
            state_dict, "block_create",
            _make_palette_drop_payload("SourceOp"),
        )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    source_ids = [
        b["block_id"]
        for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    ]
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]

    # Same ts on both payloads — order of application still produces
    # deterministic slots because the dispatcher is purely deterministic
    # per call.
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=source_ids[0],
            target_block_id=tgt,
            target_port="list_aux",
            edge_kind="aux",
            ts=100,
        ),
    )
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=source_ids[1],
            target_block_id=tgt,
            target_port="list_aux",
            edge_kind="aux",
            ts=100,
        ),
    )
    list_edges = sorted(
        [
            e for e in state_dict["root"]["edges"]
            if e["target_block_id"] == tgt
            and e["target_port"] == "list_aux"
        ],
        key=lambda e: e["target_slot"],
    )
    assert [e["target_slot"] for e in list_edges] == [0, 1]
    # Distinct sources mapped to distinct slots.
    assert {e["source_block_id"] for e in list_edges} == set(source_ids)


def test_edge_create_image_source_single_wire_rule(
    patched_registry: Any,
) -> None:
    """Wiring source → A then source → B replaces the first wire (spec §4.2).

    Output ports take at most one outgoing wire **total** (image or aux,
    never both).  The second ``edge_create`` from the same source must
    delete the first.
    """

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("SourceOp"),
    )
    # Two consumer blocks (distinct downstream targets).
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    src = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    )["block_id"]
    consumers = [
        b["block_id"]
        for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    ]

    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src,
            target_block_id=consumers[0],
            target_port="scalar_aux",
            edge_kind="aux",
        ),
    )
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src,
            target_block_id=consumers[1],
            target_port="scalar_aux",
            edge_kind="aux",
            ts=1,
        ),
    )

    src_edges = [
        e for e in state_dict["root"]["edges"]
        if e["source_block_id"] == src
    ]
    assert len(src_edges) == 1
    assert src_edges[0]["target_block_id"] == consumers[1]


# ---------------------------------------------------------------------------
# edge_delete
# ---------------------------------------------------------------------------


def test_edge_delete_removes_edge(patched_registry: Any) -> None:
    """Edge in scope.edges; after dispatch, gone."""

    state_dict, src, tgt = _build_source_consumer_state(patched_registry)
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src,
            target_block_id=tgt,
            target_port="in",
            edge_kind="image",
        ),
    )
    edge_id = state_dict["root"]["edges"][0]["edge_id"]

    new_state = _dispatch_state_update(
        state_dict, "edge_delete",
        {"kind": "edge_delete", "edge_id": edge_id, "ts": 1},
    )
    assert all(
        e["edge_id"] != edge_id for e in new_state["root"]["edges"]
    )


def test_edge_delete_list_aux_keeps_slot_count(
    patched_registry: Any,
) -> None:
    """Slot count after delete unchanged (spec §5.6: never decrements)."""

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    for _ in range(2):
        state_dict = _dispatch_state_update(
            state_dict, "block_create",
            _make_palette_drop_payload("SourceOp"),
        )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    sources = [
        b["block_id"]
        for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    ]
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]

    for sid in sources:
        state_dict = _dispatch_state_update(
            state_dict, "edge_create",
            _make_edge_create_payload(
                source_block_id=sid,
                target_block_id=tgt,
                target_port="list_aux",
                edge_kind="aux",
            ),
        )

    tgt_block_before = next(
        b for b in state_dict["root"]["blocks"] if b["block_id"] == tgt
    )
    slot_count_before = tgt_block_before["list_slot_counts"]["list_aux"]
    assert slot_count_before == 2

    edge_to_delete = state_dict["root"]["edges"][0]["edge_id"]
    new_state = _dispatch_state_update(
        state_dict, "edge_delete",
        {"kind": "edge_delete", "edge_id": edge_to_delete, "ts": 1},
    )

    tgt_block_after = next(
        b for b in new_state["root"]["blocks"] if b["block_id"] == tgt
    )
    assert (
        tgt_block_after["list_slot_counts"]["list_aux"] == slot_count_before
    )


def test_edge_delete_unknown_id_is_noop(patched_registry: Any) -> None:
    """Unknown edge_id → state unchanged (no crash)."""

    state_dict, _, _ = _build_source_consumer_state(patched_registry)
    edges_before = list(state_dict["root"].get("edges", []) or [])
    new_state = _dispatch_state_update(
        state_dict, "edge_delete",
        {"kind": "edge_delete", "edge_id": "x" * 32, "ts": 0},
    )
    assert list(new_state["root"].get("edges", []) or []) == edges_before


# ---------------------------------------------------------------------------
# list_aux_reorder
# ---------------------------------------------------------------------------


def test_list_aux_reorder_valid_permutation(patched_registry: Any) -> None:
    """Wired edges reorder; ``target_slot`` updated to new positions."""

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    for _ in range(3):
        state_dict = _dispatch_state_update(
            state_dict, "block_create",
            _make_palette_drop_payload("SourceOp"),
        )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    sources = [
        b["block_id"]
        for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    ]
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]
    for sid in sources:
        state_dict = _dispatch_state_update(
            state_dict, "edge_create",
            _make_edge_create_payload(
                source_block_id=sid,
                target_block_id=tgt,
                target_port="list_aux",
                edge_kind="aux",
            ),
        )
    # Current edges sorted by slot: slot0, slot1, slot2.
    list_edges = sorted(
        [
            e for e in state_dict["root"]["edges"]
            if e["target_port"] == "list_aux"
        ],
        key=lambda e: e["target_slot"],
    )
    e0, e1, e2 = list_edges

    # Reorder: reverse the list.
    new_order = [e2["edge_id"], e1["edge_id"], e0["edge_id"]]
    new_state = _dispatch_state_update(
        state_dict, "list_aux_reorder",
        {
            "kind": "list_aux_reorder",
            "block_id": tgt,
            "param": "list_aux",
            "new_order": new_order,
            "ts": 0,
        },
    )

    slot_by_edge = {
        e["edge_id"]: e["target_slot"]
        for e in new_state["root"]["edges"]
        if e["target_port"] == "list_aux"
    }
    assert slot_by_edge[e2["edge_id"]] == 0
    assert slot_by_edge[e1["edge_id"]] == 1
    assert slot_by_edge[e0["edge_id"]] == 2

    tgt_block = next(
        b for b in new_state["root"]["blocks"] if b["block_id"] == tgt
    )
    assert tgt_block["list_slot_counts"]["list_aux"] == 3


def test_list_aux_reorder_non_permutation_rejected(
    patched_registry: Any,
) -> None:
    """Invalid input → no-op + toast (spec §5.6)."""

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    for _ in range(2):
        state_dict = _dispatch_state_update(
            state_dict, "block_create",
            _make_palette_drop_payload("SourceOp"),
        )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    sources = [
        b["block_id"]
        for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    ]
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]
    for sid in sources:
        state_dict = _dispatch_state_update(
            state_dict, "edge_create",
            _make_edge_create_payload(
                source_block_id=sid,
                target_block_id=tgt,
                target_port="list_aux",
                edge_kind="aux",
            ),
        )
    edges_before = [
        {"edge_id": e["edge_id"], "target_slot": e["target_slot"]}
        for e in state_dict["root"]["edges"]
        if e["target_port"] == "list_aux"
    ]
    toast_count_before = len(state_dict.get("toast_queue") or [])

    # Invalid: refers to a non-existent edge_id.
    new_state = _dispatch_state_update(
        state_dict, "list_aux_reorder",
        {
            "kind": "list_aux_reorder",
            "block_id": tgt,
            "param": "list_aux",
            "new_order": ["z" * 32, edges_before[0]["edge_id"]],
            "ts": 0,
        },
    )
    edges_after = [
        {"edge_id": e["edge_id"], "target_slot": e["target_slot"]}
        for e in new_state["root"]["edges"]
        if e["target_port"] == "list_aux"
    ]
    assert sorted(edges_after, key=lambda e: e["edge_id"]) == sorted(
        edges_before, key=lambda e: e["edge_id"]
    )
    toasts = new_state.get("toast_queue") or []
    assert len(toasts) == toast_count_before + 1
    assert "Reorder rejected" in toasts[-1]["text"]


# ---------------------------------------------------------------------------
# list_aux_add_empty_slot
# ---------------------------------------------------------------------------


def test_list_aux_add_empty_slot_increments_count(
    patched_registry: Any,
) -> None:
    """Count + 1; no edge created (spec §5.6)."""

    _seed_aux_consumer_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("ConsumerOp"),
    )
    tgt = next(
        b for b in state_dict["root"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )["block_id"]
    edges_before = list(state_dict["root"].get("edges", []) or [])

    new_state = _dispatch_state_update(
        state_dict, "list_aux_add_empty_slot",
        {
            "kind": "list_aux_add_empty_slot",
            "block_id": tgt,
            "param": "list_aux",
            "ts": 0,
        },
    )

    tgt_block = next(
        b for b in new_state["root"]["blocks"] if b["block_id"] == tgt
    )
    assert tgt_block["list_slot_counts"]["list_aux"] == 1
    # No edge was created.
    assert (
        list(new_state["root"].get("edges", []) or []) == edges_before
    )

    # Fire it again — count rises to 2.
    newer_state = _dispatch_state_update(
        new_state, "list_aux_add_empty_slot",
        {
            "kind": "list_aux_add_empty_slot",
            "block_id": tgt,
            "param": "list_aux",
            "ts": 1,
        },
    )
    tgt_block = next(
        b for b in newer_state["root"]["blocks"] if b["block_id"] == tgt
    )
    assert tgt_block["list_slot_counts"]["list_aux"] == 2


# ---------------------------------------------------------------------------
# wire_select / block_select (mutual exclusion)
# ---------------------------------------------------------------------------


def test_wire_select_clears_block_selection(patched_registry: Any) -> None:
    """Block selected; wire_select fires; block_id cleared (spec §4.5)."""

    state_dict, src, tgt = _build_source_consumer_state(patched_registry)
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src,
            target_block_id=tgt,
            target_port="in",
            edge_kind="image",
        ),
    )
    edge_id = state_dict["root"]["edges"][0]["edge_id"]
    # Block selected initially (block_create sets it).
    state_dict["selected_block_id"] = src
    state_dict["selected_edge_id"] = None

    new_state = _dispatch_state_update(
        state_dict, "wire_select",
        {"kind": "wire_select", "edge_id": edge_id, "ts": 0},
    )
    assert new_state["selected_edge_id"] == edge_id
    assert new_state["selected_block_id"] is None

    # Deselection (edge_id=None) clears both — but specifically only the
    # wire selection is touched here.
    cleared = _dispatch_state_update(
        new_state, "wire_select",
        {"kind": "wire_select", "edge_id": None, "ts": 1},
    )
    assert cleared["selected_edge_id"] is None


def test_block_select_clears_wire_selection(patched_registry: Any) -> None:
    """Wire selected; block_select fires; edge_id cleared (spec §4.5)."""

    state_dict, src, tgt = _build_source_consumer_state(patched_registry)
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src,
            target_block_id=tgt,
            target_port="in",
            edge_kind="image",
        ),
    )
    edge_id = state_dict["root"]["edges"][0]["edge_id"]
    state_dict["selected_block_id"] = None
    state_dict["selected_edge_id"] = edge_id

    new_state = _dispatch_state_update(
        state_dict, "block_select",
        {"kind": "block_select", "block_id": src, "ts": 0},
    )
    assert new_state["selected_block_id"] == src
    assert new_state["selected_edge_id"] is None

    # Deselection.
    cleared = _dispatch_state_update(
        new_state, "block_select",
        {"kind": "block_select", "block_id": None, "ts": 1},
    )
    assert cleared["selected_block_id"] is None


# ---------------------------------------------------------------------------
# block_reparent — pipeline container relocation (spec §4.4 / §5.6)
# ---------------------------------------------------------------------------


def _seed_reparent_registry(empty_registry: Any) -> None:
    """Seed registry with a no-aux op + a consumer with one scalar aux.

    These are the two minimal shapes the reparent tests need: an op that can
    safely live in either scope without aux wiring, and a consumer that can
    be wired to it so we can fabricate orphan-edge scenarios.
    """

    from .conftest import _make_param

    empty_registry.ops["SourceOp"] = _make_op_info("SourceOp")
    empty_registry.ops["ConsumerOp"] = _make_op_info(
        "ConsumerOp",
        parameters={
            "scalar_aux": _make_param(
                "scalar_aux",
                has_default=True,
                is_operation=True,
                is_list=False,
            ),
        },
    )


def test_block_reparent_to_container_moves_block(
    patched_registry: Any,
) -> None:
    """Block in root scope moves to a container's nested scope.

    Drag-IN direction (root → container, ancestor → descendant).  Spec §4.4
    rules: the block is removed from root and appended to the container's
    nested scope in one dispatch.  No orphan edges (no wires touching the
    block), so the move is clean.
    """

    _seed_reparent_registry(patched_registry)
    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)
    # Drop a SourceOp into the root scope (sibling to the container).
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("SourceOp"),
    )
    src_block_id = next(
        b["block_id"] for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    )

    new_state = _dispatch_state_update(
        state_dict, "block_reparent",
        {
            "kind": "block_reparent",
            "block_id": src_block_id,
            "new_parent_block_id": container.block_id,
            "x": 0.0,
            "y": 0.0,
            "ts": 0,
        },
    )

    # Root scope no longer contains the SourceOp.
    root_class_names = [b["class_name"] for b in new_state["root"]["blocks"]]
    assert "SourceOp" not in root_class_names

    # Container's nested scope now contains it.
    container_block = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    inner_class_names = [
        b["class_name"] for b in container_block["nested"]["blocks"]
    ]
    assert inner_class_names.count("SourceOp") == 1


def test_block_reparent_sibling_container_atomic(
    patched_registry: Any,
) -> None:
    """Block moves from container A's nested scope to sibling B's nested scope.

    Spec §4.4 "Sibling-container moves are a single atomic dispatch — the
    block is removed from its current containing scope and appended to the
    target's nested scope in one tick".  Both scopes change in the same
    dispatch.
    """

    _seed_reparent_registry(patched_registry)
    container_a = _make_container_block()
    container_b = _make_container_block()
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container_a, container_b]),
    )
    state_dict = state_to_json(state)
    # Drop SourceOp into container_a's nested scope.
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload(
            "SourceOp", container_block_id=container_a.block_id,
        ),
    )
    container_a_blocks = next(
        b for b in state_dict["root"]["blocks"]
        if b["block_id"] == container_a.block_id
    )["nested"]["blocks"]
    src_block_id = next(
        b["block_id"] for b in container_a_blocks
        if b["class_name"] == "SourceOp"
    )

    new_state = _dispatch_state_update(
        state_dict, "block_reparent",
        {
            "kind": "block_reparent",
            "block_id": src_block_id,
            "new_parent_block_id": container_b.block_id,
            "x": 0.0,
            "y": 0.0,
            "ts": 0,
        },
    )

    # A's nested scope no longer has SourceOp; B's does.  Single dispatch.
    a_inner = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == container_a.block_id
    )["nested"]["blocks"]
    b_inner = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == container_b.block_id
    )["nested"]["blocks"]
    a_class_names = [b["class_name"] for b in a_inner]
    b_class_names = [b["class_name"] for b in b_inner]
    assert "SourceOp" not in a_class_names
    assert b_class_names.count("SourceOp") == 1


def test_block_reparent_drag_out_with_inner_edges_rejected_with_toast(
    patched_registry: Any,
) -> None:
    """Drag-OUT direction with orphan edges → reject + toast.

    Spec §4.4 "Drag-out direction with orphan edges → snap-back + toast
    listing edge count + names. User must manually disconnect first."
    """

    _seed_reparent_registry(patched_registry)
    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)
    # Put SourceOp + ConsumerOp inside the container, then wire them.
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload(
            "SourceOp", container_block_id=container.block_id,
        ),
    )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload(
            "ConsumerOp", container_block_id=container.block_id,
        ),
    )
    container_dict = next(
        b for b in state_dict["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    src_block_id = next(
        b["block_id"] for b in container_dict["nested"]["blocks"]
        if b["class_name"] == "SourceOp"
    )
    consumer_block_id = next(
        b["block_id"] for b in container_dict["nested"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )
    # Drill into the container so the cross-scope rule is satisfied (the edge
    # dispatcher requires source + target in the same scope).
    state_dict = _dispatch_state_update(
        state_dict, "drill_into_container",
        {
            "kind": "drill_into_container",
            "block_id": container.block_id,
            "ts": 0,
        },
    )
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src_block_id,
            target_block_id=consumer_block_id,
            target_port="scalar_aux",
            edge_kind="aux",
        ),
    )
    # Drill back out so the "current scope" (per breadcrumb) is root.  The
    # drag-out direction is the case where the *target* scope (current
    # breadcrumb scope) is an *ancestor* of the block's actual containing
    # scope — i.e. user grabs a block from inside a container while
    # viewing the parent scope and drops onto the parent scope.
    state_dict = _dispatch_state_update(
        state_dict, "drill_out", {"kind": "drill_out", "ts": 0},
    )
    initial_inner_blocks = [
        b["block_id"] for b in next(
            b for b in state_dict["root"]["blocks"]
            if b["block_id"] == container.block_id
        )["nested"]["blocks"]
    ]
    initial_inner_edges = list(
        next(
            b for b in state_dict["root"]["blocks"]
            if b["block_id"] == container.block_id
        )["nested"].get("edges", []) or []
    )
    initial_toast_count = len(state_dict.get("toast_queue") or [])
    # Move SourceOp OUT (to root) — drag-out direction; ConsumerOp stays in
    # the container so the SourceOp → ConsumerOp aux edge would orphan.
    new_state = _dispatch_state_update(
        state_dict, "block_reparent",
        {
            "kind": "block_reparent",
            "block_id": src_block_id,
            "new_parent_block_id": None,  # to root
            "x": 0.0,
            "y": 0.0,
            "ts": 1,
        },
    )

    # State unchanged (rejection).  Source still in container's nested scope.
    new_container = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    assert [
        b["block_id"] for b in new_container["nested"]["blocks"]
    ] == initial_inner_blocks
    # Edge still present.
    assert list(new_container["nested"].get("edges", []) or []) == \
        initial_inner_edges
    # Toast surfaced with the orphan-edge count.
    toasts = new_state.get("toast_queue") or []
    assert len(toasts) == initial_toast_count + 1
    assert "orphan" in toasts[-1]["text"].lower() or \
        "disconnect" in toasts[-1]["text"].lower() or \
        "inner edge" in toasts[-1]["text"].lower()


def test_block_reparent_rejects_input_image(
    patched_registry: Any,
) -> None:
    """Defense-in-depth: ``block_reparent`` of an Input Image is rejected."""

    state_dict = _empty_dag_state_dict()
    input_image_id = next(
        b["block_id"] for b in state_dict["root"]["blocks"]
        if b["class_name"] == INPUT_IMAGE_CLASS_NAME
    )
    # Build a container we could attempt to relocate into.
    container = _make_container_block()
    state_dict["root"]["blocks"].append({
        "block_id": container.block_id,
        "class_name": PIPELINE_CLASS_NAME,
        "params": {},
        "label": None,
        "nested": {
            "blocks": [{
                "block_id": _new_block_id(),
                "class_name": INPUT_IMAGE_CLASS_NAME,
                "params": {},
                "label": None,
                "nested": None,
                "collapsed": False,
                "list_slot_counts": {},
            }],
            "edges": [],
            "name": "Pipeline",
            "desc": "",
            "nrows": None,
            "ncols": None,
        },
        "collapsed": False,
        "list_slot_counts": {},
    })
    blocks_before = [b["block_id"] for b in state_dict["root"]["blocks"]]
    toast_count_before = len(state_dict.get("toast_queue") or [])

    new_state = _dispatch_state_update(
        state_dict, "block_reparent",
        {
            "kind": "block_reparent",
            "block_id": input_image_id,
            "new_parent_block_id": container.block_id,
            "x": 0.0,
            "y": 0.0,
            "ts": 0,
        },
    )

    # The InputImage stays in root.
    assert [
        b["block_id"] for b in new_state["root"]["blocks"]
    ] == blocks_before
    # Toast queued.
    toasts = new_state.get("toast_queue") or []
    assert len(toasts) == toast_count_before + 1
    assert "Input Image" in toasts[-1]["text"]


# ---------------------------------------------------------------------------
# block_collapsed_toggle (spec §4.4 / §5.6)
# ---------------------------------------------------------------------------


def test_block_collapsed_toggle_flips_bool(patched_registry: Any) -> None:
    """Container's ``collapsed`` field flips True ↔ False."""

    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)
    # The container starts uncollapsed.
    container_dict = next(
        b for b in state_dict["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    assert container_dict.get("collapsed") is False

    # First toggle → True.
    new_state = _dispatch_state_update(
        state_dict, "block_collapsed_toggle",
        {
            "kind": "block_collapsed_toggle",
            "block_id": container.block_id,
            "ts": 0,
        },
    )
    new_container = next(
        b for b in new_state["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    assert new_container.get("collapsed") is True

    # Second toggle → False.
    newer_state = _dispatch_state_update(
        new_state, "block_collapsed_toggle",
        {
            "kind": "block_collapsed_toggle",
            "block_id": container.block_id,
            "ts": 1,
        },
    )
    newer_container = next(
        b for b in newer_state["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    assert newer_container.get("collapsed") is False


# ---------------------------------------------------------------------------
# drill_into_container (spec §4.4 / §5.6)
# ---------------------------------------------------------------------------


def test_drill_into_container_pushes_breadcrumb(
    patched_registry: Any,
) -> None:
    """``state.breadcrumb`` appends the container's block_id on drill-in."""

    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)
    assert state_dict.get("breadcrumb") == []

    new_state = _dispatch_state_update(
        state_dict, "drill_into_container",
        {
            "kind": "drill_into_container",
            "block_id": container.block_id,
            "ts": 0,
        },
    )

    assert new_state["breadcrumb"] == [container.block_id]
    # Selections cleared (the new scope is its own context).
    assert new_state.get("selected_block_id") is None
    assert new_state.get("selected_edge_id") is None


# ---------------------------------------------------------------------------
# drill_out (spec §4.4 / §5.6)
# ---------------------------------------------------------------------------


def test_drill_out_default_pops_one_segment(patched_registry: Any) -> None:
    """Default ``drill_out`` removes the last breadcrumb segment."""

    container_a = _make_container_block()
    container_b = _make_container_block()
    # Nest container_b inside container_a so the breadcrumb [a, b] is real.
    container_a.nested.blocks.append(container_b)
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container_a]),
        breadcrumb=[container_a.block_id, container_b.block_id],
    )
    state_dict = state_to_json(state)

    new_state = _dispatch_state_update(
        state_dict, "drill_out", {"kind": "drill_out", "ts": 0},
    )

    assert new_state["breadcrumb"] == [container_a.block_id]


# ---------------------------------------------------------------------------
# drill_to_scope (spec §4.4 / §5.6)
# ---------------------------------------------------------------------------


def test_drill_to_scope_replaces_breadcrumb_atomically(
    patched_registry: Any,
) -> None:
    """Single dispatch replaces ``state.breadcrumb`` — no intermediate states."""

    # Build a tree: root → [container_a, container_b] where container_b has
    # an inner container_c.  We can navigate root → [b, c] atomically.
    container_a = _make_container_block()
    container_c = _make_container_block()
    container_b = _make_container_block()
    container_b.nested.blocks.append(container_c)
    state = _DagBuilderState(
        root=_DagBuilderScope(blocks=[container_a, container_b]),
        breadcrumb=[container_a.block_id],
    )
    state_dict = state_to_json(state)

    new_state = _dispatch_state_update(
        state_dict, "drill_to_scope",
        {
            "kind": "drill_to_scope",
            "target_breadcrumb": [container_b.block_id, container_c.block_id],
            "ts": 0,
        },
    )

    assert new_state["breadcrumb"] == [
        container_b.block_id,
        container_c.block_id,
    ]
    # Selections cleared.
    assert new_state.get("selected_block_id") is None
    assert new_state.get("selected_edge_id") is None


def test_drill_to_scope_stale_id_rejects_with_toast(
    patched_registry: Any,
) -> None:
    """Stale (non-existent) block_id in target_breadcrumb → reject + toast."""

    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)
    breadcrumb_before = list(state_dict.get("breadcrumb") or [])
    toast_count_before = len(state_dict.get("toast_queue") or [])

    new_state = _dispatch_state_update(
        state_dict, "drill_to_scope",
        {
            "kind": "drill_to_scope",
            "target_breadcrumb": ["z" * 32],  # stale id, no real container
            "ts": 0,
        },
    )

    # Breadcrumb unchanged.
    assert new_state["breadcrumb"] == breadcrumb_before
    # Toast surfaced.
    toasts = new_state.get("toast_queue") or []
    assert len(toasts) == toast_count_before + 1
    assert "scope" in toasts[-1]["text"].lower() or \
        "container" in toasts[-1]["text"].lower()


# ---------------------------------------------------------------------------
# block_delete_request / block_delete_confirm (spec §5.6)
# ---------------------------------------------------------------------------


def test_block_delete_request_non_empty_container_sets_pending(
    patched_registry: Any,
) -> None:
    """Non-empty container delete request sets ``pending_delete_block_id``.

    The modal opens when ``pending_delete_block_id`` is set; the user must
    Confirm or Cancel.  No actual deletion happens at this stage.
    """

    _seed_reparent_registry(patched_registry)
    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)
    # Populate the container's nested scope with at least one non-InputImage
    # block so the "non-empty" branch fires.
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload(
            "SourceOp", container_block_id=container.block_id,
        ),
    )

    blocks_before = [b["block_id"] for b in state_dict["root"]["blocks"]]
    assert state_dict.get("pending_delete_block_id") is None

    new_state = _dispatch_state_update(
        state_dict, "block_delete_request",
        {
            "kind": "block_delete_request",
            "block_id": container.block_id,
            "ts": 0,
        },
    )

    # Container NOT deleted yet (modal opens).
    assert [
        b["block_id"] for b in new_state["root"]["blocks"]
    ] == blocks_before
    # The pending-delete id matches the container.
    assert new_state["pending_delete_block_id"] == container.block_id


def test_block_delete_request_empty_container_delegates_to_confirm(
    patched_registry: Any,
) -> None:
    """Empty container (only auto-seeded Input Image) delete skips modal.

    Spec §5.6: "container with ZERO non-InputImage children → delegate to
    ``block_delete_confirm`` in same dispatch".
    """

    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)

    new_state = _dispatch_state_update(
        state_dict, "block_delete_request",
        {
            "kind": "block_delete_request",
            "block_id": container.block_id,
            "ts": 0,
        },
    )

    # The container is gone.
    assert all(
        b["block_id"] != container.block_id
        for b in new_state["root"]["blocks"]
    )
    # No pending modal was opened.
    assert new_state.get("pending_delete_block_id") is None


def test_block_delete_request_non_container_delegates_to_confirm(
    patched_registry: Any,
) -> None:
    """Non-container delete request immediately deletes (no modal).

    Spec §5.6: "non-container OR container with ZERO non-InputImage children
    → delegate to ``block_delete_confirm`` in same dispatch".
    """

    _seed_reparent_registry(patched_registry)
    state_dict = _empty_dag_state_dict()
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("SourceOp"),
    )
    src_block_id = next(
        b["block_id"] for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    )

    new_state = _dispatch_state_update(
        state_dict, "block_delete_request",
        {
            "kind": "block_delete_request",
            "block_id": src_block_id,
            "ts": 0,
        },
    )

    # SourceOp deleted directly; no pending-delete state.
    assert all(
        b["block_id"] != src_block_id
        for b in new_state["root"]["blocks"]
    )
    assert new_state.get("pending_delete_block_id") is None


def test_block_delete_confirm_recursively_clears_nested_and_edges(
    patched_registry: Any,
) -> None:
    """Deleting a container also removes its inner blocks + incident edges.

    Spec §5.6: ``block_delete_confirm`` atomically removes the block from
    its scope's blocks list, removes every edge in that scope whose
    source/target matches the block, and the nested scope is GC'd as a
    side effect of removing the parent.
    """

    _seed_reparent_registry(patched_registry)
    container = _make_container_block()
    state = _DagBuilderState(root=_DagBuilderScope(blocks=[container]))
    state_dict = state_to_json(state)
    # Put SourceOp + ConsumerOp into the container and wire them.
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload(
            "SourceOp", container_block_id=container.block_id,
        ),
    )
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload(
            "ConsumerOp", container_block_id=container.block_id,
        ),
    )
    container_dict = next(
        b for b in state_dict["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    src_block_id = next(
        b["block_id"] for b in container_dict["nested"]["blocks"]
        if b["class_name"] == "SourceOp"
    )
    consumer_block_id = next(
        b["block_id"] for b in container_dict["nested"]["blocks"]
        if b["class_name"] == "ConsumerOp"
    )
    # Also drop another op at root and wire something at root scope to make
    # sure we're not accidentally pulling unrelated state.
    state_dict = _dispatch_state_update(
        state_dict, "block_create",
        _make_palette_drop_payload("SourceOp"),
    )
    root_src_id = next(
        b["block_id"] for b in state_dict["root"]["blocks"]
        if b["class_name"] == "SourceOp"
    )
    # Drill into container, wire SRC → CONSUMER inside.
    state_dict = _dispatch_state_update(
        state_dict, "drill_into_container",
        {
            "kind": "drill_into_container",
            "block_id": container.block_id,
            "ts": 0,
        },
    )
    state_dict = _dispatch_state_update(
        state_dict, "edge_create",
        _make_edge_create_payload(
            source_block_id=src_block_id,
            target_block_id=consumer_block_id,
            target_port="scalar_aux",
            edge_kind="aux",
        ),
    )
    # Drill back out so the container is in the current scope for deletion.
    state_dict = _dispatch_state_update(
        state_dict, "drill_out", {"kind": "drill_out", "ts": 0},
    )

    # Sanity: the inner edge exists.
    container_dict_before = next(
        b for b in state_dict["root"]["blocks"]
        if b["block_id"] == container.block_id
    )
    assert len(container_dict_before["nested"].get("edges", []) or []) == 1

    new_state = _dispatch_state_update(
        state_dict, "block_delete_confirm",
        {
            "kind": "block_delete_confirm",
            "block_id": container.block_id,
            "ts": 1,
        },
    )

    # Container gone from root.
    assert all(
        b["block_id"] != container.block_id
        for b in new_state["root"]["blocks"]
    )
    # The unrelated root SourceOp survives.
    assert any(
        b["block_id"] == root_src_id
        for b in new_state["root"]["blocks"]
    )
    # The container's nested SourceOp + ConsumerOp + their edge are gone
    # (GC'd via removal of the parent).
    remaining_classes = [
        b["class_name"] for b in new_state["root"]["blocks"]
    ]
    assert remaining_classes.count("ConsumerOp") == 0


def test_block_delete_request_rejects_input_image(
    patched_registry: Any,
) -> None:
    """Defense-in-depth: ``block_delete_request`` rejects Input Image ids."""

    state_dict = _empty_dag_state_dict()
    input_image_id = next(
        b["block_id"] for b in state_dict["root"]["blocks"]
        if b["class_name"] == INPUT_IMAGE_CLASS_NAME
    )
    blocks_before = [b["block_id"] for b in state_dict["root"]["blocks"]]
    toast_count_before = len(state_dict.get("toast_queue") or [])

    new_state = _dispatch_state_update(
        state_dict, "block_delete_request",
        {
            "kind": "block_delete_request",
            "block_id": input_image_id,
            "ts": 0,
        },
    )

    # Input Image survives.
    assert [
        b["block_id"] for b in new_state["root"]["blocks"]
    ] == blocks_before
    # Toast surfaced.
    toasts = new_state.get("toast_queue") or []
    assert len(toasts) == toast_count_before + 1
    assert "Input Image" in toasts[-1]["text"]
