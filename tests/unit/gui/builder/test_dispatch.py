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
from typing import Any, Dict, Optional

import pytest

from phenotypic.gui._operation_registry import OperationInfo, ParamInfo
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_op_info(
    cls_name: str,
    parameters: Optional[Dict[str, ParamInfo]] = None,
    *,
    category: str = "Enhancer",
) -> OperationInfo:
    """Build a stub :class:`OperationInfo` for the test registry.

    Mirrors the shape used by ``test_validation.py`` / ``test_recovery.py``.
    """

    class _StubCls:
        pass

    _StubCls.__name__ = cls_name
    return OperationInfo(
        cls=_StubCls,
        name=cls_name,
        category=category,
        module="tests.fake",
        docstring="",
        parameters=parameters or {},
    )


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
