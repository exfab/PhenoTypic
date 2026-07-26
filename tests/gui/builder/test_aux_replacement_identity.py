"""Revision-bound replacement tests for linear Builder auxiliary values."""

from __future__ import annotations

from typing import Any

import pytest

from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder._callbacks import _dispatch_state_update
from phenotypic.gui.builder._state import (
    _DagBuilderState,
    state_from_json,
    state_to_json,
)


@pytest.fixture(scope="module")
def app_ctx() -> Any:
    """Provide the operation registry used by linear compatibility checks."""

    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)
    with app.server.app_context():
        yield app


def _dispatch(
    state: dict[str, Any],
    kind: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return _dispatch_state_update(state, kind, payload)


def _state_with_scalar_aux() -> tuple[dict[str, Any], str, str]:
    state = state_to_json(_DagBuilderState())
    state = _dispatch(
        state,
        "linear_palette_add",
        {"class_name": "FilamentousFungiDetector"},
    )
    consumer = next(
        block
        for block in state["root"]["blocks"]
        if block["class_name"] == "FilamentousFungiDetector"
    )
    consumer_id = consumer["block_id"]
    target = {
        "kind": "parameter",
        "scope_path": [],
        "block_id": consumer_id,
        "param": "inoculum_detector",
        "slot": None,
    }
    state = _dispatch(
        state,
        "target_select",
        {"target": target, "open_menu": False},
    )
    state = _dispatch(
        state,
        "linear_palette_add",
        {"class_name": "OtsuDetector"},
    )
    edge = next(
        edge
        for edge in state["root"]["edges"]
        if edge.get("target_block_id") == consumer_id
        and edge.get("target_port") == "inoculum_detector"
    )
    return state, consumer_id, edge["source_block_id"]


def test_scalar_aux_replace_consumes_exact_pending_target(app_ctx: Any) -> None:
    """Replace changes the scalar side value without adding a spine node."""

    state, consumer_id, old_source_id = _state_with_scalar_aux()
    target = {
        "kind": "parameter",
        "scope_path": [],
        "block_id": consumer_id,
        "param": "inoculum_detector",
        "slot": None,
    }
    begun = _dispatch(
        state,
        "linear_aux_replace_begin",
        {
            "target": target,
            "source_block_id": old_source_id,
            "nonce": "replace-1",
        },
    )
    assert begun["pending_aux_replacement"]["nonce"] == "replace-1"

    replaced = _dispatch(
        begun,
        "linear_palette_add",
        {"class_name": "MeanDetector"},
    )

    assert replaced["pending_aux_replacement"] is None
    spine_classes = [
        block["class_name"]
        for block in replaced["root"]["blocks"]
        if block["class_name"]
        not in {"InputImage", "OtsuDetector", "MeanDetector"}
    ]
    assert spine_classes == ["FilamentousFungiDetector"]
    aux_edges = [
        edge
        for edge in replaced["root"]["edges"]
        if edge.get("target_block_id") == consumer_id
        and edge.get("target_port") == "inoculum_detector"
    ]
    assert len(aux_edges) == 1
    new_source_id = aux_edges[0]["source_block_id"]
    assert new_source_id != old_source_id
    new_source = next(
        block
        for block in replaced["root"]["blocks"]
        if block["block_id"] == new_source_id
    )
    assert new_source["class_name"] == "MeanDetector"
    assert not any(
        block["block_id"] == old_source_id
        for block in replaced["root"]["blocks"]
    )


def test_stale_aux_replace_never_falls_through_to_spine(app_ctx: Any) -> None:
    """A semantic edit invalidates replacement instead of inserting at top level."""

    state, consumer_id, old_source_id = _state_with_scalar_aux()
    target = {
        "kind": "parameter",
        "scope_path": [],
        "block_id": consumer_id,
        "param": "inoculum_detector",
        "slot": None,
    }
    begun = _dispatch(
        state,
        "linear_aux_replace_begin",
        {
            "target": target,
            "source_block_id": old_source_id,
            "nonce": "replace-stale",
        },
    )
    consumer = next(
        block
        for block in begun["root"]["blocks"]
        if block["block_id"] == consumer_id
    )
    consumer.setdefault("params", {})["min_size"] = 99

    rejected = _dispatch(
        begun,
        "linear_palette_add",
        {"class_name": "MeanDetector"},
    )

    assert rejected["pending_aux_replacement"] is None
    assert not any(
        block["class_name"] == "MeanDetector"
        for block in rejected["root"]["blocks"]
    )
    assert any(
        block["block_id"] == old_source_id
        for block in rejected["root"]["blocks"]
    )
    assert "stale" in rejected["toast_queue"][-1]["text"].lower()


def test_incompatible_aux_choice_preserves_pending_target(app_ctx: Any) -> None:
    """An incompatible choice leaves the exact replacement target selected."""

    state, consumer_id, old_source_id = _state_with_scalar_aux()
    target = {
        "kind": "parameter",
        "scope_path": [],
        "block_id": consumer_id,
        "param": "inoculum_detector",
        "slot": None,
    }
    begun = _dispatch(
        state,
        "linear_aux_replace_begin",
        {
            "target": target,
            "source_block_id": old_source_id,
            "nonce": "replace-after-incompatible",
        },
    )

    incompatible = _dispatch(
        begun,
        "linear_palette_add",
        {"class_name": "NotRegisteredOperation"},
    )

    assert incompatible["pending_aux_replacement"] == (
        begun["pending_aux_replacement"]
    )
    assert not any(
        block["class_name"] == "NotRegisteredOperation"
        for block in incompatible["root"]["blocks"]
    )
    assert "does not accept" in incompatible["toast_queue"][-1]["text"]

    replaced = _dispatch(
        incompatible,
        "linear_palette_add",
        {"class_name": "MeanDetector"},
    )

    assert replaced["pending_aux_replacement"] is None
    spine_classes = [
        block["class_name"]
        for block in replaced["root"]["blocks"]
        if block["class_name"]
        not in {"InputImage", "OtsuDetector", "MeanDetector"}
    ]
    assert spine_classes == ["FilamentousFungiDetector"]
    replacement_edge = next(
        edge
        for edge in replaced["root"]["edges"]
        if edge.get("target_block_id") == consumer_id
        and edge.get("target_port") == "inoculum_detector"
    )
    assert replacement_edge["source_block_id"] != old_source_id


def test_list_aux_replace_rejects_scalar_target_shape(app_ctx: Any) -> None:
    """A list-valued parameter cannot be replaced through a scalar target."""

    state = state_to_json(_DagBuilderState())
    state = _dispatch(
        state,
        "linear_palette_add",
        {"class_name": "CompositeDetector"},
    )
    consumer = next(
        block
        for block in state["root"]["blocks"]
        if block["class_name"] == "CompositeDetector"
    )
    consumer_id = consumer["block_id"]
    slot_target = {
        "kind": "parameter_slot",
        "scope_path": [],
        "block_id": consumer_id,
        "param": "ops",
        "slot": 0,
    }
    state = _dispatch(
        state,
        "target_select",
        {"target": slot_target, "open_menu": False},
    )
    state = _dispatch(
        state,
        "linear_palette_add",
        {"class_name": "OtsuDetector"},
    )
    edge = next(
        edge
        for edge in state["root"]["edges"]
        if edge.get("target_block_id") == consumer_id
        and edge.get("target_port") == "ops"
    )

    rejected = _dispatch(
        state,
        "linear_aux_replace_begin",
        {
            "target": {
                **slot_target,
                "kind": "parameter",
                "slot": None,
            },
            "source_block_id": edge["source_block_id"],
            "nonce": "replace-malformed-list",
        },
    )

    assert rejected["pending_aux_replacement"] is None
    assert "shape is invalid" in rejected["toast_queue"][-1]["text"]
    assert any(
        block["block_id"] == edge["source_block_id"]
        for block in rejected["root"]["blocks"]
    )


def test_pending_aux_replacement_round_trips(app_ctx: Any) -> None:
    """The exact pending identity survives state serialization."""

    state, consumer_id, old_source_id = _state_with_scalar_aux()
    begun = _dispatch(
        state,
        "linear_aux_replace_begin",
        {
            "target": {
                "kind": "parameter",
                "scope_path": [],
                "block_id": consumer_id,
                "param": "inoculum_detector",
                "slot": None,
            },
            "source_block_id": old_source_id,
            "nonce": "replace-roundtrip",
        },
    )

    rebuilt = state_from_json(begun)
    assert state_to_json(rebuilt)["pending_aux_replacement"] == (
        begun["pending_aux_replacement"]
    )
