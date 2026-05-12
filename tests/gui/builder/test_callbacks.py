"""Unit tests for the pure mutation helper in :mod:`_callbacks`.

These tests exercise :func:`_dispatch_state_update` without booting a Dash
server: it's a JSON-in / JSON-out function, so we can validate the full
mutation surface in milliseconds.
"""

from __future__ import annotations

from dash import html

from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder._callbacks import _dispatch_state_update
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder._state import (
    BuilderScope,
    BuilderState,
    StepNode,
    state_to_json,
)


def _seed_state() -> dict:
    """Return a JSON state dict with two ops at the root scope."""

    state = BuilderState(
        root=BuilderScope(
            nodes=[
                StepNode(node_id="aaa", class_name="GaussianBlur"),
                StepNode(node_id="bbb", class_name="OtsuDetector"),
            ],
            name="root",
        ),
    )
    return state_to_json(state)


def test_render_views_returns_breadcrumb_children_not_nested_nav() -> None:
    """Callback payload must update ``breadcrumb.children``, not nest another nav."""

    from phenotypic.gui.builder._callbacks import _render_views

    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)
    state = BuilderState()

    with app.server.app_context():
        breadcrumb_children, _canvas, _inspector, _popover = _render_views(state)

    assert isinstance(breadcrumb_children, list)
    assert not any(isinstance(child, html.Nav) for child in breadcrumb_children)


def test_render_views_drilled_in_breadcrumb_returns_button_children() -> None:
    """Drilled-in state must render ancestor buttons + separator inline, not nest a nav."""

    from phenotypic.gui.builder._callbacks import _render_views

    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)

    state = BuilderState(
        root=BuilderScope(
            nodes=[
                StepNode(
                    node_id="pipe1",
                    class_name="ImagePipeline",
                    label="Subpipeline",
                    nested=BuilderScope(
                        nodes=[StepNode(node_id="aaa", class_name="GaussianBlur")],
                        name="Subpipeline",
                    ),
                ),
            ],
        ),
        breadcrumb=[{"node_id": "pipe1", "param": None}],
    )

    with app.server.app_context():
        breadcrumb_children, _canvas, _inspector, _popover = _render_views(state)

    assert isinstance(breadcrumb_children, list)
    assert not any(isinstance(child, html.Nav) for child in breadcrumb_children)
    # Two-segment path → ancestor button + " / " separator + active span (3 children).
    assert len(breadcrumb_children) >= 3


def test_add_node_appends_and_selects() -> None:
    """``add_node`` appends to the visible scope and updates ``selected_node_id``."""

    state = _seed_state()
    out = _dispatch_state_update(state, "add_node", {"class_name": "MeasureSize"})

    assert len(out["root"]["nodes"]) == 3
    assert out["root"]["nodes"][-1]["class_name"] == "MeasureSize"
    assert out["selected_node_id"] == out["root"]["nodes"][-1]["node_id"]
    # Original input is not mutated.
    assert len(state["root"]["nodes"]) == 2


def test_delete_node_removes_selection() -> None:
    """``delete_node`` removes the selected node and clears ``selected_node_id``."""

    state = _seed_state()
    state["selected_node_id"] = "aaa"

    out = _dispatch_state_update(state, "delete_node", {})
    remaining = [n["node_id"] for n in out["root"]["nodes"]]
    assert remaining == ["bbb"]
    assert out["selected_node_id"] is None


def test_edit_param_writes_into_node() -> None:
    """``edit_param`` writes ``params[name]`` into the addressed node."""

    state = _seed_state()

    out = _dispatch_state_update(
        state,
        "edit_param",
        {"node_id": "aaa", "name": "sigma", "value": 2.5, "omit": False},
    )
    blur = next(n for n in out["root"]["nodes"] if n["node_id"] == "aaa")
    assert blur["params"]["sigma"] == 2.5


def test_edit_param_with_omit_strips_key() -> None:
    """``edit_param`` with ``omit=True`` removes the param entirely."""

    state = _seed_state()
    state["root"]["nodes"][0]["params"]["sigma"] = 1.0

    out = _dispatch_state_update(
        state,
        "edit_param",
        {"node_id": "aaa", "name": "sigma", "omit": True},
    )
    blur = next(n for n in out["root"]["nodes"] if n["node_id"] == "aaa")
    assert "sigma" not in blur["params"]


def test_reorder_swaps_nodes() -> None:
    """``reorder`` re-sequences the visible scope to match the supplied order."""

    state = _seed_state()
    out = _dispatch_state_update(
        state,
        "reorder",
        {"order": ["bbb", "aaa"]},
    )
    assert [n["node_id"] for n in out["root"]["nodes"]] == ["bbb", "aaa"]


def test_drill_in_pushes_breadcrumb_when_node_has_nested_scope() -> None:
    """Drill-in pushes onto the breadcrumb only when the node has ``nested``."""

    state = state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[
                    StepNode(
                        node_id="pipe",
                        class_name="ImagePipeline",
                        nested=BuilderScope(name="inner"),
                    ),
                ],
            ),
            selected_node_id="pipe",
        )
    )
    out = _dispatch_state_update(state, "drill_in", {})
    assert out["breadcrumb"] == [{"node_id": "pipe", "param": None}]
    assert out["selected_node_id"] is None


def test_drill_in_param_round_trip() -> None:
    """Drilling into an op-typed param scope and back commits the inner node.

    The mutation should:

    1. Append a ``{"node_id", "param"}`` segment to the breadcrumb.
    2. Seed an empty singleton scope under the param key.
    3. After adding a node inside and walking back, the inner node's
       serialized operation marker should land on
       ``parent.params[param_name]``.
    """

    state = state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[
                    StepNode(node_id="parent", class_name="GaussianBlur"),
                ],
            ),
            selected_node_id="parent",
        )
    )

    # Drill in.
    after_drill = _dispatch_state_update(
        state,
        "drill_in_param",
        {"node_id": "parent", "param_name": "sub_op"},
    )
    assert after_drill["breadcrumb"] == [
        {"node_id": "parent", "param": "sub_op"}
    ]

    # Add a node inside the singleton scope (drill_in_param already pushed
    # the breadcrumb, so subsequent ``add_node`` lands inside the param scope).
    after_add = _dispatch_state_update(
        after_drill, "add_node", {"class_name": "OtsuDetector"}
    )
    parent = after_add["root"]["nodes"][0]
    inner_scope = parent["params"]["__op_param_scope__"]["sub_op"]
    assert len(inner_scope["nodes"]) == 1
    assert inner_scope["nodes"][0]["class_name"] == "OtsuDetector"

    # Drill back out via breadcrumb_to depth=0.
    after_out = _dispatch_state_update(
        after_add, "breadcrumb_to", {"depth": 0}
    )
    parent = after_out["root"]["nodes"][0]
    marker = parent["params"]["sub_op"]
    assert marker is not None
    assert marker["__type__"] == "operation"
    assert marker["class_name"] == "OtsuDetector"


def test_breadcrumb_to_pops_aux_slot_segment() -> None:
    """``breadcrumb_to`` pops aux-slot segments without crashing.

    Regression cover for a ``KeyError 'node_id'`` raised by
    ``_commit_param_segments`` when an aux-slot breadcrumb segment
    (``{"target_node_id", "param", "slot"}``) is popped. The legacy
    ``drill_in_param`` mechanism stores synthesized scopes that need to be
    folded back into ``parent.params[name]`` on drill-out, but aux-slot
    drills (``drill_in_aux``) embed their aux ``StepNode`` directly inside
    ``consumer.aux_ports[param][slot]`` — those segments must be skipped
    here, not commit-back-walked.
    """

    state = state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[
                    StepNode(
                        node_id="consumer",
                        class_name="FilamentousFungiDetector",
                        aux_ports={
                            "inoculum_detector": [
                                StepNode(
                                    node_id="aux1",
                                    class_name="OtsuDetector",
                                ),
                            ],
                        },
                    ),
                ],
            ),
        )
    )

    after_drill = _dispatch_state_update(
        state,
        "drill_in_aux",
        {
            "target_node_id": "consumer",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )
    assert after_drill["breadcrumb"] == [
        {
            "target_node_id": "consumer",
            "param": "inoculum_detector",
            "slot": 0,
        }
    ]

    after_out = _dispatch_state_update(
        after_drill, "breadcrumb_to", {"depth": 0}
    )
    assert after_out["breadcrumb"] == []
    # The wired aux survives the drill-out — it lives inside the consumer's
    # ``aux_ports`` slot list, not under ``params``.
    consumer = after_out["root"]["nodes"][0]
    slots = consumer["aux_ports"]["inoculum_detector"]
    assert slots[0]["class_name"] == "OtsuDetector"

    # ``drill_out`` (the single-step variant) takes the same code path.
    after_step_out = _dispatch_state_update(
        after_drill, "drill_out", {}
    )
    assert after_step_out["breadcrumb"] == []
    slots = after_step_out["root"]["nodes"][0]["aux_ports"]["inoculum_detector"]
    assert slots[0]["class_name"] == "OtsuDetector"
