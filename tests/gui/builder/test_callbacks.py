"""Unit tests for the pure mutation helper in :mod:`_callbacks`.

These tests exercise :func:`_dispatch_state_update` without booting a Dash
server: it's a JSON-in / JSON-out function, so we can validate the full
mutation surface in milliseconds.
"""

from __future__ import annotations

from copy import deepcopy

from dash import html

from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._callbacks import (
    _dispatch_state_update,
    _new_preview_request,
    _pipeline_revision,
    _preview_request_is_valid,
    _preview_result_event,
    _preview_result_is_current,
    _preview_status_presentation,
    _reserve_preview_request,
)
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder._state import (
    _DagBuilderState,
    _LegacyBuilderScope as BuilderScope,
    _LegacyBuilderState as BuilderState,
    StepNode,
    state_to_json,
)


def _seed_state() -> dict:
    """Return a JSON state dict with two ops at the root scope."""

    state = BuilderState(
        root=BuilderScope(
            nodes=[
                StepNode(node_id="aaa", class_name="BlurGauss"),
                StepNode(node_id="bbb", class_name="OtsuDetector"),
            ],
            name="root",
        ),
    )
    return state_to_json(state)


def _walk_components(component):
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk_components(child)
        return
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    yield from _walk_components(children)


def test_pipeline_revision_excludes_selection_only_state() -> None:
    state = _seed_state()
    selected = deepcopy(state)
    selected["selected_node_id"] = "bbb"

    assert _pipeline_revision(selected) == _pipeline_revision(state)


def test_pipeline_revision_changes_for_parameter_edit() -> None:
    state = _seed_state()
    edited = deepcopy(state)
    edited["root"]["nodes"][0]["params"]["sigma"] = 2.0

    assert _pipeline_revision(edited) != _pipeline_revision(state)


def test_preview_error_becomes_stale_after_semantic_edit() -> None:
    """An error from an older pipeline revision cannot remain authoritative."""

    state = _seed_state()
    status = {
        "state": "error",
        "message": "ValueError: invalid sigma",
        "pipeline_revision": _pipeline_revision(state),
    }
    assert _preview_status_presentation(status, state) == (
        "ValueError: invalid sigma",
        "small text-danger mt-2",
    )

    edited = deepcopy(state)
    edited["root"]["nodes"][0]["params"]["sigma"] = 2.0
    assert _preview_status_presentation(status, edited) == (
        "Preview stale - run again",
        "small text-warning mt-2",
    )


def test_preview_running_is_revision_bound_and_replaces_old_terminal() -> None:
    """A launch is immediately authoritative only for its exact revision."""

    state = _seed_state()
    request, running = _new_preview_request(
        kind="full",
        state_data=state,
        session_id="session-1",
        image_path=None,
        nrows=None,
        ncols=None,
    )
    request, running = _reserve_preview_request(request, running)

    assert running == {
        "state": "running",
        "message": "Preview running…",
        "request_id": request["request_id"],
        "session_id": "session-1",
        "pipeline_revision": _pipeline_revision(state),
    }
    assert _preview_status_presentation(running, state) == (
        "Preview running…",
        "small text-primary mt-2",
    )

    edited = deepcopy(state)
    edited["root"]["nodes"][0]["params"]["sigma"] = 2.0
    assert _preview_status_presentation(running, edited) == (
        "Preview stale - run again",
        "small text-warning mt-2",
    )


def test_preview_terminal_publication_rejects_superseded_request() -> None:
    """An old terminal event cannot replace a newer running generation."""

    state = _seed_state()
    old_request, old_running = _new_preview_request(
        kind="full",
        state_data=state,
        session_id="session-1",
        image_path=None,
        nrows=None,
        ncols=None,
    )
    old_request, _old_running = _reserve_preview_request(
        old_request,
        old_running,
    )
    newer_request, new_running = _new_preview_request(
        kind="full",
        state_data=state,
        session_id="session-1",
        image_path=None,
        nrows=None,
        ncols=None,
    )
    newer_request, _new_running = _reserve_preview_request(
        newer_request,
        new_running,
    )
    old_result = _preview_result_event(
        old_request,
        state="complete",
        message="Preview complete",
        intermediate_keys=["aaa"],
        preview_snapshot={
            "pipeline_revision": old_request["pipeline_revision"],
            "preview_generation": 1,
        },
    )

    assert _preview_result_is_current(old_result, old_request)
    assert not _preview_result_is_current(old_result, newer_request)


def test_preview_complete_result_requires_exact_generation_descriptor() -> None:
    """A malformed or cross-revision snapshot fails closed at publication."""

    state = _seed_state()
    request, running = _new_preview_request(
        kind="full",
        state_data=state,
        session_id="session-1",
        image_path=None,
        nrows=None,
        ncols=None,
    )
    request, _running = _reserve_preview_request(request, running)
    malformed = _preview_result_event(
        request,
        state="complete",
        message="Preview complete",
        intermediate_keys=["aaa"],
        preview_snapshot={
            "pipeline_revision": "different-revision",
            "preview_generation": True,
        },
    )

    assert not _preview_result_is_current(malformed, request)


def test_preview_request_schema_rejects_client_edits() -> None:
    """Every work field is validated before cache claim or computation."""

    state = _seed_state()
    request, running = _new_preview_request(
        kind="full",
        state_data=state,
        session_id="session-1",
        image_path=None,
        nrows=None,
        ncols=None,
    )
    request, _running = _reserve_preview_request(request, running)
    assert _preview_request_is_valid(request)

    for key, value in (
        ("request_id", "not-a-uuid"),
        ("pipeline_revision", "0" * 64),
        ("cache_request_sequence", True),
        ("image_path", ["not", "a", "path"]),
        ("nrows", float("nan")),
        ("kind", "unknown"),
    ):
        malformed = deepcopy(request)
        malformed[key] = value
        assert not _preview_request_is_valid(malformed), key

    extra = deepcopy(request)
    extra["unrecognised"] = True
    assert not _preview_request_is_valid(extra)


def test_render_views_returns_breadcrumb_children_not_nested_nav() -> None:
    """Callback payload must update ``breadcrumb.children``, not nest another nav."""

    from phenotypic.gui.builder._callbacks import _render_views

    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)
    state = BuilderState()

    with app.server.app_context():
        breadcrumb_children, _canvas, _inspector = _render_views(state)

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
                        nodes=[StepNode(node_id="aaa", class_name="BlurGauss")],
                        name="Subpipeline",
                    ),
                ),
            ],
        ),
        breadcrumb=[{"node_id": "pipe1", "param": None}],
    )

    with app.server.app_context():
        breadcrumb_children, _canvas, _inspector = _render_views(state)

    assert isinstance(breadcrumb_children, list)
    assert not any(isinstance(child, html.Nav) for child in breadcrumb_children)
    # Two-segment path → ancestor button + " / " separator + active span (3 children).
    assert len(breadcrumb_children) >= 3


def test_render_views_returns_side_loader_children_not_nested_container() -> None:
    """Inspector callback payload must not nest ``#inspector`` in itself."""

    from phenotypic.gui.builder._callbacks import _render_views

    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)

    with app.server.app_context():
        _breadcrumb, _canvas, inspector_children = _render_views(_DagBuilderState())

    assert isinstance(inspector_children, list)
    assert not any(
        getattr(child, "id", None) == ids.INSPECTOR_CONTAINER
        for child in _walk_components(inspector_children)
    )


def test_no_callback_replaces_stable_inspector_shell_children() -> None:
    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)

    output_keys = tuple(app.callback_map)
    assert not any(
        f"{ids.INSPECTOR_CONTAINER}.children" in key for key in output_keys
    )
    assert any(f"{ids.INSPECTOR_CONTENT}.children" in key for key in output_keys)


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
