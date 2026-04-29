"""Dash ``@callback`` registrations for the pipeline builder.

Phase 3 owns every callback that reads or writes
:data:`phenotypic.gui.builder._ids.STORE_BUILDER_STATE` plus the few helper
callbacks that drive the directory picker, preview, and inspector preview
panel.

Architecture notes:

* **Single fan-in mutation callback** — every trigger that mutates the
  builder state (palette adds, deletes, drill-in/out, breadcrumb jumps,
  reorders, drag, parameter edits, label edits) feeds one callback that
  resolves the trigger via :data:`dash.callback_context.triggered_id` and
  returns a fresh ``STORE_BUILDER_STATE`` plus a re-rendered canvas /
  inspector / breadcrumb.  This keeps ``allow_duplicate=True`` out of the
  store contract.
* **Stateless helpers** — :func:`_dispatch_state_update` is a pure function
  taking a serialized state dict + a kind tag + a payload and returning the
  new dict.  Tests can exercise the full mutation surface without booting
  Dash.
* **Drill-in surface** — the inspector reuses :data:`BTN_DRILL_OUT` as the
  drill-in button when a pipeline node is selected (set by Phase 2).  The
  fan-in handler distinguishes "drill-in" from "drill-out" by inspecting the
  selected node's class at trigger time, so the same component id maps to
  two semantic actions.
* **Cytoscape reorder** — we listen to ``Input(CANVAS_CYTOSCAPE, "elements")``
  and, when the visible node-id sequence (filtered to non-edge entries and
  sorted by ``position.x``) differs from the state-derived order, reorder
  ``current_scope(state).nodes`` to match.  Reorder is a no-op while no node
  has a ``position`` (cytoscape's grid layout fills positions on the second
  render).

Every state-mutating callback uses ``prevent_initial_call=True`` and wraps
its body in ``try / except`` so a callback can never crash the running app
— errors flip the toast to the failure variant instead.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import ALL, Input, Output, State, ctx, html, no_update
from flask import current_app

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._directory_browser import (
    DIR_PICKER_PATH_INPUT,
    DIR_PICKER_SYNTH_BTN,
    DIR_PICKER_TREE_CONTAINER,
    DIR_PICKER_USE_PATH_BTN,
    SYNTHETIC_SENTINEL,
    directory_tree,
    find_synthetic_plate_path,
)
from phenotypic.gui.builder._image_renderer import dataframe_to_table, to_data_uri
from phenotypic.gui.builder._layout import (
    build_breadcrumb,
    build_canvas,
    build_inspector,
)
from phenotypic.gui.builder._param_form import parse_widget_value
from phenotypic.gui.builder._session import get_cache
from phenotypic.gui.builder._state import (
    PIPELINE_CLASS_NAME,
    BuilderState,
    StepNode,
    _PARAM_SCOPE_KEY,
    _new_node_id,
    current_scope,
    from_pipeline,
    stage_of,
    state_from_json,
    state_to_json,
    to_pipeline,
)

logger = logging.getLogger(__name__)


# ``STORE_IMAGE_PATH`` lives on :mod:`._ids` now; alias here for backwards
# compatibility with callers that imported it from this module.
STORE_IMAGE_PATH = ids.STORE_IMAGE_PATH


# ---------------------------------------------------------------------------
# Pure mutation helpers (testable without Dash)
# ---------------------------------------------------------------------------


def _normalize_segment(seg: Any) -> Dict[str, Any]:
    """Return *seg* as a ``{"node_id": ..., "param": ...}`` dict.

    Accepts both legacy plain-string entries (older saved state) and the
    new dict form so dispatch logic can be agnostic to the storage format.
    """

    if isinstance(seg, str):
        return {"node_id": seg, "param": None}
    return {"node_id": seg["node_id"], "param": seg.get("param")}


def _scope_at_breadcrumb(
    state_dict: Dict[str, Any], breadcrumb: List[Any]
) -> Dict[str, Any]:
    """Walk *state_dict* ``root`` → ``breadcrumb`` and return that scope dict.

    For pipeline drill segments the walker descends into ``node["nested"]``;
    for op-typed-parameter segments it descends into the synthesised scope
    dict stored under ``node["params"]["__op_param_scope__"][param_name]``,
    seeding it from any existing operation marker dict if absent.

    Args:
        state_dict: ``state_to_json`` output.
        breadcrumb: List of breadcrumb segments (string ids or
            ``{"node_id", "param"}`` dicts).

    Returns:
        The nested scope dict (``{"nodes": [...], ...}``).  Returns the root
        scope when ``breadcrumb`` is empty.
    """

    scope = state_dict["root"]
    for raw in breadcrumb:
        seg = _normalize_segment(raw)
        node = next(
            (n for n in scope["nodes"] if n["node_id"] == seg["node_id"]),
            None,
        )
        if node is None:
            return scope
        if seg["param"] is None:
            if node.get("nested") is None:
                return scope
            scope = node["nested"]
        else:
            params = node.setdefault("params", {})
            scopes = params.setdefault(_PARAM_SCOPE_KEY, {})
            param_scope = scopes.get(seg["param"])
            if param_scope is None:
                param_scope = _seed_param_scope_from_marker(
                    params.get(seg["param"]), seg["param"], node
                )
                scopes[seg["param"]] = param_scope
            scope = param_scope
    return scope


def _seed_param_scope_from_marker(
    existing: Any, param_name: str, parent_node: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a fresh scope dict, optionally seeding a single node from *existing*.

    Mirrors :func:`phenotypic.gui.builder._state._ensure_param_scope` at the
    JSON-dict level so dispatch can stay schema-agnostic.
    """

    seed_nodes: List[Dict[str, Any]] = []
    if isinstance(existing, dict):
        marker_type = existing.get("__type__")
        if marker_type == "operation":
            class_name = existing.get("class_name") or existing.get("class")
            seed_nodes.append(
                {
                    "node_id": _new_node_id(),
                    "class_name": str(class_name),
                    "params": dict(existing.get("params") or {}),
                    "label": str(class_name),
                    "nested": None,
                }
            )
        elif marker_type in {"pipeline", "pipeline_operation"}:
            inner = existing.get("scope") or existing.get("config") or {}
            seed_nodes.append(
                {
                    "node_id": _new_node_id(),
                    "class_name": PIPELINE_CLASS_NAME,
                    "params": {},
                    "label": PIPELINE_CLASS_NAME,
                    "nested": inner if isinstance(inner, dict) else {"nodes": []},
                }
            )

    label = parent_node.get("label") or parent_node.get("class_name") or "node"
    return {
        "nodes": seed_nodes,
        "name": f"{label}.{param_name}",
        "desc": "",
        "nrows": None,
        "ncols": None,
    }


def _commit_param_segments(
    state_dict: Dict[str, Any], dropped: List[Dict[str, Any]]
) -> None:
    """Mirror any popped param-scope segments back into their parent params.

    ``dropped`` is the suffix of ``breadcrumb`` that's about to be removed
    (innermost first does *not* matter — we re-walk from root each time).
    For each segment whose ``param`` is set we collapse its synthesized
    singleton scope into a normal operation marker stored at
    ``parent.params[param_name]``.

    The scope dict under ``__op_param_scope__`` is preserved so re-entering
    the slot keeps the same node ids and incidental UI state.
    """

    for raw in dropped:
        seg = _normalize_segment(raw)
        if seg["param"] is None:
            continue
        # Walk fresh from root to find the parent node.
        scope = state_dict["root"]
        # Path leading up to (but not including) this segment is everything
        # before ``seg`` in the breadcrumb.  We don't have that path here,
        # so we instead search the whole tree depth-first for the matching
        # node id.  Param scopes attach uniquely by id; collisions would be
        # a state-model bug not addressed here.
        target = _find_node_by_id(scope, seg["node_id"])
        if target is None:
            continue
        scopes = target.get("params", {}).get(_PARAM_SCOPE_KEY) or {}
        param_scope = scopes.get(seg["param"])
        if param_scope is None:
            continue
        nodes = param_scope.get("nodes") or []
        if not nodes:
            target["params"][seg["param"]] = None
            continue
        first = nodes[0]
        if first.get("class_name") == PIPELINE_CLASS_NAME:
            target["params"][seg["param"]] = {
                "__type__": "pipeline",
                "class_name": PIPELINE_CLASS_NAME,
                "scope": first.get("nested") or {"nodes": []},
            }
        else:
            inner_params = {
                k: v
                for k, v in (first.get("params") or {}).items()
                if k != _PARAM_SCOPE_KEY
            }
            target["params"][seg["param"]] = {
                "__type__": "operation",
                "class_name": first.get("class_name"),
                "params": inner_params,
            }


def _find_node_by_id(scope: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    """Depth-first search for a node by id across all nested scopes."""

    for n in scope.get("nodes", []) or []:
        if n.get("node_id") == node_id:
            return n
        nested = n.get("nested")
        if isinstance(nested, dict):
            hit = _find_node_by_id(nested, node_id)
            if hit is not None:
                return hit
        param_scopes = (n.get("params") or {}).get(_PARAM_SCOPE_KEY) or {}
        for inner in param_scopes.values():
            if isinstance(inner, dict):
                hit = _find_node_by_id(inner, node_id)
                if hit is not None:
                    return hit
    return None


def _default_params_for(class_name: str) -> Dict[str, Any]:
    """Return a JSON-friendly default-param dict for *class_name*.

    The registry's :class:`ParamInfo.default` is used directly when it is JSON
    serialisable; non-trivial defaults (e.g. nested ops) are skipped because
    they need a follow-up "Edit ▸" interaction.

    Args:
        class_name: Registry key.

    Returns:
        A dict mapping parameter name → default value.  Empty when the class
        is not registered (the caller will simply emit an empty params dict).
    """

    from phenotypic.gui._operation_registry import get_registry

    info = get_registry().get(class_name)
    if info is None:
        return {}

    out: Dict[str, Any] = {}
    for name, p in info.parameters.items():
        if not p.has_default:
            continue
        if p.is_operation or p.is_pipeline:
            # Skip nested-op defaults; the user will fill them via "Edit ▸".
            continue
        try:
            json.dumps(p.default)
        except (TypeError, ValueError):
            continue
        out[name] = p.default
    return out


def _dispatch_state_update(
    state_dict: Dict[str, Any], kind: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply a named mutation to a serialized state dict.

    The pipeline builder funnels every state-mutating event through this
    function so callbacks stay thin and the mutation surface stays unit-
    testable.  Unknown *kind* values pass through unchanged.

    Args:
        state_dict: Output of :func:`state_to_json` (a plain ``dict``).
        kind: One of:

            ``"add_node"`` (payload: ``class_name``)
                Append a fresh :class:`StepNode` for *class_name* to the
                current scope.  Selects the new node.
            ``"add_pipeline"``
                Append a :class:`StepNode` carrying an empty nested scope.
            ``"select_node"`` (payload: ``node_id``)
                Set ``selected_node_id``.
            ``"delete_node"``
                Remove ``selected_node_id`` from the current scope.
            ``"drill_in"``
                Push ``selected_node_id`` onto the breadcrumb (only if that
                node has a nested scope).
            ``"drill_out"``
                Pop the breadcrumb tail.
            ``"breadcrumb_to"`` (payload: ``depth``)
                Truncate breadcrumb to ``depth`` entries.
            ``"reorder"`` (payload: ``order`` list of node-ids)
                Reorder the current scope's nodes by node-id sequence.
            ``"edit_param"`` (payload: ``node_id``, ``name``, ``value``,
            ``omit``)
                Set or delete ``params[name]`` on a specific node.
            ``"edit_label"`` (payload: ``node_id``, ``label``)
                Update the node label.

        payload: kind-specific data (see above).

    Returns:
        A *new* state dict reflecting the mutation; the input dict is not
        modified in place.

    Raises:
        ValueError: If a payload is missing a required key or refers to a
            node that does not exist in the relevant scope.
    """

    out = deepcopy(state_dict)
    breadcrumb = list(out.get("breadcrumb", []) or [])
    scope = _scope_at_breadcrumb(out, breadcrumb)

    if kind == "add_node":
        class_name = payload["class_name"]
        node_id = _new_node_id()
        scope["nodes"].append(
            {
                "node_id": node_id,
                "class_name": class_name,
                "params": _default_params_for(class_name),
                "label": None,
                "nested": None,
            }
        )
        out["selected_node_id"] = node_id
        return out

    if kind == "add_pipeline":
        node_id = _new_node_id()
        scope["nodes"].append(
            {
                "node_id": node_id,
                "class_name": PIPELINE_CLASS_NAME,
                "params": {},
                "label": None,
                "nested": {
                    "nodes": [],
                    "name": "Subpipeline",
                    "desc": "",
                    "nrows": None,
                    "ncols": None,
                },
            }
        )
        out["selected_node_id"] = node_id
        return out

    if kind == "select_node":
        out["selected_node_id"] = payload.get("node_id")
        return out

    if kind == "delete_node":
        sel = out.get("selected_node_id")
        if sel is None:
            return out
        scope["nodes"] = [n for n in scope["nodes"] if n["node_id"] != sel]
        out["selected_node_id"] = None
        return out

    if kind == "drill_in":
        sel = out.get("selected_node_id")
        if sel is None:
            return out
        node = next((n for n in scope["nodes"] if n["node_id"] == sel), None)
        if node is None or node.get("nested") is None:
            return out
        breadcrumb.append({"node_id": sel, "param": None})
        out["breadcrumb"] = breadcrumb
        out["selected_node_id"] = None
        return out

    if kind == "drill_out":
        if breadcrumb:
            dropped = [breadcrumb.pop()]
            _commit_param_segments(out, dropped)
        out["breadcrumb"] = breadcrumb
        out["selected_node_id"] = None
        return out

    if kind == "breadcrumb_to":
        depth = int(payload.get("depth", 0))
        dropped = breadcrumb[depth:]
        _commit_param_segments(out, dropped)
        out["breadcrumb"] = breadcrumb[:depth]
        out["selected_node_id"] = None
        return out

    if kind == "drill_in_param":
        node_id = payload["node_id"]
        param_name = payload["param_name"]
        node = next((n for n in scope["nodes"] if n["node_id"] == node_id), None)
        if node is None:
            return out
        params = node.setdefault("params", {})
        scopes = params.setdefault(_PARAM_SCOPE_KEY, {})
        if scopes.get(param_name) is None:
            scopes[param_name] = _seed_param_scope_from_marker(
                params.get(param_name), param_name, node
            )
        breadcrumb.append({"node_id": node_id, "param": param_name})
        out["breadcrumb"] = breadcrumb
        out["selected_node_id"] = None
        return out

    if kind == "reorder":
        new_order: List[str] = list(payload.get("order", []) or [])
        if not new_order:
            return out
        by_id = {n["node_id"]: n for n in scope["nodes"]}
        # Drop unknown ids and append any nodes missing from new_order at the
        # end so we can never accidentally lose data on a partial drag event.
        reordered = [by_id[nid] for nid in new_order if nid in by_id]
        leftover = [n for n in scope["nodes"] if n["node_id"] not in set(new_order)]
        scope["nodes"] = reordered + leftover
        return out

    if kind == "edit_param":
        node_id = payload["node_id"]
        node = next((n for n in scope["nodes"] if n["node_id"] == node_id), None)
        if node is None:
            return out
        if payload.get("omit"):
            node.get("params", {}).pop(payload["name"], None)
        else:
            params = node.setdefault("params", {})
            params[payload["name"]] = payload.get("value")
        return out

    if kind == "edit_label":
        node_id = payload["node_id"]
        node = next((n for n in scope["nodes"] if n["node_id"] == node_id), None)
        if node is None:
            return out
        node["label"] = payload.get("label") or None
        return out

    return out


# ---------------------------------------------------------------------------
# Toast helpers
# ---------------------------------------------------------------------------


def _toast(
    message: str, *, ok: bool = True, header: Optional[str] = None
) -> Tuple[bool, str, str, str]:
    """Build the four toast outputs (``is_open``, ``children``, ``icon``, ``header``).

    Args:
        message: Text shown inside the toast body.
        ok: ``True`` for success styling (primary), ``False`` for error
            (danger).
        header: Optional header override; defaults to "Pipeline builder" for
            success or "Error" for failure.

    Returns:
        Tuple matching ``Output(TOAST_NOTIFICATION, "is_open" / "children" /
        "icon" / "header")``.
    """

    icon = "primary" if ok else "danger"
    h = header if header is not None else ("Pipeline builder" if ok else "Error")
    return True, message, icon, h


def _format_exception(exc: BaseException) -> str:
    """Pretty short single-line summary of an exception for toast display."""

    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Layout-render helpers (re-render canvas/inspector/breadcrumb after a state
# mutation)
# ---------------------------------------------------------------------------


def _registry() -> Any:
    """Return the registry stashed on ``app.server.config`` by ``create_app``."""

    return current_app.config.get("pheno_registry")


def _render_views(state: BuilderState) -> Tuple[Any, Any, Any]:
    """Re-render breadcrumb, canvas, and inspector for a given state.

    Args:
        state: Live :class:`BuilderState` object.

    Returns:
        Tuple of ``(breadcrumb, canvas, inspector)`` Dash component subtrees.
    """

    registry = _registry()
    try:
        scope = current_scope(state)
    except KeyError:
        # Stale breadcrumb — fall back to the root.
        state = BuilderState(
            root=state.root, breadcrumb=[], selected_node_id=None
        )
        scope = state.root

    canvas = build_canvas(scope, state.selected_node_id)
    inspector = build_inspector(state, registry)
    breadcrumb = build_breadcrumb(state)
    return breadcrumb, canvas, inspector


# ---------------------------------------------------------------------------
# register_callbacks
# ---------------------------------------------------------------------------


def register_callbacks(app: dash.Dash) -> None:
    """Register every Phase-3 callback on *app*.

    Idempotent: callers that build the app should invoke this exactly once
    after the layout is set so the patterns / store ids resolve.

    Args:
        app: The :class:`dash.Dash` instance returned by ``create_app``.
    """

    # --- Hidden store for the active image path ---------------------------
    # Mounted after the fact so layout doesn't need editing.  We slot it into
    # the layout via a dedicated callback that injects the store on first
    # paint (see init_session_id below).

    # ----------------------------------------------------------------------
    # 1. Initialize STORE_SESSION_ID on first paint
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_SESSION_ID, "data"),
        Input(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        prevent_initial_call=False,
    )
    def init_session_id(_state_data: Any, current_id: Any) -> Any:
        """Populate ``STORE_SESSION_ID`` with a fresh uuid on first paint.

        Args:
            _state_data: Builder state payload (only used as a trigger).
            current_id: Current value of ``STORE_SESSION_ID``.

        Returns:
            The existing id when one exists, otherwise a new ``uuid.uuid4().hex``.
        """

        if current_id:
            return no_update
        return uuid.uuid4().hex

    # ----------------------------------------------------------------------
    # 2. Single fan-in mutation callback
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data"),
        Output(ids.BREADCRUMB_CONTAINER, "children"),
        Output("canvas-cytoscape-wrapper", "children"),  # set below
        Output(ids.INSPECTOR_CONTAINER, "children"),
        # Toast outputs surface mutation errors to the user; success path leaves
        # them as ``no_update`` so they don't clobber other callbacks' toasts.
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        # palette
        Input({"type": "palette-add", "class_name": ALL}, "n_clicks"),
        Input(ids.BTN_NEW_PIPELINE_NODE, "n_clicks"),
        # selection
        Input(ids.CANVAS_CYTOSCAPE, "tapNodeData"),
        # drill in (visible button on pipeline-node inspector); drill-out is
        # done via breadcrumb-link clicks, not a dedicated button.
        Input(ids.BTN_DRILL_IN, "n_clicks"),
        Input({"type": "breadcrumb-link", "depth": ALL}, "n_clicks"),
        # delete
        Input(ids.BTN_DELETE_NODE, "n_clicks"),
        # reorder via canvas elements
        Input(ids.CANVAS_CYTOSCAPE, "elements"),
        # parameter edits (one input per widget kind)
        Input({"type": "param-bool", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-num", "prefix": ALL, "name": ALL}, "n_blur"),
        Input({"type": "param-str", "prefix": ALL, "name": ALL}, "n_blur"),
        Input({"type": "param-enum", "prefix": ALL, "name": ALL}, "value"),
        Input({"type": "param-list", "prefix": ALL, "name": ALL}, "n_blur"),
        Input({"type": "param-tuple", "prefix": ALL, "name": ALL}, "n_blur"),
        Input(
            {"type": "param-optional-toggle", "prefix": ALL, "name": ALL},
            "value",
        ),
        # operation-typed param drill-in
        Input(
            {"type": "param-edit-nested", "prefix": ALL, "name": ALL},
            "n_clicks",
        ),
        # label
        Input(ids.INPUT_NODE_LABEL, "n_blur"),
        # values for parameter widgets (so we can resolve raw values on blur)
        State({"type": "param-num", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-num", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-str", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-str", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-list", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-list", "prefix": ALL, "name": ALL}, "id"),
        State({"type": "param-tuple", "prefix": ALL, "name": ALL}, "value"),
        State({"type": "param-tuple", "prefix": ALL, "name": ALL}, "id"),
        State(ids.INPUT_NODE_LABEL, "value"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def fan_in_state_mutation(  # noqa: C901, PLR0912, PLR0913, PLR0915
        _palette_clicks: List[int],
        _new_pipe_clicks: Optional[int],
        tap_node_data: Optional[Dict[str, Any]],
        _drill_out_clicks: Optional[int],
        _crumb_clicks: List[int],
        _delete_clicks: Optional[int],
        elements: Optional[List[Dict[str, Any]]],
        bool_vals: List[Any],
        _num_blurs: List[Any],
        _str_blurs: List[Any],
        enum_vals: List[Any],
        _list_blurs: List[Any],
        _tuple_blurs: List[Any],
        toggle_vals: List[Any],
        _edit_nested_clicks: List[Any],
        _label_blur: Any,
        num_values: List[Any],
        num_ids: List[Dict[str, Any]],
        str_values: List[Any],
        str_ids: List[Dict[str, Any]],
        list_values: List[Any],
        list_ids: List[Dict[str, Any]],
        tuple_values: List[Any],
        tuple_ids: List[Dict[str, Any]],
        label_value: Optional[str],
        state_data: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Resolve the trigger and return updated state + redrawn views.

        See module docstring for the dispatch table.
        """

        if state_data is None:
            return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )

        triggered = ctx.triggered_id
        if triggered is None:
            return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )

        new_state_dict = state_data

        try:
            # --- Pattern-matching ids ---------------------------------
            if isinstance(triggered, dict):
                t_type = triggered.get("type")
                if t_type == "palette-add":
                    # Some clicks may be zero (initial render of a button
                    # group with shared n_clicks=0).  Skip those.
                    nclicks_list = ctx.triggered[0]["value"]
                    if not nclicks_list:
                        return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )
                    new_state_dict = _dispatch_state_update(
                        state_data,
                        "add_node",
                        {"class_name": triggered["class_name"]},
                    )
                elif t_type == "breadcrumb-link":
                    new_state_dict = _dispatch_state_update(
                        state_data,
                        "breadcrumb_to",
                        {"depth": triggered["depth"]},
                    )
                elif t_type in {
                    "param-bool",
                    "param-num",
                    "param-str",
                    "param-enum",
                    "param-list",
                    "param-tuple",
                }:
                    new_state_dict = _handle_param_edit(
                        state_data,
                        triggered=triggered,
                        bool_vals=bool_vals,
                        enum_vals=enum_vals,
                        num_values=num_values,
                        num_ids=num_ids,
                        str_values=str_values,
                        str_ids=str_ids,
                        list_values=list_values,
                        list_ids=list_ids,
                        tuple_values=tuple_values,
                        tuple_ids=tuple_ids,
                    )
                elif t_type == "param-edit-nested":
                    nclicks_val = ctx.triggered[0].get("value")
                    if not nclicks_val:
                        return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )
                    new_state_dict = _dispatch_state_update(
                        state_data,
                        "drill_in_param",
                        {
                            "node_id": triggered["prefix"],
                            "param_name": triggered["name"],
                        },
                    )
                elif t_type == "param-optional-toggle":
                    new_state_dict = _handle_optional_toggle(
                        state_data,
                        triggered=triggered,
                        toggle_vals=toggle_vals,
                        num_values=num_values,
                        num_ids=num_ids,
                        str_values=str_values,
                        str_ids=str_ids,
                        list_values=list_values,
                        list_ids=list_ids,
                        tuple_values=tuple_values,
                        tuple_ids=tuple_ids,
                    )
                else:
                    return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )

            # --- Plain string ids -------------------------------------
            elif triggered == ids.BTN_NEW_PIPELINE_NODE:
                new_state_dict = _dispatch_state_update(
                    state_data, "add_pipeline", {}
                )
            elif triggered == ids.CANVAS_CYTOSCAPE:
                # Disambiguate tap (selection) vs elements (reorder).
                trigger_prop = ctx.triggered[0]["prop_id"].split(".")[-1]
                if trigger_prop == "tapNodeData":
                    if not tap_node_data:
                        return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )
                    new_state_dict = _dispatch_state_update(
                        state_data,
                        "select_node",
                        {"node_id": tap_node_data.get("id")},
                    )
                elif trigger_prop == "elements":
                    new_state_dict = _maybe_reorder_from_elements(
                        state_data, elements
                    )
                    if new_state_dict is state_data:
                        return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )
                else:
                    return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )
            elif triggered == ids.BTN_DRILL_IN:
                # The inspector renders a visible "Drill in ▸" button only on
                # ImagePipeline nodes; for any other selection the button is a
                # hidden placeholder that the user can't trigger. Either way,
                # the canonical action here is drill-in.
                new_state_dict = _dispatch_state_update(
                    state_data, "drill_in", {}
                )
            elif triggered == ids.BTN_DELETE_NODE:
                new_state_dict = _dispatch_state_update(
                    state_data, "delete_node", {}
                )
            elif triggered == ids.INPUT_NODE_LABEL:
                state = state_from_json(state_data)
                if state.selected_node_id is None:
                    return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "edit_label",
                    {
                        "node_id": state.selected_node_id,
                        "label": label_value,
                    },
                )
            else:
                return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
            )

            # --- Render ----------------------------------------------
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas, inspector = _render_views(new_state)
            return (
                new_state_dict, breadcrumb, canvas, inspector,
                no_update, no_update, no_update, no_update,
            )

        except Exception as exc:
            # Never crash the UI on a bad mutation — log full traceback and
            # surface the message via the toast instead of silently dropping
            # the user's interaction.
            logger.exception("fan_in_state_mutation failed")
            return (
                no_update, no_update, no_update, no_update,
                True,
                f"{type(exc).__name__}: {exc}",
                "danger",
                "Update failed",
            )

    # ----------------------------------------------------------------------
    # 3. Run preview
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_INTERMEDIATE_KEYS, "data"),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_RUN_PREVIEW, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        State(STORE_IMAGE_PATH, "data"),
        State(ids.INPUT_NROWS, "value"),
        State(ids.INPUT_NCOLS, "value"),
        prevent_initial_call=True,
    )
    def run_preview(
        n_clicks: Optional[int],
        state_data: Dict[str, Any],
        session_id: Optional[str],
        image_path: Optional[str],
        nrows: Optional[Any],
        ncols: Optional[Any],
    ) -> Tuple[Any, bool, str, str, str]:
        """Build the pipeline, run preview, cache intermediates."""

        if not n_clicks or state_data is None:
            return no_update, *_toast("No state to preview", ok=False)

        if not session_id:
            session_id = uuid.uuid4().hex

        try:
            t0 = time.time()

            from phenotypic import GridImage, Image
            from phenotypic.abc_ import GridOperation

            state = state_from_json(state_data)
            pipeline = to_pipeline(state.root)

            uses_grid = _pipeline_uses_grid(pipeline, GridOperation)

            if not image_path or image_path == SYNTHETIC_SENTINEL:
                from phenotypic.data._synthetic_data import load_synth_yeast_plate

                image = load_synth_yeast_plate()
            else:
                p = Path(image_path)
                if uses_grid:
                    image = GridImage.imread(
                        p,
                        nrows=int(nrows) if nrows else 8,
                        ncols=int(ncols) if ncols else 12,
                    )
                else:
                    image = Image.imread(p)

            cache = get_cache()
            cache.set_image(session_id, image, str(image_path) if image_path else None)

            result = pipeline.apply_with_intermediates(image)

            # Map intermediate keys (op-keys e.g. "GaussianBlur",
            # "GaussianBlur_2") back to BuilderState node-ids by walking ops
            # in declaration order.
            ops_keys: List[str] = list(pipeline.get_ops().keys())
            ops_nodes: List[StepNode] = [
                n
                for n in state.root.nodes
                if stage_of(n.class_name) == "ops" or n.class_name == PIPELINE_CLASS_NAME
            ]
            for op_key, node in zip(ops_keys, ops_nodes):
                inter = result.intermediates.get(op_key)
                if inter is not None:
                    cache.set_intermediate(session_id, node.node_id, inter)

            # Run measurements if any are configured.  Their output is a
            # single DataFrame; we attach it to every measurement node so
            # the inspector can show it for whichever the user selects.
            # `measure()` needs the *processed* image (objmap populated by
            # the detector chain), not the raw input.
            if pipeline.get_meas() or pipeline.get_post():
                try:
                    df = pipeline.measure(result.image)
                    meas_nodes = [
                        n
                        for n in state.root.nodes
                        if stage_of(n.class_name) in {"meas", "post"}
                    ]
                    for node in meas_nodes:
                        cache.set_intermediate(session_id, node.node_id, df)
                except Exception as meas_exc:  # noqa: BLE001
                    logger.warning("measure() failed: %s", meas_exc)

            duration = time.time() - t0
            keys = cache.known_intermediate_keys(session_id)
            return (
                keys,
                *_toast(f"Preview ran in {duration:.2f}s", ok=True),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Run preview failed")
            return (
                no_update,
                *_toast(_format_exception(exc), ok=False),
            )

    # ----------------------------------------------------------------------
    # 4. Inspector preview rendering
    # ----------------------------------------------------------------------

    # Tracks the last (session_id, node_id, intermediate-id) tuple this
    # callback rendered for. State changes that don't shift selection or
    # intermediate identity short-circuit to ``no_update`` so the inspector
    # doesn't re-encode the image / rebuild the DataTable on every keystroke
    # or drag. Plain dict (single-process Dash dev server); under multi-
    # worker deployment this would just degrade to "always render", same as
    # before.
    _preview_render_keys: Dict[str, Tuple[Optional[str], int]] = {}

    @app.callback(
        Output(ids.INSPECTOR_PREVIEW, "children"),
        Input(ids.STORE_BUILDER_STATE, "data"),
        Input(ids.STORE_INTERMEDIATE_KEYS, "data"),
        State(ids.STORE_SESSION_ID, "data"),
    )
    def render_inspector_preview(
        state_data: Optional[Dict[str, Any]],
        _keys: Optional[List[str]],
        session_id: Optional[str],
    ) -> Any:
        """Show the cached intermediate (image / DataTable) for the selection."""

        if state_data is None or not session_id:
            _preview_render_keys.pop(session_id or "", None)
            return html.Div(
                "No preview yet — click Run preview.",
                className="text-muted",
            )

        try:
            state = state_from_json(state_data)
        except Exception:  # noqa: BLE001
            return no_update

        if state.selected_node_id is None:
            last = _preview_render_keys.get(session_id)
            if last is not None and last[0] is None:
                return no_update
            _preview_render_keys[session_id] = (None, 0)
            return html.Div(
                "Select a node to view its preview.",
                className="text-muted",
            )

        try:
            scope = current_scope(state)
        except KeyError:
            return no_update
        node = next(
            (n for n in scope.nodes if n.node_id == state.selected_node_id),
            None,
        )
        if node is None:
            return no_update

        cached = get_cache().get_intermediate(session_id, state.selected_node_id)
        cache_token = id(cached) if cached is not None else 0
        last = _preview_render_keys.get(session_id)
        if last == (state.selected_node_id, cache_token):
            return no_update
        _preview_render_keys[session_id] = (state.selected_node_id, cache_token)
        if cached is None:
            return html.Div(
                "No preview yet — click Run preview.",
                className="text-muted",
            )

        # DataFrame preview for measurement / post nodes.
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except Exception:  # pragma: no cover - pandas always present in env
            pd = None  # type: ignore[assignment]

        if pd is not None and isinstance(cached, pd.DataFrame):
            return dataframe_to_table(cached)

        # Otherwise treat as Image — pick channel by stage.
        try:
            stage = stage_of(node.class_name)
        except KeyError:
            stage = "ops"
        channel = "rgb"
        if stage == "ops":
            try:
                from phenotypic.gui._operation_registry import get_registry

                info = get_registry().get(node.class_name)
                if info is not None and info.category == "Enhancer":
                    channel = "detect_mat"
                elif info is not None and info.category in {"Detector", "Refiner"}:
                    channel = "objmap"
            except Exception:  # noqa: BLE001
                pass

        try:
            uri = to_data_uri(cached, channel=channel)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            return html.Div(
                f"(Could not render preview: {_format_exception(exc)})",
                className="text-warning",
            )
        return html.Img(src=uri, style={"maxWidth": "100%"})

    # ----------------------------------------------------------------------
    # 5. Save
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_SAVE, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.INPUT_SAVE_PATH, "value"),
        prevent_initial_call=True,
    )
    def save_pipeline(
        n_clicks: Optional[int], state_data: Dict[str, Any], path_value: Optional[str]
    ) -> Tuple[bool, str, str, str]:
        if not n_clicks:
            return _toast("Nothing to save", ok=False)
        if not path_value:
            return _toast("Provide a save path first", ok=False)

        try:
            state = state_from_json(state_data)
            pipeline = to_pipeline(state.root)
            payload = pipeline.to_json()
            target = Path(path_value).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(payload if isinstance(payload, str) else json.dumps(payload))

            image_root = current_app.config.get("pheno_image_root")
            if image_root is not None:
                try:
                    target.resolve().relative_to(Path(image_root).resolve())
                except ValueError:
                    return _toast(
                        f"Saved to {target} (warning: outside --image-root)",
                        ok=True,
                    )
            return _toast(f"Saved to {target}", ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Save failed")
            return _toast(_format_exception(exc), ok=False)

    # ----------------------------------------------------------------------
    # 6. Load
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_LOAD, "n_clicks"),
        State(ids.INPUT_LOAD_PATH, "value"),
        prevent_initial_call=True,
    )
    def load_pipeline(
        n_clicks: Optional[int], path_value: Optional[str]
    ) -> Tuple[Any, ...]:
        if not n_clicks:
            return (no_update,) * 4 + _toast("Nothing to load", ok=False)
        if not path_value:
            return (no_update,) * 4 + _toast("Provide a load path first", ok=False)
        try:
            from phenotypic import ImagePipeline

            with open(Path(path_value).expanduser(), encoding="utf-8") as fh:
                content = fh.read()
            pipeline = ImagePipeline.from_json(content)
            scope = from_pipeline(pipeline)
            new_state = BuilderState(
                root=scope, breadcrumb=[], selected_node_id=None
            )
            new_state_dict = state_to_json(new_state)
            breadcrumb, canvas, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas,
                inspector,
                *_toast(f"Loaded {path_value}", ok=True),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load failed")
            return (no_update,) * 4 + _toast(_format_exception(exc), ok=False)

    # ----------------------------------------------------------------------
    # 7. Directory picker
    # ----------------------------------------------------------------------

    @app.callback(
        Output(STORE_IMAGE_PATH, "data"),
        Output(DIR_PICKER_PATH_INPUT, "value"),
        Output(DIR_PICKER_TREE_CONTAINER, "children"),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input({"type": "dir-entry", "kind": ALL, "path": ALL}, "n_clicks"),
        Input(DIR_PICKER_USE_PATH_BTN, "n_clicks"),
        Input(DIR_PICKER_SYNTH_BTN, "n_clicks"),
        State(DIR_PICKER_PATH_INPUT, "value"),
        prevent_initial_call=True,
    )
    def directory_picker_actions(
        _entry_clicks: List[int],
        _use_clicks: Optional[int],
        _synth_clicks: Optional[int],
        path_value: Optional[str],
    ) -> Tuple[Any, ...]:
        triggered = ctx.triggered_id
        if triggered is None:
            return (no_update,) * 7
        image_root: Optional[Path] = current_app.config.get("pheno_image_root")

        try:
            if isinstance(triggered, dict) and triggered.get("type") == "dir-entry":
                # Skip noisy zero-click events (Dash fires once per registered
                # entry on first paint).
                value = ctx.triggered[0].get("value")
                if not value:
                    return (no_update,) * 7
                kind = triggered["kind"]
                clicked_path = Path(triggered["path"])
                if kind == "file":
                    return (
                        str(clicked_path),
                        str(clicked_path),
                        no_update,
                        *_toast(f"Image set: {clicked_path.name}", ok=True),
                    )
                # dir or parent
                if image_root is not None:
                    tree = directory_tree(image_root, clicked_path)
                    return (
                        no_update,
                        no_update,
                        tree,
                        False,
                        no_update,
                        no_update,
                        no_update,
                    )
                return (no_update,) * 7

            if triggered == DIR_PICKER_USE_PATH_BTN:
                if not path_value:
                    return (no_update,) * 4 + _toast(
                        "Type a path first", ok=False
                    )
                p = Path(path_value).expanduser()
                if not p.exists():
                    return (no_update,) * 4 + _toast(
                        f"Path does not exist: {p}", ok=False
                    )
                return (
                    str(p),
                    str(p),
                    no_update,
                    *_toast(f"Image path set: {p}", ok=True),
                )

            if triggered == DIR_PICKER_SYNTH_BTN:
                synth = find_synthetic_plate_path()
                return (
                    str(synth),
                    str(synth),
                    no_update,
                    *_toast("Using synthetic yeast plate", ok=True),
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Directory picker failed")
            return (no_update,) * 4 + _toast(_format_exception(exc), ok=False)

        return (no_update,) * 7

    # ----------------------------------------------------------------------
    # 8. Canvas zoom / fit — clientside (calls cytoscape.js directly)
    # ----------------------------------------------------------------------
    # We previously tried to refit by re-emitting the ``layout`` prop with
    # ``fit: True`` from a server callback, but dash-cytoscape sometimes
    # ignored identical-shaped layout dicts. Calling ``cy.fit()`` /
    # ``cy.zoom()`` directly via clientside JS sidesteps that and gives a
    # snappier, animation-free response.
    #
    # ``window.phenoGetCy()`` is defined in ``assets/builder.js``.

    app.clientside_callback(
        """
        function(n_clicks, prev) {
            if (!n_clicks) return window.dash_clientside.no_update;
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (cy) cy.animate({fit: {padding: 24}}, {duration: 200});
            return (prev || 0) + 1;
        }
        """,
        Output(ids.STORE_CANVAS_CONTROL, "data", allow_duplicate=True),
        Input(ids.BTN_CANVAS_FIT, "n_clicks"),
        State(ids.STORE_CANVAS_CONTROL, "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(n_clicks, prev) {
            if (!n_clicks) return window.dash_clientside.no_update;
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (cy) {
                const z = cy.zoom() * 1.25;
                const center = {x: cy.width() / 2, y: cy.height() / 2};
                cy.zoom({level: z, renderedPosition: center});
            }
            return (prev || 0) + 1;
        }
        """,
        Output(ids.STORE_CANVAS_CONTROL, "data", allow_duplicate=True),
        Input(ids.BTN_CANVAS_ZOOM_IN, "n_clicks"),
        State(ids.STORE_CANVAS_CONTROL, "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(n_clicks, prev) {
            if (!n_clicks) return window.dash_clientside.no_update;
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (cy) {
                const z = cy.zoom() / 1.25;
                const center = {x: cy.width() / 2, y: cy.height() / 2};
                cy.zoom({level: z, renderedPosition: center});
            }
            return (prev || 0) + 1;
        }
        """,
        Output(ids.STORE_CANVAS_CONTROL, "data", allow_duplicate=True),
        Input(ids.BTN_CANVAS_ZOOM_OUT, "n_clicks"),
        State(ids.STORE_CANVAS_CONTROL, "data"),
        prevent_initial_call=True,
    )


# ---------------------------------------------------------------------------
# Helpers (private to this module)
# ---------------------------------------------------------------------------


def _pipeline_uses_grid(pipeline: Any, grid_op_cls: type) -> bool:
    """True if any step of *pipeline* (recursively) is a :class:`GridOperation`."""

    for collection in (
        pipeline.get_ops().values(),
        pipeline.get_meas().values(),
        pipeline.get_post().values() if hasattr(pipeline, "get_post") else [],
    ):
        for op in collection:
            if isinstance(op, grid_op_cls):
                return True
            # Recurse into nested ImagePipeline.
            if hasattr(op, "get_ops"):
                if _pipeline_uses_grid(op, grid_op_cls):
                    return True
    return False


def _maybe_reorder_from_elements(
    state_data: Dict[str, Any], elements: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Compute a reorder mutation from a cytoscape ``elements`` payload.

    Returns *state_data* unchanged when no positional change is detectable;
    otherwise returns a new state dict with the current scope's nodes
    re-ordered to match the visual left-to-right sequence.
    """

    if not elements:
        return state_data

    # Filter to node entries with positions.
    node_entries = [
        e for e in elements if "data" in e and "source" not in e.get("data", {})
    ]
    positioned = [e for e in node_entries if "position" in e]
    if len(positioned) < 2:
        return state_data

    positioned.sort(key=lambda e: e["position"].get("x", 0.0))
    new_order = [e["data"]["id"] for e in positioned]

    breadcrumb = list(state_data.get("breadcrumb", []) or [])
    scope = _scope_at_breadcrumb(state_data, breadcrumb)
    existing_order = [n["node_id"] for n in scope["nodes"]]
    if new_order == existing_order:
        return state_data

    # Only reorder if the *set* matches — guards against transient cytoscape
    # element pruning.
    if set(new_order) != set(existing_order):
        return state_data

    return _dispatch_state_update(state_data, "reorder", {"order": new_order})


def _handle_param_edit(
    state_data: Dict[str, Any],
    *,
    triggered: Dict[str, Any],
    bool_vals: List[Any],
    enum_vals: List[Any],
    num_values: List[Any],
    num_ids: List[Dict[str, Any]],
    str_values: List[Any],
    str_ids: List[Dict[str, Any]],
    list_values: List[Any],
    list_ids: List[Dict[str, Any]],
    tuple_values: List[Any],
    tuple_ids: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Handle a single param-widget edit by writing into the matching node."""

    from phenotypic.gui._operation_registry import get_registry

    t_type = triggered["type"]
    prefix = triggered["prefix"]
    name = triggered["name"]

    raw: Any
    if t_type == "param-bool":
        raw = ctx.triggered[0]["value"]
    elif t_type == "param-enum":
        raw = ctx.triggered[0]["value"]
    elif t_type == "param-num":
        raw = _lookup_in_state(num_ids, num_values, prefix=prefix, name=name)
    elif t_type == "param-str":
        raw = _lookup_in_state(str_ids, str_values, prefix=prefix, name=name)
    elif t_type == "param-list":
        raw = _lookup_in_state(list_ids, list_values, prefix=prefix, name=name)
    elif t_type == "param-tuple":
        raw = _lookup_in_state(tuple_ids, tuple_values, prefix=prefix, name=name)
    else:
        return state_data

    # Resolve ParamInfo to coerce the raw value back to a Python type.
    state = state_from_json(state_data)
    try:
        scope = current_scope(state)
    except KeyError:
        return state_data
    node = next((n for n in scope.nodes if n.node_id == prefix), None)
    if node is None:
        return state_data
    info = get_registry().get(node.class_name)
    if info is None:
        return state_data
    p = info.parameters.get(name)
    if p is None:
        return state_data

    try:
        coerced = parse_widget_value(raw, p)
    except Exception:  # noqa: BLE001
        return state_data

    return _dispatch_state_update(
        state_data,
        "edit_param",
        {"node_id": prefix, "name": name, "value": coerced, "omit": False},
    )


def _handle_optional_toggle(
    state_data: Dict[str, Any],
    *,
    triggered: Dict[str, Any],
    toggle_vals: List[Any],
    num_values: List[Any],
    num_ids: List[Dict[str, Any]],
    str_values: List[Any],
    str_ids: List[Dict[str, Any]],
    list_values: List[Any],
    list_ids: List[Dict[str, Any]],
    tuple_values: List[Any],
    tuple_ids: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply a "Use default" toggle change to the parameter node."""

    prefix = triggered["prefix"]
    name = triggered["name"]
    value = ctx.triggered[0]["value"]
    if value:
        # Toggle ON -> remove the param so the operation default takes over.
        return _dispatch_state_update(
            state_data,
            "edit_param",
            {"node_id": prefix, "name": name, "omit": True},
        )

    # Toggle OFF -> read the current widget value (search across all kinds).
    raw = None
    for ids_list, vals in (
        (num_ids, num_values),
        (str_ids, str_values),
        (list_ids, list_values),
        (tuple_ids, tuple_values),
    ):
        candidate = _lookup_in_state(ids_list, vals, prefix=prefix, name=name)
        if candidate is not None:
            raw = candidate
            break

    return _dispatch_state_update(
        state_data,
        "edit_param",
        {"node_id": prefix, "name": name, "value": raw, "omit": False},
    )


def _lookup_in_state(
    ids_list: List[Dict[str, Any]],
    values: List[Any],
    *,
    prefix: str,
    name: str,
) -> Any:
    """Return the value matching ``{prefix, name}`` in a parallel id/value pair."""

    for component_id, val in zip(ids_list, values):
        if (
            component_id.get("prefix") == prefix
            and component_id.get("name") == name
        ):
            return val
    return None


__all__ = [
    "register_callbacks",
    "_dispatch_state_update",
    "STORE_IMAGE_PATH",
]
