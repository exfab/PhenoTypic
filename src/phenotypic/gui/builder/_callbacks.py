"""Dash ``@callback`` registrations for the pipeline builder.

Phase 3 owns every callback that reads or writes
:data:`phenotypic.gui.builder._ids.STORE_BUILDER_STATE` plus the few helper
callbacks that drive the directory picker, preview, and inspector preview
panel.

Architecture notes:

* **Single fan-in mutation callback** — every trigger that mutates the
  builder state (palette adds, deletes, drill-in/out, breadcrumb jumps,
  reorders, drag, parameter edits, label edits, aux-port wiring) feeds one
  callback that resolves the trigger via
  :data:`dash.callback_context.triggered_id` and returns a fresh
  ``STORE_BUILDER_STATE`` plus a re-rendered canvas / inspector /
  breadcrumb.  This keeps ``allow_duplicate=True`` out of the store
  contract.
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
* **Popover-driven wire flow** — Wave 4 replaces the click-then-click model
  with a canvas-anchored popover. Tapping an aux port writes a structured
  payload to ``PORT_CLICK_STORE`` (driven by ``aux_popover.js``); the
  server-side callback then renders the popover's contents based on the
  port's current state (empty / wired / list-of-slots) and routes the
  user's action button click to one of the dispatch kinds: ``wire_create``
  (pick a class), ``wire_delete`` (disconnect), ``port_slot_add``
  (extend a list-typed port), ``drill_in_aux`` (descend into a wired
  aux), or ``set_inspector_focus`` (edit a wired aux's params in the
  inspector pane without leaving the consumer's canvas selection).

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

from phenotypic.gui._config import CFG_IMAGE_ROOT, CFG_OPERATION_REGISTRY
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._ids import LoadPickerPage
from phenotypic.gui.builder._directory_browser import (
    IMAGE_EXTS,
    PIPELINE_EXTS,
    SYNTHETIC_SENTINEL,
    directory_tree,
    find_synthetic_plate_path,
)
from phenotypic.gui.builder._image_renderer import (
    bytes_to_data_uri,
    dataframe_to_table,
    render_node_preview,
)
from phenotypic.gui.builder._layout import (
    INSPECTOR_FOCUS_AUX_BANNER_ID,
    build_breadcrumb,
    build_canvas,
    build_inspector,
    build_popover_contents,
)
from phenotypic.gui.builder._modal_browser import (
    no_root_placeholder,
    render_load_picker_body,
)
from phenotypic.gui.builder._param_form import parse_widget_value
from phenotypic.gui.builder._session import PreviewRenderError, get_cache
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


# Pre-built ``no_update`` tuples for the long-output callbacks.  Building them
# once at import time keeps the callback bodies readable and avoids re-emitting
# the same long literal on every early-return branch.
#
# Layout: state, breadcrumb, canvas, inspector, popover, 4 toast outputs.
_NOOP_FAN_IN: Tuple[Any, ...] = (no_update,) * 10


def _trigger_kind_path(triggered: Any, expected_type: str) -> Optional[Tuple[str, str]]:
    """Validate a directory-tree click and return ``(kind, path)`` if usable.

    Tree-entry callbacks all share the same trigger validation: the trigger
    must be a pattern-matched dict whose ``type`` matches the modal's
    :data:`DIR_ENTRY_TYPE_*` sentinel and whose first triggered value is
    truthy (a real click rather than the initial render's zero ``n_clicks``).

    Args:
        triggered: ``dash.callback_context.triggered_id`` value.
        expected_type: The :data:`DIR_ENTRY_TYPE_*` constant the callback
            subscribes to.

    Returns:
        ``(kind, path)`` from the trigger when validation passes, else
        ``None`` so the caller can short-circuit to ``no_update``.
    """

    if not isinstance(triggered, dict):
        return None
    if triggered.get("type") != expected_type:
        return None
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return None
    kind = triggered.get("kind")
    path = triggered.get("path")
    if not isinstance(kind, str) or not isinstance(path, str):
        return None
    return kind, path


# ---------------------------------------------------------------------------
# Pure mutation helpers (testable without Dash)
# ---------------------------------------------------------------------------


def _normalize_segment(seg: Any) -> Dict[str, Any]:
    """Return *seg* as a normalized breadcrumb-segment dict.

    Accepts:
        * legacy plain-string entries (older saved state) — treated as a
          ``node_id``;
        * the standard ``{"node_id": ..., "param": ...}`` form (descends
          into a main-ribbon node's ``nested`` scope or, when ``param`` is
          set, into the legacy synthesized op-param scope under
          ``_PARAM_SCOPE_KEY``);
        * the aux-slot drill form ``{"target_node_id": ..., "param": ...,
          "slot": ...}`` pushed by ``drill_in_aux`` (Wave 4). Mirrors the
          state-side walker ``_normalize_breadcrumb_segment`` in
          ``_state.py``.
    """

    if isinstance(seg, str):
        return {"node_id": seg, "param": None}
    if isinstance(seg, dict):
        if "target_node_id" in seg:
            return {
                "target_node_id": seg["target_node_id"],
                "param": seg.get("param"),
                "slot": seg.get("slot", 0),
            }
        if "node_id" in seg:
            return {"node_id": seg["node_id"], "param": seg.get("param")}
    return {"node_id": None, "param": None}


def _scope_at_breadcrumb(
    state_dict: Dict[str, Any], breadcrumb: List[Any]
) -> Dict[str, Any]:
    """Walk *state_dict* ``root`` → ``breadcrumb`` and return that scope dict.

    Three breadcrumb-segment shapes are handled:

    * Regular main-ribbon drill (``{"node_id": <id>, "param": None}``):
      descend into ``node["nested"]``.
    * Legacy op-typed parameter drill (``{"node_id": <id>, "param":
      <name>}``): descend into the synthesised scope under
      ``node["params"]["__op_param_scope__"][param_name]``, seeding from
      any existing operation marker dict if absent. Kept for back-compat;
      new code should use the aux-slot form below.
    * Aux-slot drill (``{"target_node_id": <id>, "param": <name>,
      "slot": <int>}``): descend into the embedded aux ``StepNode`` dict
      at ``consumer["aux_ports"][param][slot]``. If that aux has
      ``nested``, descend into it; otherwise surface the aux node as a
      single-node wrapper scope (mirroring ``_state.current_scope``).

    Args:
        state_dict: ``state_to_json`` output.
        breadcrumb: List of breadcrumb segments.

    Returns:
        The nested scope dict (``{"nodes": [...], ...}``).  Returns the
        root scope when ``breadcrumb`` is empty.
    """

    scope = state_dict["root"]
    for raw in breadcrumb:
        seg = _normalize_segment(raw)
        if "target_node_id" in seg:
            target_id = seg["target_node_id"]
            param = seg.get("param")
            slot = int(seg.get("slot") or 0)
            consumer = next(
                (n for n in scope["nodes"] if n.get("node_id") == target_id),
                None,
            )
            if consumer is None:
                return scope
            if not isinstance(param, str):
                return scope
            aux_ports = consumer.get("aux_ports") or {}
            slot_list = aux_ports.get(param)
            if not isinstance(slot_list, list) or slot < 0 or slot >= len(slot_list):
                return scope
            aux = slot_list[slot]
            if not isinstance(aux, dict):
                return scope
            nested = aux.get("nested")
            if isinstance(nested, dict):
                scope = nested
                continue
            # Single-op aux: surface as a one-node wrapper scope so
            # dispatch can address it.
            scope = {
                "nodes": [aux],
                "name": aux.get("label") or aux.get("class_name") or "aux",
                "desc": "",
                "nrows": None,
                "ncols": None,
            }
            continue
        node_id = seg.get("node_id")
        if not isinstance(node_id, str):
            return scope
        node = next(
            (n for n in scope["nodes"] if n["node_id"] == node_id),
            None,
        )
        if node is None:
            return scope
        if seg.get("param") is None:
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

    Aux-slot segments (``{"target_node_id", "param", "slot"}``) are skipped
    — the wired aux already lives persistently inside
    ``consumer.aux_ports[param][slot]``, so nothing needs to be mirrored
    back when drilling out.

    The scope dict under ``__op_param_scope__`` is preserved so re-entering
    the slot keeps the same node ids and incidental UI state.
    """

    for raw in dropped:
        seg = _normalize_segment(raw)
        if "target_node_id" in seg:
            continue
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


def _find_in_scope(
    scope: Dict[str, Any], node_id: str
) -> Optional[Dict[str, Any]]:
    """Return the main-ribbon node matching *node_id* in *scope* (one level).

    Aux StepNodes are now embedded inside their consumer's ``aux_ports``
    slot list (see Wave 1-A) so this helper only inspects the visible
    main ribbon. Callers that need to address an embedded aux must drill
    into it first (push an aux-slot breadcrumb segment) so the aux
    becomes the current scope.
    """

    return next(
        (n for n in scope.get("nodes", []) or [] if n.get("node_id") == node_id),
        None,
    )


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


def _build_fresh_aux_node(
    class_name: str, registry: Any
) -> Optional[Dict[str, Any]]:
    """Build a JSON-shaped :class:`StepNode` dict for a freshly-wired aux.

    Mirrors the per-node dict structure that ``state_to_json`` produces
    (``node_id``, ``class_name``, ``params``, ``label``, ``nested``,
    ``aux_ports``). Used by ``wire_create`` to materialise an embedded
    aux on demand without dragging in the live ``StepNode`` dataclass —
    state mutations operate on dicts so the result survives the
    ``dcc.Store`` round-trip.

    Args:
        class_name: Registry key (or :data:`PIPELINE_CLASS_NAME` for a
            pipeline-typed aux).
        registry: ``OperationRegistry`` singleton (only consulted for the
            non-pipeline branch; the pipeline sentinel is built without
            registry lookup).

    Returns:
        A node dict ready for insertion into a consumer's
        ``aux_ports[<param>][<slot>]``, or ``None`` when the class is
        unknown to the registry and not the pipeline sentinel.
    """

    is_pipeline_aux = class_name == PIPELINE_CLASS_NAME
    if not is_pipeline_aux and (
        registry is None or registry.get(class_name) is None
    ):
        return None

    nested: Optional[Dict[str, Any]]
    if is_pipeline_aux:
        nested = {
            "nodes": [],
            "name": "Subpipeline",
            "desc": "",
            "nrows": None,
            "ncols": None,
        }
        params: Dict[str, Any] = {}
    else:
        nested = None
        params = _default_params_for(class_name)

    return {
        "node_id": _new_node_id(),
        "class_name": class_name,
        "params": params,
        "label": None,
        "nested": nested,
        "aux_ports": {},
    }


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
            ``"wire_create"`` (payload: ``target_node_id``, ``param``,
            ``slot``, ``class_name``)
                Materialise a fresh aux :class:`StepNode` for
                *class_name* and embed it at
                ``consumer.aux_ports[param][slot]``. Validates type
                compatibility against the registry; rejects mismatches by
                returning the unmodified state. Auto-focuses the new aux
                in the inspector via ``inspector_focus_aux``.
            ``"wire_delete"`` (payload: ``target_node_id``, ``param``,
            ``slot``)
                Set the consumer's slot to ``None`` so the embedded aux
                is dropped (gc collects it). For list-typed ports the
                slot remains as a ``None`` placeholder; scalar ports
                keep the single slot at ``[None]``. Clears any
                ``inspector_focus_aux`` that was pointing at the cleared
                slot.
            ``"port_slot_add"`` (payload: ``node_id``, ``param``)
                Append a ``None`` slot to the consumer's ``aux_ports``
                list for a list-typed param. No-op for scalar ports.
            ``"port_slot_remove"`` (payload: ``node_id``, ``param``,
            ``slot``)
                Remove the slot at ``slot`` from the consumer's
                ``aux_ports`` list and reindex remaining slots.
            ``"drill_in_aux"`` (payload: ``target_node_id``, ``param``,
            ``slot``)
                Push a ``{"target_node_id", "param", "slot"}`` segment
                onto the breadcrumb so the user can edit a wired aux's
                nested content. Clears ``inspector_focus_aux`` because
                the canvas scope swap takes over.
            ``"set_inspector_focus"`` (payload: ``focus``,
            ``target_node_id``?, ``param``?, ``slot``?)
                Set or clear the ``inspector_focus_aux`` override.
                ``focus == "aux"`` validates the target slot exists and
                writes the focus dict; any other ``focus`` value clears
                it (returning the inspector to the canvas-selected
                consumer's params).

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

    if kind == "wire_create":
        target_node_id = payload["target_node_id"]
        param = payload["param"]
        slot = payload.get("slot")
        class_name = payload.get("class_name")

        if not isinstance(slot, int) or slot < 0:
            return out
        if not isinstance(class_name, str) or not class_name:
            return out

        consumer = _find_in_scope(scope, target_node_id)
        if consumer is None:
            return out

        registry = _registry()
        if registry is None:
            return out
        consumer_info = registry.get(consumer.get("class_name", ""))
        if consumer_info is None:
            return out
        param_info = consumer_info.parameters.get(param)
        if param_info is None:
            return out
        if not (param_info.is_operation or param_info.is_pipeline):
            return out

        # Type compatibility: the chosen class must satisfy the port's type.
        if not _source_satisfies_port(class_name, param_info, registry):
            return out

        # Slot validation: scalar ports allow only slot 0.
        if not param_info.is_list and slot != 0:
            return out

        aux_node = _build_fresh_aux_node(class_name, registry)
        if aux_node is None:
            return out

        aux_ports = consumer.setdefault("aux_ports", {})
        slot_list = aux_ports.get(param)
        if not isinstance(slot_list, list):
            slot_list = [None] if not param_info.is_list else []
            aux_ports[param] = slot_list
        while len(slot_list) <= slot:
            slot_list.append(None)
        slot_list[slot] = aux_node

        # Auto-focus the new aux in the inspector so the user can edit
        # its params without leaving the consumer's canvas selection.
        out["inspector_focus_aux"] = {
            "target_node_id": target_node_id,
            "param": param,
            "slot": slot,
        }
        return out

    if kind == "wire_delete":
        target_node_id = payload["target_node_id"]
        param = payload["param"]
        slot = payload.get("slot")

        if not isinstance(slot, int) or slot < 0:
            return out

        consumer = _find_in_scope(scope, target_node_id)
        if consumer is None:
            return out
        slot_list = (consumer.get("aux_ports") or {}).get(param)
        if not isinstance(slot_list, list) or slot >= len(slot_list):
            return out
        slot_list[slot] = None

        # If we were focused on this aux, clear the focus override so
        # the inspector falls back to the consumer's params.
        focus = out.get("inspector_focus_aux") or {}
        if (
            isinstance(focus, dict)
            and focus.get("target_node_id") == target_node_id
            and focus.get("param") == param
            and focus.get("slot") == slot
        ):
            out["inspector_focus_aux"] = None
        return out

    if kind == "port_slot_add":
        node_id = payload["node_id"]
        param = payload["param"]

        node = _find_in_scope(scope, node_id)
        if node is None:
            return out

        registry = _registry()
        if registry is None:
            return out
        info = registry.get(node.get("class_name", ""))
        if info is None:
            return out
        param_info = info.parameters.get(param)
        if param_info is None or not param_info.is_list:
            return out

        aux_ports = node.setdefault("aux_ports", {})
        slots = aux_ports.setdefault(param, [])
        slots.append(None)
        return out

    if kind == "port_slot_remove":
        node_id = payload["node_id"]
        param = payload["param"]
        slot = payload.get("slot")

        if not isinstance(slot, int) or slot < 0:
            return out

        node = _find_in_scope(scope, node_id)
        if node is None:
            return out
        slots = (node.get("aux_ports") or {}).get(param)
        if not isinstance(slots, list) or slot >= len(slots):
            return out
        slots.pop(slot)

        # Scalar ports must always carry exactly one slot ([None] when empty).
        registry = _registry()
        info = registry.get(node.get("class_name", "")) if registry else None
        param_info = info.parameters.get(param) if info else None
        if param_info is not None and not param_info.is_list and not slots:
            slots.append(None)

        # Clear focus when the removed slot's index is at or before the
        # focused index. Removing slot `slot` shifts every higher index
        # down by 1, so the focused aux is no longer the same StepNode
        # unless focused on an earlier slot. The previous condition
        # ``focus["slot"] >= len(slots)`` only caught the case where the
        # last slot was removed; removing slot 0 from a 2-slot list left
        # focus pointing at slot 0, which now silently aliased to what
        # was slot 1.
        focus = out.get("inspector_focus_aux") or {}
        if (
            isinstance(focus, dict)
            and focus.get("target_node_id") == node_id
            and focus.get("param") == param
            and isinstance(focus.get("slot"), int)
            and focus["slot"] >= slot
        ):
            out["inspector_focus_aux"] = None
        return out

    if kind == "drill_in_aux":
        target_node_id = payload["target_node_id"]
        param = payload["param"]
        slot = payload.get("slot")

        if not isinstance(slot, int) or slot < 0:
            return out

        consumer = _find_in_scope(scope, target_node_id)
        if consumer is None:
            return out
        slot_list = (consumer.get("aux_ports") or {}).get(param)
        if (
            not isinstance(slot_list, list)
            or slot >= len(slot_list)
            or slot_list[slot] is None
        ):
            return out

        breadcrumb.append(
            {
                "target_node_id": target_node_id,
                "param": param,
                "slot": slot,
            }
        )
        out["breadcrumb"] = breadcrumb
        out["selected_node_id"] = None
        # Canvas scope swap takes over — the popover (and any aux focus)
        # belong to the parent scope, not the new one.
        out["inspector_focus_aux"] = None
        return out

    if kind == "set_inspector_focus":
        focus_kind = payload.get("focus")
        if focus_kind == "aux":
            target_node_id = payload.get("target_node_id")
            param = payload.get("param")
            raw_slot = payload.get("slot")
            if (
                not isinstance(target_node_id, str)
                or not isinstance(param, str)
                or not isinstance(raw_slot, int)
                or raw_slot < 0
            ):
                return out
            consumer = _find_in_scope(scope, target_node_id)
            if consumer is None:
                return out
            slot_list = (consumer.get("aux_ports") or {}).get(param) or []
            if (
                raw_slot >= len(slot_list)
                or slot_list[raw_slot] is None
            ):
                return out
            out["inspector_focus_aux"] = {
                "target_node_id": target_node_id,
                "param": param,
                "slot": raw_slot,
            }
            return out
        # Any other focus value clears the override.
        out["inspector_focus_aux"] = None
        return out

    return out


def _source_satisfies_port(
    source_class: Optional[str], param_info: Any, registry: Any
) -> bool:
    """Return ``True`` if an aux of *source_class* may wire into *param_info*.

    Type-compatibility rules (mirroring the plan):

    * If the target port is pipeline-eligible (``param_info.is_pipeline``)
      and the source is the :data:`PIPELINE_CLASS_NAME` sentinel, the wire
      is allowed.
    * If the target port is op-eligible (``param_info.is_operation``) and
      the source is registered as an :class:`ImageOperation` subclass,
      the wire is allowed.
    * If the port is a Union of both (``is_operation and is_pipeline``),
      either source kind is accepted.
    * Otherwise the wire is rejected (returns ``False``).

    Args:
        source_class: The aux node's ``class_name``.  Either the sentinel
            ``"ImagePipeline"`` or a registry key.
        param_info: The :class:`ParamInfo` for the target port.
        registry: The :class:`OperationRegistry` singleton.

    Returns:
        ``True`` if the wire passes type validation, else ``False``.
    """

    if source_class is None:
        return False

    if source_class == PIPELINE_CLASS_NAME:
        return bool(param_info.is_pipeline)

    info = registry.get(source_class)
    if info is None:
        return False

    # Lazy import to avoid a hard dependency at module import time.
    from phenotypic.abc_ import ImageOperation

    if param_info.is_operation:
        try:
            if issubclass(info.cls, ImageOperation):
                return True
        except TypeError:
            return False
    if param_info.is_pipeline:
        # Source class is not the sentinel; pipeline-only ports require
        # the sentinel.
        return False
    return False


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

    return current_app.config.get(CFG_OPERATION_REGISTRY)


def _image_root() -> Optional[Any]:
    """Return the configured ``--image-root`` :class:`Path`, or ``None``.

    Stashed on ``app.server.config`` by ``create_app``; many callbacks
    consult it to seed their browse-dir stores.
    """

    return current_app.config.get(CFG_IMAGE_ROOT)


def _render_tree_body(
    dir_value: Optional[str],
    *,
    extensions: Any,
    select_files: bool,
    id_type: str,
) -> Any:
    """Render a :func:`directory_tree` body for a modal navigation update.

    Used by ``render_save_body`` and ``render_load_image_body``: both take a
    string from a browse-dir store, validate the configured ``--image-root``,
    and emit either a refreshed tree or the muted placeholder.
    """

    image_root = _image_root()
    if image_root is None:
        return no_root_placeholder()
    current = Path(dir_value).expanduser() if dir_value else None
    return directory_tree(
        Path(image_root),
        current=current,
        extensions=extensions,
        select_files=select_files,
        id_type=id_type,
    )


def _render_views(state: BuilderState) -> Tuple[Any, Any, Any, Any]:
    """Re-render breadcrumb, canvas, inspector, and popover for a given state.

    Args:
        state: Live :class:`BuilderState` object.

    Returns:
        Tuple ``(breadcrumb_children, canvas, inspector, popover_contents)``
        of Dash component subtrees. The breadcrumb callback target is the
        existing nav's ``children`` property, so returning a full nav here
        would nest the breadcrumb inside itself on every update. The
        popover contents are an empty list when no aux focus is active —
        the popover container's ``display`` is toggled by Wave 4
        callbacks (see :func:`register_callbacks`).
    """

    registry = _registry()
    try:
        scope = current_scope(state)
    except KeyError:
        # Stale breadcrumb — fall back to the root.
        state = BuilderState(
            root=state.root,
            breadcrumb=[],
            selected_node_id=None,
            inspector_focus_aux=None,
        )
        scope = state.root

    canvas = build_canvas(scope, state.selected_node_id)
    inspector = build_inspector(state, registry)
    breadcrumb = build_breadcrumb(state).children
    popover_contents = build_popover_contents(state, registry)
    return breadcrumb, canvas, inspector, popover_contents


def _popover_style_for(contents: Any) -> Dict[str, Any]:
    """Return the popover container's inline ``style`` for a given children list.

    The popover hides itself entirely when ``build_popover_contents``
    returns an empty list. Wave 4 callbacks call this helper so the
    container's ``display`` flips back and forth in lock-step with the
    rendered contents (otherwise an emptied popover would leave a stale
    visible chrome).
    """

    if not contents:
        return {"display": "none"}
    return {"display": "block"}


def _state_replacement_payload(
    pipeline: Any,
) -> Tuple[Dict[str, Any], Any, Any, Any, Any, Dict[str, Any]]:
    """Build the full re-render tuple for a freshly-loaded pipeline.

    Both the JSON-load and prefab-load callbacks blow away the current
    builder state and replace it with one derived from a freshly-built
    :class:`ImagePipeline`. Both then need the same six output values;
    this helper centralises the conversion + view rendering.

    Returns:
        Tuple ``(state_dict, breadcrumb, canvas, inspector,
        popover_contents, popover_style)``. The ``popover_style`` mirrors
        ``_popover_style_for(popover_contents)`` so the container's
        ``display`` flips in lock-step with the children.
    """

    scope = from_pipeline(pipeline)
    new_state = BuilderState(
        root=scope,
        breadcrumb=[],
        selected_node_id=None,
        inspector_focus_aux=None,
    )
    breadcrumb, canvas, inspector, popover_contents = _render_views(new_state)
    return (
        state_to_json(new_state),
        breadcrumb,
        canvas,
        inspector,
        popover_contents,
        _popover_style_for(popover_contents),
    )


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
        Output("canvas-cytoscape-wrapper", "children"),
        Output(ids.INSPECTOR_CONTAINER, "children"),
        Output(ids.POPOVER_CONTAINER, "children"),
        # ``style`` output keeps display:block/display:none in lock-step
        # with the children list. Without it, state mutations that cause
        # ``build_popover_contents`` to return ``[]`` (e.g. deleting a
        # node with an open popover, drilling, breadcrumb navigation)
        # clear the popover children but leave a stale visible empty
        # box anchored where the popover used to be.
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
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

        if state_data is None or ctx.triggered_id is None:
            return _NOOP_FAN_IN

        triggered = ctx.triggered_id
        new_state_dict = state_data

        try:
            # --- Pattern-matching ids ---------------------------------
            if isinstance(triggered, dict):
                t_type = triggered.get("type")
                if t_type == "palette-add":
                    # Skip clicks where ``n_clicks`` is still zero (initial
                    # render of the palette button group).
                    if not ctx.triggered[0]["value"]:
                        return _NOOP_FAN_IN
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
                    if not ctx.triggered[0].get("value"):
                        return _NOOP_FAN_IN
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
                    return _NOOP_FAN_IN

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
                        return _NOOP_FAN_IN
                    tapped_id = tap_node_data.get("id")
                    if not isinstance(tapped_id, str):
                        return _NOOP_FAN_IN
                    # Aux-port taps are handled clientside (write to
                    # ``PORT_CLICK_STORE``); they should never reach this
                    # callback. Defensive: ignore any tap that decodes as
                    # an aux port so we don't accidentally re-select an
                    # invisible port element.
                    if ids._decode_aux_port_id(tapped_id) is not None:
                        return _NOOP_FAN_IN
                    if ids._decode_main_port_id(tapped_id) is not None:
                        # Main I/O port taps are cosmetic; ignore.
                        return _NOOP_FAN_IN
                    new_state_dict = _dispatch_state_update(
                        state_data,
                        "select_node",
                        {"node_id": tapped_id},
                    )
                elif trigger_prop == "elements":
                    new_state_dict = _maybe_reorder_from_elements(
                        state_data, elements
                    )
                    if new_state_dict is state_data:
                        return _NOOP_FAN_IN
                else:
                    return _NOOP_FAN_IN
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
                    return _NOOP_FAN_IN
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "edit_label",
                    {
                        "node_id": state.selected_node_id,
                        "label": label_value,
                    },
                )
            else:
                return _NOOP_FAN_IN

            # --- Render ----------------------------------------------
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas, inspector, popover_contents = _render_views(
                new_state
            )
            return (
                new_state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_contents,
                _popover_style_for(popover_contents),
                no_update,
                no_update,
                no_update,
                no_update,
            )

        except Exception as exc:
            # Never crash the UI on a bad mutation — log the traceback and
            # surface the failure via the toast instead.
            logger.exception("fan_in_state_mutation failed")
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                True,
                _format_exception(exc),
                "danger",
                "Update failed",
            )

    # ----------------------------------------------------------------------
    # 2b. Point-picker store fan-in
    # ----------------------------------------------------------------------
    #
    # The picker widget writes its list-of-(y, x) payload into a hidden
    # ``dcc.Store`` whose pattern-matching id carries the owning node-id in
    # the ``prefix`` field. The payload is already structured (no scalar /
    # text coercion needed) so we side-step the main fan-in and dispatch
    # ``edit_param`` directly.

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(
            {"type": ids.PICKER_PARAM_STORE_TYPE, "prefix": ALL, "name": ALL},
            "data",
        ),
        State(
            {"type": ids.PICKER_PARAM_STORE_TYPE, "prefix": ALL, "name": ALL},
            "id",
        ),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def fan_in_picker_store(  # noqa: PLR0913
        store_payloads: List[Any],
        store_ids: List[Dict[str, Any]],
        state_data: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Write a picker store's list-of-pairs into the matching node's params."""

        noop = (no_update,) * 10

        if state_data is None or ctx.triggered_id is None:
            return noop
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return noop
        if triggered.get("type") != ids.PICKER_PARAM_STORE_TYPE:
            return noop

        prefix = triggered.get("prefix")
        name = triggered.get("name")
        if not prefix or not name:
            return noop

        # Find the payload that matches the triggered store id. ``zip`` over
        # the parallel id/value lists keeps the lookup independent of the
        # ``ALL`` order Dash chose at registration time.
        raw: Any = None
        for component_id, val in zip(store_ids, store_payloads):
            if (
                component_id.get("prefix") == prefix
                and component_id.get("name") == name
            ):
                raw = val
                break

        if raw is None:
            raw = []

        try:
            new_state_dict = _dispatch_state_update(
                state_data,
                "edit_param",
                {
                    "node_id": prefix,
                    "name": name,
                    "value": raw,
                    "omit": False,
                },
            )
            # Short-circuit when nothing actually changed — Confirm with no
            # edits, modal-open re-seed, etc. — so we don't pay for a full
            # graph re-render. Cheaper than walking the nested-scope tree
            # by hand to find the target node's params.
            if new_state_dict == state_data:
                return noop
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas, inspector, popover_contents = _render_views(
                new_state
            )
            return (
                new_state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_contents,
                _popover_style_for(popover_contents),
                no_update,
                no_update,
                no_update,
                no_update,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("fan_in_picker_store failed")
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                True,
                _format_exception(exc),
                "danger",
                "Update failed",
            )

    # ----------------------------------------------------------------------
    # 2c. Popover wiring (port-click → popover, dismiss, action buttons,
    # inspector-focus banner)
    # ----------------------------------------------------------------------
    #
    # The canvas-anchored popover replaces the old inspector aux-palette
    # controls. The clientside ``aux_popover.js`` glue handles positioning
    # the popover (via popper.js) and writes three stores:
    #
    #   * ``PORT_CLICK_STORE`` — fires when the user taps an aux port; the
    #     server-side callback re-renders ``POPOVER_CONTAINER`` and may
    #     auto-focus the first wired slot in the inspector.
    #   * ``POPOVER_DISMISS_STORE`` — fires on click-outside / Escape /
    #     canvas pan; the server-side callback clears
    #     ``inspector_focus_aux`` and hides the popover.
    #   * ``POPOVER_ACTION_STORE`` — written when a popover button is
    #     clicked. Today we listen to the pattern-matched action buttons
    #     directly (one ``Input`` per ``action`` value) so the dispatch
    #     stays pure Python, but ``POPOVER_ACTION_STORE`` is reserved for
    #     future cross-cutting needs (e.g. close-on-action).

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
        Input(ids.PORT_CLICK_STORE, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def open_popover_from_port_click(
        port_click: Optional[Dict[str, Any]],
        state_data: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Render the popover (and inspector) when an aux port is tapped.

        Reads the structured payload written by ``aux_popover.js`` to
        ``PORT_CLICK_STORE``, auto-focuses the first wired slot (when
        any) so the inspector mirrors the wired aux, then re-renders the
        popover contents + inspector. The cytoscape canvas is rebuilt as
        well so it can paint the matching ``aux-port--wired`` state if
        the focus dispatch changed slot occupancy.
        """

        noop = (no_update,) * 6
        if not isinstance(port_click, dict) or state_data is None:
            return noop

        target_node_id = port_click.get("target_node_id")
        param = port_click.get("param")
        if not isinstance(target_node_id, str) or not isinstance(param, str):
            return noop

        try:
            scope = _scope_at_breadcrumb(
                state_data, state_data.get("breadcrumb", []) or []
            )
            consumer = _find_in_scope(scope, target_node_id)
            if consumer is None:
                return noop

            slot_list = (consumer.get("aux_ports") or {}).get(param) or []
            first_wired = next(
                (
                    i
                    for i, val in enumerate(slot_list)
                    if isinstance(val, dict)
                ),
                None,
            )
            # Auto-focus the first wired slot when one exists so the
            # inspector mirrors the wired aux's params. When the port has
            # NO wired slots we still need a focus entry (with a ``slot=0``
            # placeholder) so ``build_popover_contents`` knows which port
            # is active — without it the popover would render empty for
            # empty ports. ``_resolve_inspector_focus_target`` defensively
            # returns ``None`` when the focused slot is empty/out-of-bounds,
            # so the inspector cleanly falls back to its canvas-selected
            # consumer view (no slot-0 pollution downstream).
            new_state_dict = deepcopy(state_data)
            new_state_dict["inspector_focus_aux"] = {
                "target_node_id": target_node_id,
                "param": param,
                "slot": first_wired if first_wired is not None else 0,
            }

            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas, inspector, popover_contents = _render_views(
                new_state
            )
            return (
                new_state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_contents,
                _popover_style_for(popover_contents),
            )
        except Exception:  # noqa: BLE001
            logger.exception("open_popover_from_port_click failed")
            return noop

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
        Input(ids.POPOVER_DISMISS_STORE, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def dismiss_popover(
        _ts: Any, state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Clear ``inspector_focus_aux`` and hide the popover on dismiss."""

        noop = (no_update,) * 6
        if state_data is None:
            return noop
        try:
            new_state_dict = _dispatch_state_update(
                state_data,
                "set_inspector_focus",
                {"focus": "consumer"},
            )
            if new_state_dict == state_data:
                # Already cleared — just hide the popover container.
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    [],
                    {"display": "none"},
                )
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas, inspector, popover_contents = _render_views(
                new_state
            )
            return (
                new_state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_contents,
                _popover_style_for(popover_contents),
            )
        except Exception:  # noqa: BLE001
            logger.exception("dismiss_popover failed")
            return noop

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(
            {
                "type": "popover-action",
                "action": ALL,
                "target_node_id": ALL,
                "param": ALL,
                "slot": ALL,
                "class_name": ALL,
            },
            "n_clicks",
        ),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def handle_popover_action(  # noqa: C901, PLR0912
        _clicks: List[Any], state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Dispatch the popover button action keyed by the trigger ``action``."""

        noop = (no_update,) * 10
        if state_data is None or ctx.triggered_id is None:
            return noop
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return noop
        if triggered.get("type") != "popover-action":
            return noop
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return noop

        action = triggered.get("action")
        target_node_id = triggered.get("target_node_id")
        param = triggered.get("param")
        raw_slot = triggered.get("slot")
        class_name = triggered.get("class_name") or None
        try:
            slot = int(raw_slot) if raw_slot is not None else 0
        except (TypeError, ValueError):
            return noop

        try:
            new_state_dict: Dict[str, Any] = state_data
            # Set by the ``drill`` branch to force-dismiss the popover; the
            # other actions leave them as ``None`` so the renderer-derived
            # ``popover_contents`` / style apply.
            drill_dismiss = False
            if action == "pick_class":
                if not isinstance(class_name, str) or not class_name:
                    return noop
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "wire_create",
                    {
                        "target_node_id": target_node_id,
                        "param": param,
                        "slot": slot,
                        "class_name": class_name,
                    },
                )
            elif action == "edit":
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "set_inspector_focus",
                    {
                        "focus": "aux",
                        "target_node_id": target_node_id,
                        "param": param,
                        "slot": slot,
                    },
                )
            elif action == "drill":
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "drill_in_aux",
                    {
                        "target_node_id": target_node_id,
                        "param": param,
                        "slot": slot,
                    },
                )
                # Drill dismisses the popover — the new canvas scope
                # belongs to a different consumer.
                drill_dismiss = True
            elif action == "disconnect":
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "wire_delete",
                    {
                        "target_node_id": target_node_id,
                        "param": param,
                        "slot": slot,
                    },
                )
            elif action == "add_slot":
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "port_slot_add",
                    {
                        "node_id": target_node_id,
                        "param": param,
                    },
                )
            else:
                return noop

            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas, inspector, popover_contents = _render_views(
                new_state
            )
            if drill_dismiss:
                popover_children: Any = []
                popover_style: Dict[str, Any] = {"display": "none"}
            else:
                popover_children = popover_contents
                popover_style = _popover_style_for(popover_contents)
            return (
                new_state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_children,
                popover_style,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("handle_popover_action failed")
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                True,
                _format_exception(exc),
                "danger",
                "Update failed",
            )

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
        Input(INSPECTOR_FOCUS_AUX_BANNER_ID, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def revert_inspector_focus(
        n_clicks: Optional[int], state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Clear ``inspector_focus_aux`` when the inspector banner is clicked."""

        noop = (no_update,) * 6
        if not n_clicks or state_data is None:
            return noop
        try:
            new_state_dict = _dispatch_state_update(
                state_data,
                "set_inspector_focus",
                {"focus": "consumer"},
            )
            if new_state_dict == state_data:
                return noop
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas, inspector, popover_contents = _render_views(
                new_state
            )
            return (
                new_state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_contents,
                _popover_style_for(popover_contents),
            )
        except Exception:  # noqa: BLE001
            logger.exception("revert_inspector_focus failed")
            return noop

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
            from phenotypic.abc_ import GridOperation

            t0 = time.time()
            state = state_from_json(state_data)
            pipeline = to_pipeline(state.root)
            uses_grid = _pipeline_uses_grid(pipeline, GridOperation)

            image = _load_preview_image(image_path, uses_grid, nrows, ncols)

            cache = get_cache()
            cache.set_image(session_id, image, str(image_path) if image_path else None)

            result = pipeline.apply_with_intermediates(image)
            _bake_preview_cache(state, pipeline, result, session_id, cache)

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

        # Sentinel: render failed during preview run — surface the message.
        if isinstance(cached, PreviewRenderError):
            return html.Div(
                f"(Could not render preview: {cached.message})",
                className="text-warning",
            )

        # DataFrame preview for measurement / post nodes.
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except Exception:  # pragma: no cover - pandas always present in env
            pd = None  # type: ignore[assignment]

        if pd is not None and isinstance(cached, pd.DataFrame):
            return dataframe_to_table(cached)

        # Pre-baked PNG bytes — wrap and ship. The cache contract stores
        # `bytes` (not bytearray); the union check is only for type narrowing.
        if isinstance(cached, bytes):
            return html.Img(
                src=bytes_to_data_uri(cached),
                style={"maxWidth": "100%"},
            )

        # Unreachable under the current cache contract; defensive log.
        logger.warning(  # pragma: no cover
            "Unexpected cache payload type %s for node %s",
            type(cached).__name__,
            state.selected_node_id,
        )
        return no_update  # pragma: no cover

    # ----------------------------------------------------------------------
    # 5. Save modal — open / close / confirm / dir-nav / body re-render
    # ----------------------------------------------------------------------
    #
    # The Save flow used to be a single text-input + button. It is now a
    # modal file browser:
    #
    #   * BTN_SAVE          → open MODAL_SAVE
    #   * BTN_SAVE_CANCEL   → close MODAL_SAVE
    #   * BTN_SAVE_CONFIRM  → write JSON to STORE_BROWSE_DIR_SAVE /
    #                         INPUT_SAVE_FILENAME, close on success, toast
    #   * dir-entry click   → update STORE_BROWSE_DIR_SAVE
    #   * STORE_BROWSE_DIR_SAVE change → re-render MODAL_SAVE_BODY tree
    #
    # The existing path-safety rules and toast variants are preserved by
    # delegating into the same ``to_pipeline().to_json()`` flow.

    @app.callback(
        Output(ids.MODAL_SAVE, "is_open", allow_duplicate=True),
        Output(ids.STORE_BROWSE_DIR_SAVE, "data", allow_duplicate=True),
        Input(ids.BTN_SAVE, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_save_modal(n_clicks: Optional[int]) -> Tuple[Any, Any]:
        """Open :data:`ids.MODAL_SAVE` and seed :data:`ids.STORE_BROWSE_DIR_SAVE`.

        Triggered by :data:`ids.BTN_SAVE`. Seeds the browse-dir store with the
        ``pheno_image_root`` Flask config value so the folder tree starts at
        the configured working directory.

        Args:
            n_clicks: Click count from :data:`ids.BTN_SAVE`.

        Returns:
            Tuple of ``(is_open, store_data)`` — ``(True, root_str)`` on a
            valid click, or ``(no_update, no_update)`` to suppress the initial
            call.
        """
        if not n_clicks:
            return no_update, no_update
        image_root = _image_root()
        return True, str(image_root) if image_root else None

    @app.callback(
        Output(ids.MODAL_SAVE, "is_open", allow_duplicate=True),
        Input(ids.BTN_SAVE_CANCEL, "n_clicks"),
        prevent_initial_call=True,
    )
    def close_save_modal(n_clicks: Optional[int]) -> Any:
        """Close :data:`ids.MODAL_SAVE` when the Cancel button is clicked.

        Args:
            n_clicks: Click count from :data:`ids.BTN_SAVE_CANCEL`.

        Returns:
            ``False`` to close the modal, or ``no_update`` to suppress the
            initial call.
        """
        if not n_clicks:
            return no_update
        return False

    @app.callback(
        Output(ids.MODAL_SAVE, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_SAVE_CONFIRM, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_BROWSE_DIR_SAVE, "data"),
        State(ids.INPUT_SAVE_FILENAME, "value"),
        prevent_initial_call=True,
    )
    def save_pipeline(
        n_clicks: Optional[int],
        state_data: Dict[str, Any],
        dir_value: Optional[str],
        filename_value: Optional[str],
    ) -> Tuple[Any, ...]:
        """Write the current pipeline to disk and close :data:`ids.MODAL_SAVE`.

        Triggered by :data:`ids.BTN_SAVE_CONFIRM`. Deserialises the builder
        state, converts it to an :class:`~phenotypic.ImagePipeline` via
        ``to_pipeline()``, calls ``to_json()``, and writes the result to
        ``dir_value / filename_value``. A toast confirms the save path; if the
        path falls outside ``--image-root`` a warning variant is shown instead
        of an error. The modal is closed on success.

        Args:
            n_clicks: Click count from :data:`ids.BTN_SAVE_CONFIRM`.
            state_data: JSON-serialised :class:`BuilderState` from
                :data:`ids.STORE_BUILDER_STATE`.
            dir_value: Currently selected directory string from
                :data:`ids.STORE_BROWSE_DIR_SAVE`.
            filename_value: Filename entered in :data:`ids.INPUT_SAVE_FILENAME`.

        Returns:
            Five-tuple ``(modal_is_open, toast_is_open, toast_msg,
            toast_icon, toast_header)``.
        """
        if not n_clicks:
            return (no_update,) * 5
        if not dir_value:
            return (no_update, *_toast("Pick a folder first", ok=False))
        if not filename_value:
            return (no_update, *_toast("Provide a filename first", ok=False))

        try:
            # Aux nodes are now embedded inside each consumer's
            # ``aux_ports`` map (Wave 1-A onward), so "orphan aux" is
            # structurally impossible — an aux only exists while wired.
            # The pre-save orphan walk that used to live here has been
            # removed.
            state = state_from_json(state_data)
            pipeline = to_pipeline(state.root)
            target = (Path(dir_value).expanduser() / filename_value).resolve()

            # Defense-in-depth: reject targets outside ``--image-root`` before
            # touching the filesystem. Normal users cannot reach such paths
            # through the tree (it is bounded by ``_is_within``), but the store
            # is reachable from devtools.
            image_root = _image_root()
            if image_root is not None:
                try:
                    target.relative_to(Path(image_root).resolve())
                except ValueError:
                    return (
                        no_update,
                        *_toast(
                            f"Refused: {target} is outside --image-root",
                            ok=False,
                        ),
                    )

            payload = pipeline.to_json()
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(payload if isinstance(payload, str) else json.dumps(payload))
            return (False, *_toast(f"Saved to {target}", ok=True))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Save failed")
            return (no_update, *_toast(_format_exception(exc), ok=False))

    @app.callback(
        Output(ids.STORE_BROWSE_DIR_SAVE, "data", allow_duplicate=True),
        Input(
            {"type": ids.DIR_ENTRY_TYPE_SAVE, "kind": ALL, "path": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def navigate_save_dir(_entry_clicks: List[int]) -> Any:
        """Update :data:`ids.STORE_BROWSE_DIR_SAVE` when a folder is clicked in the Save tree.

        The Save modal renders only directories (``select_files=False``), so
        any triggered item is either a subdirectory or a parent entry. The new
        path is written to the store, which triggers ``render_save_body`` to
        refresh the tree.

        Args:
            _entry_clicks: Pattern-matched click counts from dir-entry items
                with type :data:`ids.DIR_ENTRY_TYPE_SAVE`.

        Returns:
            The clicked path string, or ``no_update`` when the trigger is not
            a valid directory entry click.
        """
        # Save tree only renders dirs / parents (select_files=False), so any
        # validated trigger is a navigation click.
        match = _trigger_kind_path(ctx.triggered_id, ids.DIR_ENTRY_TYPE_SAVE)
        if match is None:
            return no_update
        _, path = match
        return path

    @app.callback(
        Output(ids.MODAL_SAVE_BODY, "children"),
        Input(ids.STORE_BROWSE_DIR_SAVE, "data"),
        prevent_initial_call=True,
    )
    def render_save_body(dir_value: Optional[str]) -> Any:
        """Rebuild the folder tree inside :data:`ids.MODAL_SAVE_BODY` after navigation.

        Triggered whenever :data:`ids.STORE_BROWSE_DIR_SAVE` changes (i.e.
        when the user clicks a folder entry). Renders a directory-only tree
        (``select_files=False``) rooted at ``pheno_image_root``.

        Args:
            dir_value: Currently selected directory path string from
                :data:`ids.STORE_BROWSE_DIR_SAVE`, or ``None`` if unset.

        Returns:
            A :func:`directory_tree` component, or a muted placeholder ``Div``
            when no working directory is configured.
        """
        return _render_tree_body(
            dir_value,
            extensions=None,
            select_files=False,
            id_type=ids.DIR_ENTRY_TYPE_SAVE,
        )

    # ----------------------------------------------------------------------
    # 6. Load picker modal — open / page-swap / dir-nav / JSON / Prefab
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.MODAL_LOAD_PICKER, "is_open", allow_duplicate=True),
        Output(ids.STORE_LOAD_PICKER_PAGE, "data", allow_duplicate=True),
        Output(ids.STORE_BROWSE_DIR_JSON, "data", allow_duplicate=True),
        Input(ids.BTN_LOAD, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_load_picker(n_clicks: Optional[int]) -> Tuple[Any, ...]:
        """Open :data:`ids.MODAL_LOAD_PICKER` on the chooser page.

        Triggered by :data:`ids.BTN_LOAD`. Resets the page store to
        ``"chooser"`` and seeds :data:`ids.STORE_BROWSE_DIR_JSON` with the
        configured ``pheno_image_root`` so the JSON page starts at the right
        directory if the user navigates there.

        Args:
            n_clicks: Click count from :data:`ids.BTN_LOAD`.

        Returns:
            Three-tuple ``(modal_is_open, page_token, browse_dir_str)``.
        """
        if not n_clicks:
            return no_update, no_update, no_update
        image_root = _image_root()
        return True, "chooser", str(image_root) if image_root else None

    @app.callback(
        Output(ids.STORE_LOAD_PICKER_PAGE, "data", allow_duplicate=True),
        Input(ids.BTN_LOAD_JSON_CHOICE, "n_clicks"),
        Input(ids.BTN_LOAD_PREFAB_CHOICE, "n_clicks"),
        Input(ids.BTN_LOAD_PICKER_BACK, "n_clicks"),
        prevent_initial_call=True,
    )
    def swap_load_picker_page(
        _json_clicks: Optional[int],
        _prefab_clicks: Optional[int],
        _back_clicks: Optional[int],
    ) -> Any:
        """Update :data:`ids.STORE_LOAD_PICKER_PAGE` to navigate between chooser subpages.

        Triggered by any of :data:`ids.BTN_LOAD_JSON_CHOICE`,
        :data:`ids.BTN_LOAD_PREFAB_CHOICE`, or :data:`ids.BTN_LOAD_PICKER_BACK`.
        Writing to the store in turn triggers ``render_load_picker`` to swap
        the modal body.

        Returns:
            The new page token: ``"json"``, ``"prefab"``, or ``"chooser"``,
            or ``no_update`` when triggered by an unexpected id.
        """
        triggered = ctx.triggered_id
        if triggered == ids.BTN_LOAD_JSON_CHOICE:
            return "json"
        if triggered == ids.BTN_LOAD_PREFAB_CHOICE:
            return "prefab"
        if triggered == ids.BTN_LOAD_PICKER_BACK:
            return "chooser"
        return no_update

    @app.callback(
        Output(ids.MODAL_LOAD_PICKER_BODY, "children"),
        Input(ids.STORE_LOAD_PICKER_PAGE, "data"),
        Input(ids.STORE_BROWSE_DIR_JSON, "data"),
        prevent_initial_call=True,
    )
    def render_load_picker(
        page: Optional[LoadPickerPage], dir_value: Optional[str]
    ) -> Any:
        """Rebuild :data:`ids.MODAL_LOAD_PICKER_BODY` when the page or browse directory changes.

        Handles the JSON page specially: re-renders the full
        :func:`directory_tree` (filtered to ``.json`` files) for the active
        ``dir_value``. Chooser and prefab pages are delegated to
        :func:`~_modal_browser.render_load_picker_body`. The Back button is
        a permanent sibling of this body container (see
        :func:`load_picker_modal`), so it is not re-emitted here.

        Args:
            page: Current page token from :data:`ids.STORE_LOAD_PICKER_PAGE`.
            dir_value: Currently viewed directory string from
                :data:`ids.STORE_BROWSE_DIR_JSON`.

        Returns:
            A list (or single component) of Dash components for
            :data:`ids.MODAL_LOAD_PICKER_BODY`.
        """
        if page == "json" and _image_root() is not None:
            return _render_tree_body(
                dir_value,
                extensions=PIPELINE_EXTS,
                select_files=True,
                id_type=ids.DIR_ENTRY_TYPE_JSON,
            )
        return render_load_picker_body(page or "chooser", _image_root())

    @app.callback(
        Output(ids.BTN_LOAD_PICKER_BACK, "style"),
        Output("load-picker-chooser-buttons", "style"),
        Input(ids.STORE_LOAD_PICKER_PAGE, "data"),
    )
    def toggle_load_picker_chrome(
        page: Optional[LoadPickerPage],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Show/hide back button + chooser buttons based on the active page.

        On the chooser page, the chooser buttons are visible and the back
        button is hidden. On the JSON / Prefab pages it's the inverse — the
        back button shows so the user can return to the chooser. Both stay
        in the DOM so :func:`swap_load_picker_page`'s pattern-matching
        ``Input`` subscriptions always resolve.
        """
        if page in ("json", "prefab"):
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output(ids.STORE_BROWSE_DIR_JSON, "data", allow_duplicate=True),
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
        Output(ids.MODAL_LOAD_PICKER, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(
            {"type": ids.DIR_ENTRY_TYPE_JSON, "kind": ALL, "path": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def click_json_entry(_entry_clicks: List[int]) -> Tuple[Any, ...]:
        """Handle a click on a JSON-browser tree entry.

        Directory and parent clicks update :data:`ids.STORE_BROWSE_DIR_JSON`
        to navigate the tree. File clicks read the selected ``.json`` path,
        call ``ImagePipeline.from_json()``, replace the builder state, and
        close the modal with a success toast.

        Args:
            _entry_clicks: Pattern-matched click counts from dir-entry items
                with type :data:`ids.DIR_ENTRY_TYPE_JSON`.

        Returns:
            Eleven-tuple ``(browse_dir, state_data, breadcrumb, canvas,
            inspector, popover, modal_is_open, toast_is_open, toast_msg,
            toast_icon, toast_header)``. Directory clicks populate only the
            first element; file clicks populate elements 2–11.
        """
        match = _trigger_kind_path(ctx.triggered_id, ids.DIR_ENTRY_TYPE_JSON)
        if match is None:
            return (no_update,) * 11
        kind, path_str = match
        if kind in {"dir", "parent"}:
            return (path_str, *((no_update,) * 10))

        if kind == "file":
            try:
                from phenotypic import ImagePipeline

                with open(Path(path_str).expanduser(), encoding="utf-8") as fh:
                    pipeline = ImagePipeline.from_json(fh.read())
                (
                    state_dict,
                    breadcrumb,
                    canvas,
                    inspector,
                    popover_contents,
                    popover_style,
                ) = _state_replacement_payload(pipeline)
                return (
                    no_update,
                    state_dict,
                    breadcrumb,
                    canvas,
                    inspector,
                    popover_contents,
                    popover_style,
                    False,
                    *_toast(f"Loaded {Path(path_str).name}", ok=True),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Load JSON failed")
                return (
                    (no_update,) * 8 + _toast(_format_exception(exc), ok=False)
                )

        return (no_update,) * 12

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output("canvas-cytoscape-wrapper", "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "children", allow_duplicate=True),
        Output(ids.POPOVER_CONTAINER, "style", allow_duplicate=True),
        Output(ids.MODAL_LOAD_PICKER, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(
            {"type": "prefab-card", "class_name": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def click_prefab_card(_clicks: List[int]) -> Tuple[Any, ...]:
        """Instantiate a prefab pipeline and replace the current builder state.

        Triggered by a click on any :func:`~_ids.prefab_card_id` list item.
        Imports the named class from :mod:`phenotypic.prefab`, calls its
        constructor with no arguments, converts the result to a
        :class:`BuilderScope` via ``from_pipeline()``, and replaces the
        builder state. The modal is closed and a success toast is shown.

        Args:
            _clicks: Pattern-matched click counts from prefab-card items.

        Returns:
            Ten-tuple ``(state_data, breadcrumb, canvas, inspector, popover,
            modal_is_open, toast_is_open, toast_msg, toast_icon,
            toast_header)``.
        """
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered.get("type") != "prefab-card":
            return (no_update,) * 11
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update,) * 11
        class_name = triggered.get("class_name")
        if not class_name:
            return (no_update,) * 11

        try:
            import phenotypic.prefab as prefab_module

            pipeline = getattr(prefab_module, class_name)()
            (
                state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_contents,
                popover_style,
            ) = _state_replacement_payload(pipeline)
            return (
                state_dict,
                breadcrumb,
                canvas,
                inspector,
                popover_contents,
                popover_style,
                False,
                *_toast(f"Loaded prefab {class_name}", ok=True),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load prefab failed")
            return (no_update,) * 7 + _toast(_format_exception(exc), ok=False)

    # ----------------------------------------------------------------------
    # 7. Load Image modal — open / dir-nav / file pick / synthetic shortcut
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.MODAL_LOAD_IMAGE, "is_open", allow_duplicate=True),
        Output(ids.STORE_BROWSE_DIR_IMAGE, "data", allow_duplicate=True),
        Input(ids.BTN_LOAD_IMAGE, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_load_image_modal(n_clicks: Optional[int]) -> Tuple[Any, Any]:
        """Open :data:`ids.MODAL_LOAD_IMAGE` and seed :data:`ids.STORE_BROWSE_DIR_IMAGE`.

        Triggered by :data:`ids.BTN_LOAD_IMAGE`. Seeds the browse-dir store
        with ``pheno_image_root`` so the image tree starts at the configured
        directory.

        Args:
            n_clicks: Click count from :data:`ids.BTN_LOAD_IMAGE`.

        Returns:
            Tuple of ``(is_open, store_data)`` — ``(True, root_str)`` on a
            valid click, or ``(no_update, no_update)`` to suppress the initial
            call.
        """
        if not n_clicks:
            return no_update, no_update
        image_root = _image_root()
        return True, str(image_root) if image_root else None

    @app.callback(
        Output(ids.STORE_BROWSE_DIR_IMAGE, "data", allow_duplicate=True),
        Output(ids.STORE_IMAGE_PATH, "data", allow_duplicate=True),
        Output(ids.MODAL_LOAD_IMAGE, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(
            {"type": ids.DIR_ENTRY_TYPE_IMAGE, "kind": ALL, "path": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def click_image_entry(_entry_clicks: List[int]) -> Tuple[Any, ...]:
        """Handle a click on an image-browser tree entry.

        Directory and parent clicks update :data:`ids.STORE_BROWSE_DIR_IMAGE`
        to navigate the tree. File clicks write the selected path to
        :data:`ids.STORE_IMAGE_PATH` and close the modal with a toast
        confirming the selected plate image filename.

        Args:
            _entry_clicks: Pattern-matched click counts from dir-entry items
                with type :data:`ids.DIR_ENTRY_TYPE_IMAGE`.

        Returns:
            Seven-tuple ``(browse_dir, image_path, modal_is_open,
            toast_is_open, toast_msg, toast_icon, toast_header)``. Directory
            clicks populate only the first element; file clicks populate
            elements 2–7.
        """
        match = _trigger_kind_path(ctx.triggered_id, ids.DIR_ENTRY_TYPE_IMAGE)
        if match is None:
            return (no_update,) * 7
        kind, path_str = match
        if kind in {"dir", "parent"}:
            return (path_str, *((no_update,) * 6))
        if kind == "file":
            return (
                no_update,
                path_str,
                False,
                *_toast(f"Image set: {Path(path_str).name}", ok=True),
            )
        return (no_update,) * 7

    @app.callback(
        Output(ids.MODAL_LOAD_IMAGE_BODY, "children"),
        Input(ids.STORE_BROWSE_DIR_IMAGE, "data"),
        prevent_initial_call=True,
    )
    def render_load_image_body(dir_value: Optional[str]) -> Any:
        """Rebuild the image-file tree inside :data:`ids.MODAL_LOAD_IMAGE_BODY` after navigation.

        Triggered whenever :data:`ids.STORE_BROWSE_DIR_IMAGE` changes.
        Renders :func:`directory_tree` filtered to :data:`IMAGE_EXTS`
        (plate image formats including DSLR raw formats ``.nef``, ``.cr2``,
        ``.arw``). The store itself lives outside this body container — see
        :func:`._modal_browser.load_image_modal` — so it survives body
        re-renders without re-emitting itself (which would cycle this
        callback).

        Args:
            dir_value: Currently browsed directory path string from
                :data:`ids.STORE_BROWSE_DIR_IMAGE`, or ``None`` if unset.

        Returns:
            A :func:`directory_tree` div, or a placeholder div when no
            working directory is configured.
        """
        return _render_tree_body(
            dir_value,
            extensions=IMAGE_EXTS,
            select_files=True,
            id_type=ids.DIR_ENTRY_TYPE_IMAGE,
        )

    @app.callback(
        Output(ids.STORE_IMAGE_PATH, "data", allow_duplicate=True),
        Output(ids.MODAL_LOAD_IMAGE, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_USE_SYNTHETIC, "n_clicks"),
        Input(ids.BTN_USE_SYNTHETIC_MODAL, "n_clicks"),
        prevent_initial_call=True,
    )
    def use_synthetic_plate(
        _top_clicks: Optional[int], _modal_clicks: Optional[int]
    ) -> Tuple[Any, ...]:
        """Set :data:`ids.STORE_IMAGE_PATH` to the bundled synthetic yeast plate.

        Handles both the footer-level :data:`ids.BTN_USE_SYNTHETIC` and the
        in-modal :data:`ids.BTN_USE_SYNTHETIC_MODAL`. When triggered from
        inside the modal the modal is also closed; the footer button leaves the
        modal state unchanged. Uses :func:`find_synthetic_plate_path` to
        resolve the on-disk path; falls back to :data:`SYNTHETIC_SENTINEL`
        when the file cannot be found, and the run-preview callback detects
        the sentinel and calls :func:`~phenotypic.data.load_synth_yeast_plate`
        programmatically.

        Args:
            _top_clicks: Click count from :data:`ids.BTN_USE_SYNTHETIC`
                (footer bar button).
            _modal_clicks: Click count from :data:`ids.BTN_USE_SYNTHETIC_MODAL`
                (modal footer button).

        Returns:
            Six-tuple ``(image_path, modal_is_open, toast_is_open, toast_msg,
            toast_icon, toast_header)``.
        """
        triggered = ctx.triggered_id
        if triggered not in (ids.BTN_USE_SYNTHETIC, ids.BTN_USE_SYNTHETIC_MODAL):
            return (no_update,) * 6
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update,) * 6
        synth = find_synthetic_plate_path()
        # Close the load-image modal if the user triggered from inside it.
        close_modal = False if triggered == ids.BTN_USE_SYNTHETIC_MODAL else no_update
        return (
            str(synth),
            close_modal,
            *_toast("Using synthetic yeast plate", ok=True),
        )

    @app.callback(
        Output(ids.ACTIVE_IMAGE_LABEL, "children"),
        Input(ids.STORE_IMAGE_PATH, "data"),
    )
    def update_active_image_label(image_path: Optional[str]) -> str:
        """Update :data:`ids.ACTIVE_IMAGE_LABEL` to reflect the currently loaded image.

        Triggered whenever :data:`ids.STORE_IMAGE_PATH` changes. Displays the
        filename of the loaded plate image (or a synthetic-plate label) beneath
        the "Load image" / "Use synthetic plate" buttons in the footer so the
        user can confirm which image will be used for "Run preview".

        Args:
            image_path: Path string from :data:`ids.STORE_IMAGE_PATH`, or
                ``None`` / empty string when no image is loaded.

        Returns:
            A short display string: ``"(no image loaded)"``,
            ``"synthetic yeast plate"``, or the image filename (basename only).
        """
        if not image_path:
            return "(no image loaded)"
        if image_path == SYNTHETIC_SENTINEL:
            return "synthetic yeast plate"
        return Path(image_path).name


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

    def _register_canvas_clientside(button_id: str, body: str) -> None:
        """Register a clientside callback ``button_id → body`` against ``cy``."""

        app.clientside_callback(
            f"""
            function(n_clicks, prev) {{
                if (!n_clicks) return window.dash_clientside.no_update;
                const cy = window.phenoGetCy && window.phenoGetCy();
                if (cy) {{ {body} }}
                return (prev || 0) + 1;
            }}
            """,
            Output(ids.STORE_CANVAS_CONTROL, "data", allow_duplicate=True),
            Input(button_id, "n_clicks"),
            State(ids.STORE_CANVAS_CONTROL, "data"),
            prevent_initial_call=True,
        )

    _register_canvas_clientside(
        ids.BTN_CANVAS_FIT,
        "cy.animate({fit: {padding: 24}}, {duration: 200});",
    )
    _register_canvas_clientside(
        ids.BTN_CANVAS_ZOOM_IN,
        "const z = cy.zoom() * 1.25;"
        " const c = {x: cy.width() / 2, y: cy.height() / 2};"
        " cy.zoom({level: z, renderedPosition: c});",
    )
    _register_canvas_clientside(
        ids.BTN_CANVAS_ZOOM_OUT,
        "const z = cy.zoom() / 1.25;"
        " const c = {x: cy.width() / 2, y: cy.height() / 2};"
        " cy.zoom({level: z, renderedPosition: c});",
    )

    # ----------------------------------------------------------------------
    # 9. Point picker — clientside lifecycle callbacks
    # ----------------------------------------------------------------------
    # The Python callbacks in ``_point_picker.py`` own modal open / close,
    # staged-store mutation, and the Confirm fan-out. The OSD viewer itself
    # is a clientside concern (heavy WebGL canvas; round-tripping every
    # click through Dash would be wasteful). Three callbacks bridge the
    # two layers:
    #
    #   A. Mount / remount the OSD viewer when ``PICKER_DZI_URL_STORE``
    #      changes (modal open or channel toggle).
    #   B. Redraw the marker overlay when ``PICKER_STAGED_STORE`` changes
    #      (clicks, undo, clear, or modal-open seed).
    #   C. Dispose the viewer when the modal closes — frees WebGL context.
    #
    # All three write to the same hidden ``picker-osd-mount-trigger`` store
    # so each callback has a real Output without polluting the modal's
    # state stores. The trigger store's value is never read.

    app.clientside_callback(
        """
        function(dziUrl, stagedPoints) {
            const ns = window.__phenotypicBuilderPointPicker;
            if (!ns || !ns.mountViewer) {
                return window.dash_clientside.no_update;
            }
            if (!dziUrl) {
                if (ns.disposeViewer) ns.disposeViewer();
                return window.dash_clientside.no_update;
            }
            // Defer one frame so the modal body has rendered the host div
            // when the callback fires immediately after the modal opens.
            requestAnimationFrame(function () {
                ns.mountViewer("picker-osd", dziUrl, stagedPoints || []);
            });
            return Date.now();
        }
        """,
        Output(ids.PICKER_OSD_MOUNT_TRIGGER, "data", allow_duplicate=True),
        Input(ids.PICKER_DZI_URL_STORE, "data"),
        State(ids.PICKER_STAGED_STORE, "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(stagedPoints) {
            const ns = window.__phenotypicBuilderPointPicker;
            if (!ns || !ns.redrawOverlay) {
                return window.dash_clientside.no_update;
            }
            ns.redrawOverlay(stagedPoints || []);
            return Date.now();
        }
        """,
        Output(ids.PICKER_OSD_MOUNT_TRIGGER, "data", allow_duplicate=True),
        Input(ids.PICKER_STAGED_STORE, "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(isOpen) {
            const ns = window.__phenotypicBuilderPointPicker;
            if (!ns || !ns.disposeViewer) {
                return window.dash_clientside.no_update;
            }
            if (!isOpen) {
                ns.disposeViewer();
            }
            return Date.now();
        }
        """,
        Output(ids.PICKER_OSD_MOUNT_TRIGGER, "data", allow_duplicate=True),
        Input(ids.MODAL_POINT_PICKER, "is_open"),
        prevent_initial_call=True,
    )

    # ----------------------------------------------------------------------
    # 10. Inspector "Documentation" collapse toggle.
    # ----------------------------------------------------------------------
    # The Inspector is fully rebuilt by the fan-in callback whenever the
    # selected node changes, so each new selection starts with the
    # docstring section collapsed. This callback only handles the
    # in-place expand / collapse for the currently-selected node.
    # ``_doc_section_widgets`` (in ``_layout.py``) emits hidden
    # placeholders carrying the same ids on every other branch so this
    # callback's Input / State always resolve.

    @app.callback(
        Output(ids.INSPECTOR_DOC_COLLAPSE, "is_open"),
        Input(ids.INSPECTOR_DOC_TOGGLE, "n_clicks"),
        State(ids.INSPECTOR_DOC_COLLAPSE, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_inspector_doc(
        n_clicks: Optional[int], is_open: bool
    ) -> bool:
        """Flip the docstring collapse open / closed on toggle clicks."""

        if not n_clicks:
            return is_open
        return not is_open


# ---------------------------------------------------------------------------
# Helpers (private to this module)
# ---------------------------------------------------------------------------


def _bake_preview_cache(
    state: "BuilderState",
    pipeline: Any,
    result: Any,
    session_id: str,
    cache: Any,
) -> None:
    """Render every intermediate to PNG bytes (or DataFrame) into *cache*.

    Pulled out of ``run_preview`` so the cache contract — bytes for ops,
    DataFrame for meas/post, ``PreviewRenderError`` on render failure — can
    be exercised end-to-end without booting a Dash server.

    Args:
        state: The deserialised :class:`BuilderState` driving the preview.
        pipeline: The compiled :class:`ImagePipeline`.
        result: The :class:`IntermediateResult` returned by
            :meth:`ImagePipeline.apply_with_intermediates`.
        session_id: Per-tab uuid keying the cache.
        cache: The :class:`IntermediatesCache` to populate.
    """

    # Map intermediate keys ("GaussianBlur", "GaussianBlur_2", ...) back to
    # BuilderState node-ids by walking ops in declaration order.
    ops_nodes: List[StepNode] = [
        n
        for n in state.root.nodes
        if n.class_name == PIPELINE_CLASS_NAME
        or stage_of(n.class_name) == "ops"
    ]
    # Pre-bake one PNG per ops intermediate so the inspector never has to
    # re-encode the source Image on selection. Source Images go out of scope
    # at the end of the loop body — only the bytes are retained in the cache.
    for op_key, node in zip(pipeline.get_ops().keys(), ops_nodes):
        inter = result.intermediates.get(op_key)
        if inter is None:
            continue
        try:
            png = render_node_preview(inter, node.class_name)
        except Exception as render_exc:  # noqa: BLE001
            logger.warning(
                "Preview render failed for %s (%s): %s",
                node.class_name, node.node_id, render_exc,
            )
            cache.set_intermediate(
                session_id,
                node.node_id,
                PreviewRenderError(_format_exception(render_exc)),
            )
            continue
        cache.set_intermediate(session_id, node.node_id, png)

    # Run measurements if any are configured. Their output is a single
    # DataFrame; attach it to every measurement / post node so the inspector
    # can show it for whichever the user selects. measure() needs the
    # *processed* image (objmap populated by the detector chain), not the
    # raw input.
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


def _load_preview_image(
    image_path: Optional[str],
    uses_grid: bool,
    nrows: Optional[Any],
    ncols: Optional[Any],
) -> Any:
    """Load the input :class:`Image` / :class:`GridImage` for a preview run.

    Falls back to :func:`load_synth_yeast_plate` when *image_path* is empty or
    the synthetic sentinel.  Otherwise reads from disk via :class:`GridImage`
    when the pipeline contains a :class:`GridOperation`, else :class:`Image`.
    Default grid is ``8 × 12`` when *nrows* / *ncols* are unset.
    """

    from phenotypic import GridImage, Image

    if not image_path or image_path == SYNTHETIC_SENTINEL:
        from phenotypic.data._synthetic_data import load_synth_yeast_plate

        return load_synth_yeast_plate()

    p = Path(image_path)
    if uses_grid:
        return GridImage.imread(
            p,
            nrows=int(nrows) if nrows else 8,
            ncols=int(ncols) if ncols else 12,
        )
    return Image.imread(p)


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
    "_build_fresh_aux_node",
    "_source_satisfies_port",
    "STORE_IMAGE_PATH",
]
