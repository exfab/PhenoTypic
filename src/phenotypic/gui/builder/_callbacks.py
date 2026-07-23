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
* **DAG wire flow** — wires are drawn between block ports via the
  clientside ``wire_drawing.js`` glue; the inspector handles aux-port
  edits (per-row Disconnect / reorder) through ``STORE_EDGE_EVENT``.
  The legacy popover-anchored wire flow (Wave 4) is gone (Phase 7).

Every state-mutating callback uses ``prevent_initial_call=True`` and wraps
its body in ``try / except`` so a callback can never crash the running app
— errors flip the toast to the failure variant instead.
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias

import dash
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from flask import current_app

from phenotypic.gui._config import (
    CFG_IMAGE_ROOT,
    CFG_OPERATION_REGISTRY,
    CFG_URL_PREFIX,
)
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._ids import LoadPickerPage
from phenotypic.gui.builder._preview_callbacks import build_preview_payload
from phenotypic.gui.builder._preview_tiles import preview_dzi_url
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
from phenotypic.gui.builder._linear_model import (
    LinearTarget,
    default_continuation_target,
    derive_linear_scope,
    resolve_selected_target,
    scope_at_path,
    scope_key,
    target_from_dict,
    target_to_dict,
)
from phenotypic.gui.builder._layout import (
    _resolve_dag_accepts,
    _sort_issues_for_badge,
    build_breadcrumb,
    build_canvas_elements,
    build_inspector,
    build_issue_badge,
)
from phenotypic.gui.builder._linear_layout import (
    build_linear_map_section,
    build_linear_side_loader,
)
from phenotypic.gui.builder._modal_browser import (
    no_root_placeholder,
    render_load_picker_body,
)
from phenotypic.gui.builder._param_form import parse_widget_value
from phenotypic.gui.builder._session import (
    PreviewGenerationWriter,
    PreviewKey,
    PreviewRenderError,
    get_cache,
)
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    BuilderState,
    _DagBuilderScope,
    _DagBuilderState,
    StepNode,
    _new_block_id,
    _new_node_id,
    current_scope,
    stage_of,
    state_from_json,
    state_to_json,
    to_pipeline,
)
from phenotypic.gui.builder._conversion_dag import from_pipeline_dag, to_pipeline_dag
from phenotypic.gui.builder._validation import validate
from phenotypic.gui.builder._validation import Issue
from phenotypic.gui.shell._ids import SHELL_SOURCE_IMAGE_ROOT_STORE
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import resolve_source_image_root
from phenotypic.sdk_ import CONFIG_SUFFIX_PIPELINE, ensure_typed_json_suffix

logger = logging.getLogger(__name__)


# ``STORE_IMAGE_PATH`` lives on :mod:`._ids` now; alias here for backwards
# compatibility with callers that imported it from this module.
STORE_IMAGE_PATH = ids.STORE_IMAGE_PATH


# Pre-built ``no_update`` tuples for the long-output callbacks.  Building them
# once at import time keeps the callback bodies readable and avoids re-emitting
# the same long literal on every early-return branch.
#
# Layout: state, breadcrumb, canvas_elements, inspector, 4 toast outputs.
_NOOP_FAN_IN: Tuple[Any, ...] = (no_update,) * 8


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
          into a main-ribbon node's ``nested`` scope; ``param`` is now
          a no-op since the popover-era synthesized op-param scope was
          removed in Phase 7);
        * the aux-slot drill form ``{"target_node_id": ..., "param": ...,
          "slot": ...}`` (no-op in Phase 7+ — the popover wire flow that
          pushed these segments is gone). Mirrors the state-side walker
          ``_normalize_breadcrumb_segment`` in ``_state.py``.
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
      <name>}``): the synthesised op-param scope was retired in Phase 7
      with the popover wire flow. The segment is now treated as a no-op
      (continues walking with the current scope) so older saved state
      that still carries these segments loads without crashing.
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

    Defensive note: when *state_dict* carries the DAG
    schema (``root`` is a ``{"blocks": [...], "edges": [...]}`` dict, not
    a legacy ``{"nodes": [...]}`` dict), this helper returns the root
    scope unchanged.  Every DAG dispatch kind (``block_create``,
    ``edge_create``, ``block_reparent``, etc.) re-computes its own scope
    via :func:`_dag_scope_at_breadcrumb` / :func:`_find_block_in_tree`
    and never consults the result of this legacy walker, so returning
    root is safe — the alternative is a ``KeyError: 'nodes'`` raised
    immediately at the top of :func:`_dispatch_state_update` for any
    DAG mutation triggered while the breadcrumb is non-empty (i.e. while
    drilled into a container).
    """

    scope = state_dict["root"]
    # DAG-schema short-circuit: legacy scopes have a ``nodes`` list; DAG
    # scopes have ``blocks``.  When the root carries the DAG shape we
    # don't walk the breadcrumb here — DAG dispatchers handle their own
    # scope resolution via the dedicated helpers.
    if isinstance(scope, dict) and "blocks" in scope and "nodes" not in scope:
        return scope
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
            # Op-typed-parameter drill via the legacy popover-era synthesized
            # scope is no longer supported (Phase 7 removed the synthesized
            # op-param scope machinery). Treat the segment as a no-op so
            # the walker doesn't crash on stale breadcrumbs loaded from
            # older saved state.
            return scope
    return scope


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


def _find_dag_container_scope(
    root_scope_dict: Dict[str, Any], block_id: str
) -> Optional[Dict[str, Any]]:
    """Depth-first search for a container's nested scope dict by ``block_id``.

    Dict-level mirror of
    :func:`phenotypic.gui.builder._state._find_container_scope_by_block_id`,
    used by the ``block_create`` dispatch kind (and any future DAG
    dispatch that takes a ``container_block_id`` payload field) so
    state mutations operate on JSON-shaped state without re-hydrating
    the dataclass tree on every call.

    The walk visits every block in *root_scope_dict* (and, recursively,
    every block in every container's nested scope) and returns the
    *nested scope dict* of the first block whose ``block_id`` matches.
    Only container blocks (``class_name == PIPELINE_CLASS_NAME``) carry
    a non-``None`` ``nested`` field; non-container hits return ``None``.

    Args:
        root_scope_dict: The outermost DAG scope dict — the value of
            ``state_dict["root"]`` for a payload encoded via
            :func:`state_to_json`.
        block_id: The container ``BlockNode.block_id`` whose ``nested``
            scope dict is being resolved.

    Returns:
        The matching container's nested scope dict (``{"blocks": [...],
        "edges": [...], ...}``), or ``None`` when *block_id* doesn't
        resolve to a container in the tree.
    """

    for block in root_scope_dict.get("blocks", []) or []:
        if block.get("block_id") == block_id:
            nested = block.get("nested")
            return nested if isinstance(nested, dict) else None
        nested = block.get("nested")
        if isinstance(nested, dict):
            hit = _find_dag_container_scope(nested, block_id)
            if hit is not None:
                return hit
    return None


def _find_block_in_tree(
    root_scope_dict: Dict[str, Any], block_id: str
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Locate a block dict + its containing scope dict by ``block_id``.

    Depth-first search rooted at *root_scope_dict*; recurses into every
    container's ``nested`` scope so callers can look up blocks anywhere
    in the state tree without knowing the breadcrumb depth.  Used by
    the edge dispatchers (``edge_create``, ``edge_delete``,
    ``list_aux_reorder``, etc.) which all need the block dict plus its
    enclosing scope (to read/write that scope's ``edges`` list and
    enforce the spec §4.4 cross-scope rule).

    Args:
        root_scope_dict: The outermost scope dict (``state_dict["root"]``).
        block_id: The :class:`BlockNode.block_id` to find.

    Returns:
        ``(scope_dict, block_dict)`` tuple where ``scope_dict`` is the
        scope that directly contains *block_id* and ``block_dict`` is
        the block entry inside that scope's ``blocks`` list, or ``None``
        when *block_id* doesn't resolve to a block anywhere in the
        tree.
    """

    for block in root_scope_dict.get("blocks", []) or []:
        if block.get("block_id") == block_id:
            return root_scope_dict, block
        nested = block.get("nested")
        if isinstance(nested, dict):
            hit = _find_block_in_tree(nested, block_id)
            if hit is not None:
                return hit
    return None


def _find_edge_in_tree(
    root_scope_dict: Dict[str, Any], edge_id: str
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Locate an edge dict + its containing scope dict by ``edge_id``.

    Mirrors :func:`_find_block_in_tree` but for edges.  Used by
    ``edge_delete`` (which receives only an ``edge_id`` and must
    discover the scope) and by the issue-pan helpers that pre-resolve
    a target edge for inspector wire-cards.

    Args:
        root_scope_dict: The outermost scope dict.
        edge_id: The :class:`Edge.edge_id` to find.

    Returns:
        ``(scope_dict, edge_dict)`` or ``None`` when not found.
    """

    for edge in root_scope_dict.get("edges", []) or []:
        if edge.get("edge_id") == edge_id:
            return root_scope_dict, edge
    for block in root_scope_dict.get("blocks", []) or []:
        nested = block.get("nested")
        if isinstance(nested, dict):
            hit = _find_edge_in_tree(nested, edge_id)
            if hit is not None:
                return hit
    return None


def _find_list_aux_target(
    state_dict: Dict[str, Any], edge_id: str
) -> Optional[Tuple[str, str, List[Optional[str]], int]]:
    """Resolve a list-aux edge's containing ``(block_id, param)`` + slot order.

    Helper for :func:`inspector_list_move_emit`'s permutation
    computation: given just an edge_id, walk the state tree, find the
    edge's target block + port, enumerate every list-aux edge that
    targets the same ``(block, port)``, and return them in slot order
    along with the moved edge's current position.

    The returned ``ordered_edge_ids`` list interleaves ``None`` entries
    for any empty slots in ``[0, list_slot_counts[param])`` so the
    caller can swap with the right neighbour even when one of the rows
    being swapped is empty.

    Args:
        state_dict: ``state_to_json`` JSON dump.
        edge_id: The :class:`Edge.edge_id` of the row being moved.

    Returns:
        Tuple ``(block_id, param, ordered_edge_ids, current_idx)`` or
        ``None`` when *edge_id* doesn't resolve to a list-aux edge.
    """

    root_scope = state_dict.get("root")
    if not isinstance(root_scope, dict):
        return None
    hit = _find_edge_in_tree(root_scope, edge_id)
    if hit is None:
        return None
    scope_dict, edge = hit
    if edge.get("kind") != "aux" or edge.get("target_slot") is None:
        return None
    block_id = edge.get("target_block_id")
    param = edge.get("target_port")
    if not isinstance(block_id, str) or not isinstance(param, str):
        return None

    block_hit = _find_block_in_tree(root_scope, block_id)
    if block_hit is None:
        return None
    _, block = block_hit

    # Enumerate every aux edge targeting (block_id, param) and bucket
    # by slot.  Empty slots (within ``list_slot_counts[param]``) get
    # ``None`` entries.
    slot_counts = block.get("list_slot_counts") or {}
    declared = int(slot_counts.get(param, 0))
    wired = [
        e for e in scope_dict.get("edges", []) or []
        if e.get("target_block_id") == block_id
        and e.get("target_port") == param
    ]
    if not wired:
        return None
    max_slot = max(
        int(e.get("target_slot", 0)) for e in wired
    )
    slot_count = max(declared, max_slot + 1)
    ordered: List[Optional[str]] = [None] * slot_count
    for e in wired:
        idx = int(e.get("target_slot", 0))
        if 0 <= idx < slot_count:
            ordered[idx] = e.get("edge_id")
    try:
        current_idx = ordered.index(edge_id)
    except ValueError:
        return None
    return block_id, param, ordered, current_idx


def _resolve_target_port(
    param_info: Any, target_port: str
) -> Tuple[str, bool]:
    """Return ``(canonical_param, is_list)`` for a wire's target port.

    The clientside ``wire_drawing.js`` writes the *parameter name* into
    ``target_port`` for aux wires; list-slot encoding (``"<param>[<i>]"``)
    is not used on the dispatch payload — the server resolves the slot
    from the consumer's ``list_slot_counts`` count.

    For image-flow wires, ``target_port == "in"``.

    Args:
        param_info: :class:`ParamInfo` for the aux port (``None`` for
            ``target_port == "in"``).
        target_port: Raw target-port string from the payload.

    Returns:
        ``(canonical_param, is_list)``.  ``canonical_param`` is the
        unbracketed param name; ``is_list`` is ``True`` for list-typed
        aux ports.
    """

    if target_port == "in" or param_info is None:
        return target_port, False
    # Strip a legacy ``[<i>]`` suffix defensively; new clients don't
    # send one.
    canonical = target_port.split("[", 1)[0]
    return canonical, bool(getattr(param_info, "is_list", False))


def _seed_input_image_dict(scope_dict: Dict[str, Any]) -> None:
    """Idempotently add an ``InputImage`` block dict to *scope_dict*.

    Dict-level mirror of
    :func:`phenotypic.gui.builder._state._seed_input_image` — used by
    the DAG dispatcher's defense-in-depth seeding pass on a container
    scope before a new block is appended.  No-op when the scope
    already contains at least one block with
    ``class_name == INPUT_IMAGE_CLASS_NAME``.

    Args:
        scope_dict: The scope dict to seed in place.  Mutated in place;
            the returned value is the same dict.
    """

    blocks = scope_dict.setdefault("blocks", [])
    if any(b.get("class_name") == INPUT_IMAGE_CLASS_NAME for b in blocks):
        return
    blocks.insert(
        0,
        {
            "block_id": _new_block_id(),
            "class_name": INPUT_IMAGE_CLASS_NAME,
            "params": {},
            "label": None,
            "nested": None,
            "collapsed": False,
            "list_slot_counts": {},
        },
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


def _queue_toast(
    state_dict: Dict[str, Any], text: str, *, kind: str = "info"
) -> None:
    """Append a ``{kind, text}`` entry to ``state_dict['toast_queue']``.

    The DAG dispatchers surface rejection / informational messages by
    appending to ``state.toast_queue``; the JS GUI drains the queue
    and shows each entry via the toast notification surface.  Every kind
    that surfaces a rejection toast uses the same setdefault → append
    boilerplate (``block_create`` InputImage/stale-container reject,
    ``edge_create`` cross-scope reject, ``list_aux_reorder``
    non-permutation reject); this helper centralises it so future
    dispatch kinds only need a one-liner.

    Args:
        state_dict: The in-progress dispatcher state dict (the function
            mutates ``state_dict["toast_queue"]`` in place).
        text: User-facing message.
        kind: ``"info"`` / ``"warning"`` / ``"error"`` — drives the toast
            icon + accent colour.
    """

    queue = state_dict.setdefault("toast_queue", [])
    queue.append({"kind": kind, "text": text})


def _write_pipeline_config(pipeline: Any, target: Path) -> Path:
    """Write *pipeline* using the canonical typed pipeline suffix."""
    typed_target = ensure_typed_json_suffix(target, CONFIG_SUFFIX_PIPELINE)
    typed_target.parent.mkdir(parents=True, exist_ok=True)
    pipeline.to_json(typed_target)
    return typed_target


def _dag_scope_at_breadcrumb(
    state_dict: Dict[str, Any], breadcrumb: List[str]
) -> Optional[Dict[str, Any]]:
    """Resolve the DAG scope dict pointed at by *breadcrumb*.

    Walks the DAG ``state["root"]`` tree, descending into each
    container's ``nested`` scope by ``block_id``.  Returns ``None``
    when any segment fails to resolve to a real Pipeline container
    (stale state, wrong class, etc.) so callers can short-circuit.

    Args:
        state_dict: JSON-shaped DAG state dict (``state_to_json`` output).
        breadcrumb: Ordered list of container block_ids; ``[]`` returns
            the root scope.

    Returns:
        The matching scope dict, or ``None`` when *breadcrumb* is
        invalid against the current tree.
    """

    scope = state_dict.get("root")
    if not isinstance(scope, dict):
        return None
    for segment in breadcrumb:
        if not isinstance(segment, str):
            return None
        block = next(
            (
                b for b in scope.get("blocks", []) or []
                if b.get("block_id") == segment
            ),
            None,
        )
        if block is None:
            return None
        if block.get("class_name") != PIPELINE_CLASS_NAME:
            return None
        nested = block.get("nested")
        if not isinstance(nested, dict):
            return None
        scope = nested
    return scope


def _validate_breadcrumb_path(
    root_scope: Dict[str, Any], target_breadcrumb: List[str]
) -> bool:
    """Return ``True`` if every segment resolves to a real container.

    Each entry of *target_breadcrumb* must name a Pipeline container in
    the scope produced by the previous segments.  Used by
    ``drill_to_scope`` to reject stale ids without mutating state.

    Args:
        root_scope: ``state_dict["root"]``.
        target_breadcrumb: List of container block_ids.

    Returns:
        ``True`` when the breadcrumb resolves cleanly, ``False``
        otherwise.
    """

    scope = root_scope
    for segment in target_breadcrumb:
        block = next(
            (
                b for b in scope.get("blocks", []) or []
                if b.get("block_id") == segment
            ),
            None,
        )
        if block is None:
            return False
        if block.get("class_name") != PIPELINE_CLASS_NAME:
            return False
        nested = block.get("nested")
        if not isinstance(nested, dict):
            return False
        scope = nested
    return True


def _container_is_empty(block_dict: Dict[str, Any]) -> bool:
    """Return ``True`` if *block_dict* is an empty Pipeline container.

    Per spec §5.6 the empty-container threshold for the two-stage delete
    is: the container's ``nested.blocks`` contains *only* the
    auto-seeded ``InputImage`` sentinel (i.e. the count of non-
    ``InputImage`` blocks is zero).

    Args:
        block_dict: A block dict from a scope's ``blocks`` list.

    Returns:
        ``True`` when the container has no real inner ops (only the
        auto-seeded ``InputImage``), ``False`` otherwise.  Non-container
        blocks return ``False`` since "empty" is meaningless for them.
    """

    if block_dict.get("class_name") != PIPELINE_CLASS_NAME:
        return False
    nested = block_dict.get("nested")
    if not isinstance(nested, dict):
        return True
    non_input_count = sum(
        1 for b in nested.get("blocks", []) or []
        if b.get("class_name") != INPUT_IMAGE_CLASS_NAME
    )
    return non_input_count == 0


def _linear_current_scope(
    state_dict: Dict[str, Any],
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Return the current DAG breadcrumb and scope dict for linear edits."""

    breadcrumb = [
        segment
        for segment in list(state_dict.get("breadcrumb", []) or [])
        if isinstance(segment, str)
    ]
    return breadcrumb, _dag_scope_at_breadcrumb(state_dict, breadcrumb)


_LINEAR_PENDING_NODE_PREFIX = "linear_node:"
_LINEAR_PENDING_CLEAR_PREFIX = "linear_clear:"


def _linear_pending_node_token(block_id: str) -> str:
    """Encode a pending linear node delete for the shared confirm modal."""

    return f"{_LINEAR_PENDING_NODE_PREFIX}{block_id}"


def _linear_pending_clear_token(target: LinearTarget) -> str:
    """Encode a pending side-loader clear action for the confirm modal."""

    return (
        _LINEAR_PENDING_CLEAR_PREFIX
        + json.dumps(target_to_dict(target), sort_keys=True, separators=(",", ":"))
    )


def _parse_linear_pending_action(
    pending: str,
) -> Tuple[Optional[str], Optional[Any]]:
    """Decode a tagged linear confirm-modal pending value."""

    if pending.startswith(_LINEAR_PENDING_NODE_PREFIX):
        return "node", pending.removeprefix(_LINEAR_PENDING_NODE_PREFIX)
    if pending.startswith(_LINEAR_PENDING_CLEAR_PREFIX):
        raw_target = pending.removeprefix(_LINEAR_PENDING_CLEAR_PREFIX)
        try:
            data = json.loads(raw_target)
        except json.JSONDecodeError:
            return None, None
        return "clear", data
    return None, None


def _linear_first_unsupported_issue(
    state_dict: Dict[str, Any],
) -> Optional[Issue]:
    """Return the first unsupported linear scope anywhere in the DAG."""

    try:
        state = state_from_json(state_dict)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(state, _DagBuilderState):
        return None
    issues = _linear_unsupported_issues_for_state(state)
    return issues[0] if issues else None


def _linear_reject_unsupported_edit(state_dict: Dict[str, Any]) -> bool:
    """Queue a warning and return ``True`` when linear edits are paused."""

    unsupported = _linear_first_unsupported_issue(state_dict)
    if unsupported is None:
        return False
    _queue_toast(
        state_dict,
        f"Linear editing is paused for this DAG shape: {unsupported.detail}",
        kind="warning",
    )
    return True


def _linear_unsupported_issues_for_state(
    state: _DagBuilderState,
) -> List[Issue]:
    """Build blocking issues for scopes the fixed linear map cannot edit."""

    issues: List[Issue] = []

    def visit(scope: _DagBuilderScope, scope_path: List[str]) -> None:
        model = derive_linear_scope(scope, scope_path=scope_path)
        if model.unsupported is not None:
            issues.append(
                Issue(
                    kind="unsupported_linear",
                    block_id=model.unsupported.block_id,
                    detail=model.unsupported.detail,
                    scope_path=list(scope_path),
                    severity="error",
                )
            )
        for block in scope.blocks:
            if block.nested is not None:
                visit(block.nested, [*scope_path, block.block_id])

    visit(state.root, [])
    return issues


def _state_with_issue_focus(
    state_data: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Drill to an issue scope and select the offending fixed-map block."""

    raw_target = payload.get("target_breadcrumb")
    target_breadcrumb = (
        [entry for entry in raw_target if isinstance(entry, str) and entry]
        if isinstance(raw_target, list)
        else []
    )
    new_state_dict = state_data
    if target_breadcrumb != (state_data.get("breadcrumb") or []):
        new_state_dict = _dispatch_state_update(
            state_data,
            "drill_to_scope",
            {
                "kind": "drill_to_scope",
                "target_breadcrumb": target_breadcrumb,
                "ts": payload.get("ts"),
            },
        )

    block_id = payload.get("block_id")
    if isinstance(block_id, str) and block_id:
        focused_state = deepcopy(new_state_dict)
        focused_scope = _dag_scope_at_breadcrumb(
            focused_state,
            list(focused_state.get("breadcrumb", []) or []),
        )
        if focused_scope is not None and any(
            block.get("block_id") == block_id
            for block in focused_scope.get("blocks", []) or []
        ):
            focused_state["selected_block_id"] = block_id
            focused_state["selected_edge_id"] = None
            new_state_dict = focused_state

    return new_state_dict


def _decode_linear_scope_path(encoded: Any) -> List[str]:
    """Decode a Dash-safe linear id scope path into breadcrumb segments."""

    if not isinstance(encoded, str) or encoded == "__root__":
        return []
    return [part for part in encoded.split("/") if part]


def _decode_linear_optional(value: Any) -> Optional[str]:
    """Decode a Dash-safe optional id field."""

    return value if isinstance(value, str) and value != "__none__" else None


def _decode_linear_slot(value: Any) -> Optional[int]:
    """Decode a Dash-safe optional slot field."""

    return value if isinstance(value, int) and value >= 0 else None


def _linear_target_payload_from_id(triggered: Dict[str, Any]) -> Dict[str, Any]:
    """Build a store target payload from a linear pattern id."""

    return {
        "kind": triggered.get("kind"),
        "scope_path": _decode_linear_scope_path(triggered.get("scope_path")),
        "block_id": _decode_linear_optional(triggered.get("block_id")),
        "param": _decode_linear_optional(triggered.get("param")),
        "slot": _decode_linear_slot(triggered.get("slot")),
    }


def _linear_param_target_payload_from_id(
    triggered: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a parameter target payload from a side-loader action id."""

    target = _linear_target_payload_from_id(triggered)
    target["kind"] = (
        "parameter_slot" if target.get("slot") is not None else "parameter"
    )
    return target


def _linear_preview_target_payload_from_id(
    triggered: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an image-prefix preview target from a node-action id."""

    block_id = _decode_linear_optional(triggered.get("block_id"))
    return {
        "kind": "image_output" if block_id is not None else "continuation",
        "scope_path": _decode_linear_scope_path(triggered.get("scope_path")),
        "block_id": block_id,
        "param": None,
        "slot": None,
    }


def _linear_source_block_id_from_action_id(
    triggered: Dict[str, Any],
) -> Optional[str]:
    """Decode the source block carried by a linear value action id."""

    return _decode_linear_optional(triggered.get("source_block_id"))


def _linear_selected_target(state_dict: Dict[str, Any]) -> LinearTarget:
    """Resolve the active linear target from JSON state with fallback."""

    return resolve_selected_target(state_from_json(state_dict))


def _linear_set_target(
    state_dict: Dict[str, Any], target: LinearTarget, *, open_menu: bool = False
) -> None:
    """Persist a selected target and optionally open its local menu."""

    targets = state_dict.setdefault("selected_targets_by_scope", {})
    targets[scope_key(target.scope_path)] = target_to_dict(target)
    state_dict["open_port_menu"] = target_to_dict(target) if open_menu else None


def _linear_reset_target_to_continuation(
    state_dict: Dict[str, Any], scope_path: List[str]
) -> None:
    """Select the floating continuation port for ``scope_path``."""

    _linear_set_target(
        state_dict,
        default_continuation_target(scope_path),
        open_menu=False,
    )


def _linear_block(
    scope_dict: Dict[str, Any], block_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Find a direct block child inside ``scope_dict``."""

    if not isinstance(block_id, str):
        return None
    return next(
        (
            block for block in scope_dict.get("blocks", []) or []
            if block.get("block_id") == block_id
        ),
        None,
    )


def _linear_input_block(scope_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the scope's InputImage block dict."""

    return next(
        (
            block for block in scope_dict.get("blocks", []) or []
            if block.get("class_name") == INPUT_IMAGE_CLASS_NAME
        ),
        None,
    )


def _linear_image_edge_from(
    scope_dict: Dict[str, Any], source_block_id: str
) -> Optional[Dict[str, Any]]:
    """Return the image edge leaving ``source_block_id`` if present."""

    return next(
        (
            edge for edge in scope_dict.get("edges", []) or []
            if edge.get("kind") == "image"
            and edge.get("source_block_id") == source_block_id
        ),
        None,
    )


def _linear_image_edge_to(
    scope_dict: Dict[str, Any], target_block_id: str
) -> Optional[Dict[str, Any]]:
    """Return the image edge entering ``target_block_id`` if present."""

    return next(
        (
            edge for edge in scope_dict.get("edges", []) or []
            if edge.get("kind") == "image"
            and edge.get("target_block_id") == target_block_id
        ),
        None,
    )


def _linear_spine_ids(scope_dict: Dict[str, Any]) -> List[str]:
    """Return image-spine block ids in execution order."""

    input_block = _linear_input_block(scope_dict)
    if input_block is None:
        return []
    current_id = input_block.get("block_id")
    if not isinstance(current_id, str):
        return []
    spine: List[str] = []
    seen: set[str] = set()
    while isinstance(current_id, str) and current_id not in seen:
        seen.add(current_id)
        spine.append(current_id)
        next_edge = _linear_image_edge_from(scope_dict, current_id)
        current_id = (
            next_edge.get("target_block_id") if next_edge is not None else None
        )
    return spine


def _linear_terminal_block(
    scope_dict: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the terminal block in the current linear spine."""

    spine = _linear_spine_ids(scope_dict)
    return _linear_block(scope_dict, spine[-1]) if spine else None


def _linear_new_edge(
    source_block_id: str,
    target_block_id: str,
    *,
    kind: Literal["image", "aux"],
    target_port: str,
    target_slot: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a JSON edge dict for linear mutations."""

    return {
        "edge_id": _new_block_id(),
        "source_block_id": source_block_id,
        "source_port": "out",
        "target_block_id": target_block_id,
        "target_port": target_port,
        "target_slot": target_slot,
        "kind": kind,
    }


def _linear_new_block(class_name: str) -> Dict[str, Any]:
    """Create a JSON block dict with defaults and optional nested scope."""

    nested_scope: Optional[Dict[str, Any]] = None
    if class_name == PIPELINE_CLASS_NAME:
        nested_scope = {
            "blocks": [],
            "edges": [],
            "name": "Pipeline",
            "desc": "",
            "nrows": None,
            "ncols": None,
        }
        _seed_input_image_dict(nested_scope)
    return {
        "block_id": _new_block_id(),
        "class_name": class_name,
        "params": _default_params_for(class_name),
        "label": None,
        "nested": nested_scope,
        "collapsed": False,
        "list_slot_counts": {},
    }


def _linear_insert_block_in_list(
    scope_dict: Dict[str, Any],
    new_block: Dict[str, Any],
    *,
    before_block_id: Optional[str] = None,
    after_block_id: Optional[str] = None,
) -> None:
    """Insert ``new_block`` near the visible target for stable rendering."""

    blocks = scope_dict.setdefault("blocks", [])
    insert_at = len(blocks)
    if before_block_id is not None:
        for idx, block in enumerate(blocks):
            if block.get("block_id") == before_block_id:
                insert_at = idx
                break
    elif after_block_id is not None:
        for idx, block in enumerate(blocks):
            if block.get("block_id") == after_block_id:
                insert_at = idx + 1
                break
    blocks.insert(insert_at, new_block)


def _linear_insert_spine_block(
    state_dict: Dict[str, Any],
    scope_dict: Dict[str, Any],
    target: LinearTarget,
    class_name: str,
) -> Optional[str]:
    """Insert a new main-spine block at ``target`` and reconnect image edges."""

    new_block = _linear_new_block(class_name)
    new_id = new_block["block_id"]
    edges = scope_dict.setdefault("edges", [])

    if target.kind == "continuation":
        terminal = _linear_terminal_block(scope_dict)
        if terminal is None:
            return None
        terminal_id = terminal.get("block_id")
        if not isinstance(terminal_id, str):
            return None
        _linear_insert_block_in_list(
            scope_dict, new_block, after_block_id=terminal_id
        )
        edges.append(
            _linear_new_edge(
                terminal_id, new_id, kind="image", target_port="in"
            )
        )
    elif target.kind == "image_output":
        source = _linear_block(scope_dict, target.block_id)
        if source is None:
            return None
        source_id = source.get("block_id")
        if not isinstance(source_id, str):
            return None
        old_next = _linear_image_edge_from(scope_dict, source_id)
        next_target = (
            old_next.get("target_block_id") if old_next is not None else None
        )
        if old_next is not None:
            edges[:] = [
                edge for edge in edges
                if edge.get("edge_id") != old_next.get("edge_id")
            ]
        _linear_insert_block_in_list(
            scope_dict, new_block, after_block_id=source_id
        )
        edges.append(
            _linear_new_edge(source_id, new_id, kind="image", target_port="in")
        )
        if isinstance(next_target, str):
            edges.append(
                _linear_new_edge(
                    new_id, next_target, kind="image", target_port="in"
                )
            )
    elif target.kind == "image_input":
        target_block = _linear_block(scope_dict, target.block_id)
        if target_block is None:
            return None
        target_id = target_block.get("block_id")
        if not isinstance(target_id, str):
            return None
        old_prev = _linear_image_edge_to(scope_dict, target_id)
        if old_prev is None:
            _queue_toast(
                state_dict,
                "Cannot insert before the Input Image source.",
                kind="info",
            )
            return None
        prev_source = old_prev.get("source_block_id")
        if not isinstance(prev_source, str):
            return None
        edges[:] = [
            edge for edge in edges
            if edge.get("edge_id") != old_prev.get("edge_id")
        ]
        _linear_insert_block_in_list(
            scope_dict, new_block, before_block_id=target_id
        )
        edges.append(
            _linear_new_edge(prev_source, new_id, kind="image", target_port="in")
        )
        edges.append(
            _linear_new_edge(new_id, target_id, kind="image", target_port="in")
        )
    else:
        return None

    out_path = list(target.scope_path)
    state_dict["selected_block_id"] = new_id
    state_dict["selected_edge_id"] = None
    _linear_reset_target_to_continuation(state_dict, out_path)
    return new_id


def _linear_edges_for_param(
    scope_dict: Dict[str, Any],
    block_id: str,
    param: str,
    slot: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return aux edges targeting ``block_id.param`` and optional slot."""

    matches = [
        edge for edge in scope_dict.get("edges", []) or []
        if edge.get("kind") == "aux"
        and edge.get("target_block_id") == block_id
        and edge.get("target_port") == param
    ]
    if slot is None:
        return matches
    return [edge for edge in matches if edge.get("target_slot") == slot]


def _linear_aux_subtree_ids(
    scope_dict: Dict[str, Any], source_block_id: str
) -> set[str]:
    """Collect an aux source and its upstream aux dependencies."""

    if _linear_block_is_image_incident(scope_dict, source_block_id):
        return set()

    to_delete = {source_block_id}
    if _linear_has_external_aux_output(scope_dict, source_block_id, to_delete):
        return set()

    changed = True
    while changed:
        changed = False
        for edge in scope_dict.get("edges", []) or []:
            if edge.get("kind") != "aux":
                continue
            if edge.get("target_block_id") not in to_delete:
                continue
            source_id = edge.get("source_block_id")
            if not isinstance(source_id, str) or source_id in to_delete:
                continue
            candidate_delete_ids = to_delete | {source_id}
            if _linear_block_is_image_incident(scope_dict, source_id):
                continue
            if _linear_has_external_aux_output(
                scope_dict, source_id, candidate_delete_ids
            ):
                continue
            to_delete.add(source_id)
            changed = True
    return to_delete


def _linear_block_is_image_incident(
    scope_dict: Dict[str, Any], block_id: str
) -> bool:
    """Return whether ``block_id`` participates in image flow."""

    return any(
        edge.get("kind") == "image"
        and (
            edge.get("source_block_id") == block_id
            or edge.get("target_block_id") == block_id
        )
        for edge in scope_dict.get("edges", []) or []
    )


def _linear_has_external_aux_output(
    scope_dict: Dict[str, Any], block_id: str, owned_ids: set[str]
) -> bool:
    """Return whether ``block_id`` feeds an aux target outside ``owned_ids``."""

    return any(
        edge.get("kind") == "aux"
        and edge.get("source_block_id") == block_id
        and edge.get("target_block_id") not in owned_ids
        for edge in scope_dict.get("edges", []) or []
    )


def _linear_delete_aux_sources(
    scope_dict: Dict[str, Any], source_block_ids: List[str]
) -> set[str]:
    """Delete aux source blocks plus their upstream aux dependencies."""

    delete_ids: set[str] = set()
    for source_id in source_block_ids:
        delete_ids.update(_linear_aux_subtree_ids(scope_dict, source_id))
    if not delete_ids:
        return set()

    scope_dict["blocks"] = [
        block for block in scope_dict.get("blocks", []) or []
        if block.get("block_id") not in delete_ids
    ]
    scope_dict["edges"] = [
        edge for edge in scope_dict.get("edges", []) or []
        if edge.get("source_block_id") not in delete_ids
        and edge.get("target_block_id") not in delete_ids
    ]
    return delete_ids


def _linear_compact_list_slots_dict(
    scope_dict: Dict[str, Any], block: Dict[str, Any], param: str
) -> None:
    """Renumber list aux edge slots without preserving empty gaps."""

    indexed_edges = [
        (idx, edge)
        for idx, edge in enumerate(scope_dict.get("edges", []) or [])
        if edge.get("kind") == "aux"
        and edge.get("target_block_id") == block.get("block_id")
        and edge.get("target_port") == param
    ]
    indexed_edges.sort(
        key=lambda pair: (
            pair[1].get("target_slot")
            if isinstance(pair[1].get("target_slot"), int)
            else 0,
            pair[0],
        )
    )
    for slot, (_idx, edge) in enumerate(indexed_edges):
        edge["target_slot"] = slot
    slot_counts = block.setdefault("list_slot_counts", {})
    if indexed_edges:
        slot_counts[param] = len(indexed_edges)
    else:
        slot_counts.pop(param, None)


def _linear_remove_param_edges(
    scope_dict: Dict[str, Any],
    block: Dict[str, Any],
    param: str,
    *,
    slot: Optional[int] = None,
    compact: bool = True,
) -> set[str]:
    """Remove matching aux edges and owned aux subtrees."""

    block_id = block.get("block_id")
    if not isinstance(block_id, str):
        return set()
    edges_to_remove = _linear_edges_for_param(scope_dict, block_id, param, slot)
    source_ids = [
        edge.get("source_block_id")
        for edge in edges_to_remove
        if isinstance(edge.get("source_block_id"), str)
    ]
    edge_ids = {edge.get("edge_id") for edge in edges_to_remove}
    scope_dict["edges"] = [
        edge for edge in scope_dict.get("edges", []) or []
        if edge.get("edge_id") not in edge_ids
    ]
    removed = _linear_delete_aux_sources(scope_dict, source_ids)
    if compact:
        _linear_compact_list_slots_dict(scope_dict, block, param)
    return removed


def _linear_param_info(
    block: Dict[str, Any], param: str
) -> Optional[Any]:
    """Return registry metadata for a consumer parameter."""

    from phenotypic.gui._operation_registry import get_registry

    info = get_registry().get(block.get("class_name", ""))
    if info is None:
        return None
    return info.parameters.get(param)


def _linear_class_can_fill_param(class_name: str, param_info: Any) -> bool:
    """Return whether ``class_name`` can be used as a parameter value."""

    from phenotypic.gui._operation_registry import get_registry

    return class_name in _resolve_dag_accepts(param_info, get_registry())


def _linear_fill_parameter(
    state_dict: Dict[str, Any],
    scope_dict: Dict[str, Any],
    target: LinearTarget,
    class_name: str,
) -> Optional[str]:
    """Create or replace a side-loaded parameter value."""

    if target.block_id is None or target.param is None:
        return None
    if target.kind == "parameter_slot" and target.slot is None:
        return None
    consumer = _linear_block(scope_dict, target.block_id)
    if consumer is None:
        return None
    param_info = _linear_param_info(consumer, target.param)
    if param_info is None:
        return None
    if not _linear_class_can_fill_param(class_name, param_info):
        _queue_toast(
            state_dict,
            "Selected target does not accept that operation.",
            kind="warning",
        )
        return None

    is_list = bool(getattr(param_info, "is_list", False))
    source_block = _linear_new_block(class_name)
    source_id = source_block["block_id"]
    if is_list:
        slot = target.slot if target.kind == "parameter_slot" else None
        if slot is None:
            _linear_compact_list_slots_dict(scope_dict, consumer, target.param)
            slot = int(
                consumer.setdefault("list_slot_counts", {}).get(target.param, 0)
            )
        else:
            _linear_remove_param_edges(
                scope_dict,
                consumer,
                target.param,
                slot=slot,
                compact=False,
            )
        scope_dict.setdefault("blocks", []).append(source_block)
        scope_dict.setdefault("edges", []).append(
            _linear_new_edge(
                source_id,
                target.block_id,
                kind="aux",
                target_port=target.param,
                target_slot=slot,
            )
        )
        _linear_compact_list_slots_dict(scope_dict, consumer, target.param)
    else:
        _linear_remove_param_edges(
            scope_dict, consumer, target.param, compact=False
        )
        scope_dict.setdefault("blocks", []).append(source_block)
        scope_dict.setdefault("edges", []).append(
            _linear_new_edge(
                source_id,
                target.block_id,
                kind="aux",
                target_port=target.param,
            )
        )

    if class_name == PIPELINE_CLASS_NAME:
        nested_path = list(target.scope_path) + [source_id]
        state_dict["breadcrumb"] = nested_path
        state_dict["selected_block_id"] = None
        state_dict["selected_edge_id"] = None
        _linear_reset_target_to_continuation(state_dict, nested_path)
    else:
        state_dict["selected_block_id"] = target.block_id
        state_dict["selected_edge_id"] = None
        _linear_reset_target_to_continuation(state_dict, list(target.scope_path))
    return source_id


def _linear_clear_param_needs_confirmation(
    scope_dict: Dict[str, Any],
    target: LinearTarget,
) -> bool:
    """Return whether clearing this side value removes an embedded pipeline."""

    if target.block_id is None or target.param is None:
        return False
    if target.kind == "parameter_slot" and target.slot is None:
        return False
    slot = target.slot if target.kind == "parameter_slot" else None
    edges = _linear_edges_for_param(scope_dict, target.block_id, target.param, slot)
    for edge in edges:
        source_id = edge.get("source_block_id")
        if not isinstance(source_id, str):
            continue
        source = _linear_block(scope_dict, source_id)
        if source is not None and source.get("class_name") == PIPELINE_CLASS_NAME:
            return True
    return False


def _linear_clear_param(
    state_dict: Dict[str, Any],
    scope_dict: Dict[str, Any],
    target: LinearTarget,
) -> None:
    """Clear a scalar parameter value or one list slot."""

    if target.block_id is None or target.param is None:
        return
    if target.kind == "parameter_slot" and target.slot is None:
        return
    consumer = _linear_block(scope_dict, target.block_id)
    if consumer is None:
        return
    slot = target.slot if target.kind == "parameter_slot" else None
    _linear_remove_param_edges(scope_dict, consumer, target.param, slot=slot)
    _linear_reset_target_to_continuation(state_dict, list(target.scope_path))


def _linear_drill_param_pipeline(
    state_dict: Dict[str, Any],
    scope_dict: Dict[str, Any],
    source_block_id: Optional[str],
) -> None:
    """Drill into an aux ImagePipeline source block."""

    if not isinstance(source_block_id, str):
        return
    block = _linear_block(scope_dict, source_block_id)
    if block is None or block.get("class_name") != PIPELINE_CLASS_NAME:
        return
    if not isinstance(block.get("nested"), dict):
        return
    scope_path = list(state_dict.get("breadcrumb", []) or []) + [source_block_id]
    state_dict["breadcrumb"] = scope_path
    state_dict["selected_block_id"] = None
    state_dict["selected_edge_id"] = None
    _linear_reset_target_to_continuation(state_dict, scope_path)


def _linear_rewire_spine(
    scope_dict: Dict[str, Any], spine_ids: List[str]
) -> None:
    """Replace image-flow edges with consecutive edges for ``spine_ids``."""

    old_spine_ids = set(_linear_spine_ids(scope_dict))
    scope_dict["edges"] = [
        edge for edge in scope_dict.get("edges", []) or []
        if not (
            edge.get("kind") == "image"
            and edge.get("source_block_id") in old_spine_ids
            and edge.get("target_block_id") in old_spine_ids
        )
    ]
    for source_id, target_id in zip(spine_ids, spine_ids[1:]):
        scope_dict.setdefault("edges", []).append(
            _linear_new_edge(source_id, target_id, kind="image", target_port="in")
        )


def _linear_reorder_blocks_by_spine(
    scope_dict: Dict[str, Any], spine_ids: List[str]
) -> None:
    """Move spine blocks to match execution order while preserving aux order."""

    by_id = {
        block.get("block_id"): block
        for block in scope_dict.get("blocks", []) or []
        if isinstance(block.get("block_id"), str)
    }
    spine_set = set(spine_ids)
    ordered = [by_id[block_id] for block_id in spine_ids if block_id in by_id]
    ordered.extend(
        block for block in scope_dict.get("blocks", []) or []
        if block.get("block_id") not in spine_set
    )
    scope_dict["blocks"] = ordered


def _linear_move_node(
    state_dict: Dict[str, Any],
    scope_dict: Dict[str, Any],
    block_id: str,
    direction: str,
) -> None:
    """Swap a main-spine node left or right."""

    spine_ids = _linear_spine_ids(scope_dict)
    if block_id not in spine_ids:
        return
    idx = spine_ids.index(block_id)
    if idx == 0:
        return
    swap_idx = idx - 1 if direction == "left" else idx + 1
    if swap_idx <= 0 or swap_idx >= len(spine_ids):
        return
    spine_ids[idx], spine_ids[swap_idx] = spine_ids[swap_idx], spine_ids[idx]
    _linear_rewire_spine(scope_dict, spine_ids)
    _linear_reorder_blocks_by_spine(scope_dict, spine_ids)
    state_dict["selected_block_id"] = block_id
    state_dict["selected_edge_id"] = None


def _linear_delete_spine_node(
    state_dict: Dict[str, Any],
    scope_dict: Dict[str, Any],
    block_id: str,
) -> None:
    """Delete a main-spine node, reconnecting its neighbors."""

    block = _linear_block(scope_dict, block_id)
    if block is None:
        return
    if block.get("class_name") == INPUT_IMAGE_CLASS_NAME:
        _queue_toast(state_dict, "Input Image cannot be removed.", kind="info")
        return

    spine_ids = _linear_spine_ids(scope_dict)
    if block_id not in spine_ids:
        return
    next_spine = [candidate for candidate in spine_ids if candidate != block_id]

    aux_sources = [
        edge.get("source_block_id")
        for edge in scope_dict.get("edges", []) or []
        if edge.get("kind") == "aux"
        and edge.get("target_block_id") == block_id
        and isinstance(edge.get("source_block_id"), str)
    ]
    removed_edge_ids = {
        edge.get("edge_id") for edge in scope_dict.get("edges", []) or []
        if edge.get("source_block_id") == block_id
        or edge.get("target_block_id") == block_id
    }
    scope_dict["edges"] = [
        edge for edge in scope_dict.get("edges", []) or []
        if not (
            edge.get("kind") == "aux"
            and edge.get("target_block_id") == block_id
        )
    ]
    removed_aux_ids = _linear_delete_aux_sources(scope_dict, aux_sources)
    scope_dict["blocks"] = [
        candidate for candidate in scope_dict.get("blocks", []) or []
        if candidate.get("block_id") != block_id
    ]
    scope_dict["edges"] = [
        edge for edge in scope_dict.get("edges", []) or []
        if edge.get("source_block_id") != block_id
        and edge.get("target_block_id") != block_id
    ]
    _linear_rewire_spine(scope_dict, next_spine)
    _linear_reorder_blocks_by_spine(scope_dict, next_spine)

    stale_block_ids = removed_aux_ids | {block_id}
    if out_selected := state_dict.get("selected_block_id"):
        if out_selected in stale_block_ids:
            state_dict["selected_block_id"] = None
    if state_dict.get("selected_edge_id") in removed_edge_ids:
        state_dict["selected_edge_id"] = None
    if state_dict.get("pending_delete_block_id") == block_id:
        state_dict["pending_delete_block_id"] = None
    _linear_reset_target_to_continuation(
        state_dict, list(state_dict.get("breadcrumb", []) or [])
    )


def _linear_prefix_state_for_preview(
    state: _DagBuilderState,
    target: LinearTarget,
) -> _DagBuilderState:
    """Return a temporary DAG state containing only ``target``'s prefix.

    ``Preview here`` intentionally previews the active scope only. The
    returned state has a fresh root scope made from the visible prefix spine
    plus any side-loaded aux dependencies required by those prefix blocks;
    downstream blocks are omitted so global pipeline validation does not gate
    a local inspection action.
    """

    scope, spine_blocks, selected_block_id = _linear_preview_selection(
        state, target
    )

    keep_ids = {block.block_id for block in spine_blocks}
    changed = True
    while changed:
        changed = False
        for edge in scope.edges:
            if edge.kind != "aux" or edge.target_block_id not in keep_ids:
                continue
            if edge.source_block_id not in keep_ids:
                keep_ids.add(edge.source_block_id)
                changed = True

    prefix_blocks = [
        deepcopy(block)
        for block in scope.blocks
        if block.block_id in keep_ids
    ]
    prefix_edges = [
        deepcopy(edge)
        for edge in scope.edges
        if edge.source_block_id in keep_ids and edge.target_block_id in keep_ids
    ]
    prefix_scope = _DagBuilderScope(
        blocks=prefix_blocks,
        edges=prefix_edges,
        name=scope.name,
        desc=scope.desc,
        nrows=scope.nrows,
        ncols=scope.ncols,
    )
    return _DagBuilderState(root=prefix_scope, selected_block_id=selected_block_id)


def _linear_preview_selection(
    state: _DagBuilderState,
    target: LinearTarget,
) -> tuple[_DagBuilderScope, list[BlockNode], str]:
    """Resolve the active scope, prefix spine, and selected block for preview."""

    scope = scope_at_path(state.root, target.scope_path)
    if scope is None:
        raise ValueError("Cannot preview here: the selected scope is stale.")

    model = derive_linear_scope(scope, scope_path=target.scope_path)
    if model.unsupported is not None:
        raise ValueError(
            f"Cannot preview here: unsupported map shape ({model.unsupported.reason})."
        )
    if target.kind not in {"continuation", "image_output"}:
        raise ValueError("Cannot preview here from that port.")

    spine_blocks = list(model.spine_blocks)
    if not spine_blocks:
        raise ValueError("Cannot preview here: this scope has no image source.")

    if target.kind == "continuation":
        return scope, spine_blocks, spine_blocks[-1].block_id

    selected_block_id = target.block_id
    cutoff_idx = next(
        (
            idx
            for idx, block in enumerate(spine_blocks)
            if block.block_id == selected_block_id
        ),
        None,
    )
    if selected_block_id is None or cutoff_idx is None:
        raise ValueError("Cannot preview here: the selected block is stale.")
    return scope, spine_blocks[: cutoff_idx + 1], selected_block_id


def _linear_state_with_preview_selection(
    state_data: Dict[str, Any],
    target_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Close a preview menu and select the block whose output is being previewed."""

    state = state_from_json(state_data)
    if not isinstance(state, _DagBuilderState):
        return state_data
    target = target_from_dict(target_payload, state.breadcrumb)
    _, _, selected_block_id = _linear_preview_selection(state, target)
    state.selected_block_id = selected_block_id
    state.selected_edge_id = None
    state.open_port_menu = None
    return state_to_json(state)


def _scope_contains_block_id(
    scope_dict: Dict[str, Any], block_id: str
) -> bool:
    """Return ``True`` if *block_id* is a direct child of *scope_dict*.

    Used by the reparent dispatcher's ancestry check: we ask whether the
    block's *current scope* is reachable from the *target scope* by
    descending into containers (i.e. the move is a drag-in / sibling)
    vs. ascending (drag-out / promote).
    """

    return any(
        b.get("block_id") == block_id
        for b in scope_dict.get("blocks", []) or []
    )


def _is_ancestor_scope(
    state_dict: Dict[str, Any],
    candidate_scope: Dict[str, Any],
    descendant_scope: Dict[str, Any],
) -> bool:
    """Return ``True`` if *candidate_scope* contains *descendant_scope*.

    Walks the state tree from *candidate_scope* down and checks whether
    any descendant container's nested scope is the same dict object as
    *descendant_scope*.  The ``is`` identity check is intentional —
    every scope dict in the state tree is a unique Python object after
    the dispatcher's ``deepcopy``, so identity-equality is the precise
    notion of "same scope" we want.
    """

    if candidate_scope is descendant_scope:
        return False  # same scope is *not* an ancestor in this sense
    stack: List[Dict[str, Any]] = [candidate_scope]
    while stack:
        current = stack.pop()
        for block in current.get("blocks", []) or []:
            nested = block.get("nested")
            if not isinstance(nested, dict):
                continue
            if nested is descendant_scope:
                return True
            stack.append(nested)
    return False


def _dispatch_block_reparent(
    out: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Implement the ``block_reparent`` dispatch (spec §4.4 / §5.6).

    The full algorithm is described in the inline call-site comment in
    :func:`_dispatch_state_update`.  Pulled out into its own helper so
    the dispatcher's body stays under the McCabe complexity limit.
    """

    block_id = payload.get("block_id")
    if not isinstance(block_id, str) or not block_id:
        return out
    new_parent_block_id = payload.get("new_parent_block_id")
    if new_parent_block_id is not None and not isinstance(
        new_parent_block_id, str
    ):
        return out

    root_scope = out.get("root")
    if not isinstance(root_scope, dict):
        return out

    # Locate the block + its containing scope.
    block_hit = _find_block_in_tree(root_scope, block_id)
    if block_hit is None:
        return out
    source_scope, block = block_hit

    # Reject InputImage block_ids (defense in depth — palette guard
    # already prevents the drag from starting).
    if block.get("class_name") == INPUT_IMAGE_CLASS_NAME:
        _queue_toast(
            out, "Input Image cannot be moved between scopes.", kind="info"
        )
        return out

    # Resolve the target scope.  ``None`` parent → the visible scope
    # (current breadcrumb); non-None → that container's nested scope.
    breadcrumb_list = list(out.get("breadcrumb", []) or [])
    if new_parent_block_id is None:
        target_scope = _dag_scope_at_breadcrumb(out, breadcrumb_list)
        if target_scope is None:
            return out
    else:
        target_scope = _find_dag_container_scope(
            root_scope, new_parent_block_id
        )
        if target_scope is None:
            _queue_toast(
                out,
                "Drop target container is no longer in the pipeline.",
                kind="warning",
            )
            return out

    # No-op when source and target are the same scope: the canvas
    # surfaces drags within a single scope as pure visual nudges (spec
    # §4.7 manual-drag-is-ephemeral) so the dispatcher takes no action.
    if source_scope is target_scope:
        return out

    # Identify orphaned edges: edges incident to *block_id* whose other
    # endpoint is NOT moving (we only move one block at a time in v1 —
    # multi-select reparent is deferred per spec §10).  These edges
    # would cross the new scope boundary, so they must be deleted (or
    # the move snaps back if this is a drag-out direction).
    orphan_edges: List[Dict[str, Any]] = []
    other_labels: List[str] = []
    for edge in list(source_scope.get("edges", []) or []):
        src = edge.get("source_block_id")
        tgt = edge.get("target_block_id")
        if src == block_id or tgt == block_id:
            other_id = tgt if src == block_id else src
            other_block = next(
                (
                    b for b in source_scope.get("blocks", []) or []
                    if b.get("block_id") == other_id
                ),
                None,
            )
            if other_block is not None:
                orphan_edges.append(edge)
                label = other_block.get("label") or other_block.get(
                    "class_name", "block"
                )
                other_labels.append(str(label))

    # Drag-out direction: the target scope is an *ancestor* of the
    # source scope.  If any inner edges would be orphaned, reject the
    # move with snap-back + toast (spec §4.4).
    moving_outward = _is_ancestor_scope(out, target_scope, source_scope)
    if moving_outward and orphan_edges:
        moved_label = block.get("label") or block.get("class_name", "block")
        joined = ", ".join(sorted(set(other_labels)))
        _queue_toast(
            out,
            (
                f"Can't move {moved_label} out — "
                f"{len(orphan_edges)} inner edge(s) would be orphaned "
                f"({joined}). Disconnect first."
            ),
            kind="warning",
        )
        return out

    # Drag-in / sibling / promote direction: delete the orphan edges and
    # commit the move.  Toast the deletion count (named labels included
    # so the user knows which connections vanished).
    if orphan_edges:
        orphan_ids = {e.get("edge_id") for e in orphan_edges}
        source_scope["edges"] = [
            e for e in source_scope.get("edges", []) or []
            if e.get("edge_id") not in orphan_ids
        ]
        # Clear any selection that would dangle after the orphan deletion
        # (mirrors ``edge_delete`` / ``block_delete_confirm`` cleanup so
        # the inspector wire card doesn't try to render a stale edge).
        if out.get("selected_edge_id") in orphan_ids:
            out["selected_edge_id"] = None
        joined = ", ".join(sorted(set(other_labels)))
        _queue_toast(
            out,
            (
                f"Removed {len(orphan_edges)} edge(s) that no longer "
                f"fit ({joined})."
            ),
            kind="info",
        )

    # Remove block from source scope; ensure target scope is seeded
    # (defense in depth: any scope we paste into MUST have an
    # InputImage per Rule 6).
    source_scope["blocks"] = [
        b for b in source_scope.get("blocks", []) or []
        if b.get("block_id") != block_id
    ]
    _seed_input_image_dict(target_scope)
    target_scope.setdefault("blocks", []).append(block)
    return out


#: Closed set of dispatch-kind strings routed by :func:`_dispatch_state_update`.
#:
#: Per the project's typing convention (CLAUDE.md "Code Style") a Literal
#: alias is sufficient for type-only enforcement of a closed set with no
#: user-visible documentation surface — dispatch keys never reach the GUI
#: chrome.  Adding the alias as a non-enforcing annotation on the dispatcher
#: doesn't change runtime behaviour (``Literal`` is erased at runtime), but
#: it lets type-checkers flag a typo at the call sites that build the
#: ``store-edge-event`` / ``store-palette-drop`` / ``store-builder-state``
#: payloads.  The DAG redesign added 6 new kinds (``block_create``,
#: ``edge_create``, ``edge_delete``, ``list_aux_reorder``,
#: ``list_aux_add_empty_slot``, ``wire_select``, ``block_select``), so
#: the alias is now warranted.
DispatchKind: TypeAlias = Literal[
    # Legacy linear-builder kinds.
    "add_node",
    "add_pipeline",
    "select_node",
    "delete_node",
    "drill_in",
    "drill_out",
    "breadcrumb_to",
    "reorder",
    "edit_param",
    "edit_label",
    "port_slot_add",
    "port_slot_remove",
    # Legacy aux-port mutation kinds.
    "wire_create",
    "wire_delete",
    "drill_in_aux",
    "set_inspector_focus",
    # Palette drag-and-drop.
    "block_create",
    # Wire drawing + list-aux fan-in.
    "edge_create",
    "edge_delete",
    "list_aux_reorder",
    "list_aux_add_empty_slot",
    "wire_select",
    "block_select",
    # Pipeline-container dispatchers (spec §4.4 / §5.6).
    "block_reparent",
    "block_collapsed_toggle",
    "drill_into_container",
    "drill_to_scope",
    "block_delete_request",
    "block_delete_confirm",
    # Fixed linear port-map dispatchers.
    "target_select",
    "target_menu_close",
    "linear_palette_add",
    "linear_delete_node_request",
    "linear_delete_node_confirm",
    "linear_node_move",
    "linear_clear_param",
    "linear_clear_param_confirm",
    "linear_drill_param_pipeline",
    "linear_select_aux_value",
]


def _is_dag_state_dict(state_dict: Dict[str, Any]) -> bool:
    """Return ``True`` when *state_dict* carries the DAG schema.

    Mirrors the duck-typed dispatch in :func:`state_from_json`:
    explicit ``_schema`` discriminator first, then heuristic
    ``root.blocks`` presence for older payloads that predated the
    discriminator. Used by :func:`_dispatch_state_update` to translate
    legacy dispatch kinds to their DAG equivalents.
    """

    schema = state_dict.get("_schema")
    if schema == "dag":
        return True
    if schema == "legacy":
        return False
    root = state_dict.get("root")
    return isinstance(root, dict) and "blocks" in root


def _dispatch_state_update(
    state_dict: Dict[str, Any], kind: DispatchKind, payload: Dict[str, Any]
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
            ``"port_slot_add"`` (payload: ``node_id``, ``param``)
                Append a ``None`` slot to the consumer's ``aux_ports``
                list for a list-typed param. No-op for scalar ports.
            ``"port_slot_remove"`` (payload: ``node_id``, ``param``,
            ``slot``)
                Remove the slot at ``slot`` from the consumer's
                ``aux_ports`` list and reindex remaining slots.
            ``"block_create"`` (payload: ``class_name``, ``x``, ``y``,
            ``container_block_id``, ``ts``)
                DAG-only. Append a fresh DAG :class:`BlockNode` dict to
                either the root scope (when ``container_block_id`` is
                ``None``) or to a container's nested scope (looked up
                by DFS).  Rejects ``class_name ==
                INPUT_IMAGE_CLASS_NAME`` with a toast (spec §4.8); a
                stale ``container_block_id`` short-circuits with a
                warning + toast.  Drop coords ``(x, y)`` are
                **ignored** — the leaf-first dagre pass re-lays the
                canvas on the next render (spec §4.7).

        payload: kind-specific data (see above).

    Returns:
        A *new* state dict reflecting the mutation; the input dict is not
        modified in place.

    Raises:
        ValueError: If a payload is missing a required key or refers to a
            node that does not exist in the relevant scope.
    """

    out = deepcopy(state_dict)
    is_dag = _is_dag_state_dict(out)

    # When state carries the DAG schema, translate legacy dispatch kinds
    # to their DAG equivalents so callers that don't know about the
    # state shape (toolbar buttons, label / param edit inputs, the
    # canvas tap callback, breadcrumb-link clicks, etc.) continue to
    # mutate state correctly instead of raising ``KeyError: 'nodes'``
    # against a DAG scope dict. Legacy-only kinds without a DAG analogue
    # are short-circuited to a no-op so unit tests that drive the
    # dispatcher directly don't trip over them either.
    if is_dag:
        if kind == "select_node":
            kind = "block_select"
            payload = {"block_id": payload.get("node_id")}
        elif kind == "delete_node":
            sel = out.get("selected_block_id")
            if sel is None:
                return out
            kind = "block_delete_request"
            payload = {"block_id": sel}
        elif kind == "drill_in":
            sel = out.get("selected_block_id")
            if sel is None:
                return out
            kind = "drill_into_container"
            payload = {"block_id": sel}
        elif kind == "drill_out":
            existing = list(out.get("breadcrumb", []) or [])
            if existing:
                existing = existing[:-1]
            kind = "drill_to_scope"
            payload = {"target_breadcrumb": existing}
        elif kind == "breadcrumb_to":
            depth = int(payload.get("depth", 0))
            existing = list(out.get("breadcrumb", []) or [])[:depth]
            kind = "drill_to_scope"
            payload = {"target_breadcrumb": existing}
        elif kind == "edit_param":
            if _linear_reject_unsupported_edit(out):
                return out
            block_id = payload.get("node_id")
            root_scope = out.get("root")
            if not isinstance(root_scope, dict) or not isinstance(
                block_id, str
            ):
                return out
            hit = _find_block_in_tree(root_scope, block_id)
            if hit is None:
                return out
            _, block = hit
            if payload.get("omit"):
                block.setdefault("params", {}).pop(payload["name"], None)
            else:
                block.setdefault("params", {})[payload["name"]] = (
                    payload.get("value")
                )
            return out
        elif kind == "edit_label":
            if _linear_reject_unsupported_edit(out):
                return out
            block_id = payload.get("node_id")
            root_scope = out.get("root")
            if not isinstance(root_scope, dict) or not isinstance(
                block_id, str
            ):
                return out
            hit = _find_block_in_tree(root_scope, block_id)
            if hit is None:
                return out
            _, block = hit
            block["label"] = payload.get("label") or None
            return out
        elif kind in {
            "add_node",
            "add_pipeline",
            "reorder",
            "port_slot_add",
            "port_slot_remove",
        }:
            # No DAG analogue. ``block_create`` / ``edge_create`` /
            # dispatcher-handled aux flow supersede these kinds; callers
            # that still emit them against DAG state (e.g. legacy unit
            # tests) get a clean no-op instead of an exception.
            return out

    breadcrumb = list(out.get("breadcrumb", []) or [])
    scope = _scope_at_breadcrumb(out, breadcrumb)

    if kind == "target_select":
        breadcrumb_list = [
            segment for segment in breadcrumb
            if isinstance(segment, str)
        ]
        target = target_from_dict(payload.get("target", payload), breadcrumb_list)
        _linear_set_target(
            out,
            target,
            open_menu=bool(payload.get("open_menu", True)),
        )
        return out

    if kind == "target_menu_close":
        out["open_port_menu"] = None
        return out

    if kind == "linear_palette_add":
        class_name = payload.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            return out
        if class_name == INPUT_IMAGE_CLASS_NAME:
            _queue_toast(out, "scope already has an Input Image", kind="info")
            return out
        if _linear_reject_unsupported_edit(out):
            return out
        scope_path, linear_scope = _linear_current_scope(out)
        if linear_scope is None:
            return out
        _seed_input_image_dict(linear_scope)
        target = _linear_selected_target(out)
        if target.kind in {"continuation", "image_output", "image_input"}:
            _linear_insert_spine_block(out, linear_scope, target, class_name)
            return out
        if target.kind in {"parameter", "parameter_slot"}:
            _linear_fill_parameter(out, linear_scope, target, class_name)
            return out
        _linear_reset_target_to_continuation(out, scope_path)
        return out

    if kind == "linear_clear_param":
        if _linear_reject_unsupported_edit(out):
            return out
        scope_path, linear_scope = _linear_current_scope(out)
        if linear_scope is None:
            return out
        target = target_from_dict(payload.get("target", payload), scope_path)
        if _linear_clear_param_needs_confirmation(linear_scope, target):
            out["pending_delete_block_id"] = _linear_pending_clear_token(target)
            return out
        _linear_clear_param(out, linear_scope, target)
        return out

    if kind == "linear_clear_param_confirm":
        scope_path, linear_scope = _linear_current_scope(out)
        if linear_scope is None:
            return out
        target = target_from_dict(payload.get("target", payload), scope_path)
        _linear_clear_param(out, linear_scope, target)
        out["pending_delete_block_id"] = None
        return out

    if kind == "linear_drill_param_pipeline":
        _scope_path, linear_scope = _linear_current_scope(out)
        if linear_scope is None:
            return out
        source_block_id = payload.get("source_block_id")
        if not isinstance(source_block_id, str):
            target = target_from_dict(payload.get("target", payload), breadcrumb)
            if target.block_id is not None and target.param is not None:
                matches = _linear_edges_for_param(
                    linear_scope,
                    target.block_id,
                    target.param,
                    target.slot if target.kind == "parameter_slot" else None,
                )
                source_block_id = (
                    matches[0].get("source_block_id") if matches else None
                )
        _linear_drill_param_pipeline(out, linear_scope, source_block_id)
        return out

    if kind == "linear_select_aux_value":
        source_block_id = payload.get("source_block_id")
        if not isinstance(source_block_id, str):
            return out
        _scope_path, linear_scope = _linear_current_scope(out)
        if linear_scope is None:
            return out
        if _linear_block(linear_scope, source_block_id) is None:
            return out
        out["selected_block_id"] = source_block_id
        out["selected_edge_id"] = None
        return out

    if kind == "linear_node_move":
        if _linear_reject_unsupported_edit(out):
            return out
        block_id = payload.get("block_id")
        direction = payload.get("direction")
        if not isinstance(block_id, str) or direction not in {"left", "right"}:
            return out
        _scope_path, linear_scope = _linear_current_scope(out)
        if linear_scope is None:
            return out
        _linear_move_node(out, linear_scope, block_id, direction)
        return out

    if kind == "linear_delete_node_request":
        if _linear_reject_unsupported_edit(out):
            return out
        block_id = payload.get("block_id")
        if not isinstance(block_id, str):
            return out
        out["pending_delete_block_id"] = _linear_pending_node_token(block_id)
        return out

    if kind == "linear_delete_node_confirm":
        block_id = payload.get("block_id")
        if not isinstance(block_id, str):
            return out
        _scope_path, linear_scope = _linear_current_scope(out)
        if linear_scope is None:
            return out
        _linear_delete_spine_node(out, linear_scope, block_id)
        out["pending_delete_block_id"] = None
        return out

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
            breadcrumb.pop()
        out["breadcrumb"] = breadcrumb
        out["selected_node_id"] = None
        return out

    if kind == "breadcrumb_to":
        depth = int(payload.get("depth", 0))
        out["breadcrumb"] = breadcrumb[:depth]
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
        return out

    if kind == "wire_create":
        # Embed a fresh aux StepNode at consumer.aux_ports[param][slot].
        # Payload: {target_node_id, param, slot, class_name}
        target_node_id = payload.get("target_node_id")
        param = payload.get("param", "")
        slot = int(payload.get("slot") or 0)
        class_name = payload.get("class_name", "")

        node = _find_in_scope(scope, target_node_id)
        if node is None:
            return out

        registry = _registry()
        if registry is None:
            return out

        class_info = registry.get(class_name)
        if class_info is None:
            return out  # Unknown class: no-op

        # Type-compatibility check: reject non-ImageOperation sources
        # wired to an is_operation param.
        consumer_info = registry.get(node.get("class_name", ""))
        if consumer_info is not None:
            p_info = consumer_info.parameters.get(param)
            if p_info is not None and p_info.is_operation:
                from phenotypic.abc_ import ImageOperation

                if not issubclass(class_info.cls, ImageOperation):
                    return out

        aux_node = {
            "node_id": _new_node_id(),
            "class_name": class_name,
            "params": _default_params_for(class_name),
            "label": class_name,
            "nested": None,
            "aux_ports": {},
        }

        aux_ports = node.setdefault("aux_ports", {})
        slots = aux_ports.setdefault(param, [])
        while len(slots) <= slot:
            slots.append(None)
        slots[slot] = aux_node

        out["inspector_focus_aux"] = {
            "target_node_id": target_node_id,
            "param": param,
            "slot": slot,
        }
        return out

    if kind == "wire_delete":
        # Clear an aux slot and reset inspector_focus_aux if it pointed there.
        # Payload: {target_node_id, param, slot}
        target_node_id = payload.get("target_node_id")
        param = payload.get("param", "")
        slot = int(payload.get("slot") or 0)

        node = _find_in_scope(scope, target_node_id)
        if node is None:
            return out

        aux_ports = node.get("aux_ports") or {}
        wired_slots = aux_ports.get(param)
        if not isinstance(wired_slots, list) or slot >= len(wired_slots):
            return out

        wired_slots[slot] = None

        focus = out.get("inspector_focus_aux")
        if isinstance(focus, dict):
            if (
                focus.get("target_node_id") == target_node_id
                and focus.get("param") == param
                and focus.get("slot") == slot
            ):
                out["inspector_focus_aux"] = None
        return out

    if kind == "drill_in_aux":
        # Push an aux-slot breadcrumb segment and clear inspector_focus_aux.
        # Payload: {target_node_id, param, slot}
        target_node_id = payload.get("target_node_id")
        param = payload.get("param", "")
        slot = int(payload.get("slot") or 0)

        node = _find_in_scope(scope, target_node_id)
        if node is None:
            return out

        wired_slots = (node.get("aux_ports") or {}).get(param) or []
        if slot >= len(wired_slots) or wired_slots[slot] is None:
            return out  # Empty slot: reject drill

        breadcrumb.append(
            {"target_node_id": target_node_id, "param": param, "slot": slot}
        )
        out["breadcrumb"] = breadcrumb
        out["inspector_focus_aux"] = None
        return out

    if kind == "set_inspector_focus":
        # Set or clear inspector_focus_aux.
        # Payload: {focus: "aux"|"consumer", target_node_id, param, slot}
        if payload.get("focus") == "aux":
            target_node_id = payload.get("target_node_id")
            param = payload.get("param", "")
            slot = int(payload.get("slot") or 0)

            node = _find_in_scope(scope, target_node_id)
            if node is None:
                return out

            wired_slots = (node.get("aux_ports") or {}).get(param) or []
            if slot >= len(wired_slots) or wired_slots[slot] is None:
                return out  # Empty slot: reject

            out["inspector_focus_aux"] = {
                "target_node_id": target_node_id,
                "param": param,
                "slot": slot,
            }
        else:
            out["inspector_focus_aux"] = None
        return out

    if kind == "block_create":
        # DAG-only dispatch kind written by ``assets/palette_dnd.js`` on
        # palette drop / keyboard fallback (spec §5.6).
        #
        # Payload contract (per spec §5.5):
        #   {"kind": "block_create", "class_name": str, "x": float,
        #    "y": float, "container_block_id": str | None,
        #    "ts": int}
        #
        # Algorithm:
        #   1. Reject ``class_name == INPUT_IMAGE_CLASS_NAME`` — Input
        #      Image is auto-seeded per scope and cannot be created from
        #      the palette (spec §4.8 + §4.10). Queue an info toast +
        #      short-circuit.
        #   2. Resolve ``container_block_id`` (if not None) to a
        #      container's nested scope dict via DFS in
        #      ``out["root"]``. If unresolvable, log a warning, queue a
        #      toast, and short-circuit (defense in depth — JS sends a
        #      stale id only when the user dropped on a container that
        #      was just deleted).
        #   3. Seed an InputImage block in the parent scope before
        #      appending — defense in depth on top of Phase 1's
        #      auto-seed.
        #   4. Mint a fresh BlockNode dict with default params.
        #
        # The dispatcher does NOT persist (x, y); per spec §4.7 manual
        # drag positions are ephemeral and the leaf-first dagre pass
        # re-lays the canvas on the next render.
        class_name = payload.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            return out
        if class_name == INPUT_IMAGE_CLASS_NAME:
            # Spec §4.1 mandates the toast text "scope already has an
            # Input Image" so wording stays consistent with the spec-
            # surfaced rejection copy referenced elsewhere (e.g. docs,
            # tutorials, screenshots).
            _queue_toast(out, "scope already has an Input Image", kind="info")
            return out

        container_block_id = payload.get("container_block_id")
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out

        if container_block_id is None:
            target_scope = root_scope
        else:
            resolved = _find_dag_container_scope(root_scope, container_block_id)
            if resolved is None:
                logger.warning(
                    "block_create: stale container_block_id %r — "
                    "container missing from state tree",
                    container_block_id,
                )
                _queue_toast(
                    out,
                    "Drop target container is no longer in the "
                    "pipeline; block not created.",
                    kind="warning",
                )
                return out
            target_scope = resolved

        # Defense in depth — Phase 1 already auto-seeds, but make sure
        # the target scope has its InputImage before the new block is
        # appended so the canvas never renders a scope without one.
        _seed_input_image_dict(target_scope)

        new_block_id = _new_block_id()
        # ImagePipeline containers auto-initialize an empty nested
        # BuilderScope (spec §4.4 "Every container scope is auto-seeded
        # with an InputImage block"). Non-container blocks keep
        # ``nested=None``.
        nested_scope: Optional[Dict[str, Any]] = None
        if class_name == PIPELINE_CLASS_NAME:
            nested_scope = {
                "blocks": [],
                "edges": [],
                "name": "Pipeline",
                "desc": "",
                "nrows": None,
                "ncols": None,
            }
            _seed_input_image_dict(nested_scope)
        new_block = {
            "block_id": new_block_id,
            "class_name": class_name,
            "params": _default_params_for(class_name),
            "label": None,
            "nested": nested_scope,
            "collapsed": False,
            "list_slot_counts": {},
        }
        target_scope.setdefault("blocks", []).append(new_block)
        out["selected_block_id"] = new_block_id
        # Selection focus moves to the new block; clear any wire selection
        # so the inspector picks the new block up cleanly.
        out["selected_edge_id"] = None
        return out

    if kind == "edge_create":
        # Wire-drawing dispatch (spec §5.6, §4.2, §4.3).
        #
        # Payload (per spec §5.5):
        #   {"kind": "edge_create", "source_block_id": str,
        #    "target_block_id": str, "target_port": str,
        #    "edge_kind": "image" | "aux", "ts": int}
        #
        # ``edge_kind`` (not ``kind``) carries the wire kind; the top-
        # level ``kind`` is the dispatch discriminator.  Client emits
        # NO slot index for list-aux — slot resolution is server-side
        # (spec §5.6 list-aux paragraph), which eliminates the
        # concurrent-drag race condition.
        #
        # Algorithm:
        #   1. Locate source + target blocks (and their scopes) in the
        #      tree.  Both endpoints MUST live in the same scope (spec
        #      §4.4 cross-scope rule).
        #   2. For scalar aux + image-in: delete any existing edge
        #      whose ``(target_block_id, target_port)`` pair matches
        #      (replace semantics, spec §4.2 / §4.3).
        #   3. For list aux: server-side append at
        #      ``block.list_slot_counts.get(port, 0)`` and increment.
        #   4. Mint a fresh :class:`Edge` dict; append to the shared
        #      scope's ``edges`` list.
        source_block_id = payload.get("source_block_id")
        target_block_id = payload.get("target_block_id")
        target_port = payload.get("target_port")
        edge_kind = payload.get("edge_kind")
        if (
            not isinstance(source_block_id, str)
            or not isinstance(target_block_id, str)
            or not isinstance(target_port, str)
            or edge_kind not in {"image", "aux"}
        ):
            return out
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out

        source_hit = _find_block_in_tree(root_scope, source_block_id)
        target_hit = _find_block_in_tree(root_scope, target_block_id)
        if source_hit is None or target_hit is None:
            return out
        source_scope, _ = source_hit
        target_scope_dict, target_block = target_hit

        # Cross-scope rule (spec §4.4): source + target must live in
        # the same scope.  Reject with a toast.
        if source_scope is not target_scope_dict:
            _queue_toast(
                out, "Cross-scope wires are not allowed", kind="warning"
            )
            return out

        scope_dict = source_scope

        # Resolve param info for slot logic (aux wires only).  Use the
        # module-level ``get_registry`` rather than the Flask-context
        # ``_registry`` helper so unit tests can monkeypatch the
        # registry without spinning up a full Dash app.
        from phenotypic.gui._operation_registry import get_registry

        registry = get_registry()
        param_info = None
        if edge_kind == "aux" and registry is not None:
            consumer_info = registry.get(target_block.get("class_name", ""))
            if consumer_info is not None:
                param_info = consumer_info.parameters.get(target_port)
        canonical_port, is_list = _resolve_target_port(
            param_info, target_port
        )

        # Single-wire rule for source ports (spec §4.2): an output port
        # takes at most one outgoing wire total.  Replace any existing
        # outgoing edge in the same dispatch.
        edges_list: List[Dict[str, Any]] = scope_dict.setdefault("edges", [])
        edges_list[:] = [
            e for e in edges_list
            if e.get("source_block_id") != source_block_id
        ]

        # Replace-by-deletion semantics for scalar aux + image-in
        # targets (spec §4.2 / §4.3).  List aux skips this step and
        # appends to the next free slot.
        if not is_list:
            edges_list[:] = [
                e for e in edges_list
                if not (
                    e.get("target_block_id") == target_block_id
                    and e.get("target_port") == canonical_port
                )
            ]

        target_slot: Optional[int] = None
        if is_list:
            slot_counts = target_block.setdefault("list_slot_counts", {})
            target_slot = int(slot_counts.get(canonical_port, 0))
            slot_counts[canonical_port] = target_slot + 1

        new_edge = {
            "edge_id": _new_block_id(),
            "source_block_id": source_block_id,
            "source_port": "out",
            "target_block_id": target_block_id,
            "target_port": canonical_port,
            "target_slot": target_slot,
            "kind": edge_kind,
        }
        edges_list.append(new_edge)
        return out

    if kind == "edge_delete":
        # Wire-deletion dispatch (spec §5.6).
        #
        # Payload: ``{"kind": "edge_delete", "edge_id": str, "ts": int}``.
        #
        # Algorithm:
        #   1. DFS through ``root`` + every nested scope to find the
        #      edge's containing scope.
        #   2. Remove the edge from that scope's ``edges`` list.
        #   3. For list-aux edges: DO NOT renumber remaining slots
        #      (spec §5.6 explicitly: ``list_slot_counts`` stays the
        #      same; the freed slot becomes an empty placeholder).
        edge_id = payload.get("edge_id")
        if not isinstance(edge_id, str) or not edge_id:
            return out
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out
        hit = _find_edge_in_tree(root_scope, edge_id)
        if hit is None:
            return out
        scope_dict, target_edge = hit
        scope_dict["edges"] = [
            e for e in scope_dict.get("edges", []) or []
            if e.get("edge_id") != edge_id
        ]
        # If the selected edge was the one we deleted, clear the
        # selection so the inspector doesn't try to render a stale
        # wire card.
        if out.get("selected_edge_id") == edge_id:
            out["selected_edge_id"] = None
        return out

    if kind == "list_aux_reorder":
        # List-aux reorder dispatch (spec §5.6).
        #
        # Payload: ``{"kind": "list_aux_reorder", "block_id": str,
        # "param": str, "new_order": [edge_id_or_null, ...], "ts": int}``.
        #
        # Algorithm:
        #   1. Locate block dict by DFS.
        #   2. Validate ``new_order`` is a permutation of the wired
        #      edge_ids targeting ``(block_id, param)`` interspersed
        #      with ``None``s for empty slots.
        #   3. Rebuild each edge's ``target_slot`` from its position.
        #   4. Update ``block.list_slot_counts[param] = len(new_order)``.
        #   5. Reject non-permutation inputs with a toast (no-op).
        block_id = payload.get("block_id")
        param = payload.get("param")
        reorder_request = payload.get("new_order")
        if (
            not isinstance(block_id, str)
            or not isinstance(param, str)
            or not isinstance(reorder_request, list)
        ):
            return out
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out
        hit = _find_block_in_tree(root_scope, block_id)
        if hit is None:
            return out
        scope_dict, block = hit

        # Wired edges currently targeting (block_id, param).
        wired_edges = [
            e for e in scope_dict.get("edges", []) or []
            if e.get("target_block_id") == block_id
            and e.get("target_port") == param
        ]
        wired_ids = {e.get("edge_id") for e in wired_edges}
        new_order_ids = {x for x in reorder_request if x is not None}

        # Validate permutation: every wired edge_id appears exactly
        # once in new_order, and every non-null entry in new_order is
        # a wired edge_id.
        if new_order_ids != wired_ids or len(new_order_ids) != len(
            [x for x in reorder_request if x is not None]
        ):
            _queue_toast(out, "Reorder rejected", kind="warning")
            return out

        # Rebuild target_slot from position.
        slot_for: Dict[str, int] = {}
        for i, entry in enumerate(reorder_request):
            if entry is not None:
                slot_for[entry] = i
        for edge in scope_dict.get("edges", []) or []:
            if (
                edge.get("target_block_id") == block_id
                and edge.get("target_port") == param
                and edge.get("edge_id") in slot_for
            ):
                edge["target_slot"] = slot_for[edge["edge_id"]]

        # Update the slot count (spec §5.6: count = len(new_order)).
        slot_counts = block.setdefault("list_slot_counts", {})
        slot_counts[param] = len(reorder_request)
        return out

    if kind == "list_aux_add_empty_slot":
        # List-aux empty-slot dispatch (spec §5.6).
        #
        # Payload: ``{"kind": "list_aux_add_empty_slot", "block_id":
        # str, "param": str, "ts": int}``.
        #
        # No edge is materialised — empty slots are tracked solely on
        # ``block.list_slot_counts``.  At ``to_pipeline_dag`` time,
        # slot indices in ``[0, count)`` not covered by an edge emit
        # ``None`` entries.
        block_id = payload.get("block_id")
        param = payload.get("param")
        if not isinstance(block_id, str) or not isinstance(param, str):
            return out
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out
        hit = _find_block_in_tree(root_scope, block_id)
        if hit is None:
            return out
        _scope_dict, block = hit
        slot_counts = block.setdefault("list_slot_counts", {})
        slot_counts[param] = int(slot_counts.get(param, 0)) + 1
        return out

    if kind == "wire_select":
        # Wire-selection dispatch (spec §5.6, §4.5).
        #
        # Payload: ``{"kind": "wire_select", "edge_id": str | None,
        # "ts": int}``.  ``None`` deselects.  Setting a new id clears
        # ``selected_block_id`` (mutual exclusion, spec §4.5).
        edge_id = payload.get("edge_id")
        out["selected_edge_id"] = edge_id if isinstance(edge_id, str) else None
        if out["selected_edge_id"] is not None:
            out["selected_block_id"] = None
        return out

    if kind == "block_select":
        # Block-selection dispatch (spec §5.6, §4.5).
        #
        # Payload: ``{"kind": "block_select", "block_id": str | None,
        # "ts": int}``.  ``None`` deselects.  Setting a new id clears
        # ``selected_edge_id`` (mutual exclusion, spec §4.5).
        block_id = payload.get("block_id")
        out["selected_block_id"] = (
            block_id if isinstance(block_id, str) else None
        )
        if out["selected_block_id"] is not None:
            out["selected_edge_id"] = None
        return out

    if kind == "block_reparent":
        # Pipeline-container reparent dispatch (spec §4.4 / §5.6).
        #
        # Payload (per spec §5.6):
        #   {"kind": "block_reparent", "block_id": str,
        #    "new_parent_block_id": str | None,
        #    "x": float, "y": float, "ts": int}
        #
        # ``new_parent_block_id=None`` promotes the block to the current
        # scope (the visible scope under ``state.breadcrumb``); a non-None
        # value adopts the block into that container's nested scope.
        #
        # Sibling-container moves are a single atomic dispatch — the block
        # is removed from its current containing scope and appended to the
        # target's nested scope in one tick, and the orphan-edge check
        # runs across both scopes before the move commits.
        #
        # Drag-out direction (i.e. the new parent is an *ancestor* of the
        # block's current scope) with orphan edges → snap-back + toast.
        # Drag-in / sibling direction with orphan edges → delete the
        # incompatible edges + toast the count, then commit the move.
        return _dispatch_block_reparent(out, payload)

    if kind == "block_collapsed_toggle":
        # Container collapse-toggle (spec §4.4 / §5.6).
        #
        # Payload: ``{"kind": "block_collapsed_toggle",
        # "block_id": str, "ts": int}``.
        #
        # Toggles ``block.collapsed`` for the named Pipeline container.
        # No-op for non-container blocks (i.e. anything whose
        # ``class_name != PIPELINE_CLASS_NAME``).
        block_id = payload.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            return out
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out
        hit = _find_block_in_tree(root_scope, block_id)
        if hit is None:
            return out
        _scope, block = hit
        if block.get("class_name") != PIPELINE_CLASS_NAME:
            return out
        block["collapsed"] = not bool(block.get("collapsed", False))
        return out

    if kind == "drill_into_container":
        # Drill-into-container dispatch (spec §4.4 / §5.6).
        #
        # Payload: ``{"kind": "drill_into_container",
        # "block_id": str, "ts": int}``.
        #
        # Algorithm:
        #   1. Validate ``block_id`` resolves to a container at the
        #      *current* breadcrumb depth (i.e. it must live directly in
        #      the visible scope, not a sibling drill level).
        #   2. Push ``block_id`` onto ``state.breadcrumb``.
        #   3. Clear any block / wire selection — the new scope is its
        #      own context.
        block_id = payload.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            return out
        breadcrumb_list = list(out.get("breadcrumb", []) or [])
        current_scope_dict = _dag_scope_at_breadcrumb(out, breadcrumb_list)
        if current_scope_dict is None:
            return out
        block_in_scope = next(
            (
                b for b in current_scope_dict.get("blocks", []) or []
                if b.get("block_id") == block_id
            ),
            None,
        )
        if block_in_scope is None:
            return out
        if block_in_scope.get("class_name") != PIPELINE_CLASS_NAME:
            return out
        breadcrumb_list.append(block_id)
        out["breadcrumb"] = breadcrumb_list
        out["selected_block_id"] = None
        out["selected_edge_id"] = None
        return out

    if kind == "drill_to_scope":
        # Atomic-breadcrumb-replacement dispatch (spec §4.4 / §5.6).
        #
        # Payload: ``{"kind": "drill_to_scope",
        # "target_breadcrumb": List[str], "ts": int}``.
        #
        # Algorithm:
        #   1. Validate every block_id in ``target_breadcrumb`` resolves
        #      to a real Pipeline container at the correct depth in the
        #      state tree.
        #   2. Stale (deleted) ids → reject + queue toast.  The
        #      ``viewport_ops.js`` scrim notices the toast / no state
        #      change and emits ``phenotypic:scroll-to-aborted`` itself.
        #   3. Set ``state.breadcrumb = target_breadcrumb`` atomically.
        target_breadcrumb_raw = payload.get("target_breadcrumb")
        if not isinstance(target_breadcrumb_raw, list):
            return out
        target_breadcrumb: List[str] = []
        for entry in target_breadcrumb_raw:
            if not isinstance(entry, str) or not entry:
                _queue_toast(
                    out,
                    "Cannot navigate to that scope: invalid breadcrumb entry.",
                    kind="warning",
                )
                return out
            target_breadcrumb.append(entry)
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out
        if not _validate_breadcrumb_path(root_scope, target_breadcrumb):
            _queue_toast(
                out,
                "Cannot navigate to that scope: container is no longer "
                "in the pipeline.",
                kind="warning",
            )
            return out
        out["breadcrumb"] = target_breadcrumb
        out["selected_block_id"] = None
        out["selected_edge_id"] = None
        return out

    if kind == "block_delete_request":
        # First stage of the two-stage container delete (spec §5.6).
        #
        # Payload: ``{"kind": "block_delete_request",
        # "block_id": str, "ts": int}``.
        #
        # Algorithm:
        #   1. Reject InputImage block_ids (defense in depth).
        #   2. Non-container OR empty container → delegate to
        #      ``block_delete_confirm`` immediately.
        #   3. Non-empty container → set
        #      ``state.pending_delete_block_id = block_id`` so the
        #      confirm modal opens.
        block_id = payload.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            return out
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out
        hit = _find_block_in_tree(root_scope, block_id)
        if hit is None:
            return out
        _scope, block = hit
        if block.get("class_name") == INPUT_IMAGE_CLASS_NAME:
            _queue_toast(
                out, "Input Image cannot be removed.", kind="info"
            )
            return out
        is_container = block.get("class_name") == PIPELINE_CLASS_NAME
        if not is_container or _container_is_empty(block):
            return _dispatch_state_update(
                out,
                "block_delete_confirm",
                {"kind": "block_delete_confirm", "block_id": block_id},
            )
        out["pending_delete_block_id"] = block_id
        return out

    if kind == "block_delete_confirm":
        # Second stage / single-stage delete (spec §5.6).
        #
        # Payload: ``{"kind": "block_delete_confirm",
        # "block_id": str, "ts": int}``.
        #
        # Algorithm:
        #   1. Locate the block + its containing scope by DFS.
        #   2. Reject InputImage block_ids (defense in depth).
        #   3. Atomically remove the block from its scope's
        #      ``blocks`` list AND every edge in that scope whose
        #      source/target matches the block.
        #   4. The block's ``nested`` scope (if any) is dropped
        #      wholesale by virtue of removing the parent — Python
        #      garbage-collects the dict.
        #   5. Clear ``selected_block_id`` / ``pending_delete_block_id``
        #      when they pointed at the deleted block.
        block_id = payload.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            return out
        root_scope = out.get("root")
        if not isinstance(root_scope, dict):
            return out
        hit = _find_block_in_tree(root_scope, block_id)
        if hit is None:
            # Block already gone — at least clear any matching pending id.
            if out.get("pending_delete_block_id") == block_id:
                out["pending_delete_block_id"] = None
            return out
        scope_dict, block = hit
        if block.get("class_name") == INPUT_IMAGE_CLASS_NAME:
            _queue_toast(
                out, "Input Image cannot be removed.", kind="info"
            )
            return out
        scope_dict["blocks"] = [
            b for b in scope_dict.get("blocks", []) or []
            if b.get("block_id") != block_id
        ]
        # Identify incident edge ids before removing them so we can also
        # clear ``selected_edge_id`` when it points at one of the wires we
        # are about to drop (spec §5.6: ``block_delete_confirm`` clears
        # both ``selected_block_id`` and ``selected_edge_id`` if matching).
        removed_edge_ids = {
            e.get("edge_id") for e in scope_dict.get("edges", []) or []
            if e.get("source_block_id") == block_id
            or e.get("target_block_id") == block_id
        }
        scope_dict["edges"] = [
            e for e in scope_dict.get("edges", []) or []
            if e.get("source_block_id") != block_id
            and e.get("target_block_id") != block_id
        ]
        if out.get("selected_block_id") == block_id:
            out["selected_block_id"] = None
        if out.get("selected_edge_id") in removed_edge_ids:
            out["selected_edge_id"] = None
        if out.get("pending_delete_block_id") == block_id:
            out["pending_delete_block_id"] = None
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


#: Auto-dismiss duration (ms) for toasts surfaced from the FIFO queue,
#: per spec §5.1.  Distinct from the static toast's ``duration=5000`` so
#: queue-driven toasts drain faster than ad-hoc Run/Save notifications.
_TOAST_QUEUE_DURATION_MS = 3000


# ---------------------------------------------------------------------------
# Run preview / Save gating (spec §5.6)
# ---------------------------------------------------------------------------


def _filter_blocking_issues(state_data: Optional[Dict[str, Any]]) -> List[Any]:
    """Return the blocking-severity issues for the current builder state.

    Spec §5.6: Run preview and Save pipeline are gated on validation
    severity == ``"error"``.  Advisory hints (currently
    ``stage_order_hint``) NEVER block these actions — they decorate the
    canvas with yellow borders and surface in the issue badge tooltip,
    but the user can still preview / save. Unknown classes and
    unsupported linear shapes are blocking because they cannot safely
    materialize a runtime pipeline.

    The legacy (non-DAG) state shape doesn't carry the validation
    schema, so :func:`validate` would no-op against it; the helper
    short-circuits to an empty list in that case so the gate is
    transparent on legacy state.

    Args:
        state_data: ``state_to_json`` payload from
            :data:`ids.STORE_BUILDER_STATE`.  ``None`` during the first
            paint, before any state has been published.

    Returns:
        List of :class:`Issue` records whose ``severity == "error"``.
        Empty list when state is ``None``, parses, or carries the
        legacy schema.  The list preserves the validator's emission
        order so callers can pull ``[0]`` as the user-facing first
        offence.
    """

    if state_data is None:
        return []
    try:
        state = state_from_json(state_data)
    except Exception:  # noqa: BLE001
        logger.exception("_filter_blocking_issues: state_from_json failed")
        return []
    # Validation only targets the DAG schema; the legacy state shape
    # has no ``root.blocks`` field and the validator would no-op
    # anyway.  Detect via duck-typing on the state object so the
    # legacy GUI flag-off path never accidentally gates Run/Save.
    if not hasattr(state, "selected_block_id"):
        return []
    try:
        issues = validate(state)
    except Exception:  # noqa: BLE001
        logger.exception("_filter_blocking_issues: validate raised")
        return []
    if isinstance(state, _DagBuilderState):
        issues = [*issues, *_linear_unsupported_issues_for_state(state)]
    return [i for i in issues if i.severity == "error"]


def _gate_toast_for_issue(action: str, issue: Any) -> Tuple[bool, str, str, str]:
    """Build the toast outputs surfaced when an action is gated.

    Wraps :func:`_toast` to apply a consistent message shape so the
    user sees the same "Cannot <action>: <kind> (<detail>)" prefix
    regardless of which gated callback fired.  Centralised here so
    future gates (e.g. autosave, sandbox export) can reuse the same
    copy convention.

    Args:
        action: User-facing verb describing the gated action — e.g.
            ``"run preview"`` / ``"save pipeline"``.  Spliced into the
            toast body verbatim.
        issue: The first blocking :class:`Issue` returned by
            :func:`_filter_blocking_issues`.

    Returns:
        Toast output tuple matching the standard
        ``Output(TOAST_NOTIFICATION, ...)`` quadruple.
    """

    return _toast(
        f"Cannot {action}: {issue.kind} ({issue.detail})",
        ok=False,
        header="Validation",
    )


def _toast_queue_from_state(
    state_data: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract a normalised ``toast_queue`` from a state-store payload.

    Shared by :func:`surface_toast_queue_head` and
    :func:`pop_toast_queue_on_dismiss` so the same null-tolerance and
    type-check rules apply to both ends of the FIFO consumer pair.

    Args:
        state_data: ``state_to_json`` payload from
            :data:`ids.STORE_BUILDER_STATE`.  ``None`` / non-dict yields
            an empty list.

    Returns:
        List of queue entries (each a dict).  Empty when the state has
        no queue or the payload is malformed.
    """

    if not isinstance(state_data, dict):
        return []
    queue = state_data.get("toast_queue") or []
    return list(queue)


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


def _browse_seed_from_source(
    image_root: Any,
    source_payload: object,
) -> str | None:
    """Return a Builder browse seed from shared source or ``image_root``.

    The returned path is always inside ``image_root`` when non-``None``.
    """
    if image_root is None:
        return None
    try:
        sandbox = SandboxRoot.from_path(image_root)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError):
        return None
    root = sandbox.root
    candidate = resolve_source_image_root(sandbox, source_payload)
    if candidate is None:
        return str(root)
    try:
        candidate.relative_to(root)
    except ValueError:
        return str(root)
    return str(candidate)


def _pipeline_revision(state_data: Dict[str, Any]) -> str:
    """Return a stable digest of pipeline semantics, excluding UI selection."""

    root = state_data.get("root")
    canonical = json.dumps(
        root,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def _mounted_children(component: Any) -> Any:
    """Return children for callbacks that target an existing mount point."""

    return getattr(component, "children", component)


def _render_views(state: BuilderState) -> Tuple[Any, Any, Any]:
    """Re-render breadcrumb, linear map, and side loader for a given state.

    Args:
        state: Live :class:`BuilderState` object.  Post-Phase-8 ``BuilderState``
            is permanently aliased to ``_DagBuilderState``; the legacy
            attribute names (``selected_node_id``, ``scope.nodes``) are
            gracefully handled via duck-typing so the function remains
            tolerant of the legacy fixture states used by the
            migration-test suite.

    Returns:
        Tuple ``(breadcrumb_children, linear_map_children,
        side_loader_children)``.
        The map callback target is the mounted
        :data:`ids.LINEAR_MAP_CONTAINER` ``children`` property, so this
        function returns the *children* of the section's map container
        rather than a full section wrapper. The breadcrumb callback
        target is the existing nav's ``children`` property, so returning
        a full nav here would nest the breadcrumb inside itself on every
        update. The inspector callback targets
        :data:`ids.INSPECTOR_CONTENT` and returns side-loader children,
        leaving the stable preview sibling untouched.
    """

    registry = _registry()
    is_dag = hasattr(state, "selected_block_id")
    stale_scope = False
    if is_dag:
        stale_scope = scope_at_path(state.root, state.breadcrumb) is None
    else:
        try:
            current_scope(state)
        except KeyError:
            stale_scope = True
    if stale_scope:
        # Stale breadcrumb — fall back to the root.  Rebuild the state
        # under the right schema so subsequent attribute lookups don't
        # cross legacy/DAG boundaries.
        if hasattr(state, "selected_block_id"):
            from phenotypic.gui.builder._state import _DagBuilderState

            state = _DagBuilderState(
                root=state.root,
                breadcrumb=[],
                selected_block_id=None,
                selected_edge_id=None,
                pending_delete_block_id=None,
                toast_queue=[],
            )
        else:
            from phenotypic.gui.builder._state import _LegacyBuilderState

            state = _LegacyBuilderState(
                root=state.root,
                breadcrumb=[],
                selected_node_id=None,
            )
    if is_dag:
        map_section = build_linear_map_section(state, registry)
        map_container = next(
            (
                child for child in map_section.children
                if getattr(child, "id", None) == ids.LINEAR_MAP_CONTAINER
            ),
            None,
        )
        map_children = getattr(map_container, "children", None)
        inspector = _mounted_children(build_linear_side_loader(state, registry))
    else:
        scope = current_scope(state)
        map_children = json.dumps(
            build_canvas_elements(scope, getattr(state, "selected_node_id", None))
        )
        inspector = _mounted_children(build_inspector(state, registry))
    breadcrumb = build_breadcrumb(state).children
    return breadcrumb, map_children, inspector


def _state_replacement_payload(
    pipeline: Any,
) -> Tuple[Dict[str, Any], Any, Any, Any]:
    """Build the re-render tuple for a freshly-loaded pipeline.

    Both the JSON-load and prefab-load callbacks blow away the current
    builder state and replace it with one derived from a freshly-built
    :class:`ImagePipeline`. Both then need the same four output values;
    this helper centralises the conversion + view rendering.

    Returns:
        Tuple ``(state_dict, breadcrumb, linear_map_children,
        inspector)`` — see :func:`_render_views`.
    """

    new_state = from_pipeline_dag(pipeline)
    breadcrumb, canvas_elements, inspector = _render_views(new_state)
    return (
        state_to_json(new_state),
        breadcrumb,
        canvas_elements,
        inspector,
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
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children"),
        # Toast outputs surface mutation errors to the user; success path leaves
        # them as ``no_update`` so they don't clobber other callbacks' toasts.
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        # palette
        Input({"type": "palette-add", "class_name": ALL}, "n_clicks"),
        Input(ids.BTN_NEW_PIPELINE_NODE, "n_clicks"),
        # Linear map / side-loader actions. ``surface`` disambiguates map
        # and side-loader instances that otherwise point at the same
        # logical target.
        Input(
            {
                "type": ids.LINEAR_PORT,
                "surface": ALL,
                "kind": ALL,
                "scope_path": ALL,
                "block_id": ALL,
                "param": ALL,
                "slot": ALL,
            },
            "n_clicks",
        ),
        Input(
            {
                "type": ids.LINEAR_NODE_ACTION,
                "surface": ALL,
                "action": ALL,
                "scope_path": ALL,
                "block_id": ALL,
            },
            "n_clicks",
        ),
        Input(
            {
                "type": ids.LINEAR_PARAM_ACTION,
                "surface": ALL,
                "action": ALL,
                "scope_path": ALL,
                "block_id": ALL,
                "param": ALL,
                "slot": ALL,
                "source_block_id": ALL,
            },
            "n_clicks",
        ),
        # Retired DAG drag/drop channels remain subscribed as safe no-ops so
        # loaded assets cannot mutate the fixed linear builder by accident.
        Input(ids.STORE_PALETTE_DROP, "data"),
        Input(ids.STORE_EDGE_EVENT, "data"),
        # drill in (visible button on pipeline-node inspector); drill-out is
        # done via breadcrumb-link clicks, not a dedicated button.
        Input(ids.BTN_DRILL_IN, "n_clicks"),
        Input({"type": "breadcrumb-link", "depth": ALL}, "n_clicks"),
        # Cytoscape element feedback is no longer subscribed in the
        # default builder; visible state redraws target the fixed linear
        # map container directly.
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
        _linear_port_clicks: List[int],
        _linear_node_action_clicks: List[int],
        _linear_param_action_clicks: List[int],
        palette_drop: Optional[Dict[str, Any]],
        edge_event: Optional[Dict[str, Any]],
        _drill_out_clicks: Optional[int],
        _crumb_clicks: List[int],
        bool_vals: List[Any],
        _num_blurs: List[Any],
        _str_blurs: List[Any],
        enum_vals: List[Any],
        _list_blurs: List[Any],
        _tuple_blurs: List[Any],
        toggle_vals: List[Any],
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
                        "linear_palette_add",
                        {
                            "kind": "linear_palette_add",
                            "class_name": triggered["class_name"],
                        },
                    )
                elif t_type == ids.LINEAR_PORT:
                    if not ctx.triggered[0]["value"]:
                        return _NOOP_FAN_IN
                    new_state_dict = _dispatch_state_update(
                        state_data,
                        "target_select",
                        {
                            "kind": "target_select",
                            "target": _linear_target_payload_from_id(triggered),
                            "open_menu": True,
                        },
                    )
                elif t_type == ids.LINEAR_NODE_ACTION:
                    if not ctx.triggered[0]["value"]:
                        return _NOOP_FAN_IN
                    action = triggered.get("action")
                    block_id = _decode_linear_optional(triggered.get("block_id"))
                    if action == "select":
                        if block_id is None:
                            return _NOOP_FAN_IN
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "block_select",
                            {"kind": "block_select", "block_id": block_id},
                        )
                    elif action == "target_menu_close":
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "target_menu_close",
                            {"kind": "target_menu_close"},
                        )
                    elif action == "preview_here":
                        # The dedicated preview callback handles runtime work;
                        # this branch keeps the visible inspector aligned with
                        # the output block whose prefix cache will be baked.
                        new_state_dict = _linear_state_with_preview_selection(
                            state_data,
                            _linear_preview_target_payload_from_id(triggered),
                        )
                    elif action == "drill":
                        if block_id is None:
                            return _NOOP_FAN_IN
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "drill_into_container",
                            {
                                "kind": "drill_into_container",
                                "block_id": block_id,
                            },
                        )
                    elif action == "delete":
                        if block_id is None:
                            return _NOOP_FAN_IN
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "linear_delete_node_request",
                            {
                                "kind": "linear_delete_node_request",
                                "block_id": block_id,
                            },
                        )
                    elif action in {"move_left", "move_right"}:
                        if block_id is None:
                            return _NOOP_FAN_IN
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "linear_node_move",
                            {
                                "kind": "linear_node_move",
                                "block_id": block_id,
                                "direction": (
                                    "left"
                                    if action == "move_left"
                                    else "right"
                                ),
                            },
                        )
                    else:
                        return _NOOP_FAN_IN
                elif t_type == ids.LINEAR_PARAM_ACTION:
                    if not ctx.triggered[0]["value"]:
                        return _NOOP_FAN_IN
                    action = triggered.get("action")
                    target_payload = _linear_param_target_payload_from_id(
                        triggered
                    )
                    if action == "replace":
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "target_select",
                            {
                                "kind": "target_select",
                                "target": target_payload,
                                "open_menu": False,
                            },
                        )
                    elif action == "clear":
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "linear_clear_param",
                            {
                                "kind": "linear_clear_param",
                                "target": target_payload,
                            },
                        )
                    elif action == "drill":
                        new_state_dict = _dispatch_state_update(
                            state_data,
                            "linear_drill_param_pipeline",
                            {
                                "kind": "linear_drill_param_pipeline",
                                "target": target_payload,
                                "source_block_id": (
                                    _linear_source_block_id_from_action_id(
                                        triggered
                                    )
                                ),
                            },
                        )
                    else:
                        return _NOOP_FAN_IN
                elif t_type == "breadcrumb-link":
                    # Pattern-matched Inputs fire on initial render of newly
                    # added matching components — e.g. drilling pushes a new
                    # breadcrumb-link button (depth=0) onto the nav, and Dash
                    # fires this callback once with n_clicks=0. Without this
                    # guard, ``breadcrumb_to(depth=0)`` would immediately undo
                    # the drill by truncating the breadcrumb back to root.
                    if not ctx.triggered[0]["value"]:
                        return _NOOP_FAN_IN
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
            elif triggered == ids.STORE_PALETTE_DROP:
                # The fixed linear builder retired palette drag/drop. Keep
                # the store subscribed so stale asset writes are harmless.
                return _NOOP_FAN_IN
            elif triggered == ids.STORE_EDGE_EVENT:
                # Drag-to-wire and old inspector aux-wire emitters are not
                # user-facing in the fixed map. Linear side-loader buttons use
                # their own pattern ids above.
                return _NOOP_FAN_IN
            elif triggered == ids.BTN_NEW_PIPELINE_NODE:
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "linear_palette_add",
                    {
                        "kind": "linear_palette_add",
                        "class_name": PIPELINE_CLASS_NAME,
                    },
                )
            elif triggered == ids.BTN_DRILL_IN:
                # The inspector renders a visible "Drill in ▸" button only on
                # ImagePipeline nodes; for any other selection the button is a
                # hidden placeholder that the user can't trigger. Either way,
                # the canonical action here is drill-in.
                new_state_dict = _dispatch_state_update(
                    state_data, "drill_in", {}
                )
            elif triggered == ids.INPUT_NODE_LABEL:
                state = state_from_json(state_data)
                # Duck-type the selection id: DAG state exposes
                # ``selected_block_id``, legacy state exposes
                # ``selected_node_id``. The dispatcher's edit_label
                # translation works for both shapes.
                selected_id = getattr(
                    state,
                    "selected_block_id",
                    getattr(state, "selected_node_id", None),
                )
                if selected_id is None:
                    return _NOOP_FAN_IN
                new_state_dict = _dispatch_state_update(
                    state_data,
                    "edit_label",
                    {
                        "node_id": selected_id,
                        "label": label_value,
                    },
                )
            else:
                return _NOOP_FAN_IN

            # --- Render ----------------------------------------------
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas_elements, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
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
                True,
                _format_exception(exc),
                "danger",
                "Update failed",
            )

    # ----------------------------------------------------------------------
    # 2a-0. Linear map redraw
    # ----------------------------------------------------------------------
    #
    # The default builder surface is a fixed HTML/Dash linear port map.
    # Each state mutation returns fresh ``LINEAR_MAP_CONTAINER.children``
    # from :func:`_render_views`, so the visible map redraws directly with
    # normal Dash component diffing. Cytoscape helpers remain importable
    # for legacy tests and transition cleanup, but are no longer mounted
    # or updated by this default path.

    # ----------------------------------------------------------------------
    # 2a. Validation pipeline (spec §5.6 + §5.3)
    # ----------------------------------------------------------------------
    #
    # Every state mutation feeds this callback via ``STORE_BUILDER_STATE``.
    # We re-run :func:`validate` and republish the resulting list of
    # :class:`Issue` dicts to ``STORE_ISSUES`` so the toolbar badge, the
    # red/yellow border decorations, and the Run/Save gating callbacks all
    # subscribe to a single source of truth. The validator is pure +
    # O(V+E); piping it through a separate callback keeps the fan-in body
    # focused on state mutation and avoids re-rendering the canvas /
    # inspector when only the issues list changes.

    @app.callback(
        Output(ids.STORE_ISSUES, "data"),
        Input(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_ISSUES, "data"),
        prevent_initial_call=False,
    )
    def revalidate_on_state_change(
        state_data: Optional[Dict[str, Any]],
        prev_issues: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """Re-run :func:`validate` whenever ``STORE_BUILDER_STATE`` changes.

        Args:
            state_data: ``state_to_json`` payload (legacy or DAG schema).
                ``None`` during the very first paint, before any state
                has been written.
            prev_issues: The previously published issue list from
                ``STORE_ISSUES``. Used to skip the store write (return
                ``dash.no_update``) when the freshly-computed issue list
                is identical — avoids cascading re-renders of every
                downstream subscriber when state mutations that don't
                change validity fire (selection clicks, drill-in /
                drill-out, label edits, etc.).

        Returns:
            JSON-friendly list of :class:`Issue` dicts, or
            :data:`dash.no_update` when the list is unchanged. Empty
            list when state is ``None`` or carries the legacy schema
            (the DAG validation suite is meaningless against legacy
            linear-list state).
        """

        if state_data is None:
            new_issues: List[Dict[str, Any]] = []
        else:
            try:
                state = state_from_json(state_data)
            except Exception:
                logger.exception(
                    "revalidate_on_state_change: state_from_json failed"
                )
                new_issues = []
            else:
                # Validation only targets the DAG schema; the legacy
                # state shape has no ``root.blocks`` field and the
                # validator would no-op anyway. Detect via duck-typing
                # on the state object.
                if not hasattr(state, "selected_block_id"):
                    new_issues = []
                else:
                    try:
                        issues = validate(state)
                    except Exception:
                        logger.exception(
                            "revalidate_on_state_change: validate raised"
                        )
                        new_issues = []
                    else:
                        if isinstance(state, _DagBuilderState):
                            issues = [
                                *issues,
                                *_linear_unsupported_issues_for_state(state),
                            ]
                        new_issues = [
                            {
                                "kind": issue.kind,
                                "block_id": issue.block_id,
                                "detail": issue.detail,
                                "scope_path": list(issue.scope_path),
                                "severity": issue.severity,
                            }
                            for issue in issues
                        ]
        # Skip the store write when nothing changed. ``prev_issues`` is
        # ``None`` on first paint, in which case we always publish.
        if prev_issues is not None and prev_issues == new_issues:
            return no_update
        return new_issues

    @app.callback(
        Output(ids.DOWNLOAD_RAW_STATE, "data"),
        Input(
            {
                "type": ids.LINEAR_NODE_ACTION,
                "action": "export_raw_state",
                "scope_path": ALL,
                "block_id": ALL,
                "surface": ALL,
            },
            "n_clicks",
        ),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def export_raw_builder_state(
        n_clicks: List[Optional[int]],
        state_data: Optional[Dict[str, Any]],
    ) -> Any:
        """Download raw builder JSON from the unsupported-state panel."""

        if not any(click or 0 for click in n_clicks or []):
            return no_update
        if state_data is None:
            return no_update
        payload = json.dumps(state_data, indent=2, sort_keys=True)
        return dcc.send_string(payload, "builder-state.json")

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
        Input(
            {
                "type": ids.LINEAR_NODE_ACTION,
                "action": "start_new_state",
                "scope_path": ALL,
                "block_id": ALL,
                "surface": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def start_new_builder_state(
        n_clicks: List[Optional[int]],
    ) -> Tuple[Any, ...]:
        """Reset an unsupported development DAG to a fresh linear state."""

        noop = (no_update,) * 4
        if not any(click or 0 for click in n_clicks or []):
            return noop
        new_state = BuilderState()
        new_state_dict = state_to_json(new_state)
        breadcrumb, canvas_elements, inspector = _render_views(new_state)
        return (
            new_state_dict,
            breadcrumb,
            canvas_elements,
            inspector,
        )

    # ----------------------------------------------------------------------
    # 2a-bis. Toolbar issue badge update (spec §4.6)
    # ----------------------------------------------------------------------
    #
    # Re-renders the toolbar issue badge widget (count chip + popover
    # listing one row per issue) whenever ``STORE_ISSUES`` changes.
    # The badge itself lives inside the canvas-section header (a
    # sibling of the relayout button); we splice its updated children
    # back onto a wrapping ``html.Span``.  The wrapping span uses the
    # static id :data:`ids.ISSUE_BADGE` for the badge chip and
    # :data:`ids.ISSUE_BADGE_TOOLTIP` for the popover, both of which
    # the row-click callback below subscribes to.
    #
    # ``build_issue_badge`` produces an ``html.Span([badge, popover])``;
    # we publish that span's children to a wrapping ``html.Span`` we
    # mount inside the toolbar.  Because Dash's ``Output`` targets a
    # single id, we publish the badge's ``children`` (the [badge,
    # popover] list) directly to the wrapping span's ``children``.
    #
    # NB: we INTENTIONALLY swap the entire badge+popover content on
    # every issue change.  The popover's child rows carry
    # pattern-matched ids (see :func:`ids.issue_row_id`); recreating
    # them is the correct way to keep the pattern-match callback's
    # input list in sync with the live issue list.

    @app.callback(
        Output(ids.ISSUE_BADGE, "children", allow_duplicate=True),
        Output(ids.ISSUE_BADGE, "color", allow_duplicate=True),
        Output(ids.ISSUE_BADGE_TOOLTIP, "children", allow_duplicate=True),
        Input(ids.STORE_ISSUES, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def update_issue_badge(
        issues: Optional[List[Dict[str, Any]]],
        state_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any, Any]:
        """Mirror the live ``STORE_ISSUES`` list onto the toolbar badge.

        Splits the rendered ``build_issue_badge(...)`` span back into
        its sub-pieces and publishes:

        * the badge label (``html.Span`` text content) → ``ISSUE_BADGE.children``
        * the badge severity colour (``"danger"`` / ``"warning"`` / ``"secondary"``)
          → ``ISSUE_BADGE.color``
        * the popover body rows → ``ISSUE_BADGE_TOOLTIP.children``

        Args:
            issues: Live :data:`STORE_ISSUES` payload (list of dicts
                shaped by ``revalidate_on_state_change``).  ``None``
                before the first publish.
            state_data: Live ``STORE_BUILDER_STATE`` dump used to
                resolve ``block_id`` → block label for each row.

        Returns:
            Tuple of (badge label, badge color, popover children).
        """

        try:
            state: Optional[BuilderState] = None
            if state_data is not None:
                try:
                    state = state_from_json(state_data)
                except Exception:  # noqa: BLE001
                    # Stale / corrupt state still lets us render the
                    # badge — block_label fallback handles missing
                    # block_ids gracefully.
                    state = None
            badge_span = build_issue_badge(issues=issues or [], state=state)
        except Exception:  # noqa: BLE001
            logger.exception("update_issue_badge failed")
            return no_update, no_update, no_update

        # The wrapping span contains [badge_wrapper, popover] in order.
        # ``badge_wrapper`` is an ``html.Span`` whose single child is the
        # ``dbc.Badge`` chip carrying both the label (``children``) and
        # severity colour (``color``).  Reach through the wrapper so we
        # read the props off the badge itself — ``html.Span`` has no
        # ``color`` prop and would raise ``AttributeError`` on access.
        badge_wrapper = badge_span.children[0]
        popover_widget = badge_span.children[1]
        badge_chip = badge_wrapper.children
        return (
            badge_chip.children,
            badge_chip.color,
            popover_widget.children,
        )

    # ----------------------------------------------------------------------
    # 2a-ter. Issue badge row click → fixed-map focus dispatch (spec §4.6, §5.6)
    # ----------------------------------------------------------------------
    #
    # Each tooltip row carries the pattern-matched id from
    # :func:`ids.issue_row_id`. A click writes an ``issue_focus`` payload
    # to ``STORE_VIEWPORT_OP``; the fixed-map server fan-in consumes it
    # by drilling to the issue scope and selecting the offending block.
    # The retired Cytoscape ``scroll_to`` relay remains available for
    # legacy direct viewport-op tests, but badge rows no longer depend on
    # a mounted Cytoscape canvas.
    #
    # The row id encodes (block_id, kind, idx) — the click callback
    # walks the active ``STORE_ISSUES`` list to recover the issue's
    # ``scope_path`` (which becomes ``target_breadcrumb`` per spec §5.6).
    # ``block_id`` "__scope__" sentinel (used for scope-level findings
    # like ``missing_input``) translates back to ``None`` so the fan-in
    # only drills without selecting a specific block.

    @app.callback(
        Output(ids.STORE_VIEWPORT_OP, "data", allow_duplicate=True),
        Input(
            {
                "type": "issue-row",
                "block_id": ALL,
                "kind": ALL,
                "idx": ALL,
            },
            "n_clicks",
        ),
        State(ids.STORE_ISSUES, "data"),
        prevent_initial_call=True,
    )
    def issue_row_click_dispatch(
        n_clicks_list: List[Optional[int]],
        issues: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """Translate an issue-row click into a fixed-map focus viewport op.

        Reads :data:`dash.callback_context.triggered_id` to identify the
        clicked row, then walks the live ``STORE_ISSUES`` list to
        recover the matching issue's ``scope_path``.  Emits a payload
        of the form:

        .. code-block:: python

            {
                "kind": "issue_focus",
                "block_id": <BlockNode.block_id or None>,
                "scope_path": <list[str]>,
                "target_breadcrumb": <list[str]>,
                "ts": <ms timestamp>,
            }

        Per spec §5.6, ``target_breadcrumb`` is set to the issue's
        ``scope_path`` unconditionally. The server fan-in compares
        against the current state breadcrumb and only dispatches
        ``drill_to_scope`` when the two differ.

        Args:
            n_clicks_list: Pattern-match ``n_clicks`` values; only used
                to gate against initial render (``ctx.triggered`` is the
                authoritative trigger source).
            issues: Live ``STORE_ISSUES`` list to recover full issue
                metadata (``scope_path`` is needed for the payload but
                not encoded in the row id).

        Returns:
            ``issue_focus`` payload as a plain dict, or :data:`no_update`
            when the trigger is the initial render or the issues list
            is unavailable.
        """

        # Initial render fires every pattern with ``n_clicks=None`` /
        # ``0`` — short-circuit before doing any work.
        if not ctx.triggered:
            return no_update
        triggered_value = ctx.triggered[0].get("value")
        if not triggered_value:
            return no_update
        triggered_id = ctx.triggered_id
        if not isinstance(triggered_id, dict):
            return no_update
        if triggered_id.get("type") != "issue-row":
            return no_update

        raw_block_id = triggered_id.get("block_id")
        clicked_kind = triggered_id.get("kind")
        clicked_idx = triggered_id.get("idx")
        block_id: Optional[str] = (
            None if raw_block_id == "__scope__" else raw_block_id
        )

        # Recover ``scope_path`` from the live issues list.  Match on
        # ``(kind, block_id)`` and fall back to the ``idx`` ordering
        # produced by ``_sort_issues_for_badge`` (i.e. the position in
        # the badge tooltip).  The match must align with the order
        # rendered into the tooltip — that's also the order the
        # pattern-match Inputs see.
        sorted_issues = _sort_issues_for_badge(issues or [])
        scope_path: List[str] = []
        if (
            isinstance(clicked_idx, int)
            and 0 <= clicked_idx < len(sorted_issues)
        ):
            issue = sorted_issues[clicked_idx]
            if (
                issue.get("kind") == clicked_kind
                and (issue.get("block_id") or None) == block_id
            ):
                scope_path = list(issue.get("scope_path") or [])

        payload: Dict[str, Any] = {
            "kind": "issue_focus",
            "block_id": block_id,
            "scope_path": scope_path,
            "target_breadcrumb": list(scope_path),
            "ts": int(time.time() * 1000),
        }
        return payload

    # ----------------------------------------------------------------------
    # 2a-i. Inspector wire / aux ports button → STORE_EDGE_EVENT
    # ----------------------------------------------------------------------
    #
    # The inspector wire card + aux ports section emit pattern-matched
    # button clicks (Disconnect, list-row ``✕`` remove, ``+ Add empty
    # slot``, ``▲``/``▼`` arrow reorder).  Each callback builds the
    # appropriate ``STORE_EDGE_EVENT`` payload and writes it; the
    # existing fan-in callback (above) then routes through
    # ``_dispatch_state_update`` so all state mutations stay in one
    # place.  Splitting the inspector → store hop into its own
    # pattern-match callback keeps the central fan-in's Input list
    # short (Dash imposes O(N) signature growth per added Input pattern).

    @app.callback(
        Output(ids.STORE_EDGE_EVENT, "data", allow_duplicate=True),
        Input(
            {"type": ids.BTN_INSPECTOR_DISCONNECT, "edge_id": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def inspector_disconnect_emit(_clicks: List[Any]) -> Any:
        """Emit an ``edge_delete`` payload from the inspector ``Disconnect``.

        Routes the wire-card's ``Disconnect`` button through
        :data:`STORE_EDGE_EVENT` so the central fan-in mutation
        callback handles it.  Pattern-matched against ``edge_id`` so a
        single callback serves both the wire card and any other surface
        that exposes the same pattern type.

        Args:
            _clicks: One entry per matched button (n_clicks).  The trigger
                is identified via ``ctx.triggered_id``.

        Returns:
            ``STORE_EDGE_EVENT`` payload of shape
            ``{"kind": "edge_delete", "edge_id": <id>, "ts": <int>}``;
            :data:`dash.no_update` on initial-render fan-outs (where
            ``n_clicks`` is still 0).
        """

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.BTN_INSPECTOR_DISCONNECT:
            return no_update
        if not ctx.triggered[0].get("value"):
            return no_update
        edge_id = triggered.get("edge_id")
        if not isinstance(edge_id, str) or not edge_id:
            return no_update
        return {
            "kind": "edge_delete",
            "edge_id": edge_id,
            "ts": int(time.time() * 1000),
        }

    @app.callback(
        Output(ids.STORE_EDGE_EVENT, "data", allow_duplicate=True),
        Input(
            {"type": ids.BTN_INSPECTOR_LIST_REMOVE, "edge_id": ALL},
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def inspector_list_remove_emit(_clicks: List[Any]) -> Any:
        """Emit an ``edge_delete`` payload for the list-aux ``✕`` button.

        Mirrors :func:`inspector_disconnect_emit` but listens on the
        list-row-specific ``BTN_INSPECTOR_LIST_REMOVE`` pattern so both
        surfaces can co-exist in the DOM.
        """

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.BTN_INSPECTOR_LIST_REMOVE:
            return no_update
        if not ctx.triggered[0].get("value"):
            return no_update
        edge_id = triggered.get("edge_id")
        if not isinstance(edge_id, str) or not edge_id:
            return no_update
        return {
            "kind": "edge_delete",
            "edge_id": edge_id,
            "ts": int(time.time() * 1000),
        }

    @app.callback(
        Output(ids.STORE_EDGE_EVENT, "data", allow_duplicate=True),
        Input(
            {
                "type": ids.BTN_INSPECTOR_ADD_EMPTY_SLOT,
                "block_id": ALL,
                "param": ALL,
            },
            "n_clicks",
        ),
        prevent_initial_call=True,
    )
    def inspector_add_empty_slot_emit(_clicks: List[Any]) -> Any:
        """Emit a ``list_aux_add_empty_slot`` payload for ``+ Add empty slot``.

        Pattern-matched against ``(block_id, param)`` so a single
        callback covers every list-aux param on whichever block is
        currently selected.
        """

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.BTN_INSPECTOR_ADD_EMPTY_SLOT:
            return no_update
        if not ctx.triggered[0].get("value"):
            return no_update
        block_id = triggered.get("block_id")
        param = triggered.get("param")
        if not isinstance(block_id, str) or not isinstance(param, str):
            return no_update
        return {
            "kind": "list_aux_add_empty_slot",
            "block_id": block_id,
            "param": param,
            "ts": int(time.time() * 1000),
        }

    @app.callback(
        Output(ids.STORE_EDGE_EVENT, "data", allow_duplicate=True),
        Input(
            {
                "type": ids.BTN_INSPECTOR_LIST_MOVE,
                "edge_id": ALL,
                "direction": ALL,
            },
            "n_clicks",
        ),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def inspector_list_move_emit(
        _clicks: List[Any], state_data: Optional[Dict[str, Any]]
    ) -> Any:
        """Emit a ``list_aux_reorder`` payload for ``▲``/``▼`` clicks.

        The arrow buttons are the drag-handle fallback (spec §4.5 calls
        for HTML5 drag-handles; up/down buttons are an intermediate).
        The callback resolves which block/param the edge targets,
        swaps it with its neighbour, and emits a ``list_aux_reorder``
        payload whose ``new_order`` argument is what
        :func:`_dispatch_state_update` consumes.

        Args:
            _clicks: One entry per matched button (n_clicks).
            state_data: Current ``STORE_BUILDER_STATE`` JSON dump,
                read to resolve which block / param the edge belongs
                to and to compute the slot permutation.

        Returns:
            ``STORE_EDGE_EVENT`` payload, or :data:`dash.no_update` when
            the click is ignored (initial render, missing edge, edge at
            the boundary of the list).
        """

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.BTN_INSPECTOR_LIST_MOVE:
            return no_update
        if not ctx.triggered[0].get("value"):
            return no_update
        edge_id = triggered.get("edge_id")
        direction = triggered.get("direction")
        if (
            not isinstance(edge_id, str)
            or not edge_id
            or direction not in ("up", "down")
            or not isinstance(state_data, dict)
        ):
            return no_update

        # Resolve the edge's containing scope + target (block, param).
        try:
            target = _find_list_aux_target(state_data, edge_id)
        except Exception:
            logger.exception("inspector_list_move_emit: lookup failed")
            return no_update
        if target is None:
            return no_update
        block_id, param, ordered_edge_ids, current_idx = target

        # Boundary checks: ▲ at slot 0 / ▼ at last slot → no-op.
        new_idx = current_idx - 1 if direction == "up" else current_idx + 1
        if new_idx < 0 or new_idx >= len(ordered_edge_ids):
            return no_update

        new_order = list(ordered_edge_ids)
        new_order[current_idx], new_order[new_idx] = (
            new_order[new_idx],
            new_order[current_idx],
        )
        return {
            "kind": "list_aux_reorder",
            "block_id": block_id,
            "param": param,
            "new_order": new_order,
            "ts": int(time.time() * 1000),
        }

    @app.callback(
        Output(ids.STORE_EDGE_EVENT, "data", allow_duplicate=True),
        Input(
            {
                "type": ids.STORE_INSPECTOR_LIST_REORDER,
                "block_id": ALL,
                "param": ALL,
            },
            "data",
        ),
        prevent_initial_call=True,
    )
    def inspector_list_reorder_emit(_payloads: List[Any]) -> Any:
        """Emit a ``list_aux_reorder`` payload from a drag-handle store write.

        The row layout ships with ``▲``/``▼`` arrows as the primary
        reorder surface; the spec calls for HTML5 drag-handles.
        Mounting this callback against the hidden ``dcc.Store`` family
        means a future phase can land the drag JS glue without touching
        the inspector callback wiring.  Until that JS lands, the store
        data on inspector render mirrors the current state — and the
        central dispatcher's permutation check no-ops a same-order
        payload.
        """

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        if triggered.get("type") != ids.STORE_INSPECTOR_LIST_REORDER:
            return no_update
        # Find the triggered store's payload by id within the Input bucket.
        try:
            inputs_list = ctx.inputs_list[0]
        except (IndexError, KeyError):
            return no_update
        payload: Optional[Dict[str, Any]] = None
        for entry in inputs_list:
            if entry.get("id") == triggered:
                payload = entry.get("value")
                break
        if not isinstance(payload, dict):
            return no_update
        new_order = payload.get("edge_id_order")
        if not isinstance(new_order, list):
            return no_update
        block_id = triggered.get("block_id")
        param = triggered.get("param")
        if not isinstance(block_id, str) or not isinstance(param, str):
            return no_update
        return {
            "kind": "list_aux_reorder",
            "block_id": block_id,
            "param": param,
            "new_order": new_order,
            "ts": int(time.time() * 1000),
        }

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
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
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

        noop = (no_update,) * 8

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
            breadcrumb, canvas_elements, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
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
                True,
                _format_exception(exc),
                "danger",
                "Update failed",
            )

    # ----------------------------------------------------------------------
    # 2c-i. Container dispatch fan-in (spec §4.4 / §5.6)
    # ----------------------------------------------------------------------
    #
    # The DAG canvas surfaces a handful of clientside gestures that don't
    # fit the existing palette / wire-event fan-ins:
    #
    #   * Container drill-in (double-click body, ``Drill in →`` button,
    #     breadcrumb-segment click) → ``STORE_VIEWPORT_OP`` with
    #     ``{kind: "drill_into_container" | "drill_out" |
    #     "drill_to_scope", ...}``.
    #   * Collapse chevron click → ``STORE_VIEWPORT_OP`` with
    #     ``{kind: "block_collapsed_toggle", block_id, ts}``.
    #   * Container reparent (drag block between scopes) →
    #     ``STORE_VIEWPORT_OP`` with ``{kind: "block_reparent",
    #     block_id, new_parent_block_id, x, y, ts}``.  ``palette_dnd.js``
    #     already mints this payload when the user drags an *existing*
    #     block (vs. a palette button) across container boundaries.
    #
    # All viewport-op payloads land in :data:`STORE_VIEWPORT_OP`.  This
    # callback routes the mutation-bearing kinds through
    # :func:`_dispatch_state_update`; the purely visual kinds
    # (``scroll_to`` / ``relayout`` / ``reanchor``) are handled
    # clientside and never round-trip through this callback.

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Output(ids.STORE_VIEWPORT_OP, "data", allow_duplicate=True),
        Input(ids.STORE_VIEWPORT_OP, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def viewport_op_fan_in(
        viewport_op: Optional[Dict[str, Any]],
        state_data: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """Route ``STORE_VIEWPORT_OP`` mutation kinds through the dispatcher.

        The clientside ``viewport_ops.js`` writes the same store for
        purely visual ops (``scroll_to``, ``relayout``, ``reanchor``)
        and for state-mutating ops (``drill_to_scope``,
        ``block_collapsed_toggle``).  The visual ops do not change
        state — they're consumed clientside via the chained scrim and
        never reach this callback (the JS clears the store before
        Dash debounces would refire).  The mutation ops route through
        :func:`_dispatch_state_update`.

        Stale-id abort signalling (spec §5.6 ``drill_to_scope`` row):
            When the dispatch rejects a ``drill_to_scope`` payload
            (every breadcrumb id must resolve to a current Pipeline
            container at the right depth) the state changes only in the
            toast queue — the breadcrumb itself is untouched.  In that
            case we additionally write a sentinel
            ``{"kind": "scroll_to_aborted", "ts": <now>}`` back to
            :data:`STORE_VIEWPORT_OP`.  A small clientside callback
            (registered in :func:`_register_scroll_to_aborted_relay`)
            relays the sentinel as the
            ``phenotypic:scroll-to-aborted`` DOM event so
            ``viewport_ops.js`` dismisses its active scrim immediately
            instead of waiting on the layout-stop timeout.

        Args:
            viewport_op: Payload written by ``viewport_ops.js`` /
                ``palette_dnd.js`` / inspector buttons.
            state_data: Current ``STORE_BUILDER_STATE`` dump.

        Returns:
            The standard 8-output fan-in tuple matching
            ``fan_in_state_mutation``, extended with a final
            :data:`STORE_VIEWPORT_OP` output for the abort sentinel.
        """

        noop = (no_update,) * 9
        if not isinstance(viewport_op, dict) or state_data is None:
            return noop

        kind = viewport_op.get("kind")
        if kind not in {
            "drill_into_container",
            "drill_out",
            "drill_to_scope",
            "issue_focus",
            "block_collapsed_toggle",
            "block_reparent",
        }:
            return noop

        try:
            if kind == "issue_focus":
                new_state_dict = _state_with_issue_focus(state_data, viewport_op)
                if new_state_dict == state_data:
                    return noop
                new_state = state_from_json(new_state_dict)
                breadcrumb, canvas_elements, inspector = _render_views(
                    new_state
                )
                return (
                    new_state_dict,
                    breadcrumb,
                    canvas_elements,
                    inspector,
                    *((no_update,) * 5),
                )

            new_state_dict = _dispatch_state_update(
                state_data, kind, viewport_op
            )
            # Detect drill_to_scope stale-id rejection.  The dispatcher
            # leaves the breadcrumb unchanged + queues a toast; the
            # clientside scrim needs an immediate abort signal so it can
            # dismiss without waiting on the layout-stop timeout.
            if kind == "drill_to_scope":
                old_breadcrumb = state_data.get("breadcrumb") or []
                new_breadcrumb = new_state_dict.get("breadcrumb") or []
                if old_breadcrumb == new_breadcrumb:
                    # No breadcrumb change → either a toast was queued
                    # (the rejection path) or the payload was a no-op
                    # against the current state.  Either way, emit the
                    # abort sentinel so the scrim dismisses; the toast
                    # output below will surface the rejection text on
                    # the next fan-in tick.
                    aborted_payload = {
                        "kind": "scroll_to_aborted",
                        "ts": int(time.time() * 1000),
                    }
                    if new_state_dict == state_data:
                        # Nothing else changed — just dispatch the
                        # sentinel and bail.
                        return (*((no_update,) * 8), aborted_payload)
                    # State changed (toast queue grew) — render views
                    # + emit the abort sentinel together.
                    new_state = state_from_json(new_state_dict)
                    breadcrumb, canvas_elements, inspector = _render_views(
                        new_state
                    )
                    return (
                        new_state_dict,
                        breadcrumb,
                        canvas_elements,
                        inspector,
                        *((no_update,) * 4),
                        aborted_payload,
                    )
            if new_state_dict == state_data:
                return noop
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas_elements, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
                *((no_update,) * 5),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("viewport_op_fan_in failed")
            return (
                *((no_update,) * 4),
                True,
                _format_exception(exc),
                "danger",
                "Update failed",
                no_update,
            )

    # ----------------------------------------------------------------------
    # 2c-i-bis. STORE_VIEWPORT_OP → clientside visual ops + abort relay
    # ----------------------------------------------------------------------
    # The clientside ``viewport_ops.js`` exposes four canvas-only viewport
    # ops on ``window``: ``phenotypicScrollTo``, ``phenotypicRelayout``,
    # ``phenotypicReanchor``, and ``phenotypicScrollToAbortedRelay``.
    # ``STORE_VIEWPORT_OP`` carries the dispatch payload for each:
    #
    #   * ``scroll_to`` (from issue-row click, see
    #     :func:`issue_row_click_dispatch`) → invoke
    #     ``phenotypicScrollTo(block_id, scope_path, target_breadcrumb)``;
    #     the JS mounts a scrim, walks the expand chain, and emits
    #     ``phenotypic:scroll-to-complete`` on settle.
    #   * ``relayout`` (from ``BTN_RELAYOUT`` button) → invoke
    #     ``phenotypicRelayout()``; the JS reruns dagre + fits.
    #   * ``reanchor`` (programmatic) → invoke ``phenotypicReanchor()``.
    #   * ``scroll_to_aborted`` (written back by
    #     :func:`viewport_op_fan_in` on stale-id rejection) → invoke
    #     ``phenotypicScrollToAbortedRelay()`` to dispatch the
    #     ``phenotypic:scroll-to-aborted`` DOM event so the JS's
    #     ``waitForLayoutstopOrAbort`` race wakes up immediately.
    #
    # The fan-in callback handles the *state-mutating* kinds
    # (``drill_to_scope``, ``block_collapsed_toggle``, ...) on the
    # server side; this clientside relay handles the *visual* kinds
    # so the round-trip never blocks the user's pan/fit gesture.
    app.clientside_callback(
        """
        function(payload) {
            if (!payload || typeof payload !== 'object') {
                return window.dash_clientside.no_update;
            }
            var kind = payload.kind;
            if (
                kind === 'scroll_to_aborted'
                && typeof window.phenotypicScrollToAbortedRelay === 'function'
            ) {
                window.phenotypicScrollToAbortedRelay();
            } else if (
                kind === 'scroll_to'
                && typeof window.phenotypicScrollTo === 'function'
            ) {
                window.phenotypicScrollTo(
                    payload.block_id,
                    payload.scope_path || [],
                    payload.target_breadcrumb || []
                );
            } else if (
                kind === 'relayout'
                && typeof window.phenotypicRelayout === 'function'
            ) {
                window.phenotypicRelayout();
            } else if (
                kind === 'reanchor'
                && typeof window.phenotypicReanchor === 'function'
            ) {
                window.phenotypicReanchor();
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output(ids.STORE_CANVAS_CONTROL, "data", allow_duplicate=True),
        Input(ids.STORE_VIEWPORT_OP, "data"),
        prevent_initial_call=True,
    )


    # ----------------------------------------------------------------------
    # 2c-ii. Container delete two-stage flow (spec §5.6)
    # ----------------------------------------------------------------------
    #
    # The Delete button on a container surface emits a
    # ``block_delete_request`` payload (through ``STORE_EDGE_EVENT`` or
    # the canvas right-click menu).  The dispatcher inspects the
    # container's inner-block count and either:
    #
    #   * Short-circuits to ``block_delete_confirm`` immediately
    #     (empty container or non-container block); the deletion
    #     happens in the same dispatch tick.
    #   * Sets ``state.pending_delete_block_id = block_id`` so the
    #     ``CONFIRM_DELETE_MODAL_ID`` modal opens; the user clicks
    #     Confirm (dispatches ``block_delete_confirm``) or Cancel
    #     (clears the pending field).
    #
    # The two callbacks below wire the modal buttons.  The toggle
    # callback (``toggle_confirm_delete_modal``) keys the modal's
    # ``is_open`` to ``pending_delete_block_id`` being non-None.

    @app.callback(
        Output(ids.CONFIRM_DELETE_MODAL_ID, "is_open"),
        Output(f"{ids.CONFIRM_DELETE_MODAL_ID}-body", "children"),
        Input(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=False,
    )
    def toggle_confirm_delete_modal(
        state_data: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Any]:
        """Open the confirm-delete modal when ``pending_delete_block_id`` is set.

        Per spec §5.6 the modal's ``is_open`` is driven entirely by
        ``state.pending_delete_block_id``.  This callback fires on every
        state mutation, but the modal stays closed (``is_open=False``)
        unless the dispatcher set the pending id in this tick.

        Modal body shape (spec §5.6 "Confirm-delete modal" row):
            "Delete container '<label>' and its N inner block(s)?"
        where N excludes the auto-seeded ``InputImage`` sentinel.
        """

        if not isinstance(state_data, dict):
            return False, no_update
        pending = state_data.get("pending_delete_block_id")
        if not isinstance(pending, str) or not pending:
            return False, no_update
        root_scope = state_data.get("root")
        if not isinstance(root_scope, dict):
            return False, no_update
        linear_action, linear_payload = _parse_linear_pending_action(pending)
        if linear_action == "node" and isinstance(linear_payload, str):
            hit = _find_block_in_tree(root_scope, linear_payload)
            if hit is None:
                return False, no_update
            _scope, block = hit
            label = block.get("label") or block.get("class_name") or "node"
            body_text = (
                f"Delete '{label}' and any side parameter values attached "
                "to it?"
            )
            return True, html.Div(body_text)
        if linear_action == "clear":
            target = target_from_dict(linear_payload, [])
            scope_dict = _dag_scope_at_breadcrumb(
                state_data, list(target.scope_path)
            )
            label = "parameter value"
            if (
                scope_dict is not None
                and target.block_id is not None
                and target.param is not None
            ):
                block = _linear_block(scope_dict, target.block_id)
                if block is not None:
                    block_label = (
                        block.get("label")
                        or block.get("class_name")
                        or "node"
                    )
                    label = f"{target.param} on '{block_label}'"
            body_text = f"Clear embedded pipeline value for {label}?"
            return True, html.Div(body_text)
        hit = _find_block_in_tree(root_scope, pending)
        if hit is None:
            return False, no_update
        _scope, block = hit
        label = block.get("label") or block.get("class_name") or "container"
        nested = block.get("nested") or {}
        inner_blocks = nested.get("blocks", []) or []
        inner_count = sum(
            1 for b in inner_blocks
            if b.get("class_name") != INPUT_IMAGE_CLASS_NAME
        )
        body_text = (
            f"Delete container '{label}' and its {inner_count} inner "
            f"block(s)?"
        )
        return True, html.Div(body_text)

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
        Input(ids.BTN_CONFIRM_DELETE, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def confirm_delete_block(
        n_clicks: Optional[int], state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Dispatch ``block_delete_confirm`` for the pending container.

        Reads ``state.pending_delete_block_id`` and routes it through
        :func:`_dispatch_state_update`.  Clicking Confirm on a stale
        state (pending id no longer in the tree) is harmless — the
        dispatcher's missing-block branch clears the pending field and
        returns a clean state.
        """

        noop = (no_update,) * 4
        if not n_clicks or state_data is None:
            return noop
        pending = state_data.get("pending_delete_block_id")
        if not isinstance(pending, str) or not pending:
            return noop
        try:
            linear_action, linear_payload = _parse_linear_pending_action(pending)
            if linear_action == "node" and isinstance(linear_payload, str):
                dispatch_kind = "linear_delete_node_confirm"
                dispatch_payload = {
                    "kind": dispatch_kind,
                    "block_id": linear_payload,
                    "ts": int(time.time() * 1000),
                }
            elif linear_action == "clear":
                dispatch_kind = "linear_clear_param_confirm"
                dispatch_payload = {
                    "kind": dispatch_kind,
                    "target": linear_payload,
                    "ts": int(time.time() * 1000),
                }
            else:
                dispatch_kind = "block_delete_confirm"
                dispatch_payload = {
                    "kind": dispatch_kind,
                    "block_id": pending,
                    "ts": int(time.time() * 1000),
                }
            new_state_dict = _dispatch_state_update(
                state_data,
                dispatch_kind,
                dispatch_payload,
            )
            if new_state_dict == state_data:
                return noop
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas_elements, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
            )
        except Exception:  # noqa: BLE001
            logger.exception("confirm_delete_block failed")
            return noop

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Input(ids.BTN_CANCEL_DELETE, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def cancel_delete_block(
        n_clicks: Optional[int], state_data: Dict[str, Any]
    ) -> Any:
        """Clear ``pending_delete_block_id`` so the modal closes.

        No canvas re-render needed — only the modal-toggle callback
        cares about the pending field, and writing the state store
        re-triggers it.
        """

        if not n_clicks or state_data is None:
            return no_update
        if not isinstance(state_data, dict):
            return no_update
        if state_data.get("pending_delete_block_id") is None:
            return no_update
        new_state_dict = deepcopy(state_data)
        new_state_dict["pending_delete_block_id"] = None
        return new_state_dict

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(
            {"type": "breadcrumb-link", "depth": ALL}, "n_clicks"
        ),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def breadcrumb_drill_out_dag(
        _clicks: List[int], state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Pop the breadcrumb when a segment is clicked (DAG schema).

        The legacy fan-in already handles the breadcrumb under the
        ``breadcrumb_to`` kind, but the DAG schema stores breadcrumb
        entries as plain ``block_id`` strings (no ``depth`` dict) so
        this dedicated callback dispatches ``drill_out`` with the
        right depth.

        When the active schema is the legacy linear-list (``state_data``
        lacks ``root.blocks``), the callback short-circuits to
        ``no_update`` so the legacy fan-in's branch handles it instead.
        """

        noop = (no_update,) * 8
        if state_data is None or ctx.triggered_id is None:
            return noop
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return noop
        if triggered.get("type") != "breadcrumb-link":
            return noop
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return noop
        # DAG-schema check (duck-typing): legacy state has no "root.blocks".
        if not isinstance(state_data.get("root"), dict):
            return noop
        if "blocks" not in (state_data.get("root") or {}):
            return noop
        depth = triggered.get("depth")
        if not isinstance(depth, int) or depth < 0:
            return noop
        try:
            new_state_dict = _dispatch_state_update(
                state_data,
                "drill_to_scope",
                {
                    "kind": "drill_to_scope",
                    "target_breadcrumb": list(
                        state_data.get("breadcrumb", []) or []
                    )[:depth],
                    "ts": int(time.time() * 1000),
                },
            )
            if new_state_dict == state_data:
                return noop
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas_elements, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        except Exception:  # noqa: BLE001
            logger.exception("breadcrumb_drill_out_dag failed")
            return noop

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_DRILL_IN_CONTAINER, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def drill_in_container_button(
        n_clicks: Optional[int], state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Inspector container-card ``Drill in →`` → ``drill_into_container``.

        Distinct from the legacy ``BTN_DRILL_IN`` (which dispatches the
        ``drill_in`` kind against a legacy ``StepNode`` selection).
        The inspector card mounts the dedicated
        :data:`BTN_DRILL_IN_CONTAINER` button only when the selected
        block is a Pipeline container; this callback dispatches the
        ``drill_into_container`` kind against
        ``state.selected_block_id``.
        """

        noop = (no_update,) * 8
        if not n_clicks or state_data is None:
            return noop
        if not isinstance(state_data.get("root"), dict):
            return noop
        if "blocks" not in (state_data.get("root") or {}):
            return noop
        sel = state_data.get("selected_block_id")
        if not isinstance(sel, str) or not sel:
            return noop
        try:
            new_state_dict = _dispatch_state_update(
                state_data,
                "drill_into_container",
                {
                    "kind": "drill_into_container",
                    "block_id": sel,
                    "ts": int(time.time() * 1000),
                },
            )
            if new_state_dict == state_data:
                return noop
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas_elements, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        except Exception:  # noqa: BLE001
            logger.exception("drill_in_container_button failed")
            return noop

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_DELETE_NODE, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def delete_button_dag(
        n_clicks: Optional[int], state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Toolbar Delete button → ``block_delete_request`` (DAG schema).

        Routes the existing ``BTN_DELETE_NODE`` click through the new
        two-stage delete flow.  For non-container blocks the dispatcher
        short-circuits to ``block_delete_confirm`` in the same tick so
        the user-facing behaviour for ops is identical to the legacy
        flow (immediate deletion).  Empty containers also short-
        circuit; non-empty containers open the modal.

        Short-circuits to ``no_update`` for legacy state so the legacy
        fan-in's ``delete_node`` branch handles it instead.
        """

        noop = (no_update,) * 8
        if not n_clicks or state_data is None:
            return noop
        if not isinstance(state_data.get("root"), dict):
            return noop
        if "blocks" not in (state_data.get("root") or {}):
            return noop
        sel = state_data.get("selected_block_id")
        if not isinstance(sel, str) or not sel:
            return noop
        try:
            new_state_dict = _dispatch_state_update(
                state_data,
                "block_delete_request",
                {
                    "kind": "block_delete_request",
                    "block_id": sel,
                    "ts": int(time.time() * 1000),
                },
            )
            if new_state_dict == state_data:
                return noop
            new_state = state_from_json(new_state_dict)
            breadcrumb, canvas_elements, inspector = _render_views(new_state)
            return (
                new_state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
                no_update,
                no_update,
                no_update,
                no_update,
            )
        except Exception:  # noqa: BLE001
            logger.exception("delete_button_dag failed")
            return noop

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(ids.BTN_NEW_PIPELINE_NODE, "n_clicks"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def new_pipeline_palette_click_dag(
        n_clicks: Optional[int], state_data: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Retired DAG palette fallback kept inert for callback compatibility."""

        noop = (no_update,) * 8
        _ = (n_clicks, state_data)
        return noop

    # ----------------------------------------------------------------------
    # 3. Prefix preview / Run preview
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.STORE_INTERMEDIATE_KEYS, "data", allow_duplicate=True),
        Output(ids.STORE_PREVIEW_SNAPSHOT, "data", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Input(
            {
                "type": ids.LINEAR_NODE_ACTION,
                "surface": ALL,
                "action": ALL,
                "scope_path": ALL,
                "block_id": ALL,
            },
            "n_clicks",
        ),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        State(STORE_IMAGE_PATH, "data"),
        State(ids.INPUT_NROWS, "value"),
        State(ids.INPUT_NCOLS, "value"),
        prevent_initial_call=True,
    )
    def run_linear_prefix_preview(
        _node_action_clicks: List[int],
        state_data: Dict[str, Any],
        session_id: Optional[str],
        image_path: Optional[str],
        nrows: Optional[Any],
        ncols: Optional[Any],
    ) -> Tuple[Any, ...]:
        """Run ``Preview here`` for a linear map prefix only."""

        if not isinstance(ctx.triggered_id, dict):
            return (no_update,) * 6
        triggered = ctx.triggered_id
        if triggered.get("type") != ids.LINEAR_NODE_ACTION:
            return (no_update,) * 6
        if triggered.get("action") != "preview_here":
            return (no_update,) * 6
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update,) * 6
        if state_data is None:
            return no_update, no_update, *_toast("No state to preview", ok=False)
        if not session_id:
            session_id = uuid.uuid4().hex

        try:
            from phenotypic.abc_ import GridOperation

            t0 = time.time()
            pipeline_revision = _pipeline_revision(state_data)
            cache = get_cache()
            staging = cache.begin_preview_generation(
                session_id,
                pipeline_revision,
            )
            state = state_from_json(state_data)
            if not hasattr(state, "selected_block_id"):
                return no_update, no_update, *_toast(
                    "Preview here is only available in the linear builder.",
                    ok=False,
                )
            target = target_from_dict(
                _linear_preview_target_payload_from_id(triggered),
                state.breadcrumb,
            )
            prefix_state = _linear_prefix_state_for_preview(state, target)
            pipeline = to_pipeline_dag(prefix_state)
            uses_grid = _pipeline_uses_grid(pipeline, GridOperation)

            image = _load_preview_image(image_path, uses_grid, nrows, ncols)
            cache.set_image(session_id, image, str(image_path) if image_path else None)

            result = pipeline.apply_with_intermediates(image)
            generation = _bake_preview_cache(
                prefix_state,
                pipeline,
                result,
                session_id,
                cache,
                pipeline_revision=pipeline_revision,
                staging=staging,
            )
            if generation is None:
                return (no_update,) * 6

            duration = time.time() - t0
            keys = cache.known_intermediate_keys(session_id)
            return (
                keys,
                {
                    "pipeline_revision": pipeline_revision,
                    "preview_generation": generation,
                },
                *_toast(f"Prefix preview ran in {duration:.2f}s", ok=True),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prefix preview failed")
            return (
                no_update,
                no_update,
                *_toast(_format_exception(exc), ok=False),
            )

    @app.callback(
        Output(ids.STORE_INTERMEDIATE_KEYS, "data"),
        Output(ids.STORE_PREVIEW_SNAPSHOT, "data"),
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
    ) -> Tuple[Any, ...]:
        """Build the pipeline, run preview, cache intermediates.

        Spec §5.6 gate: before doing any work, filter validation
        issues to ``severity == "error"`` and short-circuit with a
        validation toast when any blocking error is present.  Advisory
        hints (``severity == "advisory"``) NEVER block — they decorate
        the canvas and badge but the user can still preview.
        """

        if not n_clicks or state_data is None:
            return no_update, no_update, *_toast("No state to preview", ok=False)

        # Run preview gating — spec §5.6.
        errors = _filter_blocking_issues(state_data)
        if errors:
            return (
                no_update,
                no_update,
                *_gate_toast_for_issue("run preview", errors[0]),
            )

        if not session_id:
            session_id = uuid.uuid4().hex

        try:
            from phenotypic.abc_ import GridOperation

            t0 = time.time()
            pipeline_revision = _pipeline_revision(state_data)
            cache = get_cache()
            staging = cache.begin_preview_generation(
                session_id,
                pipeline_revision,
            )
            state = state_from_json(state_data)
            # Duck-type the schema: DAG state needs the DAG converter
            # (which walks ``state.root.blocks`` + ``edges`` topologically);
            # the legacy converter walks ``state.root.nodes`` and would
            # AttributeError on a ``_DagBuilderScope``.
            if hasattr(state, "selected_block_id"):
                pipeline = to_pipeline_dag(state)
            else:
                pipeline = to_pipeline(state.root)
            uses_grid = _pipeline_uses_grid(pipeline, GridOperation)

            image = _load_preview_image(image_path, uses_grid, nrows, ncols)

            cache.set_image(session_id, image, str(image_path) if image_path else None)

            result = pipeline.apply_with_intermediates(image)
            generation = _bake_preview_cache(
                state,
                pipeline,
                result,
                session_id,
                cache,
                pipeline_revision=pipeline_revision,
                staging=staging,
            )
            if generation is None:
                return (no_update,) * 6

            duration = time.time() - t0
            keys = cache.known_intermediate_keys(session_id)
            return (
                keys,
                {
                    "pipeline_revision": pipeline_revision,
                    "preview_generation": generation,
                },
                *_toast(f"Preview ran in {duration:.2f}s", ok=True),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Run preview failed")
            return (
                no_update,
                no_update,
                *_toast(_format_exception(exc), ok=False),
            )

    # ----------------------------------------------------------------------
    # 4. Inspector preview rendering
    # ----------------------------------------------------------------------

    # Plain process-local memoization is safe because the Builder remains a
    # deliberate single-process deployment. Object identity is not part of
    # the contract: a preview is identified only by its semantic revision
    # and atomically published generation.
    _preview_render_keys: Dict[str, PreviewKey] = {}

    @app.callback(
        Output(ids.INSPECTOR_PREVIEW, "children"),
        Input(ids.STORE_BUILDER_STATE, "data"),
        Input(ids.STORE_PREVIEW_SNAPSHOT, "data"),
        State(ids.STORE_SESSION_ID, "data"),
    )
    def render_inspector_preview(
        state_data: Optional[Dict[str, Any]],
        preview_snapshot: Optional[Dict[str, Any]],
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

        pipeline_revision = _pipeline_revision(state_data)
        published_revision: Optional[str] = None
        preview_generation: Optional[int] = None
        if isinstance(preview_snapshot, dict):
            raw_revision = preview_snapshot.get("pipeline_revision")
            raw_generation = preview_snapshot.get("preview_generation")
            if isinstance(raw_revision, str) and type(raw_generation) is int:
                published_revision = raw_revision
                preview_generation = raw_generation
        if (
            published_revision is not None
            and published_revision != pipeline_revision
        ):
            _preview_render_keys.pop(session_id, None)
            return html.Div("Preview stale - run again", className="text-muted")

        # Duck-type the selection id: DAG state exposes
        # ``selected_block_id``, legacy state exposes
        # ``selected_node_id``. Both shapes share the preview cache by
        # the same string id (block_id / node_id are interchangeable
        # opaque strings keyed into ``IntermediatesCache``).
        selected_id = getattr(
            state,
            "selected_block_id",
            getattr(state, "selected_node_id", None),
        )

        if selected_id is None:
            _preview_render_keys.pop(session_id, None)
            return html.Div(
                "Select a node to view its preview.",
                className="text-muted",
            )

        # Verify the selection still exists in the state tree. For DAG
        # state walk the block tree by id; for legacy state walk the
        # breadcrumb-resolved scope's node list.
        if hasattr(state, "selected_block_id"):
            root_dict = state_data.get("root")
            if not isinstance(root_dict, dict):
                return no_update
            if _find_block_in_tree(root_dict, selected_id) is None:
                return no_update
        else:
            try:
                scope = current_scope(state)
            except KeyError:
                return no_update
            node = next(
                (n for n in scope.nodes if n.node_id == selected_id),
                None,
            )
            if node is None:
                return no_update

        if published_revision is None or preview_generation is None:
            _preview_render_keys.pop(session_id, None)
            return html.Div(
                "No preview yet — click Run preview.",
                className="text-muted",
            )

        preview_key: PreviewKey = (
            session_id,
            selected_id,
            pipeline_revision,
            preview_generation,
        )
        last = _preview_render_keys.get(session_id)
        if last == preview_key:
            return no_update
        _preview_render_keys[session_id] = preview_key
        cached = get_cache().get_preview(preview_key)
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
            selected_id,
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
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def open_save_modal(
        n_clicks: Optional[int], source_payload: object
    ) -> Tuple[Any, Any]:
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
        return True, _browse_seed_from_source(image_root, source_payload)

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

        # Save gating — spec §5.6.  Mirror the Run preview filter so
        # the user sees the same "Cannot <action>" toast and the
        # modal stays open for them to fix the issue.  Advisory hints
        # never block save (yellow-border decorations are persisted
        # alongside the rest of the pipeline JSON).
        errors = _filter_blocking_issues(state_data)
        if errors:
            return (
                no_update,
                *_gate_toast_for_issue("save pipeline", errors[0]),
            )

        try:
            # Aux nodes are now embedded inside each consumer's
            # ``aux_ports`` map (Wave 1-A onward), so "orphan aux" is
            # structurally impossible — an aux only exists while wired.
            # The pre-save orphan walk that used to live here has been
            # removed.
            state = state_from_json(state_data)
            if hasattr(state, "selected_block_id"):
                pipeline = to_pipeline_dag(state)
            else:
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

            target = _write_pipeline_config(pipeline, target)
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
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def open_load_picker(
        n_clicks: Optional[int], source_payload: object
    ) -> Tuple[Any, ...]:
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
        return True, "chooser", _browse_seed_from_source(image_root, source_payload)

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
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
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
            Ten-tuple ``(browse_dir, state_data, breadcrumb,
            canvas_elements, inspector, modal_is_open, toast_is_open,
            toast_msg, toast_icon, toast_header)``. Directory clicks
            populate only the first element; file clicks populate
            elements 2–10.
        """
        match = _trigger_kind_path(ctx.triggered_id, ids.DIR_ENTRY_TYPE_JSON)
        if match is None:
            return (no_update,) * 10
        kind, path_str = match
        if kind in {"dir", "parent"}:
            return (path_str, *((no_update,) * 9))

        if kind == "file":
            try:
                from phenotypic import ImagePipeline

                with open(Path(path_str).expanduser(), encoding="utf-8") as fh:
                    pipeline = ImagePipeline.from_json(fh.read())
                (
                    state_dict,
                    breadcrumb,
                    canvas_elements,
                    inspector,
                ) = _state_replacement_payload(pipeline)
                return (
                    no_update,
                    state_dict,
                    breadcrumb,
                    canvas_elements,
                    inspector,
                    False,
                    *_toast(f"Loaded {Path(path_str).name}", ok=True),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Load JSON failed")
                return (
                    (no_update,) * 6 + _toast(_format_exception(exc), ok=False)
                )

        return (no_update,) * 10

    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Output(ids.BREADCRUMB_CONTAINER, "children", allow_duplicate=True),
        Output(ids.LINEAR_MAP_CONTAINER, "children", allow_duplicate=True),
        Output(ids.INSPECTOR_CONTENT, "children", allow_duplicate=True),
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
            Nine-tuple ``(state_data, breadcrumb, canvas_elements,
            inspector, modal_is_open, toast_is_open, toast_msg,
            toast_icon, toast_header)``.
        """
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered.get("type") != "prefab-card":
            return (no_update,) * 9
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return (no_update,) * 9
        class_name = triggered.get("class_name")
        if not class_name:
            return (no_update,) * 9

        try:
            import phenotypic.prefab as prefab_module

            pipeline = getattr(prefab_module, class_name)()
            (
                state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
            ) = _state_replacement_payload(pipeline)
            return (
                state_dict,
                breadcrumb,
                canvas_elements,
                inspector,
                False,
                *_toast(f"Loaded prefab {class_name}", ok=True),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load prefab failed")
            return (no_update,) * 5 + _toast(_format_exception(exc), ok=False)

    # ----------------------------------------------------------------------
    # 7. Load Image modal — open / dir-nav / file pick / synthetic shortcut
    # ----------------------------------------------------------------------

    @app.callback(
        Output(ids.MODAL_LOAD_IMAGE, "is_open", allow_duplicate=True),
        Output(ids.STORE_BROWSE_DIR_IMAGE, "data", allow_duplicate=True),
        Input(ids.BTN_LOAD_IMAGE, "n_clicks"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        prevent_initial_call=True,
    )
    def open_load_image_modal(
        n_clicks: Optional[int], source_payload: object
    ) -> Tuple[Any, Any]:
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
        return True, _browse_seed_from_source(image_root, source_payload)

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
    # 8b. Retired canvas element application
    # ----------------------------------------------------------------------
    # The legacy Cytoscape bridge remains mounted for transition safety,
    # but default builder mutation callbacks now redraw the HTML linear
    # map directly via ``LINEAR_MAP_CONTAINER.children``.

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
    # 9b. Node-output preview modal (zoomable OSD viewer + layer toggle).
    # ----------------------------------------------------------------------
    @app.callback(
        Output(ids.MODAL_NODE_PREVIEW, "is_open", allow_duplicate=True),
        Output(ids.STORE_PREVIEW_TARGET, "data"),
        Input({"type": ids.LINEAR_NODE_ACTION, "surface": ALL, "action": "preview",
               "scope_path": ALL, "block_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_node_preview(_clicks):
        if not isinstance(ctx.triggered_id, dict) or not ctx.triggered \
                or not ctx.triggered[0].get("value"):
            return no_update, no_update
        tid = ctx.triggered_id
        scope_path = _decode_linear_scope_path(tid.get("scope_path"))
        block_id = _decode_linear_optional(tid.get("block_id"))
        return True, {"block_id": block_id, "scope_path": scope_path}

    @app.callback(
        Output(ids.PREVIEW_LAYER_RADIO, "options"),
        Output(ids.PREVIEW_LAYER_RADIO, "value"),
        Output(ids.MODAL_NODE_PREVIEW_TITLE, "children"),
        Output(ids.PREVIEW_CAPTION, "children"),
        Output(ids.PREVIEW_DZI_URL_STORE, "data"),
        Input(ids.STORE_PREVIEW_TARGET, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        State(ids.STORE_IMAGE_PATH, "data"),
        State(ids.INPUT_NROWS, "value"),
        State(ids.INPUT_NCOLS, "value"),
        prevent_initial_call=True,
    )
    def compute_node_preview(target, session_id, state_data, image_path,
                             nrows, ncols):
        # ``init_session_id`` always populates STORE_SESSION_ID before a preview
        # click, so a falsy session_id here means the store hasn't initialised
        # yet — bail rather than minting an orphan id the layer toggle can't reuse.
        if not target or not state_data or not session_id:
            return no_update, no_update, no_update, no_update, no_update
        url_prefix = current_app.config.get(CFG_URL_PREFIX, "/")
        try:
            payload = build_preview_payload(
                session_id=session_id, state_data=state_data,
                block_id=target["block_id"], scope_path=target["scope_path"],
                image_path=image_path, nrows=nrows, ncols=ncols,
                url_prefix=url_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Node preview failed")
            return [], None, "Preview", _format_exception(exc), None
        return (payload["options"], payload["value"], payload["title"],
                payload["caption"], payload["dzi_url"])

    @app.callback(
        Output(ids.PREVIEW_DZI_URL_STORE, "data", allow_duplicate=True),
        Output(ids.PREVIEW_CAPTION, "children", allow_duplicate=True),
        Input(ids.PREVIEW_LAYER_RADIO, "value"),
        State(ids.STORE_PREVIEW_TARGET, "data"),
        State(ids.STORE_SESSION_ID, "data"),
        prevent_initial_call=True,
    )
    def switch_preview_layer(channel, target, session_id):
        if not channel or not target or not session_id:
            return no_update, no_update
        scope_path = target["scope_path"]
        shash = pc.scope_hash(scope_path)
        url_prefix = current_app.config.get(CFG_URL_PREFIX, "/")
        url = preview_dzi_url(url_prefix, session_id, shash, target["block_id"], channel)
        # Keep the W×H prefix consistent with compute_node_preview's caption.
        manifest = pc.read_manifest(session_id, scope_path) or {}
        node = manifest.get("nodes", {}).get(target["block_id"], {})
        h, w = node.get("shape", [0, 0])
        return url, f"{w}×{h} · {channel}"

    @app.callback(
        Output(ids.MODAL_NODE_PREVIEW, "is_open", allow_duplicate=True),
        Input("btn-preview-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_node_preview(_clicks):
        return False

    app.clientside_callback(
        """
        function(dziUrl) {
            const ns = window.__phenotypicNodePreview;
            if (!ns || !ns.mountViewer) { return window.dash_clientside.no_update; }
            if (!dziUrl) { if (ns.disposeViewer) ns.disposeViewer(); return window.dash_clientside.no_update; }
            requestAnimationFrame(function () { ns.mountViewer("preview-osd", dziUrl); });
            return Date.now();
        }
        """,
        Output(ids.PREVIEW_OSD_MOUNT_TRIGGER, "data", allow_duplicate=True),
        Input(ids.PREVIEW_DZI_URL_STORE, "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(isOpen) {
            const ns = window.__phenotypicNodePreview;
            if (!ns || !ns.disposeViewer) { return window.dash_clientside.no_update; }
            if (!isOpen) { ns.disposeViewer(); }
            return Date.now();
        }
        """,
        Output(ids.PREVIEW_OSD_MOUNT_TRIGGER, "data", allow_duplicate=True),
        Input(ids.MODAL_NODE_PREVIEW, "is_open"),
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

    # ----------------------------------------------------------------------
    # 11. Asset-status banner wiring.
    # ----------------------------------------------------------------------
    # Subscribes to ``STORE_ASSET_STATUS`` (written by ``assets/builder.js``'s
    # readiness-poll loop with shape
    # ``{"wire_drawing": bool, "palette_dnd": bool, "viewport_ops": bool,
    # "dagre_missing": bool}``).  These assets now power retired Cytoscape
    # affordances, so missing readiness must not disable the click-only
    # palette or show irrelevant drag/wire warnings in the linear builder.
    # The hidden relayout button remains mounted only as a callback anchor.
    @app.callback(
        Output(ids.BTN_RELAYOUT, "disabled"),
        Output(ids.PALETTE_CONTAINER, "style", allow_duplicate=True),
        Output(ids.BANNER_ASSET_STATUS, "children"),
        Output(ids.BANNER_ASSET_STATUS, "style", allow_duplicate=True),
        Input(ids.STORE_ASSET_STATUS, "data"),
        prevent_initial_call=True,
    )
    def asset_status_disables(
        _status: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Dict[str, Any], List[Any], Dict[str, Any]]:
        """Keep retired asset readiness signals inert in the linear builder.

        Args:
            status: Dict written by the clientside readiness poll —
                ``{"wire_drawing": bool, "palette_dnd": bool,
                "viewport_ops": bool, "dagre_missing": bool}``.  ``None``
                during the first ~500ms of page load (before the poll
                completes).

        Returns:
            Tuple of (hidden relayout disabled, palette style override,
            banner children, banner style).  The hidden relayout anchor
            stays disabled and the click palette remains server-routed.
        """

        return True, {}, [], {"display": "none"}

    # ----------------------------------------------------------------------
    # 12. Toolbar Re-layout / Inspector Re-anchor → STORE_VIEWPORT_OP
    # ----------------------------------------------------------------------
    # Spec §5.5: clicking ``BTN_RELAYOUT`` writes a ``{kind: "relayout"}``
    # payload to ``STORE_VIEWPORT_OP``; ``viewport_ops.js`` mirrors the
    # payload into ``window.phenotypicRelayout``.  The inspector's
    # Re-layout label button (no Dash id — cosmetic only) forwards via
    # a separate document-delegated click listener emitted alongside
    # the cytoscape stylesheet so the spec-required affordance still
    # works from the inspector pane.
    #
    # The Re-anchor button (``BTN_REANCHOR`` — Input Image card only)
    # writes a ``{kind: "reanchor"}`` payload.  Both flow through the
    # ``viewport_op_fan_in`` callback above, which short-circuits
    # purely visual ops to the clientside relay.
    app.clientside_callback(
        """
        function(relayout_clicks, reanchor_clicks, prev) {
            const trig = window.dash_clientside.callback_context.triggered;
            if (!trig || !trig.length) return window.dash_clientside.no_update;
            const id = trig[0].prop_id.split('.')[0];
            const ts = Date.now();
            if (id === 'btn-relayout' && relayout_clicks) {
                return { kind: 'relayout', ts: ts };
            }
            if (id === 'btn-reanchor' && reanchor_clicks) {
                return { kind: 'reanchor', ts: ts };
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output(ids.STORE_VIEWPORT_OP, "data", allow_duplicate=True),
        Input(ids.BTN_RELAYOUT, "n_clicks"),
        Input(ids.BTN_REANCHOR, "n_clicks"),
        State(ids.STORE_VIEWPORT_OP, "data"),
        prevent_initial_call=True,
    )

    # Spec §4.10: Run preview and Save stay disabled while STORE_ISSUES
    # carries any severity=error entry, AND briefly during the
    # state-mutate → validate window (one-frame debounce).  The first
    # branch reads STORE_ISSUES.data and disables on any error.  The
    # second branch uses STORE_BUILDER_STATE as a "validation in flight"
    # signal: a state mutation triggers revalidate_on_state_change which
    # writes STORE_ISSUES; until that write lands, the buttons stay
    # disabled.  Combined, the two inputs gate both the steady-state
    # (errors present) and the transient (validation pending) cases.
    app.clientside_callback(
        """
        function(issues, state) {
            if (!Array.isArray(issues)) issues = [];
            const has_error = issues.some(function (i) {
                return i && i.severity === 'error';
            });
            return [has_error, has_error];
        }
        """,
        Output(ids.BTN_RUN_PREVIEW, "disabled", allow_duplicate=True),
        Output(ids.BTN_SAVE, "disabled", allow_duplicate=True),
        Input(ids.STORE_ISSUES, "data"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )

    # Spec §4.5: "Selecting a different block carries over the
    # inspector's scroll position so users who are comparing two ops
    # don't lose their place."  The inspector container re-renders on
    # every block_select; without intervention scrollTop resets to 0.
    # A MutationObserver on INSPECTOR_CONTAINER captures scrollTop
    # before the subtree is replaced and restores it after the new
    # children mount.  Idempotent via document-level binding flag.
    app.clientside_callback(
        """
        function (_n) {
            if (window.__phenoInspectorScrollBound) {
                return window.dash_clientside.no_update;
            }
            window.__phenoInspectorScrollBound = true;
            let lastScrollTop = 0;
            function attach() {
                const container = document.getElementById('inspector');
                if (!container) {
                    setTimeout(attach, 200);
                    return;
                }
                // Capture scrollTop on every user scroll.
                container.addEventListener('scroll', function () {
                    lastScrollTop = container.scrollTop;
                });
                // After Dash replaces the subtree, restore scrollTop on
                // the next animation frame so the restored content is
                // measurable.
                const observer = new MutationObserver(function () {
                    if (lastScrollTop > 0) {
                        requestAnimationFrame(function () {
                            container.scrollTop = lastScrollTop;
                        });
                    }
                });
                observer.observe(container, {
                    childList: true,
                    subtree: false,
                });
            }
            attach();
            return window.dash_clientside.no_update;
        }
        """,
        Output(ids.INSPECTOR_CONTAINER, "data-scroll-init", allow_duplicate=True),
        Input(ids.INSPECTOR_CONTAINER, "id"),
        prevent_initial_call="initial_duplicate",
    )

    # ----------------------------------------------------------------------
    # 13. Toast queue consumer (spec §5.1)
    # ----------------------------------------------------------------------
    #
    # ``BuilderState.toast_queue`` is a FIFO list of ``{kind, text}``
    # payloads enqueued by dispatch helpers (``_queue_toast``).  The
    # spec calls for one toast visible at a time, 3000ms auto-dismiss,
    # dismissable on click, with FIFO drain semantics.  We implement
    # the visible-at-a-time contract by surfacing only the queue's
    # head on every state change and clearing it after a 3000ms
    # ``dbc.Toast`` ``duration`` (already configured on the existing
    # ``TOAST_NOTIFICATION`` component at layout-build time; we don't
    # rebuild the toast widget).  Click-dismiss is the default
    # ``dismissable=True`` on the toast — that fires the same close
    # action that the duration timer would.
    #
    # When the toast closes (auto or click), this callback pops the
    # head off ``toast_queue`` and writes the trimmed list back to
    # ``STORE_BUILDER_STATE`` so the next queued payload surfaces on
    # the next state mutation.  Two near-simultaneous mutations queue
    # rather than race: the queue list is mutation-only (append on
    # enqueue, pop-head on consume) and there is at most one consumer
    # active at a time per session.
    #
    # The toast is rendered with the standard ``_toast`` helper so the
    # icon / header / colour conventions are consistent with the
    # direct-toast paths (``run_preview``, ``save_pipeline``, etc.).
    @app.callback(
        Output(ids.TOAST_NOTIFICATION, "is_open", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "children", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "icon", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "header", allow_duplicate=True),
        Output(ids.TOAST_NOTIFICATION, "duration", allow_duplicate=True),
        Input(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def surface_toast_queue_head(
        state_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Surface the head of ``state.toast_queue`` on the live toast.

        Per spec §5.1: one toast visible at a time, auto-dismiss after
        :data:`_TOAST_QUEUE_DURATION_MS`, FIFO order.  This callback
        subscribes to every :data:`STORE_BUILDER_STATE` write and shows
        the head when the queue is non-empty.  When the queue is empty
        (typical case), the call short-circuits with ``no_update`` so
        other toast sources (Run preview / Save / direct-error toasts)
        are not clobbered.

        Args:
            state_data: ``state_to_json`` payload from
                :data:`ids.STORE_BUILDER_STATE`.  ``None`` during the
                first paint, before any state has been published.

        Returns:
            Five-tuple matching the standard toast outputs plus the
            ``duration`` slot.  ``no_update`` on every output when the
            queue is empty.
        """

        queue = _toast_queue_from_state(state_data)
        if not queue:
            return (no_update,) * 5
        head = queue[0]
        if not isinstance(head, dict):
            return (no_update,) * 5
        text = str(head.get("text", ""))
        kind = str(head.get("kind", "info"))
        ok = kind not in {"error", "warning"}
        # _toast returns (is_open, children, icon, header) — splice
        # the queue-toast duration on so it auto-dismisses per spec.
        toast_outputs = _toast(text, ok=ok)
        return (*toast_outputs, _TOAST_QUEUE_DURATION_MS)

    # ----------------------------------------------------------------------
    # 14. Toast dismiss → pop the queue head
    # ----------------------------------------------------------------------
    #
    # When the user clicks dismiss (or the 3000ms timer fires), the
    # ``dbc.Toast`` flips ``is_open`` from True → False.  We listen
    # for that transition and rewrite ``STORE_BUILDER_STATE`` with the
    # queue's head popped so the next queued payload can surface on
    # the next state mutation.
    #
    # Edge case: the toast may flip ``is_open=False`` because *this*
    # callback just wrote ``is_open=True`` and then the timer fired —
    # we need to be tolerant of the empty-queue / out-of-sync case
    # where the queue is already empty (the toast was set by a direct
    # _toast call, not by surface_toast_queue_head).  We short-circuit
    # with ``no_update`` when the queue is empty so we don't churn
    # state on every unrelated toast dismiss.
    @app.callback(
        Output(ids.STORE_BUILDER_STATE, "data", allow_duplicate=True),
        Input(ids.TOAST_NOTIFICATION, "is_open"),
        State(ids.STORE_BUILDER_STATE, "data"),
        prevent_initial_call=True,
    )
    def pop_toast_queue_on_dismiss(
        is_open: Optional[bool],
        state_data: Optional[Dict[str, Any]],
    ) -> Any:
        """Pop the toast queue head when the toast closes.

        Spec §5.1: dismissable on user click, auto-dismiss after
        :data:`_TOAST_QUEUE_DURATION_MS`, FIFO drain.  Both close paths
        flip ``is_open=False`` on the :data:`ids.TOAST_NOTIFICATION`
        component; this callback listens for that transition and trims
        the queue.

        Args:
            is_open: New value of ``TOAST_NOTIFICATION.is_open``.
                ``True`` means a toast just opened (no-op); ``False``
                means a dismiss / timeout fired and we should pop.
            state_data: Live ``STORE_BUILDER_STATE`` dump.

        Returns:
            The mutated state dict (with the queue head removed), or
            :data:`no_update` when no pop is needed.
        """

        if is_open:
            return no_update
        queue = _toast_queue_from_state(state_data)
        if not queue:
            return no_update
        # ``state_data`` is dict here — the helper returns an empty
        # queue when the payload is non-dict, so we already short-
        # circuited above.
        new_state = dict(state_data)  # type: ignore[arg-type]
        new_state["toast_queue"] = queue[1:]
        return new_state


# ---------------------------------------------------------------------------
# Helpers (private to this module)
# ---------------------------------------------------------------------------


def _bake_preview_cache(
    state: "BuilderState",
    pipeline: Any,
    result: Any,
    session_id: str,
    cache: Any,
    *,
    pipeline_revision: Optional[str] = None,
    staging: Optional[PreviewGenerationWriter] = None,
) -> Optional[int]:
    """Render every intermediate to PNG bytes (or DataFrame) into *cache*.

    Pulled out of ``run_preview`` so the cache contract — bytes for ops,
    DataFrame for meas/post, ``PreviewRenderError`` on render failure — can
    be exercised end-to-end without booting a Dash server.

    Two schemas are supported:

    * **DAG schema** — walks the topological image-flow order produced by
      :func:`to_pipeline_dag` and keys the cache by 32-char
      ``BlockNode.block_id``. For nested containers, recurses into each
      container's inner pipeline. Aux-only blocks have no main-flow
      preview; the inspector shows their preview via the consumer's
      intermediate cache entry instead.
    * **Legacy schema** — walks ``state.root.nodes`` in declaration order
      and keys the cache by 8-char ``StepNode.node_id``. Retained for the
      back-compat migration fixtures (``test_legacy_pipeline_json``); no
      active runtime callback feeds legacy state through this helper
      after Phase 8 retired the feature flag.

    Args:
        state: The deserialised :class:`BuilderState` driving the preview.
        pipeline: The compiled :class:`ImagePipeline`.
        result: The :class:`IntermediateResult` returned by
            :meth:`ImagePipeline.apply_with_intermediates`.
        session_id: Per-tab uuid keying the cache.
        cache: The :class:`IntermediatesCache` to populate.
        pipeline_revision: Semantic state revision associated with the bake.
            Derived from ``state`` when omitted by direct test callers.
        staging: Request-sequenced detached writer reserved before expensive
            preview work. Direct callers may omit it to reserve immediately.

    Returns:
        Atomically published preview generation, or ``None`` when superseded.
    """

    revision = pipeline_revision or _pipeline_revision(state_to_json(state))
    if staging is None:
        staging = cache.begin_preview_generation(session_id, revision)
    elif (
        staging.session_id != session_id
        or staging.pipeline_revision != revision
    ):
        raise ValueError("preview staging writer does not match bake request")
    if hasattr(state, "selected_block_id"):
        _bake_preview_cache_dag(state, pipeline, result, session_id, staging)
    else:
        _bake_preview_cache_legacy(state, pipeline, result, session_id, staging)
    return cache.publish_preview_generation(staging)


def _bake_preview_cache_legacy(
    state: "BuilderState",
    pipeline: Any,
    result: Any,
    session_id: str,
    cache: Any,
) -> None:
    """Bake previews for a legacy linear-list :class:`BuilderState`.

    See :func:`_bake_preview_cache` for the full contract.
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
        meas_nodes = [
            n
            for n in state.root.nodes
            if stage_of(n.class_name) in {"meas", "post"}
        ]
        try:
            df = pipeline.measure(result.image)
            for node in meas_nodes:
                cache.set_intermediate(session_id, node.node_id, df)
        except Exception as meas_exc:  # noqa: BLE001
            logger.warning("measure() failed: %s", meas_exc)
            error = PreviewRenderError(_format_exception(meas_exc))
            for node in meas_nodes:
                cache.set_intermediate(session_id, node.node_id, error)


def _bake_preview_cache_dag(
    state: Any,
    pipeline: Any,
    result: Any,
    session_id: str,
    cache: Any,
) -> None:
    """Bake previews for a DAG :class:`BuilderState`.

    Walks the root scope's image-flow topological order (the same order
    :func:`to_pipeline_dag` materialises) and keys the cache by the
    32-char ``BlockNode.block_id`` so the inspector's block-selection
    lookup hits without an additional id translation step.

    Aux-only blocks are skipped (they have no main-flow intermediate);
    their inspector preview is the consumer's intermediate. Nested
    containers don't recurse into their inner blocks here — the outer
    pipeline's intermediate for a container reflects the *whole*
    nested pipeline's output. Inner-block previews are regenerated on
    demand the next time the user drills in and re-runs ``Run preview``
    (see spec §6 row + Phase 7 "invalidate old caches at startup"
    note).
    """

    from phenotypic.gui.builder._conversion_dag import (
        _find_input_block,
        _topological_image_order,
    )

    try:
        input_block = _find_input_block(state.root)
    except ValueError:
        return
    order = _topological_image_order(state.root, input_block)
    non_input_blocks = [b for b in order if b.block_id != input_block.block_id]

    ops_blocks = [
        b
        for b in non_input_blocks
        if b.class_name == PIPELINE_CLASS_NAME
        or stage_of(b.class_name) == "ops"
    ]

    for op_key, block in zip(pipeline.get_ops().keys(), ops_blocks):
        inter = result.intermediates.get(op_key)
        if inter is None:
            continue
        try:
            png = render_node_preview(inter, block.class_name)
        except Exception as render_exc:  # noqa: BLE001
            logger.warning(
                "Preview render failed for %s (%s): %s",
                block.class_name, block.block_id, render_exc,
            )
            cache.set_intermediate(
                session_id,
                block.block_id,
                PreviewRenderError(_format_exception(render_exc)),
            )
            continue
        cache.set_intermediate(session_id, block.block_id, png)

    if pipeline.get_meas() or pipeline.get_post():
        meas_blocks = [
            b
            for b in non_input_blocks
            if stage_of(b.class_name) in {"meas", "post"}
        ]
        try:
            df = pipeline.measure(result.image)
            for block in meas_blocks:
                cache.set_intermediate(session_id, block.block_id, df)
        except Exception as meas_exc:  # noqa: BLE001
            logger.warning("measure() failed: %s", meas_exc)
            error = PreviewRenderError(_format_exception(meas_exc))
            for block in meas_blocks:
                cache.set_intermediate(session_id, block.block_id, error)


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
    # DAG state uses block_id as the prefix and lives in ``root.blocks``;
    # legacy state walks the breadcrumb-resolved scope's nodes list.
    if _is_dag_state_dict(state_data):
        root_dict = state_data.get("root")
        if not isinstance(root_dict, dict):
            return state_data
        hit = _find_block_in_tree(root_dict, prefix)
        if hit is None:
            return state_data
        class_name = hit[1].get("class_name", "")
    else:
        state = state_from_json(state_data)
        try:
            scope = current_scope(state)
        except KeyError:
            return state_data
        node = next((n for n in scope.nodes if n.node_id == prefix), None)
        if node is None:
            return state_data
        class_name = node.class_name
    info = get_registry().get(class_name)
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
