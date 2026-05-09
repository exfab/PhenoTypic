"""Pure-Python state model for the Dash pipeline builder.

This module defines the dataclasses that back the builder canvas and the
pure functions that translate between that in-memory state and a
:class:`phenotypic.ImagePipeline`.  The data structures are intentionally
JSON-friendly so they can travel through ``dcc.Store`` without losing
fidelity.

The module imports only stdlib + ``phenotypic``; no Dash dependencies.

Examples:
    Build a tiny scope and round-trip it through an ``ImagePipeline``:

    >>> from phenotypic.gui.builder._state import (
    ...     BuilderScope, StepNode, to_pipeline, from_pipeline,
    ... )
    >>> scope = BuilderScope(
    ...     nodes=[StepNode(node_id="a1", class_name="GaussianBlur",
    ...                     params={"sigma": 1.5})],
    ...     name="demo",
    ... )
    >>> pipeline = to_pipeline(scope)
    >>> from_pipeline(pipeline).nodes[0].class_name
    'GaussianBlur'
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from phenotypic.gui.builder._ids import StageName

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageOperation, MeasureFeatures, PostMeasurement
from phenotypic.gui._operation_registry import (
    OperationRegistry,
    get_registry,
)

logger = logging.getLogger(__name__)


# Sentinel class name used for nested ``ImagePipeline`` step nodes.
PIPELINE_CLASS_NAME = "ImagePipeline"


@dataclass
class StepNode:
    """One step in a builder canvas.

    A ``StepNode`` is the canvas-layer counterpart of a single operation in an
    :class:`~phenotypic.ImagePipeline`.  It captures the class to instantiate,
    the parameter values entered by the user, and (when the step is itself a
    nested pipeline) the inner :class:`BuilderScope`.

    Attributes:
        node_id: Short, stable identifier (8-char hex slice of a UUID4).
            Used as the cytoscape node id and to address sub-scopes via the
            breadcrumb path.
        class_name: Registry key (e.g. ``"GaussianBlur"``) or the sentinel
            ``"ImagePipeline"`` when the node represents a nested pipeline.
        params: Raw parameter values.  Scalars are stored as-is; values
            corresponding to operation-typed parameters are stored as JSON
            dicts shaped like ``{"__type__": "operation", "class_name": ...,
            "params": {...}}``.
        label: User-editable display name.  ``None`` means the canvas should
            fall back to ``class_name``.
        nested: Inner :class:`BuilderScope` populated only when
            ``class_name == "ImagePipeline"``.
        aux_ports: Per-aux-port slot occupancy. Keys are the names of
            aux-port-eligible parameters (those with
            ``param_info.is_operation or param_info.is_pipeline``).  Values
            are lists of aux-source ``node_id`` strings (or ``None`` for an
            empty slot).  Non-list ports always carry a length-1 list
            (``[aux_id]`` or ``[None]``).  List-typed ports grow/shrink via
            UI ``+`` / ``×`` controls and may be any length ≥ 0.  When an
            aux port is wired, the consumer's ``params[<port>]`` entry is
            absent (or stale): the aux fold step in :func:`to_pipeline`
            re-derives the marker dict from the aux node before
            constructing the runtime pipeline.
    """

    node_id: str
    class_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    nested: Optional["BuilderScope"] = None
    aux_ports: Dict[str, List[Optional[str]]] = field(default_factory=dict)


@dataclass
class BuilderScope:
    """Linear ordered list of steps mixing ops/meas/post.

    A scope corresponds to a single :class:`~phenotypic.ImagePipeline` once
    converted via :func:`to_pipeline`.  Stage (ops/meas/post) is inferred per
    node from its class via the operation registry.

    Attributes:
        nodes: Ordered :class:`StepNode` list.  Insertion order is the
            execution order; partitioning into ops/meas/post happens at
            convert time.
        name: Pipeline ``name`` to assign on conversion.
        desc: Pipeline ``desc`` to assign on conversion.
        nrows: Optional grid row preset (forwarded to ``ImagePipeline``).
        ncols: Optional grid column preset (forwarded to ``ImagePipeline``).
        aux_nodes: Free-floating aux operation/pipeline nodes that hang off
            the dock for this scope. They are wired into consumer ports via
            :class:`StepNode` ``aux_ports`` entries; each has a stable
            ``node_id`` so disconnect-without-delete (Wave 4) preserves
            identity across re-renders.  Aux nodes that no consumer wires
            to ("orphans") are dropped silently when :func:`to_pipeline`
            folds aux ports back into the runtime pipeline.
    """

    nodes: List[StepNode] = field(default_factory=list)
    name: str = "Pipeline"
    desc: str = ""
    nrows: Optional[int] = None
    ncols: Optional[int] = None
    aux_nodes: List[StepNode] = field(default_factory=list)


@dataclass
class BuilderState:
    """Top-level state for the Dash builder.

    Attributes:
        root: The outermost :class:`BuilderScope`; what the user sees when
            the breadcrumb is empty.
        breadcrumb: Ordered list of breadcrumb segments describing how the
            user has drilled into nested scopes.  Each segment is a dict of
            the form ``{"node_id": <id>, "param": <param_name | None>}``.
            ``param=None`` means a regular ``ImagePipeline`` drill-in (uses
            ``StepNode.nested``); ``param=<name>`` means drilling into an
            operation-typed parameter slot on a non-pipeline node.  Empty
            list means "viewing ``root``".
        selected_node_id: ``node_id`` of the currently focused step in the
            visible scope, if any.
    """

    root: BuilderScope = field(default_factory=BuilderScope)
    breadcrumb: List[Dict[str, Any]] = field(default_factory=list)
    selected_node_id: Optional[str] = None


# Sentinel key used to store synthesized singleton :class:`BuilderScope` dicts
# inside a non-pipeline node's ``params`` while the user is editing an
# operation-typed parameter through drill-down.  Stripped before
# :func:`to_pipeline` so the underlying ``ImagePipeline`` never sees it.
_PARAM_SCOPE_KEY = "__op_param_scope__"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_node_id() -> str:
    """Return a fresh 8-character node identifier.

    Returns:
        Short hex string suitable for use as both ``StepNode.node_id`` and
        the corresponding cytoscape node id.
    """

    return uuid.uuid4().hex[:8]


def _is_operation_param_marker(value: Any) -> bool:
    """Check if *value* is an op-typed parameter dict marker.

    Args:
        value: Candidate parameter value pulled from ``StepNode.params``.

    Returns:
        ``True`` when *value* looks like ``{"__type__": "operation", ...}``
        (with either ``class_name`` or ``class``), otherwise ``False``.
    """

    if not isinstance(value, dict):
        return False
    if value.get("__type__") != "operation":
        return False
    return ("class_name" in value) or ("class" in value)


def _is_pipeline_param_marker(value: Any) -> bool:
    """Check if *value* is a pipeline-typed parameter dict marker.

    Args:
        value: Candidate parameter value pulled from ``StepNode.params``.

    Returns:
        ``True`` when *value* looks like a serialized nested pipeline
        carrying a ``BuilderScope`` payload.
    """

    if not isinstance(value, dict):
        return False
    return value.get("__type__") in {"pipeline", "pipeline_operation"} and (
        "scope" in value or "config" in value
    )


def _looks_like_marker(value: Any) -> bool:
    """Return ``True`` when *value* is a serialized op or pipeline marker dict.

    Convenience predicate used by :func:`from_pipeline` to decide whether a
    ``_serialize_param_value`` result should be promoted into an aux node;
    folds the operation/pipeline checks into one call so the branches at
    extraction time stay readable.
    """

    return _is_operation_param_marker(value) or _is_pipeline_param_marker(value)


def _resolve_param_value(
    value: Any, registry: OperationRegistry
) -> Any:
    """Recursively turn a stored param value into a runtime object.

    Operation markers (``{"__type__": "operation", ...}``) are converted into
    real :class:`~phenotypic.abc_.ImageOperation` instances; pipeline markers
    are converted into :class:`~phenotypic.ImagePipeline` instances; scalars
    pass through unchanged.

    Args:
        value: Raw parameter value from a :class:`StepNode`.
        registry: Operation registry used for class lookup / instantiation.

    Returns:
        The resolved runtime value.
    """

    if _is_pipeline_param_marker(value):
        scope_dict = value.get("scope")
        if scope_dict is not None:
            return to_pipeline(_scope_from_dict(scope_dict))
        # Fallback for raw ``config`` payload (canonical to_json shape).
        return ImagePipeline.from_json(value["config"])

    if _is_operation_param_marker(value):
        class_name = value.get("class_name") or value["class"]
        inner_params = value.get("params", {}) or {}
        resolved_inner = {
            k: _resolve_param_value(v, registry) for k, v in inner_params.items()
        }
        return registry.create_instance(class_name, **resolved_inner)

    if isinstance(value, list):
        return [_resolve_param_value(v, registry) for v in value]

    return value


def _operation_to_param_dict(op: Any, registry: OperationRegistry) -> Dict[str, Any]:
    """Serialize an ``ImageOperation`` instance into a JSON-friendly dict.

    Used by :func:`from_pipeline` when capturing operation-typed params.

    Args:
        op: An :class:`ImageOperation` (or :class:`ImagePipeline`) instance.
        registry: Registry used to enumerate the inner op's parameters so we
            can recurse cleanly through nested op-typed values.

    Returns:
        A dict of shape ``{"__type__": "operation", "class_name": ...,
        "params": {...}}`` (or ``{"__type__": "pipeline", "scope": ...}`` for
        nested pipelines).
    """

    if isinstance(op, ImagePipeline):
        return {
            "__type__": "pipeline",
            "class_name": PIPELINE_CLASS_NAME,
            "scope": _scope_to_dict(from_pipeline(op)),
        }

    class_name = type(op).__name__
    info = registry.get(class_name)
    params: Dict[str, Any] = {}

    if info is None:
        # Fall back to ``__dict__`` introspection for unknown classes.
        for key, value in vars(op).items():
            if key.startswith("_"):
                continue
            params[key] = _serialize_param_value(value, registry)
        return {
            "__type__": "operation",
            "class_name": class_name,
            "params": params,
        }

    for param_name in info.parameters:
        if not hasattr(op, param_name):
            continue
        current = getattr(op, param_name)
        params[param_name] = _serialize_param_value(current, registry)

    return {
        "__type__": "operation",
        "class_name": class_name,
        "params": params,
    }


def _serialize_param_value(value: Any, registry: OperationRegistry) -> Any:
    """Convert a runtime param value into a JSON-friendly representation.

    Args:
        value: Value pulled from an op instance via ``getattr``.
        registry: Registry used when recursing into nested op params.

    Returns:
        A JSON-friendly value: scalars as-is, ``ImageOperation`` /
        ``ImagePipeline`` instances as marker dicts, lists recursively
        processed.
    """

    if isinstance(value, ImagePipeline):
        return _operation_to_param_dict(value, registry)
    if isinstance(value, ImageOperation):
        return _operation_to_param_dict(value, registry)
    if isinstance(value, (list, tuple)):
        return [_serialize_param_value(v, registry) for v in value]
    return value


# ---------------------------------------------------------------------------
# Public conversion API
# ---------------------------------------------------------------------------


def _aux_node_to_marker(
    aux_node: StepNode, registry: OperationRegistry
) -> Dict[str, Any]:
    """Serialize an aux :class:`StepNode` into a runtime-resolvable marker.

    Mirrors the shape produced by :func:`_serialize_param_value` for
    operation/pipeline instances so that :func:`_resolve_param_value` can
    instantiate the value during :func:`to_pipeline`.

    Args:
        aux_node: An aux :class:`StepNode` to fold into a consumer's params.
        registry: Registry used by recursive aux folds (for nested aux scopes).

    Returns:
        A JSON-friendly dict of shape
        ``{"__type__": "operation", "class_name": ..., "params": {...}}`` for
        non-pipeline aux nodes, or
        ``{"__type__": "pipeline", "class_name": "ImagePipeline",
          "scope": ...}`` for pipeline aux nodes.
    """

    if aux_node.class_name == PIPELINE_CLASS_NAME:
        return {
            "__type__": "pipeline",
            "class_name": PIPELINE_CLASS_NAME,
            "scope": _scope_to_dict(aux_node.nested or BuilderScope()),
        }
    return {
        "__type__": "operation",
        "class_name": aux_node.class_name,
        "params": {
            k: v for k, v in aux_node.params.items() if k != _PARAM_SCOPE_KEY
        },
    }


def _fold_aux_ports_into_params(
    scope: BuilderScope, registry: OperationRegistry
) -> BuilderScope:
    """Return a copy of *scope* with aux-port wires folded into ``params``.

    Walks each main-ribbon node's ``aux_ports`` map, looks up the wired aux
    nodes in ``scope.aux_nodes`` by id, and writes the equivalent op /
    pipeline marker dict into ``params[<port>]`` so that the existing
    :func:`_resolve_param_value` path can instantiate the value.  Scalar
    ports collapse to a single marker (or ``None`` when the slot is empty);
    list-typed ports collapse to a Python list of markers, skipping ``None``
    slots.  The folded scope's ``aux_nodes`` list is dropped so the runtime
    pipeline never sees orphan aux nodes — this is also how aux nodes that
    no consumer wires to are silently dropped on save.

    Args:
        scope: The edit-time :class:`BuilderScope`.
        registry: Registry used to resolve nested aux-of-aux folds.

    Returns:
        A fresh :class:`BuilderScope` shaped exactly like the legacy
        edit-time representation (no ``aux_nodes``, all op-typed params
        carry inline markers).  Suitable to feed back into the existing
        :func:`to_pipeline` body.
    """

    aux_index: Dict[str, StepNode] = {n.node_id: n for n in scope.aux_nodes}

    folded_nodes: List[StepNode] = []
    for node in scope.nodes:
        folded_nodes.append(_fold_node(node, aux_index, registry))

    return BuilderScope(
        nodes=folded_nodes,
        name=scope.name,
        desc=scope.desc,
        nrows=scope.nrows,
        ncols=scope.ncols,
        aux_nodes=[],  # orphans dropped silently
    )


def _fold_node(
    node: StepNode,
    aux_index: Dict[str, StepNode],
    registry: OperationRegistry,
) -> StepNode:
    """Return a copy of *node* with its aux-port wires folded into ``params``.

    Args:
        node: The consumer :class:`StepNode` whose ``aux_ports`` should be
            collapsed into inline marker values.
        aux_index: Map from aux ``node_id`` to the corresponding aux
            :class:`StepNode` in the same scope.
        registry: Registry used to recurse into aux nodes that themselves
            carry aux ports / nested scopes.

    Returns:
        A fresh :class:`StepNode` whose ``params`` mirror the runtime view.
    """

    # Recurse into a nested ImagePipeline scope so aux wires inside it are
    # folded too.  Non-pipeline nodes don't carry a ``nested`` scope today,
    # but we keep the recursion symmetric in case future work changes that.
    folded_nested: Optional[BuilderScope] = (
        _fold_aux_ports_into_params(node.nested, registry)
        if node.nested is not None
        else None
    )

    new_params = dict(node.params)

    for port_name, slots in node.aux_ports.items():
        if not slots:
            # Empty list-typed port: emit an empty list value so
            # ``_resolve_param_value`` returns ``[]`` after instantiation.
            new_params[port_name] = []
            continue

        # Resolve each slot to its aux node (if any).  Folding aux-of-aux
        # recursively requires the aux node itself to have its own
        # aux_ports collapsed before we marker-ize it.
        resolved_slots: List[Optional[Dict[str, Any]]] = []
        for aux_id in slots:
            if aux_id is None:
                resolved_slots.append(None)
                continue
            aux_node = aux_index.get(aux_id)
            if aux_node is None:
                # Stale wire (the aux node was deleted but the wire wasn't
                # cleaned up): treat as an empty slot for fold-time purposes.
                resolved_slots.append(None)
                continue
            # Aux-of-aux: collapse the aux node's own aux_ports into its
            # params before marker-izing it.  We feed it through
            # ``_fold_node`` with an empty index because aux nodes only
            # reference siblings in the same scope; nested aux scopes ride
            # along inside ``aux_node.nested`` and are folded above.
            collapsed = _fold_node(aux_node, {}, registry)
            resolved_slots.append(_aux_node_to_marker(collapsed, registry))

        # Decide whether the port is list-typed: ask the registry.  Fall
        # back to "list when len > 1 or empty list" if the registry has no
        # info (unknown class).
        info = registry.get(node.class_name)
        param_info = info.parameters.get(port_name) if info is not None else None
        is_list_port = bool(param_info and param_info.is_list)

        if is_list_port:
            new_params[port_name] = [m for m in resolved_slots if m is not None]
        else:
            # Scalar port: take slot 0, falling back to None.
            new_params[port_name] = resolved_slots[0] if resolved_slots else None

    return StepNode(
        node_id=node.node_id,
        class_name=node.class_name,
        params=new_params,
        label=node.label,
        nested=folded_nested,
        aux_ports={},
    )


def to_pipeline(scope: BuilderScope) -> ImagePipeline:
    """Convert a :class:`BuilderScope` into an :class:`ImagePipeline`.

    Each :class:`StepNode` is instantiated through the operation registry;
    nested :class:`BuilderScope` references recurse to produce inner
    :class:`ImagePipeline` instances.  Aux-port wires (Galaxy-style edit-time
    representation) are folded back into inline op markers via
    :func:`_fold_aux_ports_into_params` so the runtime pipeline never sees
    them.  The resulting instance list is partitioned by ``isinstance``
    against :class:`~phenotypic.abc_.MeasureFeatures` /
    :class:`~phenotypic.abc_.PostMeasurement` (everything else falls into
    ``ops``, including nested pipelines), then handed to
    :class:`ImagePipeline` so that ``__make_unique`` mints dict keys for us.

    Args:
        scope: The :class:`BuilderScope` to materialize.

    Returns:
        A fresh :class:`ImagePipeline` with ``ops``/``meas``/``post``
        populated in the order implied by ``scope.nodes``.  Aux nodes that
        no consumer port references are dropped silently.
    """

    registry = get_registry()
    folded_scope = _fold_aux_ports_into_params(scope, registry)
    instances: List[Any] = []

    for node in folded_scope.nodes:
        if node.class_name == PIPELINE_CLASS_NAME:
            inner_scope = node.nested or BuilderScope()
            instances.append(to_pipeline(inner_scope))
            continue

        resolved_params = {
            name: _resolve_param_value(value, registry)
            for name, value in node.params.items()
            if name != _PARAM_SCOPE_KEY
        }
        instance = registry.create_instance(node.class_name, **resolved_params)
        instances.append(instance)

    ops_list: List[Any] = []
    meas_list: List[MeasureFeatures] = []
    post_list: List[PostMeasurement] = []

    for inst in instances:
        if isinstance(inst, MeasureFeatures):
            meas_list.append(inst)
        elif isinstance(inst, PostMeasurement):
            post_list.append(inst)
        else:
            ops_list.append(inst)

    return ImagePipeline(
        ops=ops_list,
        meas=meas_list,
        post=post_list,
        name=folded_scope.name,
        desc=folded_scope.desc,
        nrows=folded_scope.nrows,
        ncols=folded_scope.ncols,
    )


def _marker_to_aux_node(marker: Dict[str, Any]) -> StepNode:
    """Materialize a serialized op/pipeline marker into a fresh aux :class:`StepNode`.

    Mirror of :func:`_aux_node_to_marker`.  Used by :func:`from_pipeline`
    when extracting op-typed parameter values from a runtime
    :class:`~phenotypic.ImagePipeline` so they can be displayed as
    Galaxy-style aux source nodes on the canvas.

    Args:
        marker: A dict produced by :func:`_serialize_param_value` /
            :func:`_operation_to_param_dict`.  May be either an operation
            marker (``{"__type__": "operation", "class_name": ...,
            "params": {...}}``) or a pipeline marker
            (``{"__type__": "pipeline", "scope": {...}}``).

    Returns:
        A fresh :class:`StepNode` ready to be appended to a
        :class:`BuilderScope` ``aux_nodes`` list.
    """

    node_id = _new_node_id()
    if _is_pipeline_param_marker(marker):
        scope_dict = marker.get("scope") or {}
        nested = _scope_from_dict(scope_dict)
        return StepNode(
            node_id=node_id,
            class_name=PIPELINE_CLASS_NAME,
            params={},
            label=PIPELINE_CLASS_NAME,
            nested=nested,
        )

    class_name = str(marker.get("class_name") or marker.get("class"))
    inner_params = dict(marker.get("params") or {})
    return StepNode(
        node_id=node_id,
        class_name=class_name,
        params=inner_params,
        label=class_name,
    )


def from_pipeline(pipeline: ImagePipeline) -> BuilderScope:
    """Convert an :class:`ImagePipeline` back into a :class:`BuilderScope`.

    Walks ``pipeline.get_ops()`` then ``get_meas()`` then ``get_post()`` to
    preserve execution order and produces one :class:`StepNode` per entry.
    Nested :class:`ImagePipeline` values inside ``_ops`` recurse via this
    function; operation-typed parameters (per the registry) are captured as
    Galaxy-style aux nodes appended to ``BuilderScope.aux_nodes`` and wired
    into the consumer's ``aux_ports`` map.  The marker dict is stripped from
    the consumer's ``params`` after extraction so the runtime ``params``
    dict no longer carries it — the ``aux_ports`` map is the source of
    truth at edit time.

    Args:
        pipeline: The pipeline to mirror.

    Returns:
        A :class:`BuilderScope` whose ``nodes`` reproduce the pipeline's
        contents in execution order, with aux-port wires extracted.
    """

    registry = get_registry()
    nodes: List[StepNode] = []
    aux_nodes: List[StepNode] = []

    pairs: List[tuple[str, Any]] = []
    pairs.extend(pipeline.get_ops().items())
    pairs.extend(pipeline.get_meas().items())
    pairs.extend(pipeline.get_post().items())

    for name, op in pairs:
        node_id = _new_node_id()

        if isinstance(op, ImagePipeline):
            nodes.append(
                StepNode(
                    node_id=node_id,
                    class_name=PIPELINE_CLASS_NAME,
                    params={},
                    label=name,
                    nested=from_pipeline(op),
                )
            )
            continue

        class_name = type(op).__name__
        info = registry.get(class_name)
        params: Dict[str, Any] = {}
        node_aux_ports: Dict[str, List[Optional[str]]] = {}

        if info is None:
            # Unknown to the registry: fall back to ``vars`` introspection.
            for key, value in vars(op).items():
                if key.startswith("_"):
                    continue
                params[key] = _serialize_param_value(value, registry)
        else:
            for param_name, param_info in info.parameters.items():
                if not hasattr(op, param_name):
                    continue
                current = getattr(op, param_name)
                aux_eligible = param_info.is_operation or param_info.is_pipeline

                if not aux_eligible:
                    params[param_name] = _serialize_param_value(current, registry)
                    continue

                if current is None:
                    # Empty aux port: track it so the inspector can render
                    # a port handle even though no aux is wired. Scalar
                    # ports always carry a length-1 ``[None]`` list.
                    node_aux_ports[param_name] = (
                        [] if param_info.is_list else [None]
                    )
                    continue

                if param_info.is_list:
                    slot_aux_ids: List[Optional[str]] = []
                    seq = current if isinstance(current, (list, tuple)) else []
                    for item in seq:
                        serialized = _serialize_param_value(item, registry)
                        if _looks_like_marker(serialized):
                            aux = _marker_to_aux_node(serialized)
                            aux_nodes.append(aux)
                            slot_aux_ids.append(aux.node_id)
                        else:
                            # Non-op item in a list-typed aux port — should
                            # not happen for aux-eligible params, but stay
                            # defensive: store inline so the legacy path
                            # still resolves it.
                            params.setdefault(param_name, []).append(serialized)
                    node_aux_ports[param_name] = slot_aux_ids
                    continue

                # Scalar aux port with a value.
                serialized = _serialize_param_value(current, registry)
                if _looks_like_marker(serialized):
                    aux = _marker_to_aux_node(serialized)
                    aux_nodes.append(aux)
                    node_aux_ports[param_name] = [aux.node_id]
                else:
                    # Defensive: keep as-is if it doesn't look like a
                    # marker (shouldn't happen, but don't lose data).
                    params[param_name] = serialized

        nodes.append(
            StepNode(
                node_id=node_id,
                class_name=class_name,
                params=params,
                label=name,
                nested=None,
                aux_ports=node_aux_ports,
            )
        )

    return BuilderScope(
        nodes=nodes,
        name=pipeline.name,
        desc=pipeline._desc if pipeline._desc is not None else "",
        nrows=pipeline.nrows,
        ncols=pipeline.ncols,
        aux_nodes=aux_nodes,
    )


def _normalize_breadcrumb_segment(seg: Any) -> Dict[str, Any]:
    """Coerce a breadcrumb segment to canonical dict form.

    Three shapes are accepted:

    * Legacy plain-string ``"<node_id>"`` → ``{"node_id": <id>, "param": None}``
    * Dict with ``node_id`` key (regular drill into a main-ribbon node):
      ``{"node_id": ..., "param": ... | None}``
    * Dict with ``aux_id`` key (Galaxy-style drill into an aux node):
      ``{"aux_id": <id>, "param": ... | None}`` — preserved verbatim so
      :func:`current_scope` can branch on its presence.

    Args:
        seg: A breadcrumb entry, either a string (legacy) or a dict.

    Returns:
        A normalised dict.  Either ``node_id`` or ``aux_id`` is present
        (never both); ``param`` defaults to ``None`` if absent.
    """

    if isinstance(seg, str):
        return {"node_id": seg, "param": None}
    if isinstance(seg, dict):
        if "aux_id" in seg:
            return {"aux_id": seg["aux_id"], "param": seg.get("param")}
        if "node_id" in seg:
            return {"node_id": seg["node_id"], "param": seg.get("param")}
    raise ValueError(f"unrecognised breadcrumb segment: {seg!r}")


def _ensure_param_scope(node: StepNode, param_name: str) -> BuilderScope:
    """Return (creating if absent) the synthesized scope for an op-typed param.

    The scope lives under ``node.params[_PARAM_SCOPE_KEY][param_name]`` as a
    dict produced by :func:`_scope_to_dict` so it round-trips through JSON
    cleanly.  This helper rehydrates it into a :class:`BuilderScope` and
    seeds it from any existing operation-marker stored in
    ``node.params[param_name]``.

    Args:
        node: Parent step node that owns the parameter slot.
        param_name: Name of the operation-typed parameter being edited.

    Returns:
        The synthesised :class:`BuilderScope` (one node max).
    """

    scopes = node.params.setdefault(_PARAM_SCOPE_KEY, {})
    scope_dict = scopes.get(param_name)
    if scope_dict is None:
        seed_node: Optional[StepNode] = None
        existing = node.params.get(param_name)
        if isinstance(existing, dict) and _is_operation_param_marker(existing):
            class_name = existing.get("class_name") or existing.get("class")
            seed_node = StepNode(
                node_id=_new_node_id(),
                class_name=str(class_name),
                params=dict(existing.get("params") or {}),
                label=str(class_name),
            )
        elif isinstance(existing, dict) and _is_pipeline_param_marker(existing):
            seed_node = StepNode(
                node_id=_new_node_id(),
                class_name=PIPELINE_CLASS_NAME,
                params={},
                label=PIPELINE_CLASS_NAME,
                nested=_scope_from_dict(existing.get("scope") or {}),
            )
        scope = BuilderScope(
            nodes=[seed_node] if seed_node is not None else [],
            name=f"{node.label or node.class_name}.{param_name}",
        )
        scopes[param_name] = _scope_to_dict(scope)
        return scope
    return _scope_from_dict(scope_dict)


def _commit_param_scope(node: StepNode, param_name: str) -> None:
    """Mirror the singleton scope back into ``node.params[param_name]``.

    Called when the user drills out of an operation-typed-parameter scope so
    that the canonical serialized form (an operation marker dict) reflects
    whatever the user assembled inside the singleton.

    Args:
        node: Parent step node owning the param slot.
        param_name: Operation-typed parameter being committed.
    """

    scopes = node.params.get(_PARAM_SCOPE_KEY) or {}
    scope_dict = scopes.get(param_name)
    if scope_dict is None:
        return

    scope = _scope_from_dict(scope_dict)
    if not scope.nodes:
        node.params[param_name] = None
        return

    inner = scope.nodes[0]
    if inner.class_name == PIPELINE_CLASS_NAME:
        node.params[param_name] = {
            "__type__": "pipeline",
            "class_name": PIPELINE_CLASS_NAME,
            "scope": _scope_to_dict(inner.nested or BuilderScope()),
        }
    else:
        node.params[param_name] = {
            "__type__": "operation",
            "class_name": inner.class_name,
            "params": {
                k: v for k, v in inner.params.items() if k != _PARAM_SCOPE_KEY
            },
        }


def current_scope(state: BuilderState) -> BuilderScope:
    """Resolve the breadcrumb to the scope the user is currently editing.

    Walks each segment in turn:

    * For a regular pipeline drill (``node_id=<id>``, ``param=None``),
      descends into ``match.nested``.
    * For an op-typed parameter drill (``node_id=<id>``, ``param=<name>``),
      descends into the synthesised singleton scope stored under
      ``match.params[_PARAM_SCOPE_KEY][param]`` (created lazily on first
      visit, seeded from any existing operation-marker dict at
      ``match.params[param]``).
    * For a Galaxy-style aux drill (``aux_id=<id>``), looks the aux node up
      in the current scope's ``aux_nodes``.  If the aux node is itself a
      pipeline (``class_name == "ImagePipeline"``), descends into its
      ``nested`` scope.  Otherwise — for a non-pipeline aux node — an
      ``param`` segment field selects which of the aux node's own
      op-typed params to drill into via the existing ``_ensure_param_scope``
      machinery; without ``param`` the walker treats the aux's ``params``
      dict as the editable scope by synthesising an internal scope shell.

    Args:
        state: The full :class:`BuilderState`.

    Returns:
        The :class:`BuilderScope` referenced by the breadcrumb.

    Raises:
        KeyError: If a ``node_id`` / ``aux_id`` in the breadcrumb cannot be
            located in its parent scope, or if the matching pipeline node
            has no nested scope.
    """

    scope = state.root
    for raw in state.breadcrumb:
        seg = _normalize_breadcrumb_segment(raw)
        param = seg.get("param")

        if "aux_id" in seg:
            aux_id = seg["aux_id"]
            aux_match = next(
                (n for n in scope.aux_nodes if n.node_id == aux_id), None
            )
            if aux_match is None:
                raise KeyError(
                    f"breadcrumb aux_id {aux_id!r} not found in current scope"
                )
            if param is None:
                if aux_match.nested is not None:
                    scope = aux_match.nested
                    continue
                # Non-pipeline aux without a ``param`` selector: synthesise
                # a wrapper scope that exposes the aux node itself for
                # editing.  This keeps the walker total — callers will
                # typically only push ``aux_id`` segments for pipeline aux
                # nodes (where ``nested`` exists), but defensively allow
                # the no-param shape so the tests covering the breadcrumb
                # round-trip don't choke on it.
                scope = BuilderScope(
                    nodes=[aux_match],
                    name=aux_match.label or aux_match.class_name,
                )
                continue
            # Drilling into an op-typed param ON the aux node itself.
            scope = _ensure_param_scope(aux_match, param)
            continue

        node_id = seg["node_id"]
        match = next((n for n in scope.nodes if n.node_id == node_id), None)
        if match is None:
            raise KeyError(
                f"breadcrumb node_id {node_id!r} not found in current scope"
            )
        if param is None:
            if match.nested is None:
                raise KeyError(
                    f"breadcrumb node_id {node_id!r} does not have a nested scope"
                )
            scope = match.nested
        else:
            scope = _ensure_param_scope(match, param)
    return scope


def stage_of(class_name: str) -> StageName:
    """Return the pipeline stage a class belongs to.

    Args:
        class_name: Registry key (e.g. ``"OtsuDetector"``) or the
            ``"ImagePipeline"`` sentinel.

    Returns:
        ``"meas"`` if the class is a :class:`MeasureFeatures` subclass,
        ``"post"`` if it's a :class:`PostMeasurement` subclass, otherwise
        ``"ops"`` (which is also the bucket for nested pipelines).

    Raises:
        KeyError: If *class_name* is not registered (and is not the
            ``ImagePipeline`` sentinel).
    """

    if class_name == PIPELINE_CLASS_NAME:
        return "ops"

    registry = get_registry()
    info = registry.get(class_name)
    if info is None:
        raise KeyError(f"Operation '{class_name}' not found in registry")

    cls = info.cls
    if issubclass(cls, MeasureFeatures):
        return "meas"
    if issubclass(cls, PostMeasurement):
        return "post"
    return "ops"


# ---------------------------------------------------------------------------
# JSON serialization for ``dcc.Store``
# ---------------------------------------------------------------------------


def _scope_to_dict(scope: BuilderScope) -> Dict[str, Any]:
    """Recursively dump a :class:`BuilderScope` to a JSON-friendly dict.

    Args:
        scope: The scope to serialize.

    Returns:
        A nested dict mirroring the :class:`BuilderScope` shape.
    """

    return {
        "nodes": [_node_to_dict(n) for n in scope.nodes],
        "name": scope.name,
        "desc": scope.desc,
        "nrows": scope.nrows,
        "ncols": scope.ncols,
        "aux_nodes": [_node_to_dict(n) for n in scope.aux_nodes],
    }


def _node_to_dict(node: StepNode) -> Dict[str, Any]:
    """Recursively dump a :class:`StepNode` to a JSON-friendly dict.

    Args:
        node: The node to serialize.

    Returns:
        A dict mirroring the :class:`StepNode` shape, with ``nested``
        recursed via :func:`_scope_to_dict``.
    """

    return {
        "node_id": node.node_id,
        "class_name": node.class_name,
        "params": node.params,
        "label": node.label,
        "nested": _scope_to_dict(node.nested) if node.nested is not None else None,
        "aux_ports": {k: list(v) for k, v in node.aux_ports.items()},
    }


def _scope_from_dict(data: Dict[str, Any]) -> BuilderScope:
    """Inverse of :func:`_scope_to_dict`.

    Args:
        data: Dict previously produced by :func:`_scope_to_dict` (or an
            equivalent JSON payload).

    Returns:
        A reconstructed :class:`BuilderScope`.
    """

    nodes = [_node_from_dict(n) for n in data.get("nodes", [])]
    aux_nodes = [_node_from_dict(n) for n in (data.get("aux_nodes") or [])]
    return BuilderScope(
        nodes=nodes,
        name=data.get("name", "Pipeline"),
        desc=data.get("desc", ""),
        nrows=data.get("nrows"),
        ncols=data.get("ncols"),
        aux_nodes=aux_nodes,
    )


def _node_from_dict(data: Dict[str, Any]) -> StepNode:
    """Inverse of :func:`_node_to_dict`.

    Args:
        data: Dict previously produced by :func:`_node_to_dict`.

    Returns:
        A reconstructed :class:`StepNode` (with nested scope recursed).
    """

    nested_data = data.get("nested")
    nested = _scope_from_dict(nested_data) if nested_data is not None else None
    raw_aux = data.get("aux_ports") or {}
    aux_ports: Dict[str, List[Optional[str]]] = {
        str(k): list(v) for k, v in raw_aux.items()
    }
    return StepNode(
        node_id=data["node_id"],
        class_name=data["class_name"],
        params=data.get("params", {}) or {},
        label=data.get("label"),
        nested=nested,
        aux_ports=aux_ports,
    )


def state_to_json(state: BuilderState) -> Dict[str, Any]:
    """Convert a :class:`BuilderState` into a JSON-friendly dict.

    Suitable for round-tripping through ``dcc.Store``; the output contains
    only stdlib-compatible types (``dict``, ``list``, ``str``, ``int``,
    ``float``, ``bool``, ``None``) provided the underlying ``StepNode.params``
    values are themselves JSON-friendly (which is the contract for the GUI
    layer).

    Args:
        state: The state to serialize.

    Returns:
        Dict with ``root``, ``breadcrumb``, and ``selected_node_id`` keys.
    """

    return {
        "root": _scope_to_dict(state.root),
        "breadcrumb": list(state.breadcrumb),
        "selected_node_id": state.selected_node_id,
    }


def state_from_json(data: Dict[str, Any]) -> BuilderState:
    """Inverse of :func:`state_to_json`.

    Args:
        data: Dict previously produced by :func:`state_to_json` or an
            equivalent JSON payload.

    Returns:
        A reconstructed :class:`BuilderState`.
    """

    root_data = data.get("root")
    root = _scope_from_dict(root_data) if root_data is not None else BuilderScope()
    raw_crumbs = data.get("breadcrumb", []) or []
    crumbs = [_normalize_breadcrumb_segment(seg) for seg in raw_crumbs]
    return BuilderState(
        root=root,
        breadcrumb=crumbs,
        selected_node_id=data.get("selected_node_id"),
    )
