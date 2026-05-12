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
            "params": {...}}`` ONLY when nothing is wired through
            ``aux_ports`` for that parameter.  When an aux port is wired,
            the consumer's ``params[<port>]`` entry is absent: the
            embedded :class:`StepNode` in ``aux_ports`` is the source of
            truth and is folded back into a marker dict at
            :func:`to_pipeline` time.
        label: User-editable display name.  ``None`` means the canvas should
            fall back to ``class_name``.
        nested: Inner :class:`BuilderScope` populated only when
            ``class_name == "ImagePipeline"``.
        aux_ports: Per-aux-port slot occupancy. Keys are the names of
            aux-port-eligible parameters (those with
            ``param_info.is_operation or param_info.is_pipeline``).  Values
            are lists whose entries are EITHER an embedded aux
            :class:`StepNode` (the wired aux source) OR ``None`` (empty
            slot).  Non-list ports always carry a length-1 list
            (``[step_node]`` or ``[None]``).  List-typed ports grow/shrink
            via UI ``+`` / ``×`` controls and may be any length ≥ 0.
            Recursive aux is supported: an embedded aux ``StepNode`` may
            itself carry its own ``aux_ports`` map.
    """

    node_id: str
    class_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    nested: Optional["BuilderScope"] = None
    aux_ports: Dict[str, List[Optional["StepNode"]]] = field(default_factory=dict)


@dataclass
class BuilderScope:
    """Linear ordered list of steps mixing ops/meas/post.

    A scope corresponds to a single :class:`~phenotypic.ImagePipeline` once
    converted via :func:`to_pipeline`.  Stage (ops/meas/post) is inferred per
    node from its class via the operation registry.

    Aux operations (op-typed parameters of a consumer node, e.g.
    ``FilamentousFungiDetector.inoculum_detector``) are NOT stored in this
    scope as peer nodes. They live as embedded :class:`StepNode` instances
    inside the consumer's ``aux_ports`` map. This eliminates the previous
    ID-reference indirection (free-floating aux + ``aux_nodes`` list) and
    means aux configuration only exists while wired.

    Attributes:
        nodes: Ordered :class:`StepNode` list.  Insertion order is the
            execution order; partitioning into ops/meas/post happens at
            convert time.
        name: Pipeline ``name`` to assign on conversion.
        desc: Pipeline ``desc`` to assign on conversion.
        nrows: Optional grid row preset (forwarded to ``ImagePipeline``).
        ncols: Optional grid column preset (forwarded to ``ImagePipeline``).
    """

    nodes: List[StepNode] = field(default_factory=list)
    name: str = "Pipeline"
    desc: str = ""
    nrows: Optional[int] = None
    ncols: Optional[int] = None


@dataclass
class BuilderState:
    """Top-level state for the Dash builder.

    Attributes:
        root: The outermost :class:`BuilderScope`; what the user sees when
            the breadcrumb is empty.
        breadcrumb: Ordered list of breadcrumb segments describing how the
            user has drilled into nested scopes.  Each segment is a dict of
            one of two shapes:

            * ``{"node_id": <id>, "param": <param_name | None>}`` —
              regular ``ImagePipeline`` drill-in when ``param=None`` (uses
              ``StepNode.nested``); op-typed parameter drill via the
              legacy ``_PARAM_SCOPE_KEY`` machinery when ``param=<name>``
              (kept for back-compat; new code should prefer the aux-slot
              form below).
            * ``{"target_node_id": <id>, "param": <name>, "slot": <int>}``
              — aux-slot drill: descend into the embedded aux
              :class:`StepNode` at ``consumer.aux_ports[param][slot]``.

            Empty list means "viewing ``root``".
        selected_node_id: ``node_id`` of the currently focused step in the
            visible scope, if any.
        inspector_focus_aux: When non-``None``, the inspector pane mirrors
            the wired aux at the given slot instead of the canvas
            selection. Shape: ``{"target_node_id": str, "param": str,
            "slot": int}``. Driven by the ``set_inspector_focus``
            dispatch kind — set when the user opens a popover with a
            wired slot (or wires a new aux), cleared when the popover
            dismisses, when the wired aux is disconnected, or when the
            user drills into the aux (canvas scope swap takes over).
    """

    root: BuilderScope = field(default_factory=BuilderScope)
    breadcrumb: List[Dict[str, Any]] = field(default_factory=list)
    selected_node_id: Optional[str] = None
    inspector_focus_aux: Optional[Dict[str, Any]] = None


# Sentinel key used to store synthesized singleton :class:`BuilderScope` dicts
# inside a non-pipeline node's ``params`` while the user is editing an
# operation-typed parameter through the legacy drill-down path.  Stripped
# before :func:`to_pipeline` so the underlying ``ImagePipeline`` never sees
# it.  LEGACY: superseded by the embedded-aux model (see
# ``StepNode.aux_ports``); kept here because ``_callbacks.py`` and
# ``_layout.py`` still reference it.  Will be removed once those modules
# migrate to the popover-anchored aux flow (Wave 3/4 of the popover
# redesign).
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
    ``_serialize_param_value`` result should be promoted into an embedded
    aux :class:`StepNode`; folds the operation/pipeline checks into one
    call so the branches at extraction time stay readable.
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


def _aux_step_node_to_marker(
    aux_node: StepNode, registry: OperationRegistry
) -> Dict[str, Any]:
    """Serialize an embedded aux :class:`StepNode` into a runtime marker.

    Mirrors the shape produced by :func:`_serialize_param_value` for
    operation/pipeline instances so that :func:`_resolve_param_value` can
    instantiate the value during :func:`to_pipeline`.  Recursively folds
    the aux node's own ``aux_ports`` (aux-of-aux) into its params before
    emitting the marker so nested wiring round-trips correctly.

    Args:
        aux_node: An aux :class:`StepNode` embedded inside a consumer's
            ``aux_ports`` slot.
        registry: Registry used by recursive aux folds (for nested aux
            scopes that themselves carry op-typed parameters).

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

    # Recursively fold the aux's own aux_ports so aux-of-aux wiring lands
    # in ``params`` for the runtime ``create_instance`` call.
    folded_params = _fold_aux_ports_for_node(aux_node, registry)
    return {
        "__type__": "operation",
        "class_name": aux_node.class_name,
        "params": {k: v for k, v in folded_params.items() if k != _PARAM_SCOPE_KEY},
    }


def _fold_aux_ports_for_node(
    node: StepNode, registry: OperationRegistry
) -> Dict[str, Any]:
    """Collapse a node's embedded ``aux_ports`` into a ``params`` dict.

    Each slot's embedded :class:`StepNode` is serialised into a marker dict
    via :func:`_aux_step_node_to_marker` (which recurses, so aux-of-aux
    is supported).  Scalar ports collapse to a single marker (or ``None``
    when the slot is empty); list-typed ports collapse to a Python list of
    markers, skipping ``None`` slots.

    Args:
        node: The :class:`StepNode` whose ``aux_ports`` should be folded.
        registry: Registry used to inspect the consumer class so we know
            which params are list-typed.

    Returns:
        A ``params``-shaped dict mirroring ``node.params`` but with each
        aux-eligible port overlaid with the wired marker(s).
    """

    folded = dict(node.params)
    info = registry.get(node.class_name)

    for port_name, slots in node.aux_ports.items():
        if not slots:
            # Empty list-typed port: emit an empty list value so
            # ``_resolve_param_value`` returns ``[]`` after instantiation.
            folded[port_name] = []
            continue

        resolved_slots: List[Optional[Dict[str, Any]]] = []
        for embedded in slots:
            if embedded is None:
                resolved_slots.append(None)
                continue
            resolved_slots.append(_aux_step_node_to_marker(embedded, registry))

        # Decide whether the port is list-typed: ask the registry.  Fall
        # back to scalar handling (slot 0) when the registry has no info.
        param_info = info.parameters.get(port_name) if info is not None else None
        is_list_port = bool(param_info and param_info.is_list)

        if is_list_port:
            folded[port_name] = [m for m in resolved_slots if m is not None]
        else:
            # Scalar port: take slot 0, falling back to None.
            folded[port_name] = resolved_slots[0] if resolved_slots else None

    return folded


def to_pipeline(scope: BuilderScope) -> ImagePipeline:
    """Convert a :class:`BuilderScope` into an :class:`ImagePipeline`.

    Each :class:`StepNode` is instantiated through the operation registry;
    nested :class:`BuilderScope` references recurse to produce inner
    :class:`ImagePipeline` instances.  Aux-port wires (embedded
    :class:`StepNode` instances in each consumer's ``aux_ports`` map) are
    folded back into inline op markers via
    :func:`_fold_aux_ports_for_node` so the runtime pipeline never sees
    them.  The resulting instance list is partitioned by ``isinstance``
    against :class:`~phenotypic.abc_.MeasureFeatures` /
    :class:`~phenotypic.abc_.PostMeasurement` (everything else falls into
    ``ops``, including nested pipelines), then handed to
    :class:`ImagePipeline` so that ``__make_unique`` mints dict keys for us.

    Args:
        scope: The :class:`BuilderScope` to materialize.

    Returns:
        A fresh :class:`ImagePipeline` with ``ops``/``meas``/``post``
        populated in the order implied by ``scope.nodes``.
    """

    registry = get_registry()
    instances: List[Any] = []

    for node in scope.nodes:
        if node.class_name == PIPELINE_CLASS_NAME:
            inner_scope = node.nested or BuilderScope()
            instances.append(to_pipeline(inner_scope))
            continue

        folded_params = _fold_aux_ports_for_node(node, registry)
        resolved_params = {
            name: _resolve_param_value(value, registry)
            for name, value in folded_params.items()
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
        name=scope.name,
        desc=scope.desc,
        nrows=scope.nrows,
        ncols=scope.ncols,
    )


def _marker_to_aux_step_node(marker: Dict[str, Any]) -> StepNode:
    """Materialise a serialised op/pipeline marker into an embedded aux node.

    Mirror of :func:`_aux_step_node_to_marker`.  Used by
    :func:`from_pipeline` when extracting op-typed parameter values from a
    runtime :class:`~phenotypic.ImagePipeline` so they can be embedded in
    the consumer's ``aux_ports`` map.  Recursively extracts any nested
    op-typed params on the aux node itself into its own ``aux_ports``
    (aux-of-aux).

    Args:
        marker: A dict produced by :func:`_serialize_param_value` /
            :func:`_operation_to_param_dict`.  May be either an operation
            marker (``{"__type__": "operation", "class_name": ...,
            "params": {...}}``) or a pipeline marker
            (``{"__type__": "pipeline", "scope": {...}}``).

    Returns:
        A fresh :class:`StepNode` ready to be stored inline in a
        consumer's ``aux_ports[<param>][<slot>]`` entry.
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
    inner_params_raw = dict(marker.get("params") or {})

    # Recursively pull aux-eligible params out of the marker into this aux
    # node's own aux_ports map so aux-of-aux preserves its wired structure.
    registry = get_registry()
    info = registry.get(class_name)

    if info is None:
        return StepNode(
            node_id=node_id,
            class_name=class_name,
            params=inner_params_raw,
            label=class_name,
        )

    own_params: Dict[str, Any] = {}
    own_aux_ports: Dict[str, List[Optional[StepNode]]] = {}

    for param_name, param_info in info.parameters.items():
        if param_name not in inner_params_raw:
            continue
        current = inner_params_raw[param_name]
        aux_eligible = param_info.is_operation or param_info.is_pipeline

        if not aux_eligible:
            own_params[param_name] = current
            continue

        if current is None:
            own_aux_ports[param_name] = [] if param_info.is_list else [None]
            continue

        if param_info.is_list:
            seq = current if isinstance(current, (list, tuple)) else []
            slot_steps: List[Optional[StepNode]] = []
            for item in seq:
                if _looks_like_marker(item):
                    slot_steps.append(_marker_to_aux_step_node(item))
                else:
                    # Non-marker inside a list-typed aux port — defensively
                    # fall back to storing inline so we don't lose data.
                    own_params.setdefault(param_name, []).append(item)
            own_aux_ports[param_name] = slot_steps
            continue

        # Scalar aux port with a value.
        if _looks_like_marker(current):
            own_aux_ports[param_name] = [_marker_to_aux_step_node(current)]
        else:
            own_params[param_name] = current

    # Preserve any params that exist on the marker but aren't enumerated
    # by the registry (defensive against schema drift).
    for k, v in inner_params_raw.items():
        if k not in own_params and k not in own_aux_ports and (
            info.parameters.get(k) is None
        ):
            own_params[k] = v

    return StepNode(
        node_id=node_id,
        class_name=class_name,
        params=own_params,
        label=class_name,
        aux_ports=own_aux_ports,
    )


def from_pipeline(pipeline: ImagePipeline) -> BuilderScope:
    """Convert an :class:`ImagePipeline` back into a :class:`BuilderScope`.

    Walks ``pipeline.get_ops()`` then ``get_meas()`` then ``get_post()`` to
    preserve execution order and produces one :class:`StepNode` per entry.
    Nested :class:`ImagePipeline` values inside ``_ops`` recurse via this
    function; operation-typed parameters (per the registry) are captured
    as embedded aux :class:`StepNode` instances directly inside the
    consumer's ``aux_ports`` map.  The marker dict is stripped from the
    consumer's ``params`` after extraction so the runtime ``params`` dict
    no longer carries it — the ``aux_ports`` map is the source of truth at
    edit time.

    Args:
        pipeline: The pipeline to mirror.

    Returns:
        A :class:`BuilderScope` whose ``nodes`` reproduce the pipeline's
        contents in execution order, with aux-port wires extracted into
        each consumer's ``aux_ports``.
    """

    registry = get_registry()
    nodes: List[StepNode] = []

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
        node_aux_ports: Dict[str, List[Optional[StepNode]]] = {}

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
                    slot_steps: List[Optional[StepNode]] = []
                    seq = current if isinstance(current, (list, tuple)) else []
                    for item in seq:
                        serialized = _serialize_param_value(item, registry)
                        if _looks_like_marker(serialized):
                            slot_steps.append(
                                _marker_to_aux_step_node(serialized)
                            )
                        else:
                            # Non-op item in a list-typed aux port — should
                            # not happen for aux-eligible params, but stay
                            # defensive: store inline so the legacy path
                            # still resolves it.
                            params.setdefault(param_name, []).append(serialized)
                    node_aux_ports[param_name] = slot_steps
                    continue

                # Scalar aux port with a value.
                serialized = _serialize_param_value(current, registry)
                if _looks_like_marker(serialized):
                    node_aux_ports[param_name] = [
                        _marker_to_aux_step_node(serialized)
                    ]
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
    )


def _normalize_breadcrumb_segment(seg: Any) -> Dict[str, Any]:
    """Coerce a breadcrumb segment to canonical dict form.

    Three shapes are accepted:

    * Legacy plain-string ``"<node_id>"`` →
      ``{"node_id": <id>, "param": None}``.
    * Dict with ``node_id`` key (regular drill into a main-ribbon node):
      ``{"node_id": ..., "param": ... | None}``.
    * Dict with ``target_node_id`` key (aux-slot drill into an embedded
      aux :class:`StepNode`):
      ``{"target_node_id": <id>, "param": <name>, "slot": <int>}``.

    Args:
        seg: A breadcrumb entry, either a string (legacy) or a dict.

    Returns:
        A normalised dict.  Either ``node_id`` or ``target_node_id`` is
        present (never both); ``param`` defaults to ``None`` for
        ``node_id`` segments.
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
    raise ValueError(f"unrecognised breadcrumb segment: {seg!r}")


def _ensure_param_scope(node: StepNode, param_name: str) -> BuilderScope:
    """Return (creating if absent) the synthesized scope for an op-typed param.

    LEGACY: predates the embedded-aux model.  The scope lives under
    ``node.params[_PARAM_SCOPE_KEY][param_name]`` as a dict produced by
    :func:`_scope_to_dict` so it round-trips through JSON cleanly.  This
    helper rehydrates it into a :class:`BuilderScope` and seeds it from
    any existing operation-marker stored in ``node.params[param_name]``.

    Kept here because ``_layout.py`` and ``_callbacks.py`` still call this
    function via the old ``param`` breadcrumb segment shape.  New code
    should prefer the aux-slot drill (``{"target_node_id", "param",
    "slot"}`` segment) which descends directly into the embedded
    ``StepNode`` in ``aux_ports``.

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

    LEGACY: pairs with :func:`_ensure_param_scope`.  Called when the user
    drills out of an operation-typed-parameter scope so the canonical
    serialised form (an operation marker dict) reflects whatever the user
    assembled inside the singleton.  Kept here because the legacy
    breadcrumb drill machinery in ``_callbacks.py`` still relies on it.

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
    * For the LEGACY op-typed parameter drill (``node_id=<id>``,
      ``param=<name>``), descends into the synthesised singleton scope
      stored under ``match.params[_PARAM_SCOPE_KEY][param]`` (created
      lazily on first visit, seeded from any existing operation-marker
      dict at ``match.params[param]``).  Kept for back-compat; new code
      should prefer the aux-slot form below.
    * For an aux-slot drill (``target_node_id=<id>``, ``param=<name>``,
      ``slot=<int>``), descends into the embedded aux :class:`StepNode`
      stored at ``consumer.aux_ports[param][slot]``.  If that aux node
      has ``nested`` (it's an ImagePipeline aux), descends into
      ``nested``.  Otherwise (single-op aux), synthesises a single-node
      :class:`BuilderScope` wrapping the aux node so the canvas can
      render it as a 1-step ribbon.

    Args:
        state: The full :class:`BuilderState`.

    Returns:
        The :class:`BuilderScope` referenced by the breadcrumb.

    Raises:
        KeyError: If a ``node_id`` / ``target_node_id`` in the breadcrumb
            cannot be located in its parent scope, if the matching
            pipeline node has no nested scope, or if the aux slot is
            empty or out of range.
    """

    scope = state.root
    for raw in state.breadcrumb:
        seg = _normalize_breadcrumb_segment(raw)

        if "target_node_id" in seg:
            target_id = seg["target_node_id"]
            param = seg["param"]
            slot = int(seg.get("slot") or 0)
            consumer = next(
                (n for n in scope.nodes if n.node_id == target_id), None
            )
            if consumer is None:
                raise KeyError(
                    f"breadcrumb target_node_id {target_id!r} not found in "
                    f"current scope"
                )
            if param is None:
                raise KeyError(
                    f"breadcrumb aux-slot segment missing 'param': {seg!r}"
                )
            slots = consumer.aux_ports.get(str(param)) or []
            if slot < 0 or slot >= len(slots):
                raise KeyError(
                    f"breadcrumb aux-slot {slot} out of range for "
                    f"{target_id!r}.{param!r} (len={len(slots)})"
                )
            aux_step = slots[slot]
            if aux_step is None:
                raise KeyError(
                    f"breadcrumb aux-slot {target_id!r}.{param!r}[{slot}] "
                    f"is empty (no aux wired)"
                )
            if aux_step.nested is not None:
                # Pipeline aux: descend into its inner scope.
                scope = aux_step.nested
            else:
                # Single-op aux: wrap it as a 1-step scope so the canvas
                # renders a 1-step ribbon.
                scope = BuilderScope(
                    nodes=[aux_step],
                    name=aux_step.label or aux_step.class_name,
                )
            continue

        node_id = seg["node_id"]
        param = seg.get("param")
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
            scope = _ensure_param_scope(match, str(param))
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
        A nested dict mirroring the :class:`BuilderScope` shape.  Aux
        operations are serialised inline inside each consumer's
        ``aux_ports`` entry (no separate ``aux_nodes`` list).
    """

    return {
        "nodes": [_node_to_dict(n) for n in scope.nodes],
        "name": scope.name,
        "desc": scope.desc,
        "nrows": scope.nrows,
        "ncols": scope.ncols,
    }


def _node_to_dict(node: StepNode) -> Dict[str, Any]:
    """Recursively dump a :class:`StepNode` to a JSON-friendly dict.

    Aux ports' embedded :class:`StepNode` slots recurse via this same
    function so the entire aux subtree round-trips through JSON.

    Args:
        node: The node to serialize.

    Returns:
        A dict mirroring the :class:`StepNode` shape.
    """

    aux_ports_serialised: Dict[str, List[Optional[Dict[str, Any]]]] = {}
    for port_name, slots in node.aux_ports.items():
        aux_ports_serialised[port_name] = [
            _node_to_dict(s) if s is not None else None for s in slots
        ]

    return {
        "node_id": node.node_id,
        "class_name": node.class_name,
        "params": node.params,
        "label": node.label,
        "nested": _scope_to_dict(node.nested) if node.nested is not None else None,
        "aux_ports": aux_ports_serialised,
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
    return BuilderScope(
        nodes=nodes,
        name=data.get("name", "Pipeline"),
        desc=data.get("desc", ""),
        nrows=data.get("nrows"),
        ncols=data.get("ncols"),
    )


def _node_from_dict(data: Dict[str, Any]) -> StepNode:
    """Inverse of :func:`_node_to_dict`.

    Recursively reconstructs embedded aux :class:`StepNode` slots from
    their serialised form so the full aux subtree survives JSON round-trip.

    Args:
        data: Dict previously produced by :func:`_node_to_dict`.

    Returns:
        A reconstructed :class:`StepNode` (with nested scope and embedded
        aux slots recursed).
    """

    nested_data = data.get("nested")
    nested = _scope_from_dict(nested_data) if nested_data is not None else None
    raw_aux = data.get("aux_ports") or {}
    aux_ports: Dict[str, List[Optional[StepNode]]] = {}
    for port_name, slots in raw_aux.items():
        aux_ports[str(port_name)] = [
            _node_from_dict(s) if isinstance(s, dict) else None for s in slots
        ]
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
        Dict with ``root``, ``breadcrumb``, ``selected_node_id``, and
        ``inspector_focus_aux`` keys.
    """

    return {
        "root": _scope_to_dict(state.root),
        "breadcrumb": list(state.breadcrumb),
        "selected_node_id": state.selected_node_id,
        "inspector_focus_aux": state.inspector_focus_aux,
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
        inspector_focus_aux=data.get("inspector_focus_aux"),
    )
