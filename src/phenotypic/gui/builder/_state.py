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
from typing import Any, Dict, List, Literal, Optional

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
    """

    node_id: str
    class_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    nested: Optional["BuilderScope"] = None


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


def to_pipeline(scope: BuilderScope) -> ImagePipeline:
    """Convert a :class:`BuilderScope` into an :class:`ImagePipeline`.

    Each :class:`StepNode` is instantiated through the operation registry;
    nested :class:`BuilderScope` references recurse to produce inner
    :class:`ImagePipeline` instances.  The resulting instance list is
    partitioned by ``isinstance`` against
    :class:`~phenotypic.abc_.MeasureFeatures` /
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
        name=scope.name,
        desc=scope.desc,
        nrows=scope.nrows,
        ncols=scope.ncols,
    )


def from_pipeline(pipeline: ImagePipeline) -> BuilderScope:
    """Convert an :class:`ImagePipeline` back into a :class:`BuilderScope`.

    Walks ``pipeline.get_ops()`` then ``get_meas()`` then ``get_post()`` to
    preserve execution order and produces one :class:`StepNode` per entry.
    Nested :class:`ImagePipeline` values inside ``_ops`` recurse via this
    function; operation-typed parameters (per the registry) are captured as
    JSON-friendly marker dicts.

    Args:
        pipeline: The pipeline to mirror.

    Returns:
        A :class:`BuilderScope` whose ``nodes`` reproduce the pipeline's
        contents in execution order.
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
                if (param_info.is_operation or param_info.is_pipeline) and current is None:
                    params[param_name] = None
                else:
                    params[param_name] = _serialize_param_value(current, registry)

        nodes.append(
            StepNode(
                node_id=node_id,
                class_name=class_name,
                params=params,
                label=name,
                nested=None,
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
    """Coerce a breadcrumb segment to ``{"node_id": ..., "param": ...}`` form.

    Accepts both the legacy plain-string form (``"<node_id>"``) and the new
    dict form so that existing JSON dumps continue to load.

    Args:
        seg: A breadcrumb entry, either a string (legacy) or a dict.

    Returns:
        A dict with required ``node_id`` and (possibly ``None``) ``param``
        keys.
    """

    if isinstance(seg, str):
        return {"node_id": seg, "param": None}
    if isinstance(seg, dict) and "node_id" in seg:
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

    * For a regular pipeline drill (``param=None``), descends into
      ``match.nested``.
    * For an op-typed parameter drill (``param=<name>``), descends into the
      synthesised singleton scope stored under
      ``match.params[_PARAM_SCOPE_KEY][param]`` (created lazily on first
      visit, seeded from any existing operation-marker dict at
      ``match.params[param]``).

    Args:
        state: The full :class:`BuilderState`.

    Returns:
        The :class:`BuilderScope` referenced by the breadcrumb.

    Raises:
        KeyError: If a ``node_id`` in the breadcrumb cannot be located in
            its parent scope, or if the matching pipeline node has no nested
            scope.
    """

    scope = state.root
    for raw in state.breadcrumb:
        seg = _normalize_breadcrumb_segment(raw)
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
            scope = _ensure_param_scope(match, param)
    return scope


def stage_of(class_name: str) -> Literal["ops", "meas", "post"]:
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
    }


def _node_to_dict(node: StepNode) -> Dict[str, Any]:
    """Recursively dump a :class:`StepNode` to a JSON-friendly dict.

    Args:
        node: The node to serialize.

    Returns:
        A dict mirroring the :class:`StepNode` shape, with ``nested``
        recursed via :func:`_scope_to_dict`.
    """

    return {
        "node_id": node.node_id,
        "class_name": node.class_name,
        "params": node.params,
        "label": node.label,
        "nested": _scope_to_dict(node.nested) if node.nested is not None else None,
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

    Args:
        data: Dict previously produced by :func:`_node_to_dict`.

    Returns:
        A reconstructed :class:`StepNode` (with nested scope recursed).
    """

    nested_data = data.get("nested")
    nested = _scope_from_dict(nested_data) if nested_data is not None else None
    return StepNode(
        node_id=data["node_id"],
        class_name=data["class_name"],
        params=data.get("params", {}) or {},
        label=data.get("label"),
        nested=nested,
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
