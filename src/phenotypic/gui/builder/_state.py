"""Pure-Python state model for the Dash pipeline builder.

This module defines the dataclasses that back the builder canvas and the
pure functions that translate between that in-memory state and a
:class:`phenotypic.ImagePipeline`.  The data structures are intentionally
JSON-friendly so they can travel through ``dcc.Store`` without losing
fidelity.

The module imports only stdlib + ``phenotypic``; no Dash dependencies.

This module is currently in a **transitional shape** (Phase 1 of the
Pipeline Builder DAG redesign — see
``docs/superpowers/specs/2026-05-12-builder-dag-redesign-design.md``):

* The **legacy** linear-list schema (``_LegacyStepNode`` /
  ``_LegacyBuilderScope`` / ``_LegacyBuilderState``) is still the active
  model when the ``PHENOTYPIC_GUI_DAG`` env var is unset (or anything
  other than the literal ``"1"``).
* The new **DAG** schema (:class:`BlockNode`, :class:`Edge`,
  ``_DagBuilderScope``, ``_DagBuilderState``) is always defined so
  sibling modules can import the dataclasses regardless of flag state.
* The exported ``BuilderScope`` / ``BuilderState`` symbols resolve at
  import time to either the legacy or DAG variant depending on the
  flag.  The legacy ``StepNode`` symbol is permanently aliased to
  ``_LegacyStepNode`` for back-compat.
* ``to_pipeline`` / ``from_pipeline`` continue to operate on the legacy
  types; new DAG conversion lives in ``_conversion_dag.py``.

Examples:
    Build a tiny scope and round-trip it through an ``ImagePipeline``
    (legacy path):

    >>> from phenotypic.gui.builder._state import (
    ...     _LegacyBuilderScope, _LegacyStepNode, to_pipeline, from_pipeline,
    ... )
    >>> scope = _LegacyBuilderScope(
    ...     nodes=[_LegacyStepNode(node_id="a1", class_name="GaussianBlur",
    ...                            params={"sigma": 1.5})],
    ...     name="demo",
    ... )
    >>> pipeline = to_pipeline(scope)
    >>> from_pipeline(pipeline).nodes[0].class_name
    'GaussianBlur'
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from phenotypic.gui.builder._ids import StageName

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageOperation, MeasureFeatures, PostMeasurement
from phenotypic.gui._operation_registry import (
    OperationRegistry,
    get_registry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag (spec §5.7a) — read ONCE at module import. Subsequent env-var
# changes are ignored for the lifetime of the process.
# ---------------------------------------------------------------------------

PHENOTYPIC_GUI_DAG: bool = os.environ.get("PHENOTYPIC_GUI_DAG", "0") == "1"


# Sentinel class names. ``PIPELINE_CLASS_NAME`` predates the redesign and
# names nested ``ImagePipeline`` step nodes; ``INPUT_IMAGE_CLASS_NAME`` is
# new (Rule 6 / spec §4.1) — every DAG scope auto-seeds one such block.
PIPELINE_CLASS_NAME = "ImagePipeline"
INPUT_IMAGE_CLASS_NAME = "InputImage"


# ---------------------------------------------------------------------------
# DAG schema (spec §5.1) — always defined so sibling modules can import.
# ---------------------------------------------------------------------------


def _new_block_id() -> str:
    """Return a fresh 32-character DAG block identifier.

    The DAG schema uses full 32-character UUID4 hex strings (distinct from
    the legacy 8-character ``StepNode.node_id`` slice) to eliminate the
    risk of collisions across nested scopes after migration.

    Returns:
        32-character lowercase hex string.
    """

    return uuid.uuid4().hex


@dataclass
class BlockNode:
    """One canvas block — single op, container, or Input Image source.

    A ``BlockNode`` is the DAG-canvas counterpart of a single operation
    (or the Input Image sentinel, or a Pipeline container) in an
    :class:`~phenotypic.ImagePipeline`.  It captures the class to
    instantiate, the scalar parameter values entered by the user, and
    (when the block is a Pipeline container) the inner
    :class:`_DagBuilderScope`.

    Aux wiring is **not** stored on the block — it lives on the
    enclosing scope's ``edges`` list as one :class:`Edge` per wire.

    Attributes:
        block_id: Stable identifier (``uuid.uuid4().hex`` — 32 chars).
        class_name: Registry key (e.g. ``"GaussianBlur"``), the
            ``"ImagePipeline"`` sentinel for container blocks, or the
            ``"InputImage"`` sentinel for the scope's source block.
        params: Scalar parameter values (no op-typed values — those come
            from edges now).
        label: User-editable display name; ``None`` falls back to
            ``class_name`` at render time.
        nested: Inner :class:`_DagBuilderScope` populated only when
            ``class_name == "ImagePipeline"``.
        collapsed: Container-only flag; ``True`` hides children on the
            canvas.
        list_slot_counts: Per list-aux param name, the total slot count
            for layout.  Empty slots = slot positions in ``[0, count)``
            not covered by any :class:`Edge`.  Increments on
            ``list_aux_add_empty_slot`` / ``edge_create`` to a list
            port; never decrements except via slot delete.
    """

    block_id: str
    class_name: str
    params: Dict[str, Any]
    label: Optional[str] = None
    nested: Optional["_DagBuilderScope"] = None
    collapsed: bool = False
    list_slot_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class Edge:
    """One wire between two block ports within a scope.

    Wires connect either an image-out port to an image-in port (``kind
    == "image"``) or an image-out port to an aux-in port (``kind ==
    "aux"``).  ``target_slot`` is set for list-typed aux ports (0-based)
    and ``None`` for scalar aux + all image-flow edges.

    Attributes:
        edge_id: Stable identifier.  Generated as ``uuid.uuid4().hex``
            on creation but accepted as-is on JSON round-trip.
        source_block_id: ``block_id`` of the upstream block.
        source_port: Source port name — currently always ``"out"`` (one
            image-output per block).  Field kept for future
            multi-output ops.
        target_block_id: ``block_id`` of the downstream block.  The
            default of ``""`` is a dataclass-ordering artifact (Python
            forbids non-default fields after defaulted ones); the
            ``__post_init__`` asserts the value is non-empty.
        target_port: Target port name: ``"in"`` for image-flow,
            ``"<param>"`` for aux (scalar or list — list slot index is
            in ``target_slot``).
        target_slot: List-aux slot index (0-based); ``None`` for scalar
            aux and for image-flow.
        kind: Edge kind — ``"image"`` for blue image-flow wires,
            ``"aux"`` for purple consumer-into-aux wires.  Redundant
            with ``target_port`` semantics but kept explicit for fast
            validation.
    """

    edge_id: str
    source_block_id: str
    source_port: str = "out"
    target_block_id: str = ""
    target_port: str = ""
    target_slot: Optional[int] = None
    kind: Literal["image", "aux"] = "image"

    def __post_init__(self) -> None:
        if not self.target_block_id:
            raise ValueError(
                "Edge.target_block_id is mandatory (per spec §5.1); "
                "the default keyword exists only so the dataclass field "
                "ordering is valid."
            )


@dataclass
class _DagBuilderScope:
    """A DAG of blocks + edges (per scope).

    A scope corresponds to one :class:`~phenotypic.ImagePipeline` once
    converted via the DAG conversion path (``_conversion_dag.py``).
    Stage (``ops``/``meas``/``post``) is inferred per block from its
    class via the operation registry at convert time; the canvas does
    not constrain block ordering by stage.

    Every scope (root *and* every nested container scope) auto-seeds an
    :class:`BlockNode` with ``class_name == INPUT_IMAGE_CLASS_NAME`` at
    construction time via :func:`_seed_input_image` (called from
    :meth:`__post_init__`).  The Rule 6 idempotency invariant is
    therefore established eagerly.

    Attributes:
        blocks: Ordered :class:`BlockNode` list.  The order is *not*
            execution order — the DAG's image-flow edges dictate
            execution order during conversion.  The first block is
            (post-seed) always the ``InputImage`` sentinel.
        edges: :class:`Edge` list backing every wire on the canvas.
        name: Pipeline ``name`` to assign on conversion.
        desc: Pipeline ``desc`` to assign on conversion.
        nrows: Optional grid row preset (forwarded to ``ImagePipeline``
            on the root scope; ignored for nested container scopes).
        ncols: Optional grid column preset (root scope only — see
            §4.5 of the spec).
    """

    blocks: List[BlockNode] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    name: str = "Pipeline"
    desc: str = ""
    nrows: Optional[int] = None
    ncols: Optional[int] = None

    def __post_init__(self) -> None:
        # Nested scopes seed themselves via their own ``__post_init__``;
        # JSON-loaded trees also pick up the seed via
        # :func:`_heal_dag_scope_tree`. We only need to seed *this*
        # scope here.
        _seed_input_image(self)


@dataclass
class _DagBuilderState:
    """Top-level DAG state for the Dash builder.

    Mirrors the legacy :class:`_LegacyBuilderState` surface but with
    block_ids / edge_ids instead of node_ids.  The popover-era
    ``inspector_focus_aux`` override is gone — the DAG path edits aux
    parameters via the inspector aux-ports section directly.

    Attributes:
        root: Outermost :class:`_DagBuilderScope` (what the user sees
            when ``breadcrumb`` is empty).
        breadcrumb: Ordered list of container block_ids the user has
            drilled into.  Each entry is the ``BlockNode.block_id`` of
            a Pipeline container in the next-outer scope.  An empty
            list means "viewing ``root``".
        selected_block_id: ``BlockNode.block_id`` of the currently
            focused block (op, container, or Input Image), if any.
            Mutually exclusive with ``selected_edge_id``.
        selected_edge_id: ``Edge.edge_id`` of the currently focused
            wire, if any.  Mutually exclusive with ``selected_block_id``.
        pending_delete_block_id: Set by ``block_delete_request`` for
            non-empty containers; drives the confirm-delete modal's
            visibility.  Cleared on Confirm/Cancel.
        toast_queue: FIFO queue of toast payloads (one visible at a
            time, 3000ms auto-dismiss).  Each entry is a free-form dict
            shaped by the callback that enqueued it; the Dash callback
            that binds to the toast component pops the head.
    """

    root: _DagBuilderScope = field(default_factory=_DagBuilderScope)
    breadcrumb: List[str] = field(default_factory=list)
    selected_block_id: Optional[str] = None
    selected_edge_id: Optional[str] = None
    pending_delete_block_id: Optional[str] = None
    toast_queue: List[Dict[str, Any]] = field(default_factory=list)


def _seed_input_image(scope: _DagBuilderScope) -> None:
    """Idempotently add an ``InputImage`` block at the head of *scope*.

    Rule 6 of the validation suite (spec §4.6) requires exactly one
    ``InputImage`` block per scope.  This helper enforces the
    invariant by:

    * Doing nothing if *scope* already has at least one block whose
      ``class_name == INPUT_IMAGE_CLASS_NAME`` — extras are detected
      and reported by the validator, not pruned here.
    * Otherwise inserting a fresh ``InputImage`` block at index 0.

    Called from :meth:`_DagBuilderScope.__post_init__` and from every
    deserialisation path (see :func:`state_from_json` for the DAG
    branch) so corrupted JSON missing an Input Image self-heals on
    load.

    Args:
        scope: The scope to seed in place.
    """

    if any(b.class_name == INPUT_IMAGE_CLASS_NAME for b in scope.blocks):
        return
    scope.blocks.insert(
        0,
        BlockNode(
            block_id=_new_block_id(),
            class_name=INPUT_IMAGE_CLASS_NAME,
            params={},
            label=None,
        ),
    )


def _find_container_scope_by_block_id(
    root: _DagBuilderScope, block_id: str
) -> Optional[_DagBuilderScope]:
    """Depth-first search for a container's *nested* scope by ``block_id``.

    Callers (palette drop, future re-parent dispatches) resolve a
    ``container_block_id`` payload field to the scope the new block
    should be appended to.  This helper walks the tree rooted at *root*
    and returns the first ``nested`` scope it finds whose parent
    :class:`BlockNode` has the matching ``block_id``.

    The search is recursive through every container's ``nested`` scope;
    only blocks whose ``class_name == PIPELINE_CLASS_NAME`` have a
    non-``None`` ``nested`` field, so the walk naturally restricts itself
    to container blocks.  Lookup is O(N) in the number of blocks across
    every scope, which is fine: typical pipelines have <50 blocks and
    the search runs once per drop.

    Args:
        root: The outermost :class:`_DagBuilderScope` (usually
            ``state.root``).
        block_id: The container ``BlockNode.block_id`` whose nested
            scope is being looked up.

    Returns:
        The matching container's :class:`_DagBuilderScope`, or ``None``
        when *block_id* doesn't resolve to a container in the tree
        (stale id, regular op block, or :data:`INPUT_IMAGE_CLASS_NAME`
        sentinel).
    """

    for block in root.blocks:
        if block.block_id == block_id:
            return block.nested
        if block.nested is not None:
            hit = _find_container_scope_by_block_id(block.nested, block_id)
            if hit is not None:
                return hit
    return None


# ---------------------------------------------------------------------------
# Legacy schema (linear list + embedded aux_ports) — renamed in place so
# the new DAG types can claim the public ``BuilderScope`` / ``BuilderState``
# symbols when the flag is on.
# ---------------------------------------------------------------------------


@dataclass
class _LegacyStepNode:
    """One step in a builder canvas (legacy linear-list schema).

    A ``_LegacyStepNode`` is the canvas-layer counterpart of a single
    operation in an :class:`~phenotypic.ImagePipeline`.  It captures the
    class to instantiate, the parameter values entered by the user, and
    (when the step is itself a nested pipeline) the inner
    :class:`_LegacyBuilderScope`.

    Re-exported under the public name :data:`StepNode` for back-compat.

    Attributes:
        node_id: Short, stable identifier (8-char hex slice of a UUID4).
            Used as the cytoscape node id and to address sub-scopes via
            the breadcrumb path.
        class_name: Registry key (e.g. ``"GaussianBlur"``) or the
            sentinel ``"ImagePipeline"`` when the node represents a
            nested pipeline.
        params: Raw parameter values.  Scalars are stored as-is; values
            corresponding to operation-typed parameters are stored as
            JSON dicts shaped like
            ``{"__type__": "operation", "class_name": ..., "params":
            {...}}`` ONLY when nothing is wired through ``aux_ports``
            for that parameter.  When an aux port is wired, the
            consumer's ``params[<port>]`` entry is absent: the embedded
            :class:`_LegacyStepNode` in ``aux_ports`` is the source of
            truth and is folded back into a marker dict at
            :func:`to_pipeline` time.
        label: User-editable display name.  ``None`` means the canvas
            should fall back to ``class_name``.
        nested: Inner :class:`_LegacyBuilderScope` populated only when
            ``class_name == "ImagePipeline"``.
        aux_ports: Per-aux-port slot occupancy. Keys are the names of
            aux-port-eligible parameters (those with
            ``param_info.is_operation or param_info.is_pipeline``).
            Values are lists whose entries are EITHER an embedded aux
            :class:`_LegacyStepNode` (the wired aux source) OR ``None``
            (empty slot).  Non-list ports always carry a length-1 list
            (``[step_node]`` or ``[None]``).  List-typed ports
            grow/shrink via UI ``+`` / ``×`` controls and may be any
            length ≥ 0.  Recursive aux is supported: an embedded aux
            ``_LegacyStepNode`` may itself carry its own ``aux_ports``
            map.
    """

    node_id: str
    class_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    nested: Optional["_LegacyBuilderScope"] = None
    aux_ports: Dict[str, List[Optional["_LegacyStepNode"]]] = field(
        default_factory=dict
    )


@dataclass
class _LegacyBuilderScope:
    """Linear ordered list of steps mixing ops/meas/post (legacy schema).

    A scope corresponds to a single :class:`~phenotypic.ImagePipeline`
    once converted via :func:`to_pipeline`.  Stage (ops/meas/post) is
    inferred per node from its class via the operation registry.

    Aux operations (op-typed parameters of a consumer node, e.g.
    ``FilamentousFungiDetector.inoculum_detector``) are NOT stored in
    this scope as peer nodes.  They live as embedded
    :class:`_LegacyStepNode` instances inside the consumer's
    ``aux_ports`` map.  This eliminates the previous ID-reference
    indirection (free-floating aux + ``aux_nodes`` list) and means aux
    configuration only exists while wired.

    Attributes:
        nodes: Ordered :class:`_LegacyStepNode` list.  Insertion order
            is the execution order; partitioning into ops/meas/post
            happens at convert time.
        name: Pipeline ``name`` to assign on conversion.
        desc: Pipeline ``desc`` to assign on conversion.
        nrows: Optional grid row preset (forwarded to
            ``ImagePipeline``).
        ncols: Optional grid column preset (forwarded to
            ``ImagePipeline``).
    """

    nodes: List[_LegacyStepNode] = field(default_factory=list)
    name: str = "Pipeline"
    desc: str = ""
    nrows: Optional[int] = None
    ncols: Optional[int] = None


@dataclass
class _LegacyBuilderState:
    """Top-level state for the Dash builder (legacy schema).

    Attributes:
        root: The outermost :class:`_LegacyBuilderScope`; what the user
            sees when the breadcrumb is empty.
        breadcrumb: Ordered list of breadcrumb segments describing how
            the user has drilled into nested scopes.  Each segment is a
            dict of one of two shapes:

            * ``{"node_id": <id>, "param": <param_name | None>}`` —
              regular ``ImagePipeline`` drill-in when ``param=None``
              (uses ``_LegacyStepNode.nested``); the popover-era
              op-typed parameter drill (``param=<name>``) was retired
              in Phase 7 and is now treated as a no-op by the walker.
            * ``{"target_node_id": <id>, "param": <name>, "slot":
              <int>}`` — aux-slot drill: descend into the embedded aux
              :class:`_LegacyStepNode` at
              ``consumer.aux_ports[param][slot]``.

            Empty list means "viewing ``root``".
        selected_node_id: ``node_id`` of the currently focused step in
            the visible scope, if any.
    """

    root: _LegacyBuilderScope = field(default_factory=_LegacyBuilderScope)
    breadcrumb: List[Dict[str, Any]] = field(default_factory=list)
    selected_node_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Public ``BuilderScope`` / ``BuilderState`` aliases — resolve to the DAG
# types when the flag is on, otherwise to the legacy types.  ``StepNode`` is
# permanently aliased to the legacy step node (no DAG counterpart — the
# DAG model uses :class:`BlockNode` instead).
# ---------------------------------------------------------------------------

StepNode = _LegacyStepNode

if PHENOTYPIC_GUI_DAG:
    BuilderScope = _DagBuilderScope  # type: ignore[misc,assignment]
    BuilderState = _DagBuilderState  # type: ignore[misc,assignment]
else:
    BuilderScope = _LegacyBuilderScope  # type: ignore[misc,assignment]
    BuilderState = _LegacyBuilderState  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Helpers (legacy)
# ---------------------------------------------------------------------------


def _new_node_id() -> str:
    """Return a fresh 8-character node identifier (legacy schema).

    Returns:
        Short hex string suitable for use as both
        :attr:`_LegacyStepNode.node_id` and the corresponding cytoscape
        node id.
    """

    return uuid.uuid4().hex[:8]


def _is_operation_param_marker(value: Any) -> bool:
    """Check if *value* is an op-typed parameter dict marker.

    Args:
        value: Candidate parameter value pulled from
            :attr:`_LegacyStepNode.params`.

    Returns:
        ``True`` when *value* looks like ``{"__type__": "operation",
        ...}`` (with either ``class_name`` or ``class``), otherwise
        ``False``.
    """

    if not isinstance(value, dict):
        return False
    if value.get("__type__") != "operation":
        return False
    return ("class_name" in value) or ("class" in value)


def _is_pipeline_param_marker(value: Any) -> bool:
    """Check if *value* is a pipeline-typed parameter dict marker.

    Args:
        value: Candidate parameter value pulled from
            :attr:`_LegacyStepNode.params`.

    Returns:
        ``True`` when *value* looks like a serialized nested pipeline
        carrying a :class:`_LegacyBuilderScope` payload.
    """

    if not isinstance(value, dict):
        return False
    return value.get("__type__") in {"pipeline", "pipeline_operation"} and (
        "scope" in value or "config" in value
    )


def _looks_like_marker(value: Any) -> bool:
    """Return ``True`` when *value* is a serialized op or pipeline marker dict.

    Convenience predicate used by :func:`from_pipeline` to decide
    whether a ``_serialize_param_value`` result should be promoted into
    an embedded aux :class:`_LegacyStepNode`; folds the
    operation/pipeline checks into one call so the branches at
    extraction time stay readable.
    """

    return _is_operation_param_marker(value) or _is_pipeline_param_marker(value)


def _resolve_param_value(
    value: Any, registry: OperationRegistry
) -> Any:
    """Recursively turn a stored param value into a runtime object.

    Operation markers (``{"__type__": "operation", ...}``) are converted
    into real :class:`~phenotypic.abc_.ImageOperation` instances;
    pipeline markers are converted into
    :class:`~phenotypic.ImagePipeline` instances; scalars pass through
    unchanged.

    Args:
        value: Raw parameter value from a :class:`_LegacyStepNode`.
        registry: Operation registry used for class lookup /
            instantiation.

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
        op: An :class:`ImageOperation` (or :class:`ImagePipeline`)
            instance.
        registry: Registry used to enumerate the inner op's parameters
            so we can recurse cleanly through nested op-typed values.

    Returns:
        A dict of shape ``{"__type__": "operation", "class_name": ...,
        "params": {...}}`` (or ``{"__type__": "pipeline", "scope":
        ...}`` for nested pipelines).
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
# Public conversion API (legacy)
# ---------------------------------------------------------------------------


def _aux_step_node_to_marker(
    aux_node: _LegacyStepNode, registry: OperationRegistry
) -> Dict[str, Any]:
    """Serialize an embedded aux :class:`_LegacyStepNode` into a runtime marker.

    Mirrors the shape produced by :func:`_serialize_param_value` for
    operation/pipeline instances so that :func:`_resolve_param_value`
    can instantiate the value during :func:`to_pipeline`.  Recursively
    folds the aux node's own ``aux_ports`` (aux-of-aux) into its params
    before emitting the marker so nested wiring round-trips correctly.

    Args:
        aux_node: An aux :class:`_LegacyStepNode` embedded inside a
            consumer's ``aux_ports`` slot.
        registry: Registry used by recursive aux folds (for nested aux
            scopes that themselves carry op-typed parameters).

    Returns:
        A JSON-friendly dict of shape
        ``{"__type__": "operation", "class_name": ..., "params":
        {...}}`` for non-pipeline aux nodes, or
        ``{"__type__": "pipeline", "class_name": "ImagePipeline",
        "scope": ...}`` for pipeline aux nodes.
    """

    if aux_node.class_name == PIPELINE_CLASS_NAME:
        return {
            "__type__": "pipeline",
            "class_name": PIPELINE_CLASS_NAME,
            "scope": _scope_to_dict(aux_node.nested or _LegacyBuilderScope()),
        }

    # Recursively fold the aux's own aux_ports so aux-of-aux wiring lands
    # in ``params`` for the runtime ``create_instance`` call.
    folded_params = _fold_aux_ports_for_node(aux_node, registry)
    return {
        "__type__": "operation",
        "class_name": aux_node.class_name,
        "params": dict(folded_params),
    }


def _fold_aux_ports_for_node(
    node: _LegacyStepNode, registry: OperationRegistry
) -> Dict[str, Any]:
    """Collapse a node's embedded ``aux_ports`` into a ``params`` dict.

    Each slot's embedded :class:`_LegacyStepNode` is serialised into a
    marker dict via :func:`_aux_step_node_to_marker` (which recurses,
    so aux-of-aux is supported).  Scalar ports collapse to a single
    marker (or ``None`` when the slot is empty); list-typed ports
    collapse to a Python list of markers, skipping ``None`` slots.

    Args:
        node: The :class:`_LegacyStepNode` whose ``aux_ports`` should be
            folded.
        registry: Registry used to inspect the consumer class so we
            know which params are list-typed.

    Returns:
        A ``params``-shaped dict mirroring ``node.params`` but with
        each aux-eligible port overlaid with the wired marker(s).
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


def to_pipeline(scope: _LegacyBuilderScope) -> ImagePipeline:
    """Convert a :class:`_LegacyBuilderScope` into an :class:`ImagePipeline`.

    Each :class:`_LegacyStepNode` is instantiated through the operation
    registry; nested :class:`_LegacyBuilderScope` references recurse to
    produce inner :class:`ImagePipeline` instances.  Aux-port wires
    (embedded :class:`_LegacyStepNode` instances in each consumer's
    ``aux_ports`` map) are folded back into inline op markers via
    :func:`_fold_aux_ports_for_node` so the runtime pipeline never sees
    them.  The resulting instance list is partitioned by ``isinstance``
    against :class:`~phenotypic.abc_.MeasureFeatures` /
    :class:`~phenotypic.abc_.PostMeasurement` (everything else falls
    into ``ops``, including nested pipelines), then handed to
    :class:`ImagePipeline` so that ``__make_unique`` mints dict keys
    for us.

    Args:
        scope: The :class:`_LegacyBuilderScope` to materialize.

    Returns:
        A fresh :class:`ImagePipeline` with ``ops``/``meas``/``post``
        populated in the order implied by ``scope.nodes``.
    """

    registry = get_registry()
    instances: List[Any] = []

    for node in scope.nodes:
        if node.class_name == PIPELINE_CLASS_NAME:
            inner_scope = node.nested or _LegacyBuilderScope()
            instances.append(to_pipeline(inner_scope))
            continue

        folded_params = _fold_aux_ports_for_node(node, registry)
        resolved_params = {
            name: _resolve_param_value(value, registry)
            for name, value in folded_params.items()
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


def _marker_to_aux_step_node(marker: Dict[str, Any]) -> _LegacyStepNode:
    """Materialise a serialised op/pipeline marker into an embedded aux node.

    Mirror of :func:`_aux_step_node_to_marker`.  Used by
    :func:`from_pipeline` when extracting op-typed parameter values
    from a runtime :class:`~phenotypic.ImagePipeline` so they can be
    embedded in the consumer's ``aux_ports`` map.  Recursively extracts
    any nested op-typed params on the aux node itself into its own
    ``aux_ports`` (aux-of-aux).

    Args:
        marker: A dict produced by :func:`_serialize_param_value` /
            :func:`_operation_to_param_dict`.  May be either an
            operation marker (``{"__type__": "operation", "class_name":
            ..., "params": {...}}``) or a pipeline marker
            (``{"__type__": "pipeline", "scope": {...}}``).

    Returns:
        A fresh :class:`_LegacyStepNode` ready to be stored inline in a
        consumer's ``aux_ports[<param>][<slot>]`` entry.
    """

    node_id = _new_node_id()
    if _is_pipeline_param_marker(marker):
        scope_dict = marker.get("scope") or {}
        nested = _scope_from_dict(scope_dict)
        return _LegacyStepNode(
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
        return _LegacyStepNode(
            node_id=node_id,
            class_name=class_name,
            params=inner_params_raw,
            label=class_name,
        )

    own_params: Dict[str, Any] = {}
    own_aux_ports: Dict[str, List[Optional[_LegacyStepNode]]] = {}

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
            slot_steps: List[Optional[_LegacyStepNode]] = []
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

    return _LegacyStepNode(
        node_id=node_id,
        class_name=class_name,
        params=own_params,
        label=class_name,
        aux_ports=own_aux_ports,
    )


def from_pipeline(pipeline: ImagePipeline) -> _LegacyBuilderScope:
    """Convert an :class:`ImagePipeline` back into a :class:`_LegacyBuilderScope`.

    Walks ``pipeline.get_ops()`` then ``get_meas()`` then ``get_post()``
    to preserve execution order and produces one
    :class:`_LegacyStepNode` per entry.  Nested :class:`ImagePipeline`
    values inside ``_ops`` recurse via this function; operation-typed
    parameters (per the registry) are captured as embedded aux
    :class:`_LegacyStepNode` instances directly inside the consumer's
    ``aux_ports`` map.  The marker dict is stripped from the consumer's
    ``params`` after extraction so the runtime ``params`` dict no
    longer carries it — the ``aux_ports`` map is the source of truth at
    edit time.

    Args:
        pipeline: The pipeline to mirror.

    Returns:
        A :class:`_LegacyBuilderScope` whose ``nodes`` reproduce the
        pipeline's contents in execution order, with aux-port wires
        extracted into each consumer's ``aux_ports``.
    """

    registry = get_registry()
    nodes: List[_LegacyStepNode] = []

    pairs: List[tuple[str, Any]] = []
    pairs.extend(pipeline.get_ops().items())
    pairs.extend(pipeline.get_meas().items())
    pairs.extend(pipeline.get_post().items())

    for name, op in pairs:
        node_id = _new_node_id()

        if isinstance(op, ImagePipeline):
            nodes.append(
                _LegacyStepNode(
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
        node_aux_ports: Dict[str, List[Optional[_LegacyStepNode]]] = {}

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
                    slot_steps: List[Optional[_LegacyStepNode]] = []
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
            _LegacyStepNode(
                node_id=node_id,
                class_name=class_name,
                params=params,
                label=name,
                nested=None,
                aux_ports=node_aux_ports,
            )
        )

    return _LegacyBuilderScope(
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
      aux :class:`_LegacyStepNode`):
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


def current_scope(state: _LegacyBuilderState) -> _LegacyBuilderScope:
    """Resolve the breadcrumb to the scope the user is currently editing.

    Walks each segment in turn:

    * For a regular pipeline drill (``node_id=<id>``, ``param=None``),
      descends into ``match.nested``.
    * For an op-typed parameter drill (``node_id=<id>``,
      ``param=<name>``), the legacy popover-era synthesized scope is no
      longer supported (Phase 7 retired the popover wire flow); the
      segment is treated as a no-op so older saved state still loads.
    * For an aux-slot drill (``target_node_id=<id>``,
      ``param=<name>``, ``slot=<int>``), descends into the embedded
      aux :class:`_LegacyStepNode` stored at
      ``consumer.aux_ports[param][slot]``.  If that aux node has
      ``nested`` (it's an ImagePipeline aux), descends into
      ``nested``.  Otherwise (single-op aux), synthesises a
      single-node :class:`_LegacyBuilderScope` wrapping the aux node
      so the canvas can render it as a 1-step ribbon.

    Args:
        state: The full :class:`_LegacyBuilderState`.

    Returns:
        The :class:`_LegacyBuilderScope` referenced by the breadcrumb.

    Raises:
        KeyError: If a ``node_id`` / ``target_node_id`` in the
            breadcrumb cannot be located in its parent scope, if the
            matching pipeline node has no nested scope, or if the aux
            slot is empty or out of range.
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
                scope = _LegacyBuilderScope(
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
            # Op-typed parameter drill via the legacy popover-era synthesized
            # scope is no longer supported (Phase 7 retired the popover
            # path). Stale ``param`` segments loaded from older saved state
            # are treated as no-ops so the walker doesn't crash.
            continue
    return scope


def stage_of(class_name: str) -> StageName:
    """Return the pipeline stage a class belongs to.

    Args:
        class_name: Registry key (e.g. ``"OtsuDetector"``) or the
            ``"ImagePipeline"`` sentinel.

    Returns:
        ``"meas"`` if the class is a :class:`MeasureFeatures` subclass,
        ``"post"`` if it's a :class:`PostMeasurement` subclass,
        otherwise ``"ops"`` (which is also the bucket for nested
        pipelines).

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
# JSON serialization for ``dcc.Store`` (legacy + DAG, dispatching by schema)
# ---------------------------------------------------------------------------


def _scope_to_dict(scope: _LegacyBuilderScope) -> Dict[str, Any]:
    """Recursively dump a :class:`_LegacyBuilderScope` to a JSON-friendly dict.

    Args:
        scope: The scope to serialize.

    Returns:
        A nested dict mirroring the :class:`_LegacyBuilderScope` shape.
        Aux operations are serialised inline inside each consumer's
        ``aux_ports`` entry (no separate ``aux_nodes`` list).
    """

    return {
        "nodes": [_node_to_dict(n) for n in scope.nodes],
        "name": scope.name,
        "desc": scope.desc,
        "nrows": scope.nrows,
        "ncols": scope.ncols,
    }


def _node_to_dict(node: _LegacyStepNode) -> Dict[str, Any]:
    """Recursively dump a :class:`_LegacyStepNode` to a JSON-friendly dict.

    Aux ports' embedded :class:`_LegacyStepNode` slots recurse via this
    same function so the entire aux subtree round-trips through JSON.

    Args:
        node: The node to serialize.

    Returns:
        A dict mirroring the :class:`_LegacyStepNode` shape.
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


def _scope_from_dict(data: Dict[str, Any]) -> _LegacyBuilderScope:
    """Inverse of :func:`_scope_to_dict`.

    Args:
        data: Dict previously produced by :func:`_scope_to_dict` (or
            an equivalent JSON payload).

    Returns:
        A reconstructed :class:`_LegacyBuilderScope`.
    """

    nodes = [_node_from_dict(n) for n in data.get("nodes", [])]
    return _LegacyBuilderScope(
        nodes=nodes,
        name=data.get("name", "Pipeline"),
        desc=data.get("desc", ""),
        nrows=data.get("nrows"),
        ncols=data.get("ncols"),
    )


def _node_from_dict(data: Dict[str, Any]) -> _LegacyStepNode:
    """Inverse of :func:`_node_to_dict`.

    Recursively reconstructs embedded aux :class:`_LegacyStepNode`
    slots from their serialised form so the full aux subtree survives
    JSON round-trip.

    Args:
        data: Dict previously produced by :func:`_node_to_dict`.

    Returns:
        A reconstructed :class:`_LegacyStepNode` (with nested scope and
        embedded aux slots recursed).
    """

    nested_data = data.get("nested")
    nested = _scope_from_dict(nested_data) if nested_data is not None else None
    raw_aux = data.get("aux_ports") or {}
    aux_ports: Dict[str, List[Optional[_LegacyStepNode]]] = {}
    for port_name, slots in raw_aux.items():
        aux_ports[str(port_name)] = [
            _node_from_dict(s) if isinstance(s, dict) else None for s in slots
        ]
    return _LegacyStepNode(
        node_id=data["node_id"],
        class_name=data["class_name"],
        params=data.get("params", {}) or {},
        label=data.get("label"),
        nested=nested,
        aux_ports=aux_ports,
    )


# ---------------------------------------------------------------------------
# DAG JSON encoding helpers (spec §8.4 fixture shape)
# ---------------------------------------------------------------------------


def _dag_scope_to_dict(scope: _DagBuilderScope) -> Dict[str, Any]:
    """Recursively dump a :class:`_DagBuilderScope` to a JSON-friendly dict.

    Args:
        scope: The scope to serialize.

    Returns:
        ``{"blocks": [...], "edges": [...], "name": ..., "desc": ...,
        "nrows": ..., "ncols": ...}``.  Each block's ``nested`` field
        recurses to the same shape, or ``null``.
    """

    return {
        "blocks": [_dag_block_to_dict(b) for b in scope.blocks],
        "edges": [_dag_edge_to_dict(e) for e in scope.edges],
        "name": scope.name,
        "desc": scope.desc,
        "nrows": scope.nrows,
        "ncols": scope.ncols,
    }


def _dag_block_to_dict(block: BlockNode) -> Dict[str, Any]:
    """Recursively dump a :class:`BlockNode` to a JSON-friendly dict.

    Args:
        block: The block to serialize.

    Returns:
        A dict mirroring the :class:`BlockNode` shape; ``nested`` is
        either the recursive scope dict or ``None``.
    """

    return {
        "block_id": block.block_id,
        "class_name": block.class_name,
        "params": block.params,
        "label": block.label,
        "nested": (
            _dag_scope_to_dict(block.nested) if block.nested is not None else None
        ),
        "collapsed": block.collapsed,
        "list_slot_counts": dict(block.list_slot_counts),
    }


def _dag_edge_to_dict(edge: Edge) -> Dict[str, Any]:
    """Dump an :class:`Edge` to a JSON-friendly dict.

    Args:
        edge: The edge to serialize.

    Returns:
        A dict mirroring the :class:`Edge` shape.
    """

    return {
        "edge_id": edge.edge_id,
        "source_block_id": edge.source_block_id,
        "source_port": edge.source_port,
        "target_block_id": edge.target_block_id,
        "target_port": edge.target_port,
        "target_slot": edge.target_slot,
        "kind": edge.kind,
    }


def _dag_scope_from_dict(data: Dict[str, Any]) -> _DagBuilderScope:
    """Inverse of :func:`_dag_scope_to_dict`.

    The constructed :class:`_DagBuilderScope` runs its
    :meth:`__post_init__` and therefore calls :func:`_seed_input_image`
    after loading — corrupt payloads missing an ``InputImage`` block
    are healed in place.

    Args:
        data: Dict previously produced by :func:`_dag_scope_to_dict`
            (or an equivalent JSON payload).

    Returns:
        A reconstructed :class:`_DagBuilderScope` with the Rule 6
        invariant restored.
    """

    blocks = [_dag_block_from_dict(b) for b in data.get("blocks", [])]
    edges = [_dag_edge_from_dict(e) for e in data.get("edges", [])]
    return _DagBuilderScope(
        blocks=blocks,
        edges=edges,
        name=data.get("name", "Pipeline"),
        desc=data.get("desc", ""),
        nrows=data.get("nrows"),
        ncols=data.get("ncols"),
    )


def _dag_block_from_dict(data: Dict[str, Any]) -> BlockNode:
    """Inverse of :func:`_dag_block_to_dict`.

    Recursively reconstructs nested scopes so the entire container tree
    survives JSON round-trip.

    Args:
        data: Dict previously produced by :func:`_dag_block_to_dict`.

    Returns:
        A reconstructed :class:`BlockNode`.
    """

    nested_data = data.get("nested")
    nested = (
        _dag_scope_from_dict(nested_data) if nested_data is not None else None
    )
    return BlockNode(
        block_id=data["block_id"],
        class_name=data["class_name"],
        params=data.get("params", {}) or {},
        label=data.get("label"),
        nested=nested,
        collapsed=bool(data.get("collapsed", False)),
        list_slot_counts=dict(data.get("list_slot_counts") or {}),
    )


def _dag_edge_from_dict(data: Dict[str, Any]) -> Edge:
    """Inverse of :func:`_dag_edge_to_dict`.

    Args:
        data: Dict previously produced by :func:`_dag_edge_to_dict`.

    Returns:
        A reconstructed :class:`Edge`.
    """

    kind_raw = data.get("kind", "image")
    if kind_raw not in {"image", "aux"}:
        raise ValueError(
            f"Edge.kind must be 'image' or 'aux', got {kind_raw!r}"
        )
    return Edge(
        edge_id=data["edge_id"],
        source_block_id=data["source_block_id"],
        source_port=data.get("source_port", "out"),
        target_block_id=data["target_block_id"],
        target_port=data.get("target_port", ""),
        target_slot=data.get("target_slot"),
        kind=kind_raw,  # type: ignore[arg-type]  # Literal narrowing above.
    )


def _heal_dag_scope_tree(scope: _DagBuilderScope) -> None:
    """Recursively re-run :func:`_seed_input_image` on every nested scope.

    :meth:`_DagBuilderScope.__post_init__` only fires the seed pass on
    the scope itself; if a container's ``nested`` scope was constructed
    via ``_dag_scope_from_dict`` it will have its own ``__post_init__``
    fire automatically.  This helper exists for callers that hand-roll
    a tree (e.g. the deserialiser branch in :func:`state_from_json`)
    and want a defense-in-depth pass over every descendant scope.

    Args:
        scope: The root scope to heal in place.
    """

    _seed_input_image(scope)
    for block in scope.blocks:
        if block.nested is not None:
            _heal_dag_scope_tree(block.nested)


def state_to_json(state: Any) -> Dict[str, Any]:
    """Convert a :class:`_LegacyBuilderState` or :class:`_DagBuilderState`
    into a JSON-friendly dict.

    The returned shape carries a top-level ``"_schema"`` discriminator
    so :func:`state_from_json` can decode unambiguously:

    * ``"_schema": "legacy"`` — encoded legacy linear-list state.
    * ``"_schema": "dag"`` — encoded DAG state.

    Suitable for round-tripping through ``dcc.Store``; the output
    contains only stdlib-compatible types provided the underlying
    ``params`` values are themselves JSON-friendly.

    Args:
        state: A :class:`_LegacyBuilderState` or :class:`_DagBuilderState`
            instance.

    Returns:
        Dict with ``"_schema"`` plus schema-specific keys (``root``,
        ``breadcrumb``, ``selected_node_id`` / ``selected_block_id``,
        etc.).

    Raises:
        TypeError: If *state* is not one of the recognised state
            dataclasses.
    """

    # Duck-typed dispatch (resilient to ``importlib.reload`` in tests that
    # would otherwise spawn fresh class objects and break ``isinstance``).
    if hasattr(state, "selected_block_id"):
        return {
            "_schema": "dag",
            "root": _dag_scope_to_dict(state.root),
            "breadcrumb": list(state.breadcrumb),
            "selected_block_id": state.selected_block_id,
            "selected_edge_id": state.selected_edge_id,
            "pending_delete_block_id": state.pending_delete_block_id,
            "toast_queue": list(state.toast_queue),
        }
    if hasattr(state, "selected_node_id"):
        return {
            "_schema": "legacy",
            "root": _scope_to_dict(state.root),
            "breadcrumb": list(state.breadcrumb),
            "selected_node_id": state.selected_node_id,
        }
    raise TypeError(
        f"state_to_json expects a _LegacyBuilderState or _DagBuilderState; "
        f"got {type(state).__name__}"
    )


def state_from_json(data: Dict[str, Any]) -> Any:
    """Inverse of :func:`state_to_json`.

    Dispatches by the top-level ``"_schema"`` key.  When the key is
    absent, the loader inspects the root dict's keys: ``"blocks"`` /
    ``"edges"`` mean DAG; ``"nodes"`` means legacy.  This guarantees
    forward-compatibility with older payloads that predated the
    discriminator while keeping the new schema unambiguous.

    After decoding a DAG payload, :func:`_heal_dag_scope_tree` walks
    every reachable scope so Rule 6 (one ``InputImage`` per scope) is
    re-established even if the JSON arrived corrupt.

    Args:
        data: Dict previously produced by :func:`state_to_json` or an
            equivalent JSON payload.

    Returns:
        A reconstructed :class:`_LegacyBuilderState` or
        :class:`_DagBuilderState` depending on the encoded schema.
    """

    schema = data.get("_schema")
    if schema == "dag" or _looks_like_dag_payload(data):
        return _state_from_json_dag(data)
    return _state_from_json_legacy(data)


def _looks_like_dag_payload(data: Dict[str, Any]) -> bool:
    """Heuristic dispatch for legacy payloads that predate ``_schema``.

    A DAG-shaped payload has either a top-level ``root.blocks`` /
    ``root.edges`` (the canonical case) or, when the payload IS the
    scope itself (no wrapping ``BuilderState``), top-level ``blocks`` /
    ``edges``.

    Args:
        data: A loaded JSON payload.

    Returns:
        ``True`` if *data* matches a DAG shape, ``False`` otherwise.
    """

    if "blocks" in data and "edges" in data:
        return True
    root = data.get("root")
    if isinstance(root, dict) and "blocks" in root and "edges" in root:
        return True
    return False


def _state_from_json_dag(data: Dict[str, Any]) -> _DagBuilderState:
    """Decode a DAG-shaped state payload.

    Accepts both the full-state shape (``{"_schema": "dag", "root":
    {...}, ...}``) and a bare scope dict (``{"blocks": [...], "edges":
    [...]}``); the latter is the on-disk shape of the per-fixture
    files under ``tests/fixtures/builder_dag/``.

    Args:
        data: The loaded JSON payload.

    Returns:
        A reconstructed :class:`_DagBuilderState`.
    """

    if "blocks" in data and "edges" in data and "root" not in data:
        # Bare scope payload — wrap it in a state with default everything.
        root = _dag_scope_from_dict(data)
        state = _DagBuilderState(root=root)
        _heal_dag_scope_tree(state.root)
        return state

    root_data = data.get("root") or {}
    root = _dag_scope_from_dict(root_data)
    state = _DagBuilderState(
        root=root,
        breadcrumb=list(data.get("breadcrumb") or []),
        selected_block_id=data.get("selected_block_id"),
        selected_edge_id=data.get("selected_edge_id"),
        pending_delete_block_id=data.get("pending_delete_block_id"),
        toast_queue=list(data.get("toast_queue") or []),
    )
    _heal_dag_scope_tree(state.root)
    return state


def _state_from_json_legacy(data: Dict[str, Any]) -> _LegacyBuilderState:
    """Decode a legacy linear-list state payload.

    Args:
        data: The loaded JSON payload.

    Returns:
        A reconstructed :class:`_LegacyBuilderState`.
    """

    root_data = data.get("root")
    root = (
        _scope_from_dict(root_data)
        if root_data is not None
        else _LegacyBuilderScope()
    )
    raw_crumbs = data.get("breadcrumb", []) or []
    crumbs = [_normalize_breadcrumb_segment(seg) for seg in raw_crumbs]
    return _LegacyBuilderState(
        root=root,
        breadcrumb=crumbs,
        selected_node_id=data.get("selected_node_id"),
    )
