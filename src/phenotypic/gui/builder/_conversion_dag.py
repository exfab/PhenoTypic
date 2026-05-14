"""DAG ↔ :class:`~phenotypic.ImagePipeline` conversion for the new builder.

This module is the conversion seam between the DAG-shaped
:class:`~phenotypic.gui.builder._state.BuilderState` (block + edge model
introduced in Phase 1 of the builder redesign — see spec §5.4) and the
runtime :class:`~phenotypic.ImagePipeline`.

The two public functions mirror today's :func:`to_pipeline` /
:func:`from_pipeline` but operate on the DAG schema:

* :func:`to_pipeline_dag` runs validation first and **raises**
  :class:`ValueError` on any blocking issue. On a clean state it
  topologically walks the image-flow edges from the ``InputImage``
  sentinel block, folds aux-typed edges into each consumer's ``params``
  dict via :func:`_block_to_marker`, partitions the result by
  :class:`MeasureFeatures` / :class:`PostMeasurement` membership, and
  hands ``ops`` / ``meas`` / ``post`` lists to a fresh
  :class:`ImagePipeline`.

* :func:`from_pipeline_dag` walks an existing pipeline's ops / meas /
  post lists, mints one :class:`BlockNode` per entry, draws image-flow
  edges between consecutive entries, and recursively extracts
  op-typed parameters as aux edges + embedded aux blocks. Shared
  operation instances (the same Python ``id(op)`` in both ``_ops`` and a
  consumer's aux param — the legacy popover model permitted this) are
  deep-copied into a fresh :class:`BlockNode` and a single info toast is
  queued summarising how many were cloned.

Both functions are stateless and side-effect-free apart from the toast
queue mutation in :func:`from_pipeline_dag`.

Examples:
    Round-trip a tiny linear chain through the DAG conversion:

    >>> from phenotypic.gui.builder._conversion_dag import (
    ...     to_pipeline_dag, from_pipeline_dag,
    ... )
    >>> # state is a BuilderState authored against the spec schema
    >>> # pipeline = to_pipeline_dag(state)
    >>> # state2 = from_pipeline_dag(pipeline)
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageOperation, MeasureFeatures, PostMeasurement
from phenotypic.gui._operation_registry import OperationRegistry, get_registry
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    Edge,
    _DagBuilderScope as BuilderScope,
    _DagBuilderState as BuilderState,
    _new_block_id,
    _seed_input_image,
)
from phenotypic.gui.builder._validation import Issue, validate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# to_pipeline_dag — DAG → ImagePipeline
# ---------------------------------------------------------------------------


def to_pipeline_dag(state: BuilderState) -> ImagePipeline:
    """Convert a DAG-shaped :class:`BuilderState` to an :class:`ImagePipeline`.

    Steps (spec §5.4):

    1. Validate the state; raise :class:`ValueError` on any blocking
       issue (``severity == "error"``).
    2. Walk image-flow edges from the ``InputImage`` block in the root
       scope to produce a topological order of non-input blocks.
    3. For each block, fold any aux edges targeting it into its
       ``params`` dict — scalar aux at ``params[port]``, list-aux at
       ``params[port][slot]`` with empty slots set to ``None``.
    4. Partition the resulting instances by
       :class:`~phenotypic.abc_.MeasureFeatures` /
       :class:`~phenotypic.abc_.PostMeasurement` membership into
       ``ops`` / ``meas`` / ``post`` lists.
    5. Construct a fresh :class:`ImagePipeline` with the root scope's
       ``name``, ``desc``, ``nrows``, ``ncols``.

    Aux-only blocks (those without an outgoing image-flow edge) are
    never returned as top-level entries — they are folded into the
    consumer's ``params`` via :func:`_block_to_marker` recursion.

    Args:
        state: The :class:`BuilderState` to materialise.

    Returns:
        A fresh :class:`ImagePipeline`.

    Raises:
        ValueError: If :func:`validate` returns any issue with
            ``severity == "error"``. The message lists each rule kind
            and the offending block's label so the GUI toast can
            surface a deterministic explanation.

    Examples:
        Build a one-op DAG (Input Image → GaussianBlur) and materialise
        it as an :class:`~phenotypic.ImagePipeline`. The synth-yeast
        fixture is loaded only to anchor the example in the project's
        microbiology context — the conversion does not touch image
        data:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.gui.builder._state import (
        ...     BuilderScope, BuilderState, BlockNode, Edge,
        ...     _new_block_id,
        ... )
        >>> from phenotypic.gui.builder._conversion_dag import to_pipeline_dag
        >>> scope = BuilderScope(name="demo", desc="one-op chain")
        >>> input_block = scope.blocks[0]
        >>> blur = BlockNode(
        ...     block_id=_new_block_id(),
        ...     class_name="GaussianBlur",
        ...     params={},
        ...     label="blur",
        ... )
        >>> scope.blocks.append(blur)
        >>> scope.edges.append(Edge(
        ...     edge_id=_new_block_id(),
        ...     source_block_id=input_block.block_id,
        ...     target_block_id=blur.block_id,
        ...     target_port="in",
        ...     kind="image",
        ... ))
        >>> state = BuilderState(root=scope)
        >>> pipeline = to_pipeline_dag(state)
        >>> len(pipeline.get_ops())
        1
        >>> _ = load_synth_yeast_plate()  # microbiology context anchor
    """
    errors = [iss for iss in validate(state) if iss.severity == "error"]
    if errors:
        _raise_validation_error(state.root, errors)

    registry = get_registry()
    instances = _materialise_scope(state.root, registry)

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
        name=state.root.name,
        desc=state.root.desc,
        nrows=state.root.nrows,
        ncols=state.root.ncols,
    )


def _raise_validation_error(scope: BuilderScope, errors: List[Issue]) -> None:
    """Raise a :class:`ValueError` summarising every blocking issue.

    The message format is fixed by spec §5.4: a count, an em-dash, and a
    comma-separated list of ``kind(label)`` pairs so the toast surface
    can show a deterministic phrase regardless of issue ordering.

    Args:
        scope: Root :class:`BuilderScope` (used to resolve block labels
            for friendlier error messages).
        errors: Issues filtered to ``severity == "error"``.

    Raises:
        ValueError: Always; constructed from the supplied issues.
    """

    labels = _collect_block_labels(scope)

    def _label_for(block_id: Optional[str]) -> str:
        if block_id is None:
            return "scope"
        return labels.get(block_id, block_id[:8])

    summary = ", ".join(f"{iss.kind}({_label_for(iss.block_id)})" for iss in errors)
    raise ValueError(
        f"Cannot materialise pipeline: {len(errors)} validation error(s) — "
        f"{summary}"
    )


def _collect_block_labels(scope: BuilderScope) -> Dict[str, str]:
    """Walk every block (including nested containers) to map ``block_id`` → label.

    Used by :func:`_raise_validation_error` to render friendly names in
    the exception message.

    Args:
        scope: :class:`BuilderScope` to walk.

    Returns:
        Dict mapping every block's id to its label / class name.
    """

    labels: Dict[str, str] = {}
    stack: List[BuilderScope] = [scope]
    while stack:
        current = stack.pop()
        for block in current.blocks:
            labels[block.block_id] = block.label or block.class_name
            if block.nested is not None:
                stack.append(block.nested)
    return labels


def _materialise_scope(
    scope: BuilderScope, registry: OperationRegistry
) -> List[Any]:
    """Materialise a scope's blocks into runtime instances in topological order.

    Walks image-flow edges starting from the ``InputImage`` sentinel to
    produce an order; folds aux edges into each block's params; and
    instantiates each non-input block via the registry (or recursively
    via :func:`_materialise_scope` for nested containers).

    Args:
        scope: The :class:`BuilderScope` to walk.
        registry: Operation registry used for class lookup +
            ``create_instance`` calls.

    Returns:
        A list of runtime instances in execution order, **excluding**
        the ``InputImage`` sentinel.

    Raises:
        ValueError: If the topological walk reveals a structural
            invariant violation that validation didn't catch (defensive
            — should never fire on a validated state).
    """

    input_block = _find_input_block(scope)
    order = _topological_image_order(scope, input_block)

    aux_edges_by_target = _index_aux_edges_by_target(scope.edges)
    blocks_by_id = {b.block_id: b for b in scope.blocks}

    instances: List[Any] = []
    aux_only_block_ids = _aux_only_block_ids(scope)

    for block in order:
        if block.block_id == input_block.block_id:
            continue
        if block.block_id in aux_only_block_ids:
            # Defensive: aux-only blocks must not appear in the
            # image-flow order; the topological walk that produced
            # ``order`` only emits blocks reachable via outgoing image
            # edges, so this branch is unreachable. Keep the guard so a
            # future refactor that broadens the walk does not silently
            # ship aux-only blocks as top-level ops.
            raise ValueError(
                f"aux-only block {block.block_id!r} surfaced in topological "
                "order; this indicates a bug in the image-flow walk"
            )
        params = _build_params_with_aux(
            block=block,
            aux_edges=aux_edges_by_target.get(block.block_id, []),
            blocks_by_id=blocks_by_id,
            aux_edges_by_target=aux_edges_by_target,
            registry=registry,
        )
        instances.append(_instantiate_block(block, params, registry))

    return instances


def _find_input_block(scope: BuilderScope) -> BlockNode:
    """Return the ``InputImage`` block of *scope* or raise.

    Validation rule 6 catches missing inputs before
    :func:`to_pipeline_dag` reaches this helper, but the conversion
    layer keeps a defensive guard so callers that bypass
    :func:`to_pipeline_dag` (e.g. internal recursion) still get a clear
    error.

    Args:
        scope: The :class:`BuilderScope` to search.

    Returns:
        The first ``InputImage`` block in scope order.

    Raises:
        ValueError: If no ``InputImage`` block exists.
    """

    for block in scope.blocks:
        if block.class_name == INPUT_IMAGE_CLASS_NAME:
            return block
    raise ValueError("scope is missing an InputImage block (Rule 6)")


def _topological_image_order(
    scope: BuilderScope, input_block: BlockNode
) -> List[BlockNode]:
    """Walk image-flow edges from ``InputImage`` to produce execution order.

    Validation rule 1 guarantees no forks (each output has ≤1 outgoing
    image wire), so the image-flow subgraph is a chain (or several
    chains, but only the one rooted at ``InputImage`` is honoured here
    — orphans are caught by Rule 2). Rule 4 guarantees acyclicity.

    Args:
        scope: The :class:`BuilderScope` whose edges drive the walk.
        input_block: The starting ``InputImage`` block.

    Returns:
        Ordered list of :class:`BlockNode` objects starting with
        ``input_block`` and proceeding via image-flow edges.
    """

    next_by_source: Dict[str, str] = {}
    for edge in scope.edges:
        if edge.kind == "image":
            # Rule 1 (fork) already validated: at most one outgoing
            # image edge per source. Last-write-wins matches the
            # behaviour of a downstream renderer that picked the most
            # recently created edge.
            next_by_source[edge.source_block_id] = edge.target_block_id

    blocks_by_id = {b.block_id: b for b in scope.blocks}
    order: List[BlockNode] = []
    visited: set[str] = set()
    current_id: Optional[str] = input_block.block_id
    while current_id is not None:
        if current_id in visited:
            # Cycle detection already runs in validate(); guard here so
            # an unvalidated direct call doesn't infinite-loop.
            break
        visited.add(current_id)
        block = blocks_by_id.get(current_id)
        if block is None:
            break
        order.append(block)
        current_id = next_by_source.get(current_id)
    return order


def _index_aux_edges_by_target(edges: List[Edge]) -> Dict[str, List[Edge]]:
    """Group aux edges by target block id for O(1) lookup.

    Args:
        edges: All edges in the scope.

    Returns:
        Dict mapping ``target_block_id`` to the list of aux edges that
        terminate there. Image edges are excluded.
    """

    grouped: Dict[str, List[Edge]] = defaultdict(list)
    for edge in edges:
        if edge.kind == "aux":
            grouped[edge.target_block_id].append(edge)
    return grouped


def _aux_only_block_ids(scope: BuilderScope) -> set[str]:
    """Return the set of block ids that have no outgoing image-flow edge.

    Aux-only blocks are sources of aux wires only; they are folded into
    consumer params rather than appearing as top-level ops.

    Args:
        scope: The :class:`BuilderScope` to scan.

    Returns:
        Set of ``block_id`` strings for aux-only blocks (excluding the
        ``InputImage`` sentinel, which has no outgoing image edge by
        construction but is never an aux source).
    """

    sources_with_image_out: set[str] = set()
    for edge in scope.edges:
        if edge.kind == "image":
            sources_with_image_out.add(edge.source_block_id)
    aux_only: set[str] = set()
    for block in scope.blocks:
        if block.class_name == INPUT_IMAGE_CLASS_NAME:
            continue
        if block.block_id in sources_with_image_out:
            continue
        # No outgoing image edge → aux source or terminal node.
        aux_only.add(block.block_id)
    # Terminal nodes (the final consumer in the chain) also have no
    # outgoing image edge. Remove them: terminal nodes have an
    # **incoming** image edge.
    image_in_targets: set[str] = set()
    for edge in scope.edges:
        if edge.kind == "image":
            image_in_targets.add(edge.target_block_id)
    return {bid for bid in aux_only if bid not in image_in_targets}


def _build_params_with_aux(
    *,
    block: BlockNode,
    aux_edges: List[Edge],
    blocks_by_id: Dict[str, BlockNode],
    aux_edges_by_target: Dict[str, List[Edge]],
    registry: OperationRegistry,
) -> Dict[str, Any]:
    """Fold aux edges + scalar params into a runtime params dict for *block*.

    Each aux edge resolves its source block to a marker via
    :func:`_block_to_marker`. Scalar aux ports collapse to a single
    marker (or ``None``); list-aux ports collapse to a list with
    ``None`` filling any empty slots ``[0, count)`` not covered by an
    edge.

    Args:
        block: Consumer :class:`BlockNode`.
        aux_edges: Aux edges targeting ``block``.
        blocks_by_id: Block-id lookup for the current scope (so edges
            can resolve their sources).
        aux_edges_by_target: Pre-computed index of all aux edges in the
            current scope (needed so :func:`_block_to_marker` recursion
            can fold aux-of-aux on its own source block).
        registry: Registry for class lookup (needed by
            :func:`_block_to_marker` recursion and to detect list-typed
            params).

    Returns:
        Params dict ready for ``registry.create_instance(class_name,
        **params)``.
    """

    params: Dict[str, Any] = dict(block.params)
    info = registry.get(block.class_name)

    # Group aux edges by target port so list-aux slots assemble.
    edges_by_port: Dict[str, List[Edge]] = defaultdict(list)
    for edge in aux_edges:
        edges_by_port[edge.target_port].append(edge)

    for port_name, port_edges in edges_by_port.items():
        param_info = info.parameters.get(port_name) if info is not None else None
        is_list = bool(param_info and param_info.is_list)
        if is_list:
            slot_count = block.list_slot_counts.get(port_name, 0)
            # Edges' explicit slot indices may exceed the recorded
            # ``list_slot_counts`` if the state was hand-edited; grow
            # to fit so we never drop a wire.
            for edge in port_edges:
                if edge.target_slot is not None:
                    slot_count = max(slot_count, edge.target_slot + 1)
            slots: List[Any] = [None] * slot_count
            for edge in port_edges:
                slot_index = edge.target_slot if edge.target_slot is not None else 0
                source_block = blocks_by_id.get(edge.source_block_id)
                if source_block is None:
                    continue
                marker = _block_to_marker(
                    source_block, blocks_by_id, aux_edges_by_target, registry
                )
                if slot_index < len(slots):
                    slots[slot_index] = marker
            params[port_name] = slots
        else:
            # Scalar aux: take the first edge's source.
            edge = port_edges[0]
            source_block = blocks_by_id.get(edge.source_block_id)
            if source_block is not None:
                params[port_name] = _block_to_marker(
                    source_block, blocks_by_id, aux_edges_by_target, registry
                )

    # Honour explicit ``list_slot_counts`` for ports that have empty
    # slots but no wired edges — surface as an empty list with ``None``
    # entries so the runtime sees a deterministic placeholder.
    for port_name, slot_count in block.list_slot_counts.items():
        if port_name in edges_by_port:
            continue
        param_info = info.parameters.get(port_name) if info is not None else None
        if param_info is None or not param_info.is_list:
            continue
        params[port_name] = [None] * slot_count

    return params


def _block_to_marker(
    block: BlockNode,
    blocks_by_id: Dict[str, BlockNode],
    aux_edges_by_target: Dict[str, List[Edge]],
    registry: OperationRegistry,
) -> Dict[str, Any]:
    """Serialise an aux-source block into a runtime marker dict.

    Marker shape matches the legacy
    :func:`phenotypic.gui.builder._state._fold_aux_ports_for_node`
    convention:

    * Container source: a ``pipeline_instance`` marker carrying a
      pre-materialised :class:`ImagePipeline` (the container is
      recursively converted via :func:`_materialise_scope`).
    * Operation source: ``{"__type__": "operation", "class_name":
      <name>, "params": {...}}`` where ``params`` is the source
      block's scalar params merged with any aux edges that target it
      (aux-of-aux recursion via the same edge index).

    Args:
        block: Source :class:`BlockNode`.
        blocks_by_id: Block-id lookup for the source's enclosing scope.
        aux_edges_by_target: Pre-computed index of every aux edge in
            the enclosing scope, keyed by ``target_block_id``. Used to
            fold aux-of-aux on the source block.
        registry: Registry for class lookup.

    Returns:
        Marker dict consumable by ``_resolve_marker``.
    """

    if block.class_name == PIPELINE_CLASS_NAME:
        # Nested container as aux source: materialise its inner scope
        # to an :class:`ImagePipeline` via recursive
        # :func:`_materialise_scope`.
        inner = block.nested or BuilderScope()
        inner_instances = _materialise_scope(inner, registry)
        inner_pipeline = _split_and_build_pipeline(inner_instances, inner)
        return {
            "__type__": "pipeline_instance",
            "instance": inner_pipeline,
        }

    # Build the source block's effective params (its scalar params +
    # aux folds from aux-of-aux edges in the same scope).
    aux_edges_for_block = aux_edges_by_target.get(block.block_id, [])
    params = _build_params_with_aux(
        block=block,
        aux_edges=aux_edges_for_block,
        blocks_by_id=blocks_by_id,
        aux_edges_by_target=aux_edges_by_target,
        registry=registry,
    )
    return {
        "__type__": "operation",
        "class_name": block.class_name,
        "params": params,
    }


def _instantiate_block(
    block: BlockNode, params: Dict[str, Any], registry: OperationRegistry
) -> Any:
    """Convert a block + folded params into a runtime instance.

    Container blocks recurse through :func:`_materialise_scope` to
    produce an inner :class:`ImagePipeline`; ordinary blocks dispatch
    to ``registry.create_instance``.

    Args:
        block: The :class:`BlockNode` to instantiate.
        params: Folded params (scalar + aux markers).
        registry: Operation registry.

    Returns:
        Runtime instance (``ImageOperation`` or ``ImagePipeline``).

    Raises:
        ValueError: If the block has an unknown class name (validation
            should have flagged this with an advisory, but we still
            raise if asked to materialise the unknown class).
    """

    if block.class_name == PIPELINE_CLASS_NAME:
        inner_scope = block.nested or BuilderScope()
        inner_instances = _materialise_scope(inner_scope, registry)
        return _split_and_build_pipeline(inner_instances, inner_scope)

    info = registry.get(block.class_name)
    if info is None:
        raise ValueError(
            f"class {block.class_name!r} is not in the operation registry "
            "(advisory issue should have surfaced this before to_pipeline_dag)"
        )

    resolved_params = {
        name: _resolve_marker(value, registry) for name, value in params.items()
    }
    return registry.create_instance(block.class_name, **resolved_params)


def _resolve_marker(value: Any, registry: OperationRegistry) -> Any:
    """Walk a folded param value and resolve marker dicts to instances.

    Markers minted by :func:`_block_to_marker` are dicts shaped like
    ``{"__type__": "operation", ...}`` (or ``"pipeline_instance"`` for
    pre-materialised containers). Scalars / lists pass through; nested
    markers recurse.

    Args:
        value: Folded param value (scalar, list, or marker).
        registry: Registry for class lookup.

    Returns:
        Runtime value (scalar or :class:`ImageOperation` /
        :class:`ImagePipeline` instance).
    """

    if isinstance(value, dict) and value.get("__type__") == "pipeline_instance":
        # Already a materialised ImagePipeline instance handed in by
        # :func:`_block_to_marker` for container sources.
        return value["instance"]
    if isinstance(value, dict) and value.get("__type__") == "operation":
        class_name = value["class_name"]
        inner = value.get("params") or {}
        resolved_inner = {k: _resolve_marker(v, registry) for k, v in inner.items()}
        return registry.create_instance(class_name, **resolved_inner)
    if isinstance(value, list):
        return [_resolve_marker(v, registry) for v in value]
    return value


def _split_and_build_pipeline(
    instances: List[Any], scope: BuilderScope
) -> ImagePipeline:
    """Partition *instances* into ops / meas / post and construct a pipeline.

    Used both at the top level (by :func:`to_pipeline_dag`) and for
    container sub-pipelines. Container scopes leave ``nrows``/``ncols``
    as ``None`` per spec §4.5.

    Args:
        instances: Materialised runtime instances in execution order.
        scope: The scope they came from (carries ``name`` / ``desc`` /
            ``nrows`` / ``ncols``).

    Returns:
        A fresh :class:`ImagePipeline`.
    """

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


# ---------------------------------------------------------------------------
# from_pipeline_dag — ImagePipeline → DAG
# ---------------------------------------------------------------------------


def from_pipeline_dag(pipeline: ImagePipeline) -> BuilderState:
    """Convert an :class:`ImagePipeline` to a DAG-shaped :class:`BuilderState`.

    Steps (spec §5.4):

    1. Construct an empty :class:`BuilderState` (the root
       :class:`BuilderScope` auto-seeds an ``InputImage`` block via
       its ``__post_init__``).
    2. Walk ``pipeline.get_ops() + get_meas() + get_post()`` minting
       one :class:`BlockNode` per entry and adding ``"image"`` edges
       between consecutive entries (plus from ``InputImage`` to the
       first entry).
    3. For each block, extract op-typed parameters via the registry's
       :class:`ParamInfo` metadata into embedded aux blocks + aux
       edges. Recursive walks handle aux-of-aux + container aux.
    4. Track an ``id(op) → block_id`` map: when the same Python
       instance appears in both ``_ops`` and a consumer's aux param
       (legacy shared-instance edge case), deep-copy the aux source
       into a fresh block and append an info toast to
       ``state.toast_queue``.
    5. Copy ``pipeline.name``, ``pipeline._desc``, ``pipeline.nrows``,
       and ``pipeline.ncols`` to the **root** scope; container scopes
       leave ``nrows``/``ncols`` as ``None``.

    Args:
        pipeline: The :class:`ImagePipeline` to mirror.

    Returns:
        A :class:`BuilderState` whose root scope reproduces the
        pipeline.

    Examples:
        Mirror a tiny :class:`~phenotypic.ImagePipeline` back into a
        DAG :class:`BuilderState`. The auto-seeded ``InputImage``
        block sits at index 0 of the root scope's ``blocks`` list and
        the mirrored op follows at index 1:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.gui.builder._conversion_dag import (
        ...     from_pipeline_dag,
        ... )
        >>> pipeline = ImagePipeline(ops=[GaussianBlur()], name="demo")
        >>> state = from_pipeline_dag(pipeline)
        >>> [b.class_name for b in state.root.blocks]
        ['InputImage', 'GaussianBlur']
        >>> _ = load_synth_yeast_plate()  # microbiology context anchor
    """

    state = BuilderState()
    registry = get_registry()
    dedup: Dict[int, str] = {}
    clone_counter = _CloneCounter()

    _populate_scope_from_pipeline(
        scope=state.root,
        pipeline=pipeline,
        registry=registry,
        dedup=dedup,
        clone_counter=clone_counter,
        is_root=True,
    )

    if clone_counter.count > 0:
        state.toast_queue.append(
            {
                "kind": "info",
                "text": (
                    f"Loaded {clone_counter.count} shared operation "
                    "instance(s) as independent copies. The GUI does "
                    "not support sharing the same operation between "
                    "two consumers."
                ),
            }
        )

    return state


class _CloneCounter:
    """Mutable counter passed through the recursive walk.

    Plain ``int`` would force every helper to return its incremented
    value; a small object keeps the call sites readable.
    """

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def bump(self) -> None:
        self.count += 1


def _populate_scope_from_pipeline(
    scope: BuilderScope,
    pipeline: ImagePipeline,
    registry: OperationRegistry,
    dedup: Dict[int, str],
    clone_counter: _CloneCounter,
    is_root: bool,
) -> None:
    """Populate *scope* in place with blocks + edges from *pipeline*.

    The root scope's ``__post_init__`` already seeded an
    ``InputImage`` block; this helper appends the rest.

    Args:
        scope: Target :class:`BuilderScope` (already constructed +
            auto-seeded with ``InputImage``).
        pipeline: Source :class:`ImagePipeline`.
        registry: Operation registry.
        dedup: ``id(op) → block_id`` map for shared-instance detection.
            Maintained across the whole recursion so a deepcopy in one
            sub-pipeline doesn't fool a sibling.
        clone_counter: Mutable clone counter for the info toast.
        is_root: ``True`` for the top-level scope; ``False`` for
            recursive container sub-scopes. Only the root copies
            ``nrows`` / ``ncols`` from the pipeline (spec §4.5).
    """

    # Copy scope-level metadata.
    scope.name = pipeline.name
    scope.desc = pipeline._desc if pipeline._desc is not None else ""
    if is_root:
        scope.nrows = pipeline.nrows
        scope.ncols = pipeline.ncols
    # else: leave nrows/ncols as None per spec §4.5

    # The root's __post_init__ already seeded an InputImage. Ensure it
    # exists (the dataclass may have been constructed without
    # __post_init__ running in some edge cases).
    _seed_input_image(scope)
    input_block = _find_input_block(scope)

    pairs: List[Tuple[str, Any]] = []
    pairs.extend(pipeline.get_ops().items())
    pairs.extend(pipeline.get_meas().items())
    pairs.extend(pipeline.get_post().items())

    prev_block_id = input_block.block_id

    for label, op in pairs:
        block = _block_from_op(op, label, registry, dedup, clone_counter, scope)
        dedup[id(op)] = block.block_id
        scope.blocks.append(block)
        scope.edges.append(
            Edge(
                edge_id=_new_block_id(),
                source_block_id=prev_block_id,
                source_port="out",
                target_block_id=block.block_id,
                target_port="in",
                target_slot=None,
                kind="image",
            )
        )
        prev_block_id = block.block_id


def _block_from_op(
    op: Any,
    label: str,
    registry: OperationRegistry,
    dedup: Dict[int, str],
    clone_counter: _CloneCounter,
    scope: BuilderScope,
) -> BlockNode:
    """Build a :class:`BlockNode` for *op* and append its aux edges to *scope*.

    Walks ``op``'s op-typed parameters via the registry's
    :class:`ParamInfo` and mints embedded aux blocks + aux edges
    targeting the new block. Shared-instance clones are detected here.

    Args:
        op: Source runtime instance (``ImageOperation`` or
            ``ImagePipeline``).
        label: Pipeline-level label for the block (becomes
            ``block.label``).
        registry: Operation registry.
        dedup: Shared-instance map.
        clone_counter: Mutable clone counter.
        scope: The target scope (aux blocks + edges land here).

    Returns:
        Newly-constructed :class:`BlockNode` (already populated; the
        caller appends it to ``scope.blocks``).
    """

    block_id = _new_block_id()

    if isinstance(op, ImagePipeline):
        nested = BuilderScope()
        _populate_scope_from_pipeline(
            scope=nested,
            pipeline=op,
            registry=registry,
            dedup=dedup,
            clone_counter=clone_counter,
            is_root=False,
        )
        return BlockNode(
            block_id=block_id,
            class_name=PIPELINE_CLASS_NAME,
            params={},
            label=label,
            nested=nested,
        )

    class_name = type(op).__name__
    info = registry.get(class_name)
    params: Dict[str, Any] = {}
    list_slot_counts: Dict[str, int] = {}

    block = BlockNode(
        block_id=block_id,
        class_name=class_name,
        params=params,
        label=label,
        list_slot_counts=list_slot_counts,
    )

    if info is None:
        # Unknown class: best-effort serialisation from ``vars``.
        for attr_name, value in vars(op).items():
            if attr_name.startswith("_"):
                continue
            if not isinstance(value, (ImageOperation, ImagePipeline)):
                params[attr_name] = value
        return block

    for param_name, param_info in info.parameters.items():
        if not hasattr(op, param_name):
            continue
        current = getattr(op, param_name)
        aux_eligible = param_info.is_operation or param_info.is_pipeline
        if not aux_eligible:
            params[param_name] = current
            continue
        if current is None:
            continue
        if param_info.is_list:
            _attach_list_aux_param(
                consumer_block_id=block_id,
                port_name=param_name,
                sequence=current,
                scope=scope,
                registry=registry,
                dedup=dedup,
                clone_counter=clone_counter,
                list_slot_counts=list_slot_counts,
            )
        else:
            _attach_scalar_aux_param(
                consumer_block_id=block_id,
                port_name=param_name,
                source_op=current,
                scope=scope,
                registry=registry,
                dedup=dedup,
                clone_counter=clone_counter,
            )

    return block


def _attach_scalar_aux_param(
    consumer_block_id: str,
    port_name: str,
    source_op: Any,
    scope: BuilderScope,
    registry: OperationRegistry,
    dedup: Dict[int, str],
    clone_counter: _CloneCounter,
) -> None:
    """Mint an aux source block + edge for a scalar op-typed param.

    Args:
        consumer_block_id: ``block_id`` of the consumer block.
        port_name: Param name (becomes ``target_port`` on the edge).
        source_op: The op-typed value pulled from the consumer.
        scope: Target scope (aux block + edge land here).
        registry: Operation registry.
        dedup: Shared-instance map.
        clone_counter: Mutable clone counter.
    """

    cloned, source_op = _maybe_clone(source_op, dedup, clone_counter)
    aux_block = _block_from_op(
        source_op,
        type(source_op).__name__,
        registry,
        dedup,
        clone_counter,
        scope,
    )
    if not cloned:
        dedup[id(source_op)] = aux_block.block_id
    scope.blocks.append(aux_block)
    scope.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=aux_block.block_id,
            source_port="out",
            target_block_id=consumer_block_id,
            target_port=port_name,
            target_slot=None,
            kind="aux",
        )
    )


def _attach_list_aux_param(
    consumer_block_id: str,
    port_name: str,
    sequence: Any,
    scope: BuilderScope,
    registry: OperationRegistry,
    dedup: Dict[int, str],
    clone_counter: _CloneCounter,
    list_slot_counts: Dict[str, int],
) -> None:
    """Mint aux blocks + edges for a list-typed op-typed param.

    Empty (``None``) entries reserve slot positions without minting
    edges. ``list_slot_counts[port_name]`` records the total slot
    count so empty slots interspersed in the original list survive
    round-trip.

    Args:
        consumer_block_id: ``block_id`` of the consumer block.
        port_name: Param name (becomes ``target_port`` on each edge).
        sequence: The list-shaped value pulled from the consumer.
        scope: Target scope (aux blocks + edges land here).
        registry: Operation registry.
        dedup: Shared-instance map.
        clone_counter: Mutable clone counter.
        list_slot_counts: Mutable map for the consumer's
            ``list_slot_counts`` (port → total slot count).
    """

    items = list(sequence) if isinstance(sequence, (list, tuple)) else []
    list_slot_counts[port_name] = len(items)
    for slot_index, source_op in enumerate(items):
        if source_op is None:
            continue
        cloned, source_op = _maybe_clone(source_op, dedup, clone_counter)
        aux_block = _block_from_op(
            source_op,
            type(source_op).__name__,
            registry,
            dedup,
            clone_counter,
            scope,
        )
        if not cloned:
            dedup[id(source_op)] = aux_block.block_id
        scope.blocks.append(aux_block)
        scope.edges.append(
            Edge(
                edge_id=_new_block_id(),
                source_block_id=aux_block.block_id,
                source_port="out",
                target_block_id=consumer_block_id,
                target_port=port_name,
                target_slot=slot_index,
                kind="aux",
            )
        )


def _maybe_clone(
    source_op: Any,
    dedup: Dict[int, str],
    clone_counter: _CloneCounter,
) -> Tuple[bool, Any]:
    """Deep-copy *source_op* if its Python id has already been seen.

    Args:
        source_op: Candidate aux source op.
        dedup: Shared-instance map (``id(op) → block_id``).
        clone_counter: Mutable counter; incremented on each clone.

    Returns:
        Tuple ``(cloned, op)`` where ``op`` is the (possibly cloned)
        source. ``cloned=True`` means a fresh instance is returned
        and the caller should NOT register its id in ``dedup``
        (because the consumer downstream of this clone owns it
        independently).
    """

    if source_op is None:
        return False, source_op
    if id(source_op) in dedup:
        clone_counter.bump()
        return True, copy.deepcopy(source_op)
    return False, source_op
