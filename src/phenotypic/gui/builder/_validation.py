"""Pure-Python validation layer for the DAG builder.

Validation is a free function over a :class:`BuilderState`; every state
mutation through the dispatcher pipes the new state through
:func:`validate` and the resulting list of :class:`Issue` objects drives
the canvas's red/yellow border decoration plus the toolbar's
preview/save enable state.

The blocking rules plus advisory stage-order hint are enumerated in spec
``2026-05-12-builder-dag-redesign-design.md`` §5.3. This module contains
no side effects; it depends only on :mod:`phenotypic.gui.builder._state`
(the dataclasses) and the operation registry (for required-aux
introspection and stage classification).

Examples:
    Empty scope (only auto-seeded ``InputImage``) is valid:

    >>> from phenotypic.gui.builder._state import BuilderScope, BuilderState
    >>> from phenotypic.gui.builder._validation import validate
    >>> state = BuilderState(root=BuilderScope())
    >>> validate(state)
    []
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from phenotypic.abc_ import MeasureFeatures, PostMeasurement
from phenotypic.gui._operation_registry import OperationInfo, get_registry
from phenotypic.gui.builder._state import (
    BlockNode,
    BuilderScope,
    BuilderState,
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
)


IssueKind = Literal[
    "fork",
    "stub",
    "required_aux",
    "cycle",
    "container_mode",
    "missing_input",
    "duplicate_input",
    "stage_order_hint",
    "unknown_class",
    "unsupported_linear",
]
IssueSeverity = Literal["error", "advisory"]
StageName = Literal["ops", "meas", "post", "pipeline"]

_STAGE_ORDER: Dict[StageName, int] = {
    "ops": 0,
    "meas": 1,
    "post": 2,
    "pipeline": 0,
}


@dataclass
class Issue:
    """A single validation finding for a block or scope.

    Attributes:
        kind: One of the nine taxonomy entries (see :data:`IssueKind`).
            The first seven are blocking errors that disable preview/save;
            ``stage_order_hint`` is advisory and decorates with a yellow
            border instead of red. ``unknown_class`` and
            ``unsupported_linear`` are blocking because the runtime cannot
            safely materialize a pipeline from them.
        block_id: ``None`` for scope-level findings (the only one that
            actually uses ``None`` is ``missing_input``); otherwise the
            offender's block id.
        detail: Free-form human-readable explanation. Used by tooltips
            and toasts; not part of the test-normalisation key.
        scope_path: List of container block_id values walked from the
            root scope to the offender's scope. Empty when the issue
            lives in the root scope. The UI consumes this to pan/zoom
            across container boundaries when a badge is clicked.
        severity: ``"error"`` for the seven blocking rules,
            ``"advisory"`` for ``stage_order_hint``. Populated by the rule
            that emits the issue; callers gate Run/Save by filtering on this
            field.
    """

    kind: IssueKind
    block_id: Optional[str]
    detail: str
    scope_path: List[str] = field(default_factory=list)
    severity: IssueSeverity = "error"


def validate(state: BuilderState) -> List[Issue]:
    """Run all validation rules across every scope reachable from the root.

    Returns issues with ``scope_path`` populated so the UI can pan/zoom
    across container boundaries when an issue badge is clicked.

    Args:
        state: The current :class:`BuilderState`. Only the
            ``state.root`` scope (and its transitively nested scopes via
            ``BlockNode.nested``) is inspected; ``selected_block_id``,
            ``breadcrumb``, ``toast_queue`` etc. are ignored.

    Returns:
        A flat list of :class:`Issue` records in deterministic order:
        rules emit in the order ``missing_input`` / ``duplicate_input``
        → ``fork`` → ``stub`` → ``required_aux`` / ``unknown_class`` →
        ``cycle`` → ``container_mode`` → ``stage_order_hint``, and
        nested-scope issues are appended after the parent scope's
        issues. The ordering is stable for snapshot-style tests.

    Examples:
        An empty scope (only the auto-seeded ``InputImage`` block)
        passes every rule and returns an empty issue list. The
        synth-yeast fixture is loaded only to anchor the example in
        the project's microbiology context — validation does not
        touch image data:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.gui.builder._state import (
        ...     BuilderScope, BuilderState,
        ... )
        >>> from phenotypic.gui.builder._validation import validate
        >>> state = BuilderState(root=BuilderScope())
        >>> validate(state)
        []
        >>> _ = load_synth_yeast_plate()  # microbiology context anchor
    """

    return _validate_scope(state.root, scope_path=[])


def _validate_scope(scope: BuilderScope, scope_path: List[str]) -> List[Issue]:
    """Apply every rule to a single scope, then recurse into containers.

    Args:
        scope: The scope under inspection (root or nested).
        scope_path: List of container ``block_id``s walked from the root
            to this scope. Each emitted :class:`Issue` copies this list
            into its own ``scope_path`` field.

    Returns:
        List of :class:`Issue` records for this scope and its
        transitively nested scopes.
    """

    issues: List[Issue] = []
    registry = get_registry()

    # Single-pass edge index reused by Rules 1/2/3/4/5/7. Building the
    # adjacency / reverse-adjacency / counters / aux-wired map ONCE
    # collapses the rules' total cost from O(V·E) (Rule 2's BFS used
    # to scan ``scope.edges`` on every frontier expansion) to O(V+E).
    image_out_count: Dict[str, int] = defaultdict(int)
    image_in_count: Dict[Tuple[str, str], int] = defaultdict(int)
    total_out_count: Dict[str, int] = defaultdict(int)
    aux_wired: Dict[Tuple[str, str], int] = defaultdict(int)
    # ``adjacency`` is the unified ``source -> [target, ...]`` map over
    # ALL edges; consumed by the cycle detector (Rule 4).
    adjacency: Dict[str, List[str]] = defaultdict(list)
    # Image-flow forward edges only (Rule 2 walks image edges forward).
    image_forward: Dict[str, List[str]] = defaultdict(list)
    # Aux edges keyed by their consumer (target) — Rule 2 walks aux
    # backwards from a consumer to its aux source. Each entry is the
    # list of source block ids that feed *target* over an aux wire.
    aux_reverse: Dict[str, List[str]] = defaultdict(list)
    # Pipeline container left/right wiring buckets (Rule 5).
    container_left_wired: Dict[str, bool] = {}
    container_right_kinds: Dict[str, set] = defaultdict(set)

    pipeline_block_ids: set = {
        b.block_id for b in scope.blocks if b.class_name == PIPELINE_CLASS_NAME
    }

    for edge in scope.edges:
        src = edge.source_block_id
        tgt = edge.target_block_id
        total_out_count[src] += 1
        adjacency[src].append(tgt)
        if edge.kind == "image":
            image_out_count[src] += 1
            image_in_count[(tgt, edge.target_port)] += 1
            image_forward[src].append(tgt)
            if tgt in pipeline_block_ids and edge.target_port == "in":
                container_left_wired[tgt] = True
        else:  # aux
            aux_wired[(tgt, edge.target_port)] += 1
            aux_reverse[tgt].append(src)
        if src in pipeline_block_ids:
            container_right_kinds[src].add(edge.kind)

    # Rule 6 — exactly one Input Image per scope.
    input_blocks = [
        b for b in scope.blocks if b.class_name == INPUT_IMAGE_CLASS_NAME
    ]
    if not input_blocks:
        issues.append(
            Issue(
                kind="missing_input",
                block_id=None,
                detail="scope has no Input Image",
                scope_path=list(scope_path),
            )
        )
    elif len(input_blocks) > 1:
        for extra in input_blocks[1:]:
            issues.append(
                Issue(
                    kind="duplicate_input",
                    block_id=extra.block_id,
                    detail="extra Input Image",
                    scope_path=list(scope_path),
                )
            )
    root_id = input_blocks[0].block_id if input_blocks else None

    # Rule 1 — image-flow forks.
    #
    # Three sub-cases all reported as ``kind="fork"``:
    #   (a) a single source has >1 outgoing ``image`` edge.
    #   (b) a single ``(target_block_id, "in")`` port has >1 incoming
    #       ``image`` edge.
    #   (c) a single source has >1 outgoing wires *total* across image
    #       and aux (spec §4.2 — "at most one outgoing wire, total").
    seen_fork_source: set = set()
    for block_id, n in image_out_count.items():
        if n > 1:
            issues.append(
                Issue(
                    kind="fork",
                    block_id=block_id,
                    detail="image-out has >1 wire",
                    scope_path=list(scope_path),
                )
            )
            seen_fork_source.add(block_id)
    for (block_id, port), n in image_in_count.items():
        if n > 1 and port == "in":
            issues.append(
                Issue(
                    kind="fork",
                    block_id=block_id,
                    detail="image-in has >1 wire",
                    scope_path=list(scope_path),
                )
            )
    for block_id, n in total_out_count.items():
        # Skip sources we already flagged for "image-out has >1 wire" so
        # we don't double-count the same logical fork. ``mixed_kind``
        # fan-out (one image-out + one aux-out from the same source)
        # ends up here.
        if n > 1 and block_id not in seen_fork_source:
            issues.append(
                Issue(
                    kind="fork",
                    block_id=block_id,
                    detail="block has >1 outgoing wires (image + aux combined)",
                    scope_path=list(scope_path),
                )
            )

    # Rule 2 — stubs. BFS from the InputImage block. Image edges only
    # forward; aux edges traversed both forward and backward so an
    # aux-producer (sink in image flow) is still reachable from a
    # consumer along the chain. Uses the prebuilt indices so the walk is
    # O(V+E) instead of O(V·E).
    reachable: set = set()
    if root_id is not None:
        frontier: List[str] = [root_id]
        while frontier:
            curr = frontier.pop()
            if curr in reachable:
                continue
            reachable.add(curr)
            # Forward across all edges (image + aux outgoing).
            frontier.extend(adjacency.get(curr, ()))
            # Backward across aux edges so an aux producer (no image
            # output, only an aux wire feeding a consumer) is still
            # reachable from its consumer.
            frontier.extend(aux_reverse.get(curr, ()))
    for block in scope.blocks:
        if block.block_id in reachable:
            continue
        # Extra ``InputImage`` blocks are already reported as
        # ``duplicate_input`` (Rule 6) — flagging them additionally as a
        # stub is redundant noise.  Skip them here so the badge count
        # accurately reflects the user's mental model: one offence per
        # extra Input Image.
        if block.class_name == INPUT_IMAGE_CLASS_NAME:
            continue
        issues.append(
            Issue(
                kind="stub",
                block_id=block.block_id,
                detail="not reachable from Input Image",
                scope_path=list(scope_path),
            )
        )

    # Rule 3 — required aux ports must be wired.
    #
    # ``ParamInfo.default`` is normalised to ``None`` by the registry
    # (line ~404 of ``_operation_registry.py``); the right predicate is
    # ``not p.has_default``, NOT ``p.default is inspect.Parameter.empty``.
    # The latter is always False on a ParamInfo instance because the
    # registry replaced ``inspect.Parameter.empty`` with ``None`` before
    # it stored the value.
    # ``aux_wired`` is the (target, port) -> count map built in the
    # top-of-function single-pass scan; reused here as-is.
    for block in scope.blocks:
        # The InputImage sentinel never has registered params; skip it
        # so we don't emit a spurious ``unknown_class`` issue.
        if block.class_name == INPUT_IMAGE_CLASS_NAME:
            continue
        # The Pipeline sentinel represents a container; its "params" are
        # the nested scope's blocks. We don't enforce required aux on
        # containers themselves — the nested scope's own validation
        # walks its inner blocks.
        if block.class_name == PIPELINE_CLASS_NAME:
            continue
        info = registry.get(block.class_name)
        if info is None:
            issues.append(
                Issue(
                    kind="unknown_class",
                    block_id=block.block_id,
                    detail=f"class '{block.class_name}' not in registry",
                    scope_path=list(scope_path),
                )
            )
            continue
        for param_name, p in info.parameters.items():
            if not (p.is_operation or p.is_pipeline):
                # Scalar param — out of scope for Rule 3.
                continue
            if p.has_default:
                # Optional aux — empty is fine.
                continue
            # Required aux. Count wired edges into this port.
            if aux_wired.get((block.block_id, param_name), 0) == 0:
                issues.append(
                    Issue(
                        kind="required_aux",
                        block_id=block.block_id,
                        detail=f"{param_name} is required",
                        scope_path=list(scope_path),
                    )
                )

    # Rule 4 — cycle detection over ALL edges (image + aux).
    #
    # We use Tarjan's strongly-connected-components algorithm: O(V+E)
    # iterative, no risk of recursion-limit issues even on deeply
    # nested scopes. Any node that participates in a non-trivial SCC
    # (size > 1, OR size 1 with a self-loop) is reported as
    # ``kind="cycle"``. ``adjacency`` was built in the top-of-function
    # single-pass scan; reused here.
    cycle_members = _find_cycle_nodes(adjacency, scope.blocks)
    for block_id in cycle_members:
        issues.append(
            Issue(
                kind="cycle",
                block_id=block_id,
                detail="block participates in a cycle",
                scope_path=list(scope_path),
            )
        )

    # Rule 5 — container left/right wiring consistency.
    #
    # A pipeline container has two operating modes:
    #   * "Consumer-fed" (left wired image-in): the container is fed
    #     downstream as a single op; its right ports must wire image-out
    #     (blue). Wiring aux out is illegal.
    #   * "Aux-fed" (left unwired): the container's right wire is an
    #     aux marker for a consumer; image-out is illegal.
    # Either incoherent combination raises ``kind="container_mode"``.
    # ``container_left_wired`` / ``container_right_kinds`` were built
    # in the top-of-function single-pass scan; reused here.
    for block_id in pipeline_block_ids:
        left_wired = container_left_wired.get(block_id, False)
        right_kinds = container_right_kinds.get(block_id, ())
        if left_wired and "aux" in right_kinds:
            issues.append(
                Issue(
                    kind="container_mode",
                    block_id=block_id,
                    detail="left wired but right wires to aux",
                    scope_path=list(scope_path),
                )
            )
        if not left_wired and "image" in right_kinds:
            issues.append(
                Issue(
                    kind="container_mode",
                    block_id=block_id,
                    detail="right wires to image but left is unwired",
                    scope_path=list(scope_path),
                )
            )

    # Rule 7 (advisory) — stage ordering hint.
    #
    # Walk every image-flow edge; if the source's stage > the target's
    # stage, emit a yellow-border advisory. The runtime partitions by
    # ``isinstance`` so a misordered chain still works (each stage runs
    # in its own block), but the canvas warns the user that the visual
    # ordering does not match the execution ordering. ``image_forward``
    # was built in the top-of-function single-pass scan; reused here.
    if root_id is not None:
        order_of: Dict[str, int] = {}
        for block in scope.blocks:
            stage = _safe_stage(block.class_name, registry=registry)
            order_of[block.block_id] = _STAGE_ORDER.get(stage, 0)
        for source_id, targets in image_forward.items():
            src_stage = order_of.get(source_id)
            if src_stage is None:
                continue
            for target_id in targets:
                tgt_stage = order_of.get(target_id)
                if tgt_stage is not None and src_stage > tgt_stage:
                    issues.append(
                        Issue(
                            kind="stage_order_hint",
                            block_id=source_id,
                            detail=(
                                "runs in a later stage than its downstream "
                                "block; runtime partitions by isinstance."
                            ),
                            scope_path=list(scope_path),
                            severity="advisory",
                        )
                    )

    # Recurse into containers AFTER the parent scope's rules so the
    # parent scope's issues are listed before the child scope's.
    for block in scope.blocks:
        if block.nested is not None:
            issues.extend(
                _validate_scope(
                    block.nested,
                    scope_path=[*scope_path, block.block_id],
                )
            )
    return issues


def _safe_stage(
    class_name: str,
    registry: Optional[Any] = None,
) -> StageName:
    """Classify an operation by stage for Rule 7's ordering hint.

    Sentinels (``InputImage``, ``ImagePipeline``) collapse to ``"ops"``
    so they don't trip the advisory at the head of every chain. Real
    operation classes are looked up in the registry and then matched
    against :class:`~phenotypic.abc_.MeasureFeatures` and
    :class:`~phenotypic.abc_.PostMeasurement`; everything else falls
    back to ``"ops"``.

    Args:
        class_name: Block's ``class_name`` (registry key or sentinel).
        registry: Optional pre-fetched registry; useful for tests that
            inject a fake or for hot loops that have already called
            :func:`get_registry`. Defaults to the singleton.

    Returns:
        One of ``"ops"``, ``"meas"``, ``"post"``, or ``"pipeline"``.
    """

    if class_name == INPUT_IMAGE_CLASS_NAME:
        return "ops"
    if class_name == PIPELINE_CLASS_NAME:
        return "pipeline"
    reg = registry if registry is not None else get_registry()
    info: Optional[OperationInfo] = reg.get(class_name)
    if info is None:
        return "ops"
    cls = info.cls
    try:
        if issubclass(cls, PostMeasurement):
            return "post"
        if issubclass(cls, MeasureFeatures):
            return "meas"
    except TypeError:
        # ``cls`` is not a class (e.g. a stub for unknown_class); fall
        # back to ops.
        return "ops"
    return "ops"


def _find_cycle_nodes(
    adjacency: Dict[str, List[str]],
    blocks: List[BlockNode],
) -> List[str]:
    """Return the ``block_id``s that participate in any cycle.

    Implements Tarjan's strongly-connected-components algorithm
    iteratively to avoid Python's recursion limit on deep graphs. A
    block participates in a cycle iff it is in an SCC of size > 1 OR
    it is a single-node SCC with a self-loop.

    Args:
        adjacency: Directed graph as ``source_block_id -> [target_block_id, ...]``.
            Only blocks with outgoing edges need to appear as keys;
            sinks are inferred from ``blocks``.
        blocks: All blocks in the scope (so isolated nodes that don't
            appear in ``adjacency`` still enter the search and we don't
            miss a self-loop on a node with no outgoing edges other
            than to itself).

    Returns:
        Sorted list of block_ids in cycles.
    """

    # Collect every node mentioned by either ``blocks`` or
    # ``adjacency`` so the SCC search visits the full vertex set.
    all_nodes: set = set(b.block_id for b in blocks)
    for src, targets in adjacency.items():
        all_nodes.add(src)
        for t in targets:
            all_nodes.add(t)

    index_counter = [0]
    stack: List[str] = []
    on_stack: Dict[str, bool] = {}
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    result: List[str] = []

    # Iterative Tarjan: each "call" is a (node, neighbour-iterator)
    # frame on ``work_stack``. We pop a frame, advance to the next
    # neighbour, and either push a child frame or backtrack and update
    # the parent's lowlink.
    for start in all_nodes:
        if start in indices:
            continue
        work_stack: List[Tuple[str, int]] = [(start, 0)]
        # When we encounter ``start`` for the first time, initialise its
        # SCC bookkeeping; the rest of the algorithm operates on the
        # invariant that any node on ``work_stack`` already has indices.
        indices[start] = index_counter[0]
        lowlinks[start] = index_counter[0]
        index_counter[0] += 1
        stack.append(start)
        on_stack[start] = True

        while work_stack:
            node, neighbour_idx = work_stack[-1]
            neighbours = adjacency.get(node, [])
            if neighbour_idx < len(neighbours):
                # Advance the parent's neighbour cursor BEFORE the
                # potential recursion so we resume at the right spot.
                work_stack[-1] = (node, neighbour_idx + 1)
                child = neighbours[neighbour_idx]
                if child not in indices:
                    indices[child] = index_counter[0]
                    lowlinks[child] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(child)
                    on_stack[child] = True
                    work_stack.append((child, 0))
                elif on_stack.get(child, False):
                    lowlinks[node] = min(lowlinks[node], indices[child])
                # else: cross-edge to a finished SCC; ignore.
            else:
                # All neighbours processed; finalise this node.
                if lowlinks[node] == indices[node]:
                    # Pop the SCC rooted at ``node``.
                    scc: List[str] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == node:
                            break
                    # Determine cycle-ness.
                    if len(scc) > 1:
                        result.extend(scc)
                    elif len(scc) == 1:
                        # Single-node SCC: cycle iff it has a self-loop.
                        only = scc[0]
                        if only in adjacency and only in adjacency[only]:
                            result.append(only)
                # Backtrack: update parent's lowlink with ours.
                work_stack.pop()
                if work_stack:
                    parent = work_stack[-1][0]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[node])

    # Deterministic ordering for snapshot-style tests.
    return sorted(result)


__all__ = [
    "Issue",
    "IssueKind",
    "IssueSeverity",
    "StageName",
    "validate",
]
