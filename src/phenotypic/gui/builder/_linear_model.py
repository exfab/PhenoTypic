"""Pure helpers for the fixed linear builder port-map.

The visible builder constrains the existing DAG state to one image spine
plus side-loaded operation/pipeline parameters. This module contains the
small, side-effect-free model used by layout and callbacks to derive that
linear view without changing the public pipeline serialization contract.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    PIPELINE_CLASS_NAME,
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
)


TargetKind = Literal[
    "continuation",
    "image_output",
    "image_input",
    "parameter",
    "parameter_slot",
]

_TARGET_KINDS: set[str] = {
    "continuation",
    "image_output",
    "image_input",
    "parameter",
    "parameter_slot",
}

ROOT_SCOPE_KEY = "__root__"


@dataclass(frozen=True)
class LinearTarget:
    """Palette add/fill target for one builder scope."""

    kind: TargetKind
    scope_path: List[str]
    block_id: Optional[str] = None
    param: Optional[str] = None
    slot: Optional[int] = None


@dataclass(frozen=True)
class UnsupportedLinearState:
    """Reason a DAG scope cannot be represented by the linear editor."""

    reason: str
    detail: str
    block_id: Optional[str] = None


@dataclass(frozen=True)
class LinearScopeModel:
    """Derived linear view of one DAG scope."""

    scope: _DagBuilderScope
    scope_path: List[str]
    input_block: Optional[BlockNode]
    spine_blocks: List[BlockNode]
    terminal_block: Optional[BlockNode]
    aux_owned_block_ids: set[str]
    unknown_block_ids: set[str]
    unsupported: Optional[UnsupportedLinearState] = None


def scope_key(scope_path: Iterable[str]) -> str:
    """Return a stable key for per-scope UI metadata."""

    parts = [str(part) for part in scope_path]
    return ROOT_SCOPE_KEY if not parts else "/".join(parts)


def default_continuation_target(scope_path: Iterable[str]) -> LinearTarget:
    """Return the fallback target for a scope."""

    return LinearTarget(kind="continuation", scope_path=list(scope_path))


def target_to_dict(target: LinearTarget) -> Dict[str, Any]:
    """Serialize a :class:`LinearTarget` for ``dcc.Store`` state."""

    return {
        "kind": target.kind,
        "scope_path": list(target.scope_path),
        "block_id": target.block_id,
        "param": target.param,
        "slot": target.slot,
    }


def target_from_dict(data: Any, scope_path: Iterable[str]) -> LinearTarget:
    """Coerce arbitrary JSON-ish data to a valid target object."""

    fallback = default_continuation_target(scope_path)
    if not isinstance(data, dict):
        return fallback
    kind = data.get("kind")
    if kind not in _TARGET_KINDS:
        return fallback
    raw_scope = data.get("scope_path")
    resolved_scope = (
        [str(part) for part in raw_scope]
        if isinstance(raw_scope, list)
        else list(scope_path)
    )
    raw_slot = data.get("slot")
    slot = raw_slot if isinstance(raw_slot, int) and raw_slot >= 0 else None
    block_id = data.get("block_id")
    param = data.get("param")
    return LinearTarget(
        kind=kind,  # type: ignore[arg-type]
        scope_path=resolved_scope,
        block_id=block_id if isinstance(block_id, str) else None,
        param=param if isinstance(param, str) else None,
        slot=slot,
    )


def resolve_selected_target(state: _DagBuilderState) -> LinearTarget:
    """Return the active target for ``state.breadcrumb`` with fallback."""

    scope_path = list(state.breadcrumb)
    key = scope_key(scope_path)
    raw_target = getattr(state, "selected_targets_by_scope", {}).get(key)
    target = target_from_dict(raw_target, scope_path)
    scope = scope_at_path(state.root, scope_path)
    if scope is None:
        return default_continuation_target(scope_path)
    if is_target_valid(target, scope):
        return target
    return default_continuation_target(scope_path)


def scope_at_path(
    root_scope: _DagBuilderScope, scope_path: Iterable[str]
) -> Optional[_DagBuilderScope]:
    """Resolve a DAG breadcrumb path to a scope."""

    scope = root_scope
    for block_id in scope_path:
        block = next(
            (candidate for candidate in scope.blocks if candidate.block_id == block_id),
            None,
        )
        if (
            block is None
            or block.class_name != PIPELINE_CLASS_NAME
            or block.nested is None
        ):
            return None
        scope = block.nested
    return scope


def is_target_valid(target: LinearTarget, scope: _DagBuilderScope) -> bool:
    """Return whether ``target`` still points at an object in ``scope``."""

    if target.kind == "continuation":
        return True
    blocks_by_id = {block.block_id: block for block in scope.blocks}
    if target.block_id not in blocks_by_id:
        return False
    if target.kind in {"image_output", "image_input"}:
        return True
    if target.kind == "parameter":
        return target.param is not None
    if target.kind == "parameter_slot":
        return target.param is not None and target.slot is not None
    return False


def derive_linear_scope(
    scope: _DagBuilderScope, *, scope_path: Iterable[str]
) -> LinearScopeModel:
    """Derive the fixed linear spine for one DAG scope.

    The derivation accepts only a unique image chain rooted at the
    scope's ``InputImage`` plus aux-owned side blocks. Development DAGs
    with forks, joins, cycles, orphan blocks, or shared aux sources are
    classified as unsupported so layout can render a defensive panel.
    """

    path = list(scope_path)
    input_blocks = [
        block for block in scope.blocks
        if block.class_name == INPUT_IMAGE_CLASS_NAME
    ]
    if not input_blocks:
        return _unsupported_model(
            scope, path, "missing_input", "scope has no Input Image"
        )
    if len(input_blocks) > 1:
        return _unsupported_model(
            scope,
            path,
            "duplicate_input",
            "scope has more than one Input Image",
            input_blocks[1].block_id,
        )

    input_block = input_blocks[0]
    blocks_by_id = {block.block_id: block for block in scope.blocks}
    for edge in scope.edges:
        if (
            edge.source_block_id not in blocks_by_id
            or edge.target_block_id not in blocks_by_id
        ):
            return _unsupported_model(
                scope,
                path,
                "dangling_edge",
                "edge endpoint is missing from this scope",
                edge.source_block_id,
            )

    image_edges = [edge for edge in scope.edges if edge.kind == "image"]
    out_by_source: dict[str, list[Edge]] = defaultdict(list)
    in_by_target: dict[str, list[Edge]] = defaultdict(list)
    for edge in image_edges:
        out_by_source[edge.source_block_id].append(edge)
        in_by_target[edge.target_block_id].append(edge)

    fork_source = next(
        (source for source, edges in out_by_source.items() if len(edges) > 1),
        None,
    )
    if fork_source is not None:
        return _unsupported_model(
            scope,
            path,
            "image_fork",
            "image output feeds more than one downstream input",
            fork_source,
        )
    join_target = next(
        (target for target, edges in in_by_target.items() if len(edges) > 1),
        None,
    )
    if join_target is not None:
        return _unsupported_model(
            scope,
            path,
            "image_join",
            "image input receives more than one upstream output",
            join_target,
        )

    spine = _walk_image_spine(input_block, blocks_by_id, out_by_source)
    if spine is None:
        return _unsupported_model(
            scope,
            path,
            "image_cycle",
            "image flow cycles before reaching a terminal node",
            input_block.block_id,
        )
    spine_ids = {block.block_id for block in spine}
    for edge in image_edges:
        if edge.source_block_id not in spine_ids or edge.target_block_id not in spine_ids:
            return _unsupported_model(
                scope,
                path,
                "extra_image_chain",
                "image edge is outside the Input Image spine",
                edge.source_block_id,
            )

    aux_edges = [edge for edge in scope.edges if edge.kind == "aux"]
    aux_out_by_source: dict[str, list[Edge]] = defaultdict(list)
    for edge in aux_edges:
        target_block = blocks_by_id.get(edge.target_block_id)
        if target_block is not None and target_block.class_name == INPUT_IMAGE_CLASS_NAME:
            return _unsupported_model(
                scope,
                path,
                "input_as_aux_target",
                "Input Image cannot receive aux parameter values",
                edge.target_block_id,
            )
        aux_out_by_source[edge.source_block_id].append(edge)
    for source_id, edges in aux_out_by_source.items():
        if source_id == input_block.block_id:
            return _unsupported_model(
                scope,
                path,
                "input_as_aux",
                "Input Image cannot be used as an aux source",
                source_id,
            )
        if source_id in spine_ids:
            return _unsupported_model(
                scope,
                path,
                "shared_aux_source",
                "main-spine block is also used as an aux source",
                source_id,
            )
        if len(edges) > 1:
            return _unsupported_model(
                scope,
                path,
                "shared_aux_source",
                "aux source feeds more than one consumer",
                source_id,
            )

    aux_owned = _collect_aux_owned_block_ids(aux_edges, spine_ids)
    visible_ids = spine_ids | aux_owned
    for block in scope.blocks:
        if block.class_name == INPUT_IMAGE_CLASS_NAME:
            continue
        if block.block_id not in visible_ids:
            return _unsupported_model(
                scope,
                path,
                "orphan_block",
                "block is neither on the image spine nor owned by an aux port",
                block.block_id,
            )

    return LinearScopeModel(
        scope=scope,
        scope_path=path,
        input_block=input_block,
        spine_blocks=spine,
        terminal_block=spine[-1],
        aux_owned_block_ids=aux_owned,
        unknown_block_ids=_unknown_block_ids(scope, visible_ids),
        unsupported=None,
    )


def compact_list_aux_slots(
    scope: _DagBuilderScope, block_id: str, param: str
) -> None:
    """Renumber list aux edges for ``block_id.param`` without empty gaps."""

    block = next((b for b in scope.blocks if b.block_id == block_id), None)
    if block is None:
        return
    indexed_edges = [
        (idx, edge)
        for idx, edge in enumerate(scope.edges)
        if edge.kind == "aux"
        and edge.target_block_id == block_id
        and edge.target_port == param
    ]
    indexed_edges.sort(
        key=lambda pair: (
            pair[1].target_slot if pair[1].target_slot is not None else 0,
            pair[0],
        )
    )
    for slot, (_idx, edge) in enumerate(indexed_edges):
        edge.target_slot = slot
    if indexed_edges:
        block.list_slot_counts[param] = len(indexed_edges)
    else:
        block.list_slot_counts.pop(param, None)


def _walk_image_spine(
    input_block: BlockNode,
    blocks_by_id: Dict[str, BlockNode],
    out_by_source: Dict[str, List[Edge]],
) -> Optional[List[BlockNode]]:
    """Return the image chain from ``input_block`` or ``None`` on cycle."""

    spine: List[BlockNode] = []
    visited: set[str] = set()
    current_id: Optional[str] = input_block.block_id
    while current_id is not None:
        if current_id in visited:
            return None
        visited.add(current_id)
        block = blocks_by_id.get(current_id)
        if block is None:
            break
        spine.append(block)
        next_edges = out_by_source.get(current_id, [])
        current_id = next_edges[0].target_block_id if next_edges else None
    return spine


def _collect_aux_owned_block_ids(
    aux_edges: List[Edge], spine_ids: set[str]
) -> set[str]:
    """Return aux blocks transitively owned by the visible spine."""

    owned: set[str] = set()
    changed = True
    while changed:
        changed = False
        allowed_targets = spine_ids | owned
        for edge in aux_edges:
            if edge.target_block_id not in allowed_targets:
                continue
            if edge.source_block_id in owned:
                continue
            owned.add(edge.source_block_id)
            changed = True
    return owned


def _unknown_block_ids(
    scope: _DagBuilderScope, visible_ids: set[str]
) -> set[str]:
    """Return visible operation blocks missing from the operation registry."""

    from phenotypic.gui import _operation_registry

    registry = _operation_registry.get_registry()
    unknown: set[str] = set()
    for block in scope.blocks:
        if block.block_id not in visible_ids:
            continue
        if block.class_name in (INPUT_IMAGE_CLASS_NAME, PIPELINE_CLASS_NAME):
            continue
        if registry.get(block.class_name) is None:
            unknown.add(block.block_id)
    return unknown


def _unsupported_model(
    scope: _DagBuilderScope,
    scope_path: List[str],
    reason: str,
    detail: str,
    block_id: Optional[str] = None,
) -> LinearScopeModel:
    """Build a model carrying an unsupported-state reason."""

    input_block = next(
        (block for block in scope.blocks if block.class_name == INPUT_IMAGE_CLASS_NAME),
        None,
    )
    spine = [input_block] if input_block is not None else []
    return LinearScopeModel(
        scope=scope,
        scope_path=list(scope_path),
        input_block=input_block,
        spine_blocks=spine,
        terminal_block=input_block,
        aux_owned_block_ids=set(),
        unknown_block_ids=set(),
        unsupported=UnsupportedLinearState(
            reason=reason,
            detail=detail,
            block_id=block_id,
        ),
    )


__all__ = [
    "LinearScopeModel",
    "LinearTarget",
    "TargetKind",
    "UnsupportedLinearState",
    "compact_list_aux_slots",
    "default_continuation_target",
    "derive_linear_scope",
    "is_target_valid",
    "resolve_selected_target",
    "scope_at_path",
    "scope_key",
    "target_from_dict",
    "target_to_dict",
]
