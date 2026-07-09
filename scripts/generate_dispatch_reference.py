"""Generator for the dispatch-kind reference RST page.

Walks ``phenotypic.gui.builder._callbacks._dispatch_state_update``,
matches each ``if kind == "X":`` branch against a hand-curated payload
schema map (mirroring spec §5.6), and writes
``docs/source/api_reference/gui/builder_dispatch.rst``.

Run with ``--check`` to fail nonzero if the committed RST does not
match the regenerated output (used by CI to detect drift between the
spec and the docs).

Determinism: the script imports
:data:`phenotypic.gui.builder._callbacks.DispatchKind` to enumerate the
kinds in their declaration order, then renders one section per kind
using only the hand-curated mapping below. No registry lookups, no
filesystem ordering — same input → same output bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, get_args

try:
    from scripts._reference_generator import write_or_check_generated_file
except ModuleNotFoundError:  # pragma: no cover - path-script execution
    from _reference_generator import write_or_check_generated_file

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_RST = (
    REPO_ROOT
    / "docs"
    / "source"
    / "api_reference"
    / "gui"
    / "builder_dispatch.rst"
)


# ---------------------------------------------------------------------------
# Hand-curated dispatch-kind metadata (mirrors spec §5.6 verbatim).
# ---------------------------------------------------------------------------
#
# Each entry maps a Literal value from
# ``phenotypic.gui.builder._callbacks.DispatchKind`` to:
#   * group: which dispatch family it belongs to.
#   * payload: the payload-schema bullet list (rendered as a code block).
#   * notes: spec-derived behavioural notes.
#
# The mapping is the source of truth here because the dispatch kinds'
# payload schemas live as inline comments in the dispatcher body and
# cannot be reliably extracted by AST walking; the hand-curated map
# keeps the RST output deterministic.

_DISPATCH_KINDS: List[Dict[str, Any]] = [
    # ----------------------------------------------------------------
    # Legacy linear-builder kinds (kept for back-compat; the DAG model
    # routes none of these).
    # ----------------------------------------------------------------
    {
        "kind": "add_node",
        "group": "Legacy linear builder",
        "payload": "``{class_name: str}``",
        "notes": (
            "Append a fresh ``StepNode`` for ``class_name`` to the "
            "current legacy scope. Selects the new node."
        ),
    },
    {
        "kind": "add_pipeline",
        "group": "Legacy linear builder",
        "payload": "``{}``",
        "notes": "Append a ``StepNode`` carrying an empty nested scope.",
    },
    {
        "kind": "select_node",
        "group": "Legacy linear builder",
        "payload": "``{node_id: str}``",
        "notes": "Set ``selected_node_id``.",
    },
    {
        "kind": "delete_node",
        "group": "Legacy linear builder",
        "payload": "``{}``",
        "notes": "Remove ``selected_node_id`` from the current scope.",
    },
    {
        "kind": "drill_in",
        "group": "Legacy linear builder",
        "payload": "``{}``",
        "notes": (
            "Push ``selected_node_id`` onto the breadcrumb (only if "
            "that node has a nested scope)."
        ),
    },
    {
        "kind": "drill_out",
        "group": "Legacy linear builder",
        "payload": "``{}``",
        "notes": "Pop the breadcrumb tail.",
    },
    {
        "kind": "breadcrumb_to",
        "group": "Legacy linear builder",
        "payload": "``{depth: int}``",
        "notes": "Truncate breadcrumb to ``depth`` entries.",
    },
    {
        "kind": "reorder",
        "group": "Legacy linear builder",
        "payload": "``{order: list[str]}``",
        "notes": "Reorder the current scope's nodes by node-id sequence.",
    },
    {
        "kind": "edit_param",
        "group": "Legacy linear builder",
        "payload": "``{node_id: str, name: str, value: Any, omit: bool}``",
        "notes": "Set or delete ``params[name]`` on a specific node.",
    },
    {
        "kind": "edit_label",
        "group": "Legacy linear builder",
        "payload": "``{node_id: str, label: str}``",
        "notes": "Update the node label.",
    },
    {
        "kind": "port_slot_add",
        "group": "Legacy linear builder",
        "payload": "``{node_id: str, param: str}``",
        "notes": (
            "Append a ``None`` slot to the consumer's ``aux_ports`` "
            "list for a list-typed param. No-op for scalar ports."
        ),
    },
    {
        "kind": "port_slot_remove",
        "group": "Legacy linear builder",
        "payload": "``{node_id: str, param: str, slot: int}``",
        "notes": (
            "Remove the slot at ``slot`` from the consumer's "
            "``aux_ports`` list and reindex remaining slots."
        ),
    },
    # ----------------------------------------------------------------
    # Legacy aux-port mutation kinds. These branches remain in
    # _dispatch_state_update for compatibility with old inspector aux-wire
    # payloads. The current fixed linear map does not emit them from
    # STORE_EDGE_EVENT.
    # ----------------------------------------------------------------
    {
        "kind": "wire_create",
        "group": "Legacy aux-port mutation",
        "payload": (
            "``{target_node_id: str, param: str, slot: int, "
            "class_name: str}``"
        ),
        "notes": (
            "Embed a fresh aux ``StepNode`` under "
            "``consumer.aux_ports[param][slot]``. The dispatcher "
            "validates the target node, registry class, and operation-type "
            "compatibility before writing, then sets ``inspector_focus_aux`` "
            "to the filled slot. Retained for legacy inspector aux-wire "
            "payloads; the fixed linear map routes side-value creation "
            "through ``linear_palette_add``."
        ),
    },
    {
        "kind": "wire_delete",
        "group": "Legacy aux-port mutation",
        "payload": "``{target_node_id: str, param: str, slot: int}``",
        "notes": (
            "Clear one legacy aux slot by setting "
            "``consumer.aux_ports[param][slot] = None``. If "
            "``inspector_focus_aux`` points at the same slot, it is cleared "
            "too."
        ),
    },
    {
        "kind": "drill_in_aux",
        "group": "Legacy aux-port mutation",
        "payload": "``{target_node_id: str, param: str, slot: int}``",
        "notes": (
            "Push an aux-slot breadcrumb segment for an occupied legacy aux "
            "slot and clear ``inspector_focus_aux``. Empty slots and missing "
            "target nodes are no-ops."
        ),
    },
    {
        "kind": "set_inspector_focus",
        "group": "Legacy aux-port mutation",
        "payload": (
            "``{focus: \"aux\" | \"consumer\", target_node_id: str, "
            "param: str, slot: int}``"
        ),
        "notes": (
            "Set ``inspector_focus_aux`` to an occupied legacy aux slot when "
            "``focus == \"aux\"``; any other focus clears it. Missing nodes "
            "or empty slots are rejected without mutation."
        ),
    },
    # ----------------------------------------------------------------
    # DAG palette drag-and-drop.
    # ----------------------------------------------------------------
    {
        "kind": "block_create",
        "group": "DAG canvas: palette drag-and-drop",
        "payload": (
            "``{class_name: str, x: float, y: float, "
            "container_block_id: str | None, ts: int}``"
        ),
        "notes": (
            "Append a fresh :class:`BlockNode` to the root scope "
            "(when ``container_block_id`` is ``None``) or to a "
            "container's nested scope (DFS lookup, innermost-wins "
            "hit-test per spec §4.4). "
            "Rejects ``class_name == \"InputImage\"`` with a toast — "
            "Input Image is auto-seeded per scope and cannot be "
            "created from the palette (spec §4.8 + §4.10). "
            "Drop coordinates ``(x, y)`` are **not** persisted; the "
            "leaf-first dagre pass re-lays the canvas on the next "
            "render (spec §4.7)."
        ),
    },
    # ----------------------------------------------------------------
    # DAG canvas: wire-drawing + list-aux fan-in.
    # ----------------------------------------------------------------
    {
        "kind": "edge_create",
        "group": "DAG canvas: wire drawing",
        "payload": (
            "``{source_block_id: str, target_block_id: str, "
            "target_port: str, edge_kind: \"image\" | \"aux\", "
            "ts: int}``"
        ),
        "notes": (
            "Mint an :class:`Edge` between two ports inside the same "
            "scope. "
            "For scalar aux + image-flow: replaces any existing wire "
            "from ``source_block_id`` (single-wire rule, spec §4.2 / "
            "§4.3). "
            "For list aux: **server-side append** — the dispatcher "
            "resolves ``target_slot = block.list_slot_counts.get("
            "target_port, 0)`` and increments "
            "``list_slot_counts[target_port]`` by 1 (eliminates the "
            "concurrent-drag race; the client emits no slot index). "
            "Cross-scope wires are rejected with a toast (spec §4.4)."
        ),
    },
    {
        "kind": "edge_delete",
        "group": "DAG canvas: wire drawing",
        "payload": "``{edge_id: str, ts: int}``",
        "notes": (
            "Remove a single edge by ``edge_id`` (DFS across every "
            "scope). For list-aux edges, remaining edges' "
            "``target_slot`` values are NOT renumbered — the freed "
            "slot becomes an empty placeholder. Use "
            "``list_aux_reorder`` to compact."
        ),
    },
    {
        "kind": "list_aux_reorder",
        "group": "DAG canvas: list-aux fan-in",
        "payload": (
            "``{block_id: str, param: str, new_order: "
            "list[str | None], ts: int}``"
        ),
        "notes": (
            "Update the canonical execution order of a list-typed aux "
            "port. ``new_order`` is a permutation of the wired "
            "edge_ids interspersed with ``None`` placeholders for "
            "empty slots; the dispatcher rebuilds each edge's "
            "``target_slot`` from its position in the new order and "
            "updates ``block.list_slot_counts[param] = "
            "len(new_order)``. Non-permutation inputs are rejected "
            "with a toast (no-op)."
        ),
    },
    {
        "kind": "list_aux_add_empty_slot",
        "group": "DAG canvas: list-aux fan-in",
        "payload": "``{block_id: str, param: str, ts: int}``",
        "notes": (
            "Increment ``block.list_slot_counts[param]`` by 1. No "
            "edge is materialised — empty slots live solely on the "
            "consumer block. At ``to_pipeline_dag`` time, slot "
            "indices in ``[0, count)`` not covered by an edge emit "
            "``None`` entries."
        ),
    },
    {
        "kind": "wire_select",
        "group": "DAG canvas: selection",
        "payload": "``{edge_id: str | None, ts: int}``",
        "notes": (
            "Set ``selected_edge_id``. ``None`` deselects. Setting a "
            "new id clears ``selected_block_id`` (mutual exclusion, "
            "spec §4.5)."
        ),
    },
    {
        "kind": "block_select",
        "group": "DAG canvas: selection",
        "payload": "``{block_id: str | None, ts: int}``",
        "notes": (
            "Set ``selected_block_id``. ``None`` deselects. Setting "
            "a new id clears ``selected_edge_id`` (mutual exclusion, "
            "spec §4.5)."
        ),
    },
    # ----------------------------------------------------------------
    # DAG canvas: Pipeline containers (spec §4.4 / §5.6).
    # ----------------------------------------------------------------
    {
        "kind": "block_reparent",
        "group": "DAG canvas: Pipeline containers",
        "payload": (
            "``{block_id: str, new_parent_block_id: str | None, "
            "x: float, y: float, ts: int}``"
        ),
        "notes": (
            "Move a block between scopes. "
            "``new_parent_block_id == None`` promotes the block to the "
            "current scope; a non-None value adopts it into that "
            "container's nested scope (sibling-container moves are a "
            "single atomic dispatch). Rejects ``InputImage`` block_ids "
            "(the source must remain in its scope). "
            "Drag-out direction (new parent is an ancestor) with "
            "orphan edges → snap-back + toast; drag-in / sibling "
            "direction with orphan edges → drop the incompatible "
            "edges + toast the count, then commit the move "
            "(spec §4.4)."
        ),
    },
    {
        "kind": "block_collapsed_toggle",
        "group": "DAG canvas: Pipeline containers",
        "payload": "``{block_id: str, ts: int}``",
        "notes": (
            "Toggle ``block.collapsed`` for a Pipeline container. "
            "No-op on non-container blocks (anything whose "
            "``class_name != \"ImagePipeline\"``). The visual "
            "expand/collapse state is per-block (per spec §4.4)."
        ),
    },
    {
        "kind": "drill_into_container",
        "group": "DAG canvas: Pipeline containers",
        "payload": "``{block_id: str, ts: int}``",
        "notes": (
            "Push ``block_id`` onto ``state.breadcrumb``. Validated: "
            "``block_id`` must resolve to a Pipeline container at the "
            "current breadcrumb depth (siblings at other depths are "
            "rejected silently)."
        ),
    },
    {
        "kind": "drill_to_scope",
        "group": "DAG canvas: Pipeline containers",
        "payload": "``{target_breadcrumb: list[str], ts: int}``",
        "notes": (
            "Atomic breadcrumb replacement — replaces "
            "``state.breadcrumb`` with ``target_breadcrumb`` in one "
            "dispatch. Each block_id in ``target_breadcrumb`` must "
            "resolve to a real Pipeline container at the right depth "
            "in ``state.root``; stale ids → reject + toast. Used by "
            "``scroll_to`` for cross-scope navigation."
        ),
    },
    {
        "kind": "block_delete_request",
        "group": "DAG canvas: Pipeline containers",
        "payload": "``{block_id: str, ts: int}``",
        "notes": (
            "First stage of the two-stage container delete. Rejects "
            "``InputImage`` block_ids (defense in depth). "
            "Non-container OR *empty* container (only the auto-seeded "
            "``InputImage`` sentinel inside) → delegates to "
            "``block_delete_confirm`` immediately. "
            "Non-empty container → sets "
            "``state.pending_delete_block_id = block_id`` which opens "
            "the confirm-delete modal (body: \"Delete container "
            "<label> and its N inner block(s)?\" where N excludes "
            "``InputImage``)."
        ),
    },
    {
        "kind": "block_delete_confirm",
        "group": "DAG canvas: Pipeline containers",
        "payload": "``{block_id: str, ts: int}``",
        "notes": (
            "Second stage (or single stage if no confirmation was "
            "needed). Atomically removes the block, its ``nested`` "
            "scope (if any) including every inner block + edge, and "
            "every edge in the containing scope whose source or "
            "target is the block. Clears ``selected_block_id`` / "
            "``selected_edge_id`` / ``pending_delete_block_id`` when "
            "they pointed at the deleted block or one of its edges."
        ),
    },
    # ----------------------------------------------------------------
    # Fixed linear port-map dispatchers.
    # ----------------------------------------------------------------
    {
        "kind": "target_select",
        "group": "Fixed linear port map",
        "payload": (
            "``{target: LinearTarget, open_menu: bool}`` or a serialized "
            "``LinearTarget`` payload directly"
        ),
        "notes": (
            "Persist the selected insertion/fill target for the current "
            "linear scope and optionally open the port menu. The fan-in "
            "callback emits this from linear port clicks and parameter "
            "replace actions."
        ),
    },
    {
        "kind": "target_menu_close",
        "group": "Fixed linear port map",
        "payload": "``{}``",
        "notes": "Close the fixed linear target menu without changing selection.",
    },
    {
        "kind": "linear_palette_add",
        "group": "Fixed linear port map",
        "payload": "``{class_name: str}``",
        "notes": (
            "Add ``class_name`` at the selected fixed-linear target. "
            "Continuation/image-output/image-input targets insert on the "
            "image spine; parameter targets fill scalar or list aux ports. "
            "``InputImage`` is rejected because each scope owns exactly one "
            "auto-seeded input. Unsupported linear shapes pause the edit and "
            "queue a warning toast."
        ),
    },
    {
        "kind": "linear_delete_node_request",
        "group": "Fixed linear port map",
        "payload": "``{block_id: str}``",
        "notes": (
            "Stage deletion by setting a linear pending-delete token for "
            "any string ``block_id``. Unsupported shapes are no-ops here; "
            "existence and spine validation happen in "
            "``linear_delete_node_confirm``."
        ),
    },
    {
        "kind": "linear_delete_node_confirm",
        "group": "Fixed linear port map",
        "payload": "``{block_id: str}``",
        "notes": (
            "Delete a fixed-linear spine block, reconnecting the image spine "
            "and removing side values owned by that block. This branch "
            "performs the existence and spine validation; missing or "
            "non-spine ids leave the graph unchanged. Clears the pending "
            "delete token afterward."
        ),
    },
    {
        "kind": "linear_node_move",
        "group": "Fixed linear port map",
        "payload": "``{block_id: str, direction: \"left\" | \"right\"}``",
        "notes": (
            "Move a fixed-linear spine block one position left or right. "
            "Unsupported shapes, invalid block ids, and invalid directions "
            "are no-ops."
        ),
    },
    {
        "kind": "linear_clear_param",
        "group": "Fixed linear port map",
        "payload": "``{target: LinearTarget}``",
        "notes": (
            "Clear one scalar side value or one list-slot side value at a "
            "fixed-linear parameter target. If clearing would remove an "
            "embedded ``ImagePipeline`` source, the dispatcher stages a "
            "confirmation token instead of deleting immediately."
        ),
    },
    {
        "kind": "linear_clear_param_confirm",
        "group": "Fixed linear port map",
        "payload": "``{target: LinearTarget}``",
        "notes": (
            "Confirm a previously staged fixed-linear parameter clear, remove "
            "the side-value edge(s), reset the scope target to continuation, "
            "and clear the pending-delete token."
        ),
    },
    {
        "kind": "linear_drill_param_pipeline",
        "group": "Fixed linear port map",
        "payload": (
            "``{target: LinearTarget, source_block_id: str | None}``"
        ),
        "notes": (
            "Drill into an aux ``ImagePipeline`` source selected from a "
            "fixed-linear parameter target. When ``source_block_id`` is "
            "omitted, the dispatcher resolves the source from the target's "
            "current aux edge."
        ),
    },
    {
        "kind": "linear_select_aux_value",
        "group": "Fixed linear port map",
        "payload": "``{source_block_id: str}``",
        "notes": (
            "Select an existing aux source block in the current fixed-linear "
            "scope and clear edge selection. Missing source ids are no-ops. "
            "This retained dispatcher branch is not emitted by the current "
            "fixed-linear UI, whose side-value controls route through "
            "replace, clear, and drill actions."
        ),
    },
]


# ---------------------------------------------------------------------------
# RST rendering.
# ---------------------------------------------------------------------------


_RST_HEADER = """Pipeline builder dispatch kinds
===============================

.. note::

   This page is generated by ``scripts/generate_dispatch_reference.py``
   from
   :data:`phenotypic.gui.builder._callbacks.DispatchKind` plus a
   hand-curated mapping that mirrors spec §5.6 verbatim. Run the
   script after touching either to regenerate; the ``--check`` flag
   is wired into CI to catch drift.

Every state-mutating event in the pipeline builder funnels through a
single fan-in callback that dispatches via
:func:`phenotypic.gui.builder._callbacks._dispatch_state_update`.
Each ``kind`` value is one of the literals enumerated below; the
payload schemas come straight from spec §5.6 of the *Builder canvas —
DAG redesign* design document.

Dispatch flow per mutation:

1. **Pre-mutation:** the dispatcher runs ``_seed_input_image`` on every
   reachable scope (idempotent guard for validation Rule 6).
2. **Mutation:** the named ``kind`` branch applies to a deep-copied
   state dict.
3. **Post-mutation:** the fan-in callback re-runs
   :func:`phenotypic.gui.builder._validation.validate`; the resulting
   list lands in ``STORE_ISSUES``, drives the toolbar count badge,
   and gates Run preview / Save pipeline (see
   :doc:`builder_validation`).

The kinds are documented in declaration order so the literal alias
and this page never diverge.
"""


def _render_rst(entries: List[Dict[str, Any]]) -> str:
    """Render the full RST page from the dispatch-kind metadata.

    Args:
        entries: Mapping entries in declaration order.

    Returns:
        Full RST document as a string. Output is deterministic — the
        same ``entries`` input always produces the same byte sequence.
    """

    chunks: List[str] = [_RST_HEADER]

    # Group entries by ``group`` for section headers.
    seen_groups: List[str] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        group = entry["group"]
        if group not in grouped:
            grouped[group] = []
            seen_groups.append(group)
        grouped[group].append(entry)

    for group in seen_groups:
        chunks.append("")
        chunks.append(group)
        chunks.append("-" * len(group))
        chunks.append("")
        for entry in grouped[group]:
            kind = entry["kind"]
            chunks.append(f"``{kind}``")
            chunks.append("~" * (len(kind) + 4))
            chunks.append("")
            chunks.append(f"**Payload schema:** {entry['payload']}")
            chunks.append("")
            chunks.append(entry["notes"])
            chunks.append("")

    return "\n".join(chunks).rstrip("\n") + "\n"


def _enumerate_dispatch_kinds() -> List[str]:
    """Pull the active :data:`DispatchKind` literal values.

    Returns:
        The kinds in the order they're declared in
        ``_callbacks.DispatchKind``.
    """

    from phenotypic.gui.builder._callbacks import DispatchKind  # noqa: PLC0415

    return list(get_args(DispatchKind))


def _check_coverage(entries: List[Dict[str, Any]]) -> None:
    """Assert the mapping covers every declared :data:`DispatchKind` literal.

    Raises:
        SystemExit: With a non-zero status when the mapping drifts from
            the literal alias.
    """

    declared = _enumerate_dispatch_kinds()
    mapped = [entry["kind"] for entry in entries]

    missing = [k for k in declared if k not in set(mapped)]
    extra = [k for k in mapped if k not in set(declared)]

    if missing or extra:
        msg_lines = [
            "Dispatch-kind reference mapping is out of sync with "
            "_callbacks.DispatchKind:",
        ]
        if missing:
            msg_lines.append(f"  missing from _DISPATCH_KINDS: {missing}")
        if extra:
            msg_lines.append(f"  extra in _DISPATCH_KINDS: {extra}")
        print("\n".join(msg_lines), file=sys.stderr)
        raise SystemExit(2)


def main(argv: List[str] | None = None) -> int:
    """Entry point for the dispatch reference generator.

    Args:
        argv: Optional CLI argument list (for tests). ``None`` falls
            back to ``sys.argv``.

    Returns:
        Process exit code (``0`` on success, ``1`` on drift in
        ``--check`` mode, ``2`` on mapping incoherence).
    """

    parser = argparse.ArgumentParser(
        description="Generate builder_dispatch.rst from "
        "DispatchKind + hand-curated mapping.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail nonzero if the committed RST differs from the "
        "regenerated output.",
    )
    args = parser.parse_args(argv)

    _check_coverage(_DISPATCH_KINDS)
    rendered = _render_rst(_DISPATCH_KINDS)

    return write_or_check_generated_file(
        output_path=OUTPUT_RST,
        rendered=rendered,
        check=args.check,
        regenerate_command="uv run python scripts/generate_dispatch_reference.py",
    )


if __name__ == "__main__":
    raise SystemExit(main())
