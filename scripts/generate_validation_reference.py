"""Generator for the validation-rule reference RST page.

Reads the :data:`phenotypic.gui.builder._validation.IssueKind` literal
to enumerate every emitted ``Issue.kind`` value, then renders one
section per kind from the hand-curated rule table below. Writes to
``docs/source/api_reference/gui/builder_validation.rst``.

Run with ``--check`` to fail nonzero if the committed RST does not
match the regenerated output (used by CI to detect drift between the
spec and the docs).

Determinism: rules are emitted in the order they appear in
:data:`_RULES`, which mirrors the rule-number ordering from spec §4.6.
Same input → same output bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, get_args

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_RST = (
    REPO_ROOT
    / "docs"
    / "source"
    / "api_reference"
    / "gui"
    / "builder_validation.rst"
)


# ---------------------------------------------------------------------------
# Hand-curated validation-rule table (mirrors spec §4.6).
# ---------------------------------------------------------------------------
#
# Each entry binds a ``rule`` ordinal (matching the spec) to one or
# more :data:`IssueKind` literal values. The mapping is the source of
# truth here because the rule numbers are documentation-only: at
# runtime the rules are pure functions inside
# ``_validate_scope`` and don't carry an ordinal attribute.

_RULES: List[Dict[str, Any]] = [
    {
        "rule": 1,
        "kinds": ["fork"],
        "severity": "error",
        "title": "Image-flow ports have at most one wire",
        "mechanic": (
            "Per-port wire-count over ``edge.kind == \"image\"``, plus a "
            "total-fan-out check across image + aux (one outgoing wire "
            "from any source, total)."
        ),
        "offender": (
            "The block whose output or input violates the rule. Three "
            "sub-cases all surface as ``kind=\"fork\"``: a source with "
            ">1 outgoing image edge, an ``(target_block_id, \"in\")`` "
            "port with >1 incoming image edge, and a source with >1 "
            "outgoing wires *total* across image and aux."
        ),
    },
    {
        "rule": 2,
        "kinds": ["stub"],
        "severity": "error",
        "title": "All blocks reachable from Input Image",
        "mechanic": (
            "BFS from the ``InputImage`` block across image-flow edges "
            "forward and aux edges in both directions. Any block not "
            "visited by the walk is flagged as a stub."
        ),
        "offender": (
            "Each unreachable block (rendered with a dashed red "
            "border). Extra ``InputImage`` blocks are excluded from "
            "the stub set so they're flagged once as "
            "``duplicate_input`` instead of double-flagged."
        ),
    },
    {
        "rule": 3,
        "kinds": ["required_aux"],
        "severity": "error",
        "title": "Required aux ports must be wired",
        "mechanic": (
            "For each block, walk the registry's "
            "``OperationInfo.parameters``; for every op-typed parameter "
            "(``param.is_operation or param.is_pipeline``) without a "
            "default (``not param.has_default``), require at least one "
            "aux edge targeting ``(block_id, param_name)``."
        ),
        "offender": (
            "The consumer block. The empty required port renders with "
            "a red ring."
        ),
    },
    {
        "rule": 4,
        "kinds": ["cycle"],
        "severity": "error",
        "title": "No cycles in the edge graph",
        "mechanic": (
            "Iterative Tarjan's strongly-connected-components over the "
            "combined edge graph (image + aux). Any block participating "
            "in a non-trivial SCC (size > 1 OR size 1 with a self-loop) "
            "is reported."
        ),
        "offender": (
            "Every block in the strongly-connected cycle (sorted "
            "lexicographically for deterministic test output)."
        ),
    },
    {
        "rule": 5,
        "kinds": ["container_mode"],
        "severity": "error",
        "title": "Container left/right wiring consistency",
        "mechanic": (
            "For each Pipeline container, evaluate whether the outer "
            "left image-input is wired and what kind of port the right "
            "output wires to. The two valid modes are "
            "*consumer-fed* (left wired, right wires to image) and "
            "*aux-fed* (left unwired, right wires to aux). Mixed modes "
            "are rejected."
        ),
        "offender": "The container block whose wiring is inconsistent.",
    },
    {
        "rule": 6,
        "kinds": ["missing_input", "duplicate_input"],
        "severity": "error",
        "title": "Exactly one Input Image per scope",
        "mechanic": (
            "Count ``InputImage`` blocks in each scope. Zero → emit "
            "``missing_input`` as a scope-level issue. Two or more → "
            "emit one ``duplicate_input`` issue per extra block."
        ),
        "offender": (
            "For ``missing_input``: reported as a scope-level issue "
            "(``block_id=None``); the dispatcher's auto-seed normally "
            "heals this on the next state-load pass. For "
            "``duplicate_input``: the extra block(s)."
        ),
    },
    {
        "rule": 7,
        "kinds": ["stage_order_hint"],
        "severity": "advisory",
        "title": "Stage ordering respects ops → meas → post",
        "mechanic": (
            "Walk each image-flow edge; if the source block's stage "
            "(via ``_safe_stage(class_name)``) is later in the canonical "
            "order than the target's, emit a yellow-border advisory. "
            "The runtime partitions by ``isinstance`` so a misordered "
            "chain still works — this is a non-blocking nudge."
        ),
        "offender": (
            "The *source* block of the out-of-order edge (yellow "
            "border + \"?\" badge)."
        ),
    },
    # Advisory issue that doesn't map to a numbered spec rule but still
    # emits from the validator.
    {
        "rule": None,
        "kinds": ["unknown_class"],
        "severity": "advisory",
        "title": "Class not in the operation registry",
        "mechanic": (
            "Registry lookup for each non-sentinel block's "
            "``class_name``. A miss emits ``unknown_class`` as an "
            "advisory so the rule-3 walk can skip the block cleanly "
            "without raising."
        ),
        "offender": (
            "The block whose class is unknown to the registry (yellow "
            "border + \"?\" badge). Typically caused by registry drift "
            "(loading a ``pipeline.json`` saved by a newer build)."
        ),
    },
]


# ---------------------------------------------------------------------------
# RST rendering.
# ---------------------------------------------------------------------------


_RST_HEADER = """Pipeline builder validation rules
=================================

.. note::

   This page is generated by
   ``scripts/generate_validation_reference.py`` from
   :data:`phenotypic.gui.builder._validation.IssueKind` plus a
   hand-curated rule table that mirrors spec §4.6 verbatim. Run the
   script after touching either to regenerate; the ``--check`` flag
   is wired into CI to catch drift.

The DAG builder's validation surface is a pure function
:func:`phenotypic.gui.builder._validation.validate` that walks every
scope reachable from ``state.root`` and emits a flat list of
:class:`~phenotypic.gui.builder._validation.Issue` records. Six
**blocking** rules (severity ``"error"``) disable
``Run preview`` and ``Save pipeline``; the advisory hints (severity
``"advisory"``) decorate the canvas with yellow borders but never
block.

Rules emit in deterministic order:

1. ``missing_input`` / ``duplicate_input`` (Rule 6)
2. ``fork`` (Rule 1)
3. ``stub`` (Rule 2)
4. ``required_aux`` / ``unknown_class`` (Rule 3)
5. ``cycle`` (Rule 4)
6. ``container_mode`` (Rule 5)
7. ``stage_order_hint`` (Rule 7, advisory)

Nested-scope issues are appended after the parent scope's issues so
snapshot-style tests stay stable.

Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 5 12 10 35

   * - Rule
     - Kind
     - Severity
     - Title
"""


def _render_rst(entries: List[Dict[str, Any]]) -> str:
    """Render the full RST page from the rule metadata.

    Args:
        entries: Rule entries in declaration order.

    Returns:
        Full RST document as a string. Output is deterministic — the
        same ``entries`` input always produces the same byte sequence.
    """

    chunks: List[str] = [_RST_HEADER]

    # Summary table rows.
    for entry in entries:
        rule_label = (
            str(entry["rule"]) if entry["rule"] is not None else "—"
        )
        kinds_label = ", ".join(f"``{k}``" for k in entry["kinds"])
        chunks.append(f"   * - {rule_label}")
        chunks.append(f"     - {kinds_label}")
        chunks.append(f"     - {entry['severity']}")
        chunks.append(f"     - {entry['title']}")
    chunks.append("")

    # Per-rule detail sections.
    chunks.append("Per-rule reference")
    chunks.append("------------------")
    chunks.append("")
    for entry in entries:
        rule_label = (
            f"Rule {entry['rule']}" if entry["rule"] is not None else "Advisory"
        )
        heading = f"{rule_label} — {entry['title']}"
        chunks.append(heading)
        chunks.append("~" * len(heading))
        chunks.append("")
        chunks.append(
            f"**Issue kind(s):** {', '.join(f'``{k}``' for k in entry['kinds'])}"
        )
        chunks.append("")
        chunks.append(f"**Severity:** {entry['severity']}")
        chunks.append("")
        chunks.append(f"**Mechanic:** {entry['mechanic']}")
        chunks.append("")
        chunks.append(f"**Offender:** {entry['offender']}")
        chunks.append("")

    return "\n".join(chunks).rstrip("\n") + "\n"


def _enumerate_issue_kinds() -> List[str]:
    """Pull the :data:`IssueKind` literal values.

    Returns:
        The kinds in their declaration order.
    """

    from phenotypic.gui.builder._validation import IssueKind  # noqa: PLC0415

    return list(get_args(IssueKind))


def _check_coverage(entries: List[Dict[str, Any]]) -> None:
    """Assert the rule table covers every :data:`IssueKind` literal value.

    Raises:
        SystemExit: With a non-zero status when the table drifts from
            the literal alias.
    """

    declared = _enumerate_issue_kinds()
    mapped: List[str] = []
    for entry in entries:
        mapped.extend(entry["kinds"])

    missing = [k for k in declared if k not in set(mapped)]
    extra = [k for k in mapped if k not in set(declared)]

    if missing or extra:
        msg_lines = [
            "Validation-rule reference mapping is out of sync with "
            "_validation.IssueKind:",
        ]
        if missing:
            msg_lines.append(f"  missing from _RULES: {missing}")
        if extra:
            msg_lines.append(f"  extra in _RULES: {extra}")
        print("\n".join(msg_lines), file=sys.stderr)
        raise SystemExit(2)


def main(argv: List[str] | None = None) -> int:
    """Entry point for the validation reference generator.

    Args:
        argv: Optional CLI argument list (for tests). ``None`` falls
            back to ``sys.argv``.

    Returns:
        Process exit code (``0`` on success, ``1`` on drift in
        ``--check`` mode, ``2`` on mapping incoherence).
    """

    parser = argparse.ArgumentParser(
        description="Generate builder_validation.rst from IssueKind + "
        "hand-curated rule mapping.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail nonzero if the committed RST differs from the "
        "regenerated output.",
    )
    args = parser.parse_args(argv)

    _check_coverage(_RULES)
    rendered = _render_rst(_RULES)

    if args.check:
        if not OUTPUT_RST.exists():
            print(
                f"{OUTPUT_RST} does not exist; run "
                "scripts/generate_validation_reference.py without --check.",
                file=sys.stderr,
            )
            return 1
        existing = OUTPUT_RST.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                f"{OUTPUT_RST} is out of date; regenerate with "
                "`uv run python scripts/generate_validation_reference.py`.",
                file=sys.stderr,
            )
            return 1
        return 0

    OUTPUT_RST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RST.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT_RST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
