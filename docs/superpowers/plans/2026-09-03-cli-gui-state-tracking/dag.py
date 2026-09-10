#!/usr/bin/env python3
"""Derive the file-overlap veto table from each task's ``Files:`` block.

``EXECUTION.md`` has said *"Regenerate with ``dag.py`` before trusting it"*
since it was written, and pointed at ``scratchpad/dag.py`` -- a session-local
path that no longer exists. So the one table whose whole job is to stop two
agents being dispatched onto one file could not actually be regenerated, and
the instruction to distrust it had no way to be followed. This is that script,
committed beside the plan where the mutation harnesses and spikes already live.

**What it is for.** Two clusters may run in parallel only if they share no
file. A stale table silently authorises a collision, and the failure mode is
not a merge conflict -- it is two agents editing the same file through
different tools, where the later write wins and the earlier one vanishes with
nothing failing.

**What it cannot see.** Only files a task's ``Files:`` block names. A task that
turns out to need a file nobody listed -- as P2 Task 1 did with
``_cli_slurm_lifecycle.py``, discovered while building -- is invisible here
until the block is corrected. That is the argument for correcting the block
rather than only the commit message: this table is generated from the plan, so
an uncorrected plan produces a wrong veto forever.

Usage::

    uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/dag.py
    uv run python .../dag.py --check     # non-zero if EXECUTION.md is stale
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

PLAN_DIR = Path(__file__).resolve().parent

#: A whole ``- Create:`` / ``- Modify:`` / ``- Test:`` / ``- Delete:`` line.
#: The REST of the line is captured, not the first path on it, because a single
#: line may name several: P3 Task 2 has
#: ``- Test: `tests/.../test_image_record.py`, `tests/.../test_run_state.py``` .
_FILE_LINE = re.compile(
    r"^-\s+(?:Create|Modify|Test|Delete):\s+(.+)$", re.MULTILINE
)

#: Every backticked span inside such a line.
_BACKTICKED = re.compile(r"`([^`]+)`")

#: A trailing line reference in any of the three shapes the plan actually uses:
#: ``:123``, ``:123-145``, and ``:2394,2428,2439,2874,3725``.
#:
#: **The comma form is why this is a named constant with a comment.** A first
#: version stripped only the first two, so P6 Task 0's
#: ``phenotypicCLI.py:2394,2428,...`` stayed a distinct key and the script
#: reported that P6 T0 does not touch ``phenotypicCLI.py`` -- which read as the
#: plan's Files block being incomplete, when the block was right and the parser
#: was wrong. It was caught only by opening the block instead of believing the
#: output. A veto table generator that under-reports overlaps fails in the
#: direction that costs work, so its parsing gets the same scrutiny as the plan.
_LINE_REF = re.compile(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
_TASK_HEADING = re.compile(r"^##+\s+Task\s+([0-9]+[a-z]?):", re.MULTILINE)


def _phase_docs() -> list[Path]:
    return sorted(PLAN_DIR.glob("phase-[0-7]-*.md"))


def _phase_label(doc: Path) -> str:
    """``phase-2-identity-schema.md`` -> ``P2``."""
    return "P" + doc.name.split("-", 2)[1]


def _tasks_with_files(doc: Path) -> list[tuple[str, set[str]]]:
    """Return ``[(task_id, {file, ...}), ...]`` for one phase document.

    A ``Files:`` block belongs to the task heading above it, so the document is
    split on headings and each chunk scanned. Blocks before the first heading
    (phase preamble) are ignored -- they name context, not ownership.
    """
    text = doc.read_text(encoding="utf-8")
    marks = [(m.start(), m.group(1)) for m in _TASK_HEADING.finditer(text)]
    if not marks:
        return []

    out: list[tuple[str, set[str]]] = []
    for i, (start, task_id) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        paths: set[str] = set()
        for line in _FILE_LINE.findall(text[start:end]):
            for span in _BACKTICKED.findall(line):
                # `:255` is a second line reference on the same file, not a
                # path -- P3 Task 2 writes ``_cli_completion.py:163`, `:255``.
                if span.startswith(":"):
                    continue
                # Drop the line reference: two tasks touching different lines
                # of one file still cannot run in parallel.
                paths.add(_LINE_REF.sub("", span.split(" ")[0]))
        # Plan-internal references (OPEN-QUESTIONS.md, the spec) are not code
        # the cluster edits in the sense this veto is about, but the spec IS
        # edited by some tasks and two agents must not both edit it either.
        out.append((f"{_phase_label(doc)} T{task_id}", paths))
    return out


def build_overlaps() -> dict[str, list[str]]:
    owners: dict[str, list[str]] = defaultdict(list)
    for doc in _phase_docs():
        for task, paths in _tasks_with_files(doc):
            for path in paths:
                owners[path].append(task)
    return {p: t for p, t in owners.items() if len(t) > 1}


def render(overlaps: dict[str, list[str]]) -> str:
    rows = sorted(overlaps.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    lines = ["| Touchers | File | Tasks |", "|---|---|---|"]
    for path, tasks in rows:
        lines.append(f"| {len(tasks)} | `{path}` | {', '.join(tasks)} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any overlap is missing from EXECUTION.md",
    )
    args = parser.parse_args()

    overlaps = build_overlaps()
    table = render(overlaps)

    if not args.check:
        print(table)
        return 0

    execution = (PLAN_DIR / "EXECUTION.md").read_text(encoding="utf-8")
    missing = [p for p in overlaps if f"`{p}`" not in execution]
    if missing:
        print("EXECUTION.md is STALE -- these overlapping files are absent:")
        for path in sorted(missing):
            print(f"  {path}  ({', '.join(overlaps[path])})")
        print("\nRegenerated table:\n")
        print(table)
        return 1
    print(f"EXECUTION.md names all {len(overlaps)} overlapping files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
