"""Derive the task DAG from each task's `Files:` block.

Regenerate whenever a `Files:` block changes:

    uv run python docs/superpowers/specs/.../refinery/build_dag.py

Two parsing rules earn their keep, both learned from getting it wrong:

1. **Blockquote lines are excluded.** Corrections are recorded in `>` blocks that
   frequently name a file as the one NOT to touch -- Task 3.7's note names
   `_cli_staged_strategy.py` precisely to say the flag does not go there. A naive
   parser reads that as a dependency and invents a conflict.
2. **"Read (do not edit)" is not a conflict.** Two tasks reading one file may run
   in parallel; only writers collide.

Output is a conflict table (what forbids parallelism) plus per-task sizes.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

PLANS = Path(__file__).resolve().parents[3] / "plans/2026-08-18-ome-zarr-image-store"

TASK_RE = re.compile(r"^### (Task (\d+)\.(\w+)):\s*(.*)$", re.M)
# Accept an optional `:123` / `:12-34` line reference after the extension: the
# plan routinely writes `phenotypicCLI.py:400`, and a pattern anchored on the
# closing backtick silently misses every one of them.
PATH_RE = re.compile(
    r"`([\w./_-]+\.(?:py|md|toml|yml|yaml|json|xsd|schema))(?::[\d-]+)?`"
)
# A Files: block ends at the next bold section or the first step checkbox.
FILES_END = re.compile(r"\n\*\*[A-Z]|\n- \[ \] \*\*Step")

# Store-internal names that appear in prose but are not repo files.
NOT_REPO_FILES = {"zarr.json", "METADATA.ome.xml"}

# refinery/ -> <spec>/ -> specs/ -> superpowers/ -> docs/ -> repo root
REPO = Path(__file__).resolve().parents[5]


def _canonical(path: str, index: dict[str, str]) -> str | None:
    """Resolve a possibly-bare filename to its real repo path.

    The plan mixes `src/phenotypic/_cli/_cli_process_single.py` with a bare
    `_cli_execution_strategies.py` on the very same line. Keyed naively they
    become two different files and the conflict between them disappears --
    which is the failure mode that makes a conflict table unsafe to cluster on.
    """
    if path in NOT_REPO_FILES:
        return None
    if "/" in path:
        return path
    # Fall back to the bare name rather than dropping it: repo-root files like
    # pyproject.toml are not under the indexed subtrees, and silently discarding
    # them loses a real conflict (0.1 vs 0.2).
    return index.get(Path(path).name, path)


def _repo_index() -> dict[str, str]:
    """basename -> repo-relative path, for basenames that are unambiguous."""
    seen: dict[str, list[str]] = defaultdict(list)
    for sub in ("src", "tests", ".github"):
        base = REPO / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                seen[p.name].append(str(p.relative_to(REPO)))
    return {n: v[0] for n, v in seen.items() if len(v) == 1}

WRITE_KINDS = ("create", "modify", "delete", "extend", "rewrite", "test")


def parse() -> dict:
    index = _repo_index()
    tasks: dict[str, dict] = {}
    for doc in sorted(PLANS.glob("phase-*.md")):
        body = doc.read_text()
        marks = list(TASK_RE.finditer(body))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
            seg = body[m.start():end]
            tid = m.group(1).replace("Task ", "")
            title = m.group(4).strip()

            fm = re.search(r"\*\*Files:\*\*(.*)", seg, re.S)
            files: dict[str, set[str]] = defaultdict(set)
            if fm:
                block = fm.group(1)
                stop = FILES_END.search(block)
                block = block[: stop.start()] if stop else block
                for line in block.splitlines():
                    stripped = line.lstrip()
                    if stripped.startswith(">"):
                        continue                      # a correction note, not a dependency
                    low = line.lower()
                    if "read (do not edit)" in low or "run-only" in low:
                        kind = "read"
                    else:
                        kind = next((k for k in WRITE_KINDS if k in low), None)
                    if kind is None:
                        # A continuation line of the previous entry: attribute it there.
                        kind = getattr(parse, "_last", None)
                    else:
                        parse._last = kind
                    for raw in PATH_RE.findall(line):
                        path = _canonical(raw, index)
                        if path:
                            files[kind or "modify"].add(path)

            tasks[tid] = {
                "phase": m.group(2),
                "title": title,
                "cut": "CUT" in title or title.startswith("~~"),
                "files": files,
                "lines": len(seg.splitlines()),
                "doc": doc.name,
            }
    return tasks


def main() -> int:
    tasks = parse()
    live = {k: v for k, v in tasks.items() if not v["cut"]}

    def order(t: str) -> tuple:
        p, s = t.split(".", 1)
        return int(p), s

    print(f"{len(tasks)} tasks ({len(live)} live, {len(tasks) - len(live)} cut)\n")

    # ---- conflicts: two tasks WRITING one file --------------------------
    writers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    readers: dict[str, list[str]] = defaultdict(list)
    for tid in sorted(live, key=order):
        for kind, paths in live[tid]["files"].items():
            for p in paths:
                (readers[p].append(tid) if kind == "read"
                 else writers[p].append((tid, kind)))

    print("=== FILES WITH >1 WRITER — these forbid parallelism ===")
    conflicts = {f: ts for f, ts in writers.items() if len({t for t, _ in ts}) > 1}
    for f, ts in sorted(conflicts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        who = ", ".join(f"{t}({k})" for t, k in sorted(set(ts), key=lambda x: order(x[0])))
        print(f"  {f}\n      {who}")

    print("\n=== READ-ONLY SHARING (safe to parallelize) ===")
    for f, ts in sorted(readers.items()):
        if len(ts) > 1 or f in writers:
            print(f"  {f}: read by {', '.join(sorted(set(ts), key=order))}"
                  + (f"; WRITTEN by {sorted({t for t, _ in writers[f]})}" if f in writers else ""))

    print("\n=== PER-TASK SIZE ===")
    for tid in sorted(live, key=order):
        t = live[tid]
        nf = sum(len(v) for v in t["files"].values())
        print(f"  {tid:<6} {t['lines']:>5} lines  {nf:>2} files   {t['title'][:52]}")

    # ---- phases that can overlap ----------------------------------------
    print("\n=== CROSS-PHASE WRITE OVERLAP ===")
    by_phase: dict[str, set[str]] = defaultdict(set)
    for tid in live:
        for kind, paths in live[tid]["files"].items():
            if kind != "read":
                by_phase[live[tid]["phase"]] |= paths
    phases = sorted(by_phase)
    for a_i, a in enumerate(phases):
        for b in phases[a_i + 1:]:
            shared = by_phase[a] & by_phase[b]
            if shared:
                print(f"  Phase {a} <-> Phase {b}: {len(shared)} shared")
                for f in sorted(shared):
                    print(f"      {f}")
    clean = [(a, b) for a_i, a in enumerate(phases) for b in phases[a_i + 1:]
             if not (by_phase[a] & by_phase[b])]
    print(f"\n  Phase pairs with NO shared writes: "
          f"{', '.join(f'{a}/{b}' for a, b in clean)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
