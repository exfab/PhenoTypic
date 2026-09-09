#!/usr/bin/env python3
"""Turn a finished regression array's junit XML into per-file durations.

    uv run python harvest_regression_durations.py <ARRAY_ID> [--out <tsv>]

Writes ``regression_durations.tsv`` (``seconds<TAB>path``), which
``regression_shard.sbatch`` reads to pack shards by measured runtime instead
of by file count. Without it the array balances file COUNT, and its wall is
set by whichever shard happened to collect the heavy files -- measured at
4m31s to 42m31s against a 285-minute total, so 3.6x of the gate's wall time
was the imbalance rather than the work.

Why junit rather than ``--durations``: ``--durations=0`` prints every test to
the log, and ``--durations-min`` silently drops the tail -- so a file made of
four hundred fast tests reads as free, which is exactly the file that unpacks
a shard. The XML carries a time for every test and costs no log lines.

Stdlib only, and it imports no ``phenotypic``: this reads test artifacts, and
should keep working when the package under test does not.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path("/bigdata/exfab/anguy344/slurm_logs")


def resolve_classname(classname: str, repo: Path) -> str | None:
    """Map a junit ``classname`` back to the test file that produced it.

    ``classname`` is a dotted module path, and for class-based tests it
    carries the class as a final component (``tests.unit.cli.test_x.TestFoo``).
    Trying the longest prefix first and shortening resolves both shapes
    without guessing which one this is -- the filesystem decides.
    """
    parts = classname.split(".")
    while parts:
        candidate = repo / (Path(*parts).as_posix() + ".py")
        if candidate.is_file():
            return candidate.relative_to(repo).as_posix()
        parts.pop()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("array_id", help="SLURM array job id, e.g. 28227908")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name(
        "regression_durations.tsv"))
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve()
                    .parents[4])
    args = ap.parse_args()

    xmls = sorted(LOG_DIR.glob(f"junit_{args.array_id}_*.xml"))
    if not xmls:
        print(f"No junit XML for array {args.array_id} under {LOG_DIR}.",
              file=sys.stderr)
        print("The array must have run a shard script that passes "
              "--junitxml (added 2026-09-09).", file=sys.stderr)
        return 2

    totals: dict[str, float] = defaultdict(float)
    unresolved: set[str] = set()
    tests = 0
    for xml in xmls:
        for case in ET.parse(xml).getroot().iter("testcase"):
            classname = case.get("classname") or ""
            path = resolve_classname(classname, args.repo)
            if path is None:
                unresolved.add(classname)
                continue
            totals[path] += float(case.get("time") or 0.0)
            tests += 1

    if not totals:
        print("Parsed no testcases -- refusing to write an empty durations "
              "file that would silently restore count-based packing.",
              file=sys.stderr)
        return 3

    rows = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    args.out.write_text("".join(f"{sec:.3f}\t{path}\n" for path, sec in rows))

    total = sum(totals.values())
    print(f"shards parsed:  {len(xmls)}")
    print(f"tests timed:    {tests}")
    print(f"files measured: {len(totals)}")
    print(f"total runtime:  {total / 60:.1f} min")
    print(f"perfect 24-way: {total / 60 / 24:.1f} min")
    print(f"heaviest file:  {rows[0][1] / 60:.1f} min  {rows[0][0]}")
    print(f"wrote:          {args.out}")
    if unresolved:
        # Named, not counted: an unresolved classname is silently dropped
        # work, and the shard it belonged to then looks cheaper than it is.
        print(f"\nUNRESOLVED classnames ({len(unresolved)}) -- their time is "
              f"missing from the packing:", file=sys.stderr)
        for name in sorted(unresolved):
            print(f"  {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
