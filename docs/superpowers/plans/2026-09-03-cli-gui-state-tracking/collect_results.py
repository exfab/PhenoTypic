"""Aggregate a sharded gate run into a comparable failure set.

Sharding makes the raw pytest summaries useless on their own: 48 of them, each
with its own counts, and the interesting question is never "how many failed" but
"which failed, and is that set the same as the baseline's".

That distinction is load-bearing here rather than stylistic. This suite has a
documented population of load-sensitive flakes -- tests with wall-clock budgets
for spawned children (a 1 s import, a 20 s multiprocessing join, a 0.5 s patched
read deadline) that pass alone and fail on a contended node. Their COUNT moves
with node load and with how the shards happened to pack. Their NAMES do not.
So: compare names, never counts.

A second trap this closes: shard membership is `index % SHARDS` over the sorted
file list, so ADDING A TEST FILE reshuffles every shard. A flake therefore
appears to "move between shards" with nothing having changed. Comparing by name
is immune to that too.

Usage:
    uv run python collect_results.py <results_dir> [--baseline <other_dir>]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# defusedxml, not xml.etree: the stdlib parsers accept external entities and
# expand nested entities without bound. These particular files are our own
# junit output on our own filesystem, so the threat model is thin -- but the
# dependency is already installed, so declining it would buy nothing.
from defusedxml import ElementTree as ET  # type: ignore[import-untyped]


def _load(results_dir: pathlib.Path) -> tuple[dict[str, str], dict[str, int]]:
    """Return ({test_id: outcome}, totals) across every shard's junit XML."""
    outcomes: dict[str, str] = {}
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "shards": 0}

    shards = sorted(results_dir.glob("shard_*.xml"))
    if not shards:
        raise SystemExit(f"no shard_*.xml under {results_dir}")

    # A shard that has not finished yet leaves no XML, and its tests then
    # simply do not appear -- so a partial run reports FEWER failures than a
    # complete one and reads as an improvement. Nothing else here notices,
    # because every check operates on the shards that ARE present.
    #
    # Caught by running it against a 48-shard array while 4 were still
    # RUNNING: 44 shards, 9,967 tests, "REGRESSIONS (0)". The diff was against
    # an 11,106-test baseline and the 1,139 missing tests were silent.
    present = {int(s.stem.split("_")[1]) for s in shards}
    gaps = sorted(set(range(max(present) + 1)) - present)
    if gaps:
        raise SystemExit(
            f"INCOMPLETE: shards {gaps} are missing from {results_dir}.\n"
            f"Found {len(present)} of {max(present) + 1} expected.\n"
            "A missing shard's tests are absent, not passing -- so this run\n"
            "would under-report failures and a regression in a missing shard\n"
            "would read as 'no regressions'. Wait for the array to finish\n"
            "(`squeue -j <id>`), then re-run."
        )

    for shard in shards:
        totals["shards"] += 1
        try:
            root = ET.parse(shard).getroot()
        except ET.ParseError as exc:  # defusedxml re-exports this
            # A truncated XML means the shard died mid-write -- an OOM kill or a
            # walltime cut. That is a harness fault, not a test result, and it
            # must not be silently read as "no failures in this shard".
            print(f"!! {shard.name}: unparseable ({exc}) -- shard did not finish",
                  file=sys.stderr)
            totals["errors"] += 1
            continue
        for case in root.iter("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            test_id = f"{classname}::{name}" if classname else name
            totals["tests"] += 1
            if case.find("failure") is not None:
                outcomes[test_id] = "failed"
                totals["failures"] += 1
            elif case.find("error") is not None:
                outcomes[test_id] = "error"
                totals["errors"] += 1
            elif case.find("skipped") is not None:
                totals["skipped"] += 1
    return outcomes, totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path, default=None)
    args = parser.parse_args()

    outcomes, totals = _load(args.results_dir)
    bad = sorted(t for t, o in outcomes.items() if o in ("failed", "error"))

    print(f"shards={totals['shards']}  tests={totals['tests']}  "
          f"failed={totals['failures']}  errors={totals['errors']}  "
          f"skipped={totals['skipped']}")
    print()

    if not args.baseline:
        print(f"{len(bad)} failing test(s):")
        for test_id in bad:
            print(f"  {test_id}")
        print("\nNo baseline given. Re-run with --baseline to classify these as "
              "regressions or pre-existing.")
        return 1 if bad else 0

    base_outcomes, _ = _load(args.baseline)
    base_bad = {t for t, o in base_outcomes.items() if o in ("failed", "error")}

    regressions = sorted(set(bad) - base_bad)
    fixed = sorted(base_bad - set(bad))
    persistent = sorted(set(bad) & base_bad)

    print(f"REGRESSIONS ({len(regressions)}) -- failing here, passing at baseline:")
    for test_id in regressions:
        print(f"  {test_id}")
    print()
    print(f"pre-existing ({len(persistent)}) -- failing in both:")
    for test_id in persistent:
        print(f"  {test_id}")
    print()
    print(f"newly passing ({len(fixed)}):")
    for test_id in fixed:
        print(f"  {test_id}")

    if regressions:
        print("\nBefore treating any of these as real: run it ALONE. This suite's "
              "known flakes pass in isolation and fail under shard contention.")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
