"""P2 Task 1: prove every test in the restart-epoch suite can fail.

Ten mutations across three targets — `_cli/_cli_identity.py` (the counter),
`_cli/_cli_slurm_lifecycle.py` (the writer that stamps it) and
`sdk_/_run_state.py` (rule 2's fence that reads it) — plus
`sdk_/_io_constants.py` for the preserve set.

**Why the writer and the reader are both mutated.** The fence is only as good
as the field it compares, and the two can fail independently: a writer that
stops stamping leaves the reader comparing against a default, and a reader
that stops comparing leaves the writer stamping a field nobody consults.
Neither shows up in the other's tests, which is why
`test_the_lifecycle_record_carries_the_epoch_current_at_publication` exists
outside the fence at all.

**The asymmetry mutation is the one worth reading.** `_live_authority`
deliberately fences the SLURM lifecycle record and deliberately does *not*
fence the GUI owner record. That is the one place this change covers half a
surface on purpose, so it is where a future contributor "completes" the job.
The mutation stamps an epoch onto the owner record too — a one-line change
that reads as finishing what Task 1 started — and
`test_a_live_gui_owner_still_reports_active_across_a_restart` is what goes
red.

Run from the worktree root::

    uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/\
mutation_harnesses/p2_task1_restart_epoch.py

Safety and concurrency rules are identical to the Task 0 harness: full-path
backups outside the repo, restore in a `finally`, sha256 compared at the end,
and **do not edit a target while this runs** — it holds pristine source in
memory and writes it back after every mutation, so a concurrent edit is
silently reverted.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ENV = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}

TARGETS = (
    "src/phenotypic/_cli/_cli_identity.py",
    "src/phenotypic/_cli/_cli_slurm_lifecycle.py",
    "src/phenotypic/sdk_/_run_state.py",
    "src/phenotypic/sdk_/_io_constants.py",
)
SUITE = "tests/unit/cli/test_run_identity.py"

# (label, old, new, tests that MUST fail)
MUTATIONS: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "scientific_config_digest becomes a WRAPPER instead of an alias --"
        " equal today, and one edit away from not being, with nothing failing"
        " at the moment it stops. D-C's whole point is that the generation and"
        " work_id cannot be allowed to drift apart about what counts as"
        " scientific configuration, and identity is the only agreement that"
        " cannot.",
        "scientific_config_digest = processing_configuration_digest\n",
        "def scientific_config_digest(config):\n"
        "    return processing_configuration_digest(config)\n",
        ("test_scientific_config_digest_is_the_work_id_digest_itself",),
    ),
    (
        "the restart epoch is dropped from the preserve set -- a restart"
        " resets the counter that fences it, which is not a fence",
        "    {TERMINAL_FAILURES_JSONL, RESTART_EPOCH_JSON}\n",
        "    {TERMINAL_FAILURES_JSONL}\n",
        ("test_restart_epoch_survives_clear_machine_state",),
    ),
    (
        "a corrupt counter raises instead of degrading to 0 -- one bad byte"
        " becomes a reason the user cannot restart at all",
        "    try:\n"
        "        # UnicodeDecodeError is a ValueError, so undecodable bytes "
        "and\n"
        "        # malformed JSON are one case here, as they are one case to "
        "a caller.\n"
        "        document = json.loads(raw)\n"
        "    except ValueError:\n"
        "        return 0\n",
        "    document = json.loads(raw)\n",
        ("test_reading_a_corrupt_restart_epoch_is_zero_not_an_error",),
    ),
    (
        "bool is accepted as an epoch, so `true` reads as 1 -- a fence"
        " silently advanced by a type error",
        "    if not isinstance(epoch, int) or isinstance(epoch, bool) or "
        "epoch < 0:\n",
        "    if not isinstance(epoch, int):\n",
        ("test_a_boolean_is_not_a_restart_epoch",),
    ),
    (
        "a failed write is swallowed and returns quietly -- the next"
        " invocation mints the generation the abandoned workers already hold",
        "    atomic_write_json(restart_epoch_path(root), "
        "{_EPOCH_KEY: updated})\n",
        "    try:\n"
        "        atomic_write_json(restart_epoch_path(root), "
        "{_EPOCH_KEY: updated})\n"
        "    except OSError:\n"
        "        pass\n",
        ("test_a_failed_write_raises_rather_than_returning_quietly",),
    ),
    (
        "the writer stops stamping the epoch. NOTE WHAT THIS DOES *NOT*"
        " BREAK: the fence tests still pass, because an unstamped record"
        " degrades to 0 and 0 >= a bumped epoch is False -- so the run is"
        " still fenced, for the wrong reason. Only the writer's own tests"
        " see it, which is exactly why they exist outside the fence.",
        '            "restart_epoch": read_restart_epoch(output_dir),\n',
        "",
        (
            "test_the_lifecycle_record_carries_the_epoch_current_at_"
            "publication",
            "test_an_existing_active_fence_is_not_re_dated",
        ),
    ),
    (
        "an existing active fence is RE-DATED to the current epoch, so a"
        " worker from before the restart looks current again -- the precise"
        " failure the fence exists to prevent."
        " THE MUTATION MUST PERSIST, and the first version did not: it set"
        " the key on `existing`, which is the in-memory dict"
        " `load_slurm_lifecycle` returned, and the early return writes"
        " nothing. The file stayed correct, so the test passed and the"
        " mutation reported NOT PROVED against a test that was right all"
        " along. `_live_authority` reads the FILE, so an in-memory re-stamp"
        " is a no-op for the fence -- it did not model the bug at all.",
        "            # Deliberately NOT re-stamped: this fence was published "
        "earlier,\n"
        "            # and the epoch live at *that* moment is the one it "
        "asserts.\n"
        "            return existing\n",
        "            existing[\"restart_epoch\"] = read_restart_epoch("
        "output_dir)\n"
        "            atomic_write_json(\n"
        "                lifecycle_state_path(output_dir), existing\n"
        "            )\n"
        "            return existing\n",
        ("test_an_existing_active_fence_is_not_re_dated",),
    ),
    (
        "rule 2's first half is dropped -- a lifecycle record from a"
        " superseded epoch reports the run alive, which is a stale authority"
        " outranking a valid verdict. Takes down BOTH fenced-authority tests,"
        " which is right rather than broad: deleting the comparison stops"
        " fencing a stamped stale record AND a record that predates the"
        " field, and those are two separate claims about the same line.",
        "        and _record_restart_epoch(lifecycle) >= "
        "identity.restart_epoch\n",
        "",
        (
            "test_a_pre_restart_authority_does_not_report_the_run_active",
            "test_a_record_without_an_epoch_is_fenced_on_a_restarted_run",
        ),
    ),
    (
        "the fence is strict, so a record at the CURRENT epoch is refused --"
        " no run is ever active and every 'is refused' test still passes",
        "        and _record_restart_epoch(lifecycle) >= "
        "identity.restart_epoch\n",
        "        and _record_restart_epoch(lifecycle) > "
        "identity.restart_epoch\n",
        (
            "test_a_current_authority_still_reports_the_run_active",
            "test_a_record_without_an_epoch_still_counts_on_an_unrestarted_"
            "run",
        ),
    ),
    (
        "a missing epoch field degrades UPWARD instead of to 0, so a doubtful"
        " authority is believed rather than fenced -- away from incomplete,"
        " against INV-VERDICT's direction",
        "    epoch = record.get(\"restart_epoch\")\n"
        "    if not isinstance(epoch, int) or isinstance(epoch, bool):\n"
        "        return 0\n",
        "    epoch = record.get(\"restart_epoch\")\n"
        "    if not isinstance(epoch, int) or isinstance(epoch, bool):\n"
        "        import sys as _s\n"
        "        return _s.maxsize\n",
        ("test_a_record_without_an_epoch_is_fenced_on_a_restarted_run",),
    ),
    (
        "THE ASYMMETRY: the GUI owner record is epoch-fenced too -- a"
        " one-line change that reads as finishing the job, and kills a LIVE"
        " process's claim on the strength of a counter it never read",
        "        if (\n"
        "            isinstance(pid, int)\n"
        "            and not isinstance(pid, bool)\n"
        "            and _process_is_alive(pid)\n"
        "        ):\n",
        "        if (\n"
        "            isinstance(pid, int)\n"
        "            and not isinstance(pid, bool)\n"
        "            and _process_is_alive(pid)\n"
        "            and _record_restart_epoch(owner) >= "
        "identity.restart_epoch\n"
        "        ):\n",
        ("test_a_live_gui_owner_still_reports_active_across_a_restart",),
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_test_names() -> set[str]:
    import ast

    tree = ast.parse(Path(SUITE).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _failed_tests() -> set[str]:
    proc = subprocess.run(
        ["uv", "run", "pytest", SUITE, "-q", "--no-header", "-rf"],
        capture_output=True,
        text=True,
        env={**_ENV},
    )
    failed: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED ") or stripped.startswith("ERROR "):
            name = stripped.split("::", 1)[-1].split(" ", 1)[0]
            failed.add(name)
    return failed


def _owner(sources: dict[Path, str], old: str) -> Path | None:
    """Return the one target containing ``old`` exactly once, else ``None``."""
    owners = [path for path, text in sources.items() if text.count(old) == 1]
    return owners[0] if len(owners) == 1 else None


def main() -> int:
    targets = [Path(name).resolve() for name in TARGETS]
    missing = [t for t in targets if not t.is_file()]
    if missing:
        print(f"ABORT: run me from the worktree root -- {missing} not found")
        return 4
    backup_dir = Path(tempfile.mkdtemp(prefix="phenotypic-mutation-"))
    print(f"backup: {backup_dir}")
    sources: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    for name, target in zip(TARGETS, targets):
        backup = backup_dir / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        sources[target] = target.read_text(encoding="utf-8")
        originals[target] = _sha256(target)

    rows: list[tuple[str, str, str]] = []
    try:
        defined = _suite_test_names()
        named = {name for _l, _o, _n, exp in MUTATIONS for name in exp}
        unknown = named - defined
        if unknown:
            print(
                "ABORT: MUTATIONS names tests that do not exist: "
                f"{sorted(unknown)}"
            )
            return 3
        unclaimed = defined - named
        print(
            f"suite defines {len(defined)} tests; {len(named)} are claimed by "
            f"a mutation"
        )
        if unclaimed:
            print(f"NOT COVERED by any mutation: {sorted(unclaimed)}")

        unowned = [
            label
            for label, old, _new, _exp in MUTATIONS
            if _owner(sources, old) is None
        ]
        if unowned:
            print(
                "ABORT: these anchors match no target exactly once: "
                f"{[label[:60] for label in unowned]}"
            )
            return 3

        baseline = _failed_tests()
        if baseline:
            print(f"ABORT: suite is not green to begin with: {baseline}")
            return 2
        print("baseline: suite green\n")

        for label, old, new, expected in MUTATIONS:
            target = _owner(sources, old)
            assert target is not None
            source = sources[target]
            target.write_text(source.replace(old, new, 1), encoding="utf-8")
            failed = _failed_tests()
            target.write_text(source, encoding="utf-8")

            missing_tests = set(expected) - failed
            extra = failed - set(expected)
            if missing_tests:
                verdict = "NOT PROVED"
                detail = f"did not fail: {sorted(missing_tests)}"
            elif extra:
                verdict, detail = (
                    "PROVED (broad)",
                    f"also failed: {sorted(extra)}",
                )
            else:
                verdict, detail = "PROVED", f"exactly {sorted(expected)}"
            rows.append((label, verdict, detail))
            print(
                f"{verdict:<14} [{target.name}] {label}\n"
                f"               {detail}"
            )
    finally:
        for name, target in zip(TARGETS, targets):
            shutil.copy2(backup_dir / name, target)
            restored = _sha256(target)
            status = "OK" if restored == originals[target] else "MISMATCH"
            print(f"restored {name}: {status} ({restored[:12]})")

    print("\n--- summary ---")
    for label, verdict, detail in rows:
        print(f"{verdict:<14} | {label} | {detail}")
    unproved = [r for r in rows if r[1] not in {"PROVED", "PROVED (broad)"}]
    print(f"\nMUTATIONS_ALL_PROVED={not unproved}")
    return 0 if not unproved else 1


if __name__ == "__main__":
    sys.exit(main())
