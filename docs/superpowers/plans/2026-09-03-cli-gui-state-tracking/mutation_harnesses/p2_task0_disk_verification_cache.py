"""P2 Task 0: prove every test in the on-disk cache suite can fail.

Eighteen mutations across **three** modules -- ``sdk_/_verification_cache.py``
(the tier itself), ``sdk_/_run_state.py`` (the wiring that reads and writes it)
and ``sdk_/_io_constants.py`` (``clear_machine_state``, which deletes it) --
each reintroducing one specific bug and asserting that the named test goes red.

**Why three modules and not one.** Spec §9.1's corruption cases split cleanly
in two: the reader can be correct while the resolver never calls it, and the
resolver can call it and then ignore what it says. A harness restricted to the
cache module could not prove a single one of the six end-to-end cases, and one
restricted to the resolver could not prove any of the reader's. The third
target is one mutation, and it is the one that decides whether a restart
carries a pre-restart verdict across the fence the restart exists to raise.

Run from the worktree root::

    uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/\
mutation_harnesses/p2_task0_disk_verification_cache.py

**Two gates run before any mutation is applied**, in this order:

1. **Name integrity.** Every test a ``MUTATIONS`` entry claims must exist, by
   AST -- a typo would otherwise report ``NOT PROVED`` and send the reader to
   investigate a test that was never written.
2. **Coverage.** Any test no mutation claims is printed. An unproved test looks
   exactly like a proved one from the outside.

Then the suite must be green, because mutation results against a red suite are
noise.

**Some mutations are inherently broad and say so in their label.** Deleting the
resolver's ``persist_states`` call takes out every test that reads the file
afterwards, which is a property of removing the write path rather than a
weakness in those tests. Their expected lists name only the tests that fail
*because of what the mutation means*; the rest arrive as ``PROVED (broad)``.

**Concurrency.** This script holds each target's pristine source in memory for
its whole run and writes it back after every mutation, so **an edit made to any
target while it runs is silently reverted at the end**. The usual worry is a
harness that fails to restore; the live hazard is one that restores *too well*,
over work that arrived after it started. Do not edit a target while this runs,
and announce start and finish if anyone else is working in the tree. A pytest
run by someone else mid-mutation is misleading too: most mutations here fail
exactly one test, indistinguishable from a genuine one-test regression.

Safety: every target is copied to a temp directory **by full relative path**
before anything is touched, restored in a ``finally``, and its sha256 compared
at the end. Backing up by basename is what let an earlier harness clobber the
wrong file while reporting a clean restore. The backup lives outside the repo,
so an interrupted run leaves no stray file in the working tree.
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
    "src/phenotypic/sdk_/_verification_cache.py",
    "src/phenotypic/sdk_/_run_state.py",
    "src/phenotypic/sdk_/_io_constants.py",
)
SUITE = "tests/unit/sdk_/test_verification_cache_disk.py"

# (label, old, new, tests that MUST fail)
MUTATIONS: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "the reader ignores the recorded identity [§9.1 case 3]",
        "    if document.get(_IDENTITY_KEY) != identity_digest:\n"
        "        return None\n",
        "",
        (
            "test_a_cache_from_another_identity_is_refused",
            "test_a_stale_identity_falls_through_to_deep",
        ),
    ),
    (
        "the reader ignores the schema version -- a file written by a build"
        " whose verification RULES differed is honoured",
        "    if document.get(_SCHEMA_VERSION_KEY) != "
        "VERIFICATION_CACHE_VERSION:\n        return None\n",
        "",
        ("test_a_different_schema_version_is_refused",),
    ),
    (
        "a read failure propagates instead of degrading [§9.1 cases 4 and 5]."
        " This is the one that turns an archived, read-only, perfectly"
        " readable run into a tree the GUI refuses to open",
        "    try:\n"
        "        raw = verification_cache_path(Path(output_dir))"
        ".read_bytes()\n"
        "    except OSError:\n"
        "        return None\n",
        "    raw = verification_cache_path(Path(output_dir)).read_bytes()\n",
        (
            "test_an_absent_cache_file_is_a_miss",
            "test_a_cache_file_that_cannot_be_read_is_a_miss",
            "test_a_deleted_cache_file_falls_through_to_deep",
            "test_an_unreadable_cache_file_falls_through_to_deep",
        ),
    ),
    (
        "a parse failure propagates instead of degrading [§9.1 case 6]",
        "    try:\n"
        "        # UnicodeDecodeError is a ValueError, so undecodable bytes "
        "and\n"
        "        # malformed JSON are one case here, as they are one case to "
        "a caller.\n"
        "        document = json.loads(raw)\n"
        "    except ValueError:\n"
        "        return None\n",
        "    document = json.loads(raw)\n",
        (
            "test_an_unparseable_cache_file_is_a_miss",
            "test_an_unparseable_cache_file_falls_through_to_deep",
        ),
    ),
    (
        "partial trust: one malformed entry is skipped rather than discarding"
        " the document, so the half of a file that happens to parse is read"
        " [inherently broad: the single-bad-entry documents in the bool and"
        " verdict tests then load as an EMPTY map rather than None, which is"
        " a property of skipping rather than a weakness in those tests]",
        "        if rebuilt is None:\n            return None\n",
        "        if rebuilt is None:\n            continue\n",
        ("test_one_malformed_entry_discards_the_whole_document",),
    ),
    (
        "bool is accepted as a stat-tuple component, so [true, true] becomes"
        " the stat tuple (1, 1)",
        "        if not _is_plain_int(size) or not _is_plain_int(mtime_ns):",
        "        if not isinstance(size, int) or not isinstance("
        "mtime_ns, int):",
        ("test_a_boolean_masquerading_as_a_stat_tuple_is_refused",),
    ),
    (
        "the verdict is not checked against the closed set of three",
        "    if verdict not in _VERDICTS:\n        return None\n",
        "",
        ("test_an_unrecognized_verdict_is_refused",),
    ),
    (
        "entries with no stat tuples are persisted -- permanently non-current,"
        " so the file grows by every unverified image while licensing nothing",
        "                for work_id, entry in entries.items()\n"
        "                if entry.stat_tuples\n",
        "                for work_id, entry in entries.items()\n",
        ("test_unverified_entries_are_not_persisted",),
    ),
    (
        "a failed write raises instead of returning False -- a read-only"
        " output becomes an unreadable one. The whole block is replaced rather"
        " than just the handler: dropping `except` alone is a SyntaxError, and"
        " a target that will not import fails every test in the suite, which"
        " proves nothing about any of them",
        "    try:\n"
        "        payload: dict[str, object] = {\n"
        "            _SCHEMA_VERSION_KEY: VERIFICATION_CACHE_VERSION,\n"
        "            _IDENTITY_KEY: identity_digest,\n"
        "            _ENTRIES_KEY: {\n"
        "                work_id: _entry_to_json(entry)\n"
        "                for work_id, entry in entries.items()\n"
        "                if entry.stat_tuples\n"
        "            },\n"
        "        }\n"
        "        atomic_write_json(verification_cache_path(root), payload)\n"
        "    except (OSError, TypeError, ValueError):\n"
        "        return False\n"
        "    return True",
        "    payload: dict[str, object] = {\n"
        "        _SCHEMA_VERSION_KEY: VERIFICATION_CACHE_VERSION,\n"
        "        _IDENTITY_KEY: identity_digest,\n"
        "        _ENTRIES_KEY: {\n"
        "            work_id: _entry_to_json(entry)\n"
        "            for work_id, entry in entries.items()\n"
        "            if entry.stat_tuples\n"
        "        },\n"
        "    }\n"
        "    atomic_write_json(verification_cache_path(root), payload)\n"
        "    return True",
        (
            "test_persisting_into_a_read_only_output_is_not_an_error",
            "test_an_unserializable_stage_value_is_not_an_error",
        ),
    ),
    (
        "the write catch is narrowed to OSError, so an unserializable stage"
        " value escapes a function documented never to raise",
        "    except (OSError, TypeError, ValueError):\n",
        "    except OSError:\n",
        ("test_an_unserializable_stage_value_is_not_an_error",),
    ),
    (
        "the writer creates .phenotypic/ -- resolving the state of a tree this"
        " package has never written to leaves a directory behind",
        "    if not phenotypic_cache_dir(root).is_dir():\n"
        "        return False\n",
        "",
        ("test_persisting_never_creates_the_machine_state_directory",),
    ),
    (
        "the tiers are consulted in the wrong order, so a long-lived process"
        " pays the JSON read on every poll",
        "    in_process = cached_states(output_dir, identity_digest)\n"
        "    if in_process is not None:\n"
        "        return in_process\n"
        "    return load_persisted_states(output_dir, identity_digest)",
        "    on_disk = load_persisted_states(output_dir, identity_digest)\n"
        "    if on_disk is not None:\n"
        "        return on_disk\n"
        "    return cached_states(output_dir, identity_digest)",
        ("test_warm_states_prefers_tier_one",),
    ),
    (
        "tier 2 is written and never read -- U-11's 1403 s ships unfixed, and"
        " every corruption test in the suite still passes."
        " ANCHORED ON THE FALLBACK LINE ALONE, not on the whole function like"
        " the mutation above, so the two ways the read path can break stay"
        " independently attributable: this one deletes the fallback while the"
        " other reorders the tiers, and a shared anchor would have made a"
        " report of either one ambiguous about which was applied.",
        "    return load_persisted_states(output_dir, identity_digest)",
        "    return None",
        (
            "test_warm_states_falls_back_to_the_persisted_tier",
            "test_a_cold_process_reuses_the_persisted_tier",
            "test_a_fully_warm_shallow_pass_does_not_rewrite_the_file",
        ),
    ),
    (
        "the resolver never writes tier 2 [inherently broad: every test that"
        " reads or edits the file after a deep pass fails too, which is a"
        " property of deleting the write path rather than a weakness in those"
        " tests]",
        "    if escalated:\n"
        "        persist_states(output_dir, identity.digest(), entries)\n",
        "",
        (
            "test_a_deep_pass_persists_the_cache",
            "test_a_cold_process_reuses_the_persisted_tier",
        ),
    ),
    (
        "the resolver trusts a cached entry without re-stating it -- the one"
        " mutation that manufactures a `complete` from a forged file",
        "        if entry is not None and entry_is_still_current("
        "output_dir, entry):",
        "        if entry is not None:",
        (
            "test_a_moved_stat_tuple_falls_through_to_deep",
            "test_a_forged_persisted_cache_cannot_manufacture_complete",
        ),
    ),
    (
        "a warm cache reports depth=shallow even when part of the pass was"
        " deep -- 'mostly shallow' silently becomes 'shallow'",
        '        if requested_depth == "shallow" and warm is not None and '
        "not escalated\n",
        '        if requested_depth == "shallow" and warm is not None\n',
        (
            "test_an_absent_entry_falls_through_to_deep",
            "test_a_moved_stat_tuple_falls_through_to_deep",
        ),
    ),
    (
        "the writer drops `stages`, so a shallow pass served from disk loses"
        " every advisory that is a projection over them",
        '        "stages": {\n'
        "            name: dict(body) for name, body in "
        "state.stages.items()\n"
        "        },\n",
        '        "stages": {},\n',
        (
            "test_persist_then_load_round_trips",
            "test_the_persisted_tier_carries_the_advisories_with_it",
        ),
    ),
    (
        "the reader drops `stages` -- the same advisory loss, from the other"
        " side of the file",
        "            stages=stages,\n",
        "            stages={},\n",
        (
            "test_the_reader_rebuilds_every_field_of_an_entry",
            "test_persist_then_load_round_trips",
            "test_the_persisted_tier_carries_the_advisories_with_it",
        ),
    ),
    (
        "the reader drops `reason`, so 'which images are missing, and why?'"
        " stops being answerable from a cached state",
        "            reason=reason,\n",
        "            reason=None,\n",
        ("test_the_reader_rebuilds_every_field_of_an_entry",),
    ),
    (
        "clear_machine_state PRESERVES the verification cache, like"
        " restart_epoch.json -- a restart then inherits its own pre-restart"
        " verdicts across the fence it exists to raise",
        "            if child.name == TERMINAL_FAILURES_JSONL:\n"
        "                continue\n",
        "            if child.name in (\n"
        "                TERMINAL_FAILURES_JSONL,\n"
        "                VERIFICATION_CACHE_JSON,\n"
        "            ):\n"
        "                continue\n",
        ("test_clear_machine_state_deletes_the_persisted_cache",),
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_test_names() -> set[str]:
    """Return every ``test_*`` function the suite defines, by AST."""
    import ast

    tree = ast.parse(Path(SUITE).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _failed_tests() -> set[str]:
    """Return the set of test names that failed in one suite run."""
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
            # "FAILED tests/...::test_name - AssertionError: ..."
            name = stripped.split("::", 1)[-1].split(" ", 1)[0]
            failed.add(name)
    return failed


def _owner(sources: dict[Path, str], old: str) -> Path | None:
    """Return the one target containing ``old`` exactly once, else ``None``.

    Ambiguity is refused rather than resolved: an anchor matching two targets
    would silently mutate whichever the dict happened to yield first.
    """
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
        # Full relative path, never the basename: a prior harness clobbered
        # 774 lines of the wrong file that way while reporting a clean
        # restore.
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
            assert target is not None  # pre-validated above
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
