"""P1 Task 3 Step 6: prove every test in the cache suite can fail.

Twelve mutations, each reintroducing one specific bug into
``src/phenotypic/sdk_/_verification_cache.py`` and asserting that the named
test -- and, for the strict ones, ONLY the named tests -- goes red.
**A mutation not demonstrated is a mutation not tested**, and INV-VERDICT's
suite is the worst possible place for a test that cannot fail: every test in
it asserts that something did *not* happen.

Run from the worktree root::

    uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/\
mutation_harnesses/p1_task3_verification_cache.py

**Two gates run before any mutation is applied**, in this order:

1. **Name integrity.** Every test a ``MUTATIONS`` entry claims must exist, by
   AST. Without this a typo reports ``NOT PROVED`` and reads as a weak test,
   when in fact no test was ever written -- the failure blames the wrong
   artifact and the investigation starts in the wrong place. Same class of
   error as ``F822`` in ``__all__``: a name asserted against nothing.
2. **Coverage.** Any test no mutation claims is printed. This is what stops
   the suite decaying one well-intentioned addition at a time; an unproved
   test looks exactly like a proved one from the outside.

Then the suite must be green, because mutation results against a red suite
are noise.

**Concurrency (learned the hard way, 2026-09-04).** This script holds the
pristine source in memory for its whole run and writes it back after every
mutation, so **an edit made to the target while it runs is silently reverted
at the end**. The usual worry with a mutation harness is that it fails to
restore; the live hazard is the opposite -- it restores *too well*, over work
that arrived after it started, and a hash check catches that only as an
unexplained mismatch afterwards. Do not edit the target while this runs, and
announce start and finish if anyone else is working in the tree. A pytest run
by someone else mid-mutation is misleading too: mutation #3 fails exactly one
test, indistinguishable from a genuine one-test regression.

Safety: the target is copied to a temp directory before anything is touched,
restored in a ``finally``, and its sha256 compared at the end. The backup
lives outside the repo, so an interrupted run leaves no stray file in the
working tree. Nothing but the target is modified.
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

TARGET = Path("src/phenotypic/sdk_/_verification_cache.py").resolve()
SUITE = "tests/unit/sdk_/test_verification_cache.py"

# (label, old, new, tests that MUST fail)
MUTATIONS: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "cached_states ignores identity_digest",
        "    stored_digest, states = entry\n"
        "    if stored_digest != identity_digest:\n"
        "        return None\n"
        "    return MappingProxyType(states)",
        "    _stored_digest, states = entry\n"
        "    return MappingProxyType(states)",
        (
            "test_a_stale_identity_never_matches",
            "test_an_identity_change_replaces_the_output_entry_wholesale",
            # End-to-end: with the fence ignored, a bumped
            # scientific_config_digest no longer discards the stale entries,
            # so a shallow pass reuses them and reports depth="shallow"
            # instead of escalating. (P1 cluster 1.3 handover test.)
            "test_an_identity_change_forces_reverification",
        ),
    ),
    (
        "an empty stat map counts as current (the all([]) inversion)",
        "    if not entry.stat_tuples:\n        return False\n    root = Path",
        "    root = Path",
        (
            "test_a_forged_entry_never_licenses_reuse",
            # The same inversion, observed through the resolver: the forged
            # entries carry stat_tuples={}, so making an empty map "current"
            # licenses their verdict="verified" and turns an INCOMPLETE run
            # complete. This is the end-to-end half spec §14 calls the
            # highest-value test in the change, and the cache-side mutation
            # above is what it depends on. (P1 cluster 1.3 handover test.)
            "test_a_forged_entry_cannot_manufacture_complete",
        ),
    ),
    (
        "ctime_ns replaces mtime_ns in the currency check",
        "        if (info.st_size, info.st_mtime_ns) != tuple(expected):",
        "        if (info.st_size, info.st_ctime_ns) != tuple(expected):",
        (
            "test_ctime_is_not_part_of_the_currency_check",
            "test_the_currency_check_never_reads_ctime",
        ),
    ),
    (
        "remember_states merges instead of replacing",
        "    _CACHE[_cache_key(output_dir)] = (identity_digest, dict(entries))",
        "    key = _cache_key(output_dir)\n"
        "    previous = _CACHE.get(key)\n"
        "    merged = dict(previous[1]) if previous is not None else {}\n"
        "    merged.update(entries)\n"
        "    _CACHE[key] = (identity_digest, merged)",
        ("test_an_identity_change_replaces_the_output_entry_wholesale",),
    ),
    (
        "clear_verification_cache ignores output_dir",
        "    if output_dir is None:\n"
        "        _CACHE.clear()\n"
        "        return\n"
        "    _CACHE.pop(_cache_key(output_dir), None)",
        "    _CACHE.clear()",
        (
            "test_clear_scoped_to_one_output_does_not_clear_another",
            # End-to-end: clearing output `a` also empties `b`, so a shallow
            # pass over the untouched `b` escalates and reports depth="deep".
            # (P1 cluster 1.3 handover test.)
            "test_clear_scoped_to_one_output_does_not_clear_another"
            "_end_to_end",
        ),
    ),
    (
        "clear_verification_cache(None) is a no-op"
        " [inherently broad: this is the function the autouse _isolate_cache"
        " fixture calls, so entries leak between tests and every"
        " tracked_output_count() assertion goes red too -- a property of"
        " mutating the fixture's own reset, not a weakness in those tests]",
        "    if output_dir is None:\n"
        "        _CACHE.clear()\n"
        "        return\n"
        "    _CACHE.pop(_cache_key(output_dir), None)",
        "    if output_dir is None:\n"
        "        return\n"
        "    _CACHE.pop(_cache_key(output_dir), None)",
        ("test_clearing_every_output_leaves_nothing_tracked",),
    ),
    (
        "cached_states hands out the live dict",
        "    return MappingProxyType(states)",
        "    return states",
        ("test_the_returned_map_cannot_be_forged_in_place",),
    ),
    (
        "the regular-file guard is dropped, so a directory can be current",
        "        if not stat.S_ISREG(info.st_mode):\n            return False\n",
        "",
        ("test_a_store_directory_is_never_current",),
    ),
    (
        "remember_states aliases the caller's map",
        "(identity_digest, dict(entries))",
        "(identity_digest, entries)",
        ("test_remember_states_does_not_alias_the_callers_map",),
    ),
    (
        "_cache_key does not canonicalise the path",
        "    try:\n"
        "        return str(Path(output_dir).resolve())\n"
        "    except OSError:\n"
        "        return os.path.abspath(str(output_dir))",
        "    return str(output_dir)",
        ("test_two_spellings_of_one_output_share_one_slot",),
    ),
    (
        "a stat failure raises instead of degrading",
        "        try:\n"
        "            info = (root / relative).stat()\n"
        "        except OSError:\n"
        "            return False\n",
        "        info = (root / relative).stat()\n",
        (
            "test_a_deleted_artifact_is_not_current",
            "test_an_unreadable_path_degrades_rather_than_raising",
        ),
    ),
    (
        "the stat tuple is never compared at all",
        "        # ctime_ns is absent by design -- see the module docstring "
        "(audit S3).\n"
        "        if (info.st_size, info.st_mtime_ns) != tuple(expected):\n"
        "            return False\n",
        "",
        (
            "test_a_rewritten_artifact_is_not_current",
            "test_a_same_size_rewrite_is_caught_by_mtime",
            # End-to-end: the tampered overlay changes both size and
            # mtime_ns, so dropping the comparison keeps the warm entry
            # "current", the resolver skips its deep pass, and a tampered run
            # reports complete. (P1 cluster 1.3 handover test.)
            "test_a_tampered_artifact_falls_through_even_with_a_warm_cache",
        ),
    ),
    (
        "cached_states never returns a warm map -- the cache is written and"
        " never read. THE SAME BUG HAS A SECOND FORM IN ANOTHER MODULE:"
        " `_run_state._resolve_images` can stop consulting the cache without"
        " this function changing at all, so no single harness catches both"
        " and neither mutation is redundant with the other. This one is the"
        " cache-side form; the resolver-side form is P1 cluster 1.3's M2."
        " [inherently broad: every test that stores entries and then reads"
        " them back goes red too, which is a property of deleting the read"
        " path rather than a weakness in those tests]",
        "    entry = _CACHE.get(_cache_key(output_dir))\n",
        "    entry = None\n",
        ("test_a_warm_cache_is_actually_used",),
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_test_names() -> set[str]:
    """Return every ``test_*`` function the suite defines, by AST.

    Guards the script against itself. A typo in a MUTATIONS entry's expected
    name would otherwise read as NOT PROVED -- the mutation blamed for a test
    that never existed -- which is the same class of error as F822 in
    ``__all__``: a name asserted against nothing.
    """
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


def main() -> int:
    if not TARGET.is_file():
        print(f"ABORT: run me from the worktree root -- {TARGET} not found")
        return 4
    backup_dir = Path(tempfile.mkdtemp(prefix="phenotypic-mutation-"))
    backup = backup_dir / TARGET.name
    print(f"backup: {backup}")
    shutil.copy2(TARGET, backup)
    original = _sha256(TARGET)
    source = TARGET.read_text(encoding="utf-8")
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

        baseline = _failed_tests()
        if baseline:
            print(f"ABORT: suite is not green to begin with: {baseline}")
            return 2
        print("baseline: suite green\n")

        for label, old, new, expected in MUTATIONS:
            if source.count(old) != 1:
                rows.append(
                    (label, "SKIPPED", f"anchor matched {source.count(old)}x")
                )
                continue
            TARGET.write_text(source.replace(old, new, 1), encoding="utf-8")
            failed = _failed_tests()
            TARGET.write_text(source, encoding="utf-8")

            missing = set(expected) - failed
            extra = failed - set(expected)
            if missing:
                verdict = "NOT PROVED"
                detail = f"did not fail: {sorted(missing)}"
            elif extra:
                verdict, detail = (
                    "PROVED (broad)",
                    f"also failed: {sorted(extra)}",
                )
            else:
                verdict, detail = "PROVED", f"exactly {sorted(expected)}"
            rows.append((label, verdict, detail))
            print(f"{verdict:<14} {label}\n               {detail}")
    finally:
        shutil.copy2(backup, TARGET)
        restored = _sha256(TARGET)
        print(
            f"\nrestored: {'OK' if restored == original else 'MISMATCH'} "
            f"({restored[:12]})"
        )

    print("\n--- summary ---")
    for label, verdict, detail in rows:
        print(f"{verdict:<14} | {label} | {detail}")
    unproved = [r for r in rows if r[1] not in {"PROVED", "PROVED (broad)"}]
    print(f"\nMUTATIONS_ALL_PROVED={not unproved}")
    return 0 if not unproved else 1


if __name__ == "__main__":
    sys.exit(main())
