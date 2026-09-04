"""Read-only: is every test proved, and does every mutation still apply?

Three checks, none of which runs pytest or touches a file:

1. **Name integrity** — every test a ``MUTATIONS`` entry claims exists.
2. **Coverage** — every test in the suite is claimed by some mutation.
3. **Anchor drift** — every mutation's ``old`` text still matches its target
   exactly once, so no mutation has quietly become a no-op.

The first two are the harness's own startup gates without the thirteen pytest
invocations behind them. Add a test, run this, and know in a second whether it
is proved or merely present.

That second is the point. A mutation suite decays one well-intentioned
addition at a time: an unproved test looks exactly like a proved one from the
outside, passes forever, and is discovered only when someone finally breaks
the code it was supposed to guard and nothing goes red.

Run from the worktree root::

    uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/\
mutation_harnesses/check_mutation_coverage.py

Exits non-zero when a ``MUTATIONS`` entry names a test that does not exist, or
when a test no mutation claims exists.

**Controls are the exception, and they are declared rather than inferred.** A
control's job is to fail when the implementation becomes *too eager* -- it is
proved by the *absence* of a mutation making it fire spuriously, so no mutation
will ever claim it. Requiring one would force a false choice: a permanently red
gate, or contrived mutations written to satisfy this script rather than to catch
a bug. The second is worse, because it degrades the signal for everyone after.

A harness declares them::

    CONTROLS = (
        "test_a_clean_tree_carries_no_advisories",
        "test_a_matching_metadata_snapshot_raises_no_advisory",
    )

They are excluded from the coverage requirement and **printed on their own
line**, not silently exempted -- an undeclared exemption is how a real gap hides
behind a green gate, which is the failure this whole file exists to catch. A name
in ``CONTROLS`` that is not in the suite is an error, exactly as a typo in a
mutation's expected-test list is.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HARNESSES = Path(__file__).resolve().parent


Mutation = tuple[str, str, str, tuple[str, ...]]


def _mutations_of(harness: Path) -> list[Mutation]:
    """Return a harness's ``MUTATIONS`` table without importing it.

    Parsed rather than imported so this stays read-only in the strongest
    sense: nothing in the harness runs, so no target file can be touched by
    the act of checking it.
    """
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    for node in tree.body:
        target = getattr(node, "target", None)
        if target is not None and getattr(target, "id", "") == "MUTATIONS":
            return ast.literal_eval(node.value)
    raise SystemExit(f"{harness.name}: no MUTATIONS table found")


def _suite_of(harness: Path) -> Path:
    """Return the suite path a harness declares in its ``SUITE`` constant."""
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(t, "id", "") == "SUITE" for t in node.targets):
            return Path(ast.literal_eval(node.value))
    raise SystemExit(f"{harness.name}: no SUITE constant found")


def _controls_of(harness: Path) -> set[str]:
    """Return the harness's declared CONTROLS, or an empty set."""
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "CONTROLS"
            for t in node.targets
        ):
            continue
        return {
            elt.value
            for elt in ast.walk(node.value)
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    return set()


def _targets_of(harness: Path) -> list[Path]:
    """Return every module the harness mutates.

    Accepts ``TARGETS = (...)`` or a single ``TARGET``. A suite routinely spans
    modules -- ``test_run_state.py`` proves claims about ``_run_state``,
    ``_state_types`` AND ``_cli_completion`` -- and a harness restricted to one
    of them cannot prove the tests about the others. Splitting into two
    harnesses does not help: they would share a SUITE, so the coverage check
    would run twice and fail the second time (measured by cluster 1.3).
    """
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not names & {"TARGET", "TARGETS"}:
            continue
        return [
            Path(elt.value)
            for elt in ast.walk(node.value)
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
    raise SystemExit(f"{harness.name}: no TARGET or TARGETS constant found")


def _target_of(harness: Path) -> Path:
    """Return the source file a harness mutates, from ``TARGET``."""
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(t, "id", "") == "TARGET" for t in node.targets):
            # TARGET = Path("...").resolve()
            return Path(ast.literal_eval(node.value.func.value.args[0]))
    raise SystemExit(f"{harness.name}: no TARGET constant found")


def _broken_anchors(target: Path, mutations: list[Mutation]) -> list[str]:
    """Return labels whose ``old`` text no longer matches the target once.

    A drifted anchor makes the harness print ``SKIPPED`` for that mutation,
    which reads as *not run* rather than *not proved* and is easy to skim
    past in a twelve-row report. Refactoring the target is exactly when it
    happens, and exactly when nobody is thinking about the harness.
    """
    source = target.read_text(encoding="utf-8")
    return [
        f"{label[:60]} (matches {source.count(old)}x)"
        for label, old, _new, _expected in mutations
        if source.count(old) != 1
    ]


def _test_names(suite: Path) -> set[str]:
    tree = ast.parse(suite.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def main() -> int:
    harnesses = sorted(
        path
        for path in HARNESSES.glob("*.py")
        if path.name != Path(__file__).name
    )
    if not harnesses:
        print("no harnesses found")
        return 1

    failures = 0
    for harness in harnesses:
        mutations = _mutations_of(harness)
        suite = _suite_of(harness)
        if not suite.is_file():
            print(f"{harness.name}: ABORT -- suite {suite} not found; run "
                  f"from the worktree root")
            failures += 1
            continue
        defined = _test_names(suite)
        # A parametrized expected name is `test_x[case]`; the AST gives the
        # bare `test_x`. Compare on the stem, keep the full id for failure
        # matching in the runner (gap found by cluster 1.3 on day one).
        named = {
            name.split("[")[0]
            for _l, _o, _n, expected in mutations
            for name in expected
        }

        drifted = []
        for target in _targets_of(harness):
            if target.is_file():
                drifted += _broken_anchors(target, mutations)

        print(f"\n{harness.name}  ->  {suite}")
        print(f"  mutations defined      : {len(mutations)}")
        print(f"  suite tests defined    : {len(defined)}")
        print(f"  tests claimed by a mut : {len(named)}")
        controls = _controls_of(harness)
        # A control is proved by the ABSENCE of a mutation making it fire, so
        # it can never be claimed. Excluded from the requirement, printed
        # anyway: a silent exemption is how a real gap hides behind a green
        # gate.
        unknown = sorted((named - defined) | (controls - defined))
        uncovered = sorted(defined - named - controls)
        print(f"  declared controls      : {sorted(controls)}")
        print(f"  unknown (typos)        : {unknown}")
        print(f"  NOT covered by any mut : {uncovered}")
        print(f"  drifted anchors        : {drifted}")
        failures += bool(unknown) + bool(uncovered) + bool(drifted)

    print(f"\nCOVERAGE_OK={not failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
