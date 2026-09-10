"""CAN-8 / §11's last row: one completion predicate, not two.

Task 0 deletes ``current_run_is_complete``, ``current_success_counts`` and
``current_aggregate_is_current`` from ``_cli_completion.py`` and converts every
caller onto ``resolve_run_state``. Two parsers of one question drift -- this
phase deletes ``_latest_event_states`` for exactly that reason, and would ship
a new instance of it CLI-side.

⛔ Every guard here asserts on the **parse**, never on text. A ``grep`` matches
docstrings and comments, so a text guard goes red when prose mentions a deleted
name and gets "fixed" by editing the prose to satisfy the search -- which
leaves the code unchanged and the guard meaningless. ``ast`` is the instrument
for every question in this file.

For each guard, what input would make it fire is stated in its docstring
(register entry 62).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

#: The three predicates Task 0 deletes.
RETIRED = frozenset(
    {
        "current_run_is_complete",
        "current_success_counts",
        "current_aggregate_is_current",
    }
)

#: Readers whose definition must stay unique across the whole package --
#: the three above plus the ``sdk_`` destinations the migration targets.
#: Destinations are watched too because the defect faces both ways: a second
#: ``run_proof`` in ``_cli_completion.py`` is the same drift as a second
#: ``valid_image_success`` in ``_run_state.py``.
UNIQUE_READERS = frozenset(
    {
        "valid_image_success",
        "valid_run_completion",
        "valid_aggregate_snapshot",
        "current_success_inventory",
        "run_proof",
        "run_proof_is_current",
        # Task 0's own addition, and the one most likely to grow a second
        # definition: a four-line config read is cheap to re-implement
        # locally by someone who does not know it exists. It is the ONLY
        # name this task adds to `_cli_completion`'s public surface -- Task 0
        # nets minus three, plus one -- so it is watched for the same reason
        # the `sdk_` destinations are: the failure runs both ways.
        "state_requires_success_markers",
    }
)


def _package_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
    assert root.is_dir(), f"package root does not exist: {root}"
    return root


def _python_files(*roots: Path) -> list[Path]:
    files = [
        path
        for root in roots
        for path in (root.rglob("*.py") if root.is_dir() else [root])
        if "__pycache__" not in path.parts
    ]
    assert files, f"no python files under {roots}; the walk found nothing"
    return files


def _called_and_imported(path: Path) -> set[str]:
    """Names this module CALLS or IMPORTS -- not names it merely mentions."""
    reached: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                reached.add(func.id)
            elif isinstance(func, ast.Attribute):
                reached.add(func.attr)
        elif isinstance(node, ast.ImportFrom):
            reached.update(alias.name for alias in node.names)
    return reached


def test_only_one_completion_predicate_survives() -> None:
    """The three O(N)-hashing readers are neither called nor imported CLI-side.

    Scoped to ``_cli`` + ``sdk_`` + ``phenotypicCLI.py``, not all of ``src``:
    the GUI's holders are migrated by Tasks 2-6, so a whole-tree assertion is
    red by construction at the end of Task 0. Task 7 owns the whole-tree one.

    **Fires when:** any of the three is still called or imported from those
    paths. It does NOT fire on a docstring or comment naming them, which is
    deliberate -- `_cli_completion.py`'s own prose references
    `current_aggregate_is_current`, and a text guard would demand that prose be
    edited to make a code assertion pass.
    """
    root = _package_root()
    hits = [
        f"{path.relative_to(root)}: {sorted(reached & RETIRED)}"
        for path in _python_files(
            root / "_cli", root / "sdk_", root / "phenotypicCLI.py"
        )
        if (reached := _called_and_imported(path)) & RETIRED
    ]
    assert not hits, "the old O(N)-hashing readers survive CLI-side:\n" + "\n".join(hits)


def test_the_resume_worklist_uses_the_cache_assisted_path() -> None:
    """§9's caller table, row 2 -- and §9.2's headline scenario IS this call.

    Asserted on CALL NODES. Two earlier forms could not do this: a substring
    test over the whole module (satisfied by a comment), and a scope narrowed
    to ``_prepare_incremental_startup`` -- which is thirty-one lines and holds
    no completion reader at all, the readers being inside ``phenotypic_cli``
    itself. The axis that works here is node type, not location.

    **Fires when:** `phenotypicCLI` stops calling `resolve_run_state`, or still
    calls one of the retired readers.
    """
    called = _called_and_imported(_package_root() / "phenotypicCLI.py")

    assert "resolve_run_state" in called, (
        "phenotypicCLI never calls resolve_run_state -- §9.2's headline "
        "scenario still re-hashes every marker"
    )
    survivors = sorted(called & RETIRED)
    assert not survivors, f"retired readers still called from the CLI: {survivors}"


def test_no_migrated_reader_gained_a_second_definition() -> None:
    """A deletion guard that counts ZERO of the old name is blind to a NEW copy
    under the same name in another module.

    The plan says so of itself: both its greps search only the three deleted
    predicate names, so a second `valid_image_success` inside `_run_state.py`
    passes every other gate. Counts DEFINITIONS, so prose naming a function is
    not a hit.

    **Fires when:** any watched reader is defined in two places at once --
    which is what "moved" looks like when the original was never deleted.
    """
    root = _package_root()
    seen: dict[str, list[str]] = defaultdict(list)
    for path in _python_files(root):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in UNIQUE_READERS:
                    seen[node.name].append(
                        f"{path.relative_to(root)}:{node.lineno}"
                    )

    duplicated = {name: at for name, at in seen.items() if len(at) > 1}
    assert not duplicated, f"a migrated reader has two definitions: {duplicated}"


def test_the_completion_verdict_has_exactly_four_states() -> None:
    """A REGRESSION guard, not a red-green step -- it passes today and always
    has. ``contradictory`` was never in ``Completion``; P1 defined it with four
    values. Spec §4.3 deletes it as a *state*, which is a claim about what the
    verdict may report, and this is what stops a fifth literal arriving.

    **Fires when:** someone adds a state to `Completion` -- which is the only
    way `contradictory` could come back.
    """
    import typing

    from phenotypic.sdk_._state_types import Completion

    assert set(typing.get_args(Completion)) == {
        "complete",
        "incomplete",
        "failed",
        "active",
    }


@pytest.mark.parametrize(
    "retired,module,symbol",
    [
        ("current_run_is_complete", "phenotypic.sdk_", "resolve_run_state"),
        (
            "current_aggregate_is_current",
            "phenotypic.sdk_._run_state",
            "_run_proof_covers_current_inventory",
        ),
    ],
)
def test_each_retired_predicate_has_a_reachable_replacement(
    retired: str, module: str, symbol: str
) -> None:
    """Each retirement is complete only when its replacement is REACHABLE.

    **An earlier version of this test could not fail**, and it is worth saying
    how, because it is the defect this file's own docstring warns about. It
    parametrized ``(retired, replacement)`` where ``replacement`` was a prose
    string, then asserted ``retired in RETIRED`` (always true -- both params
    are drawn from it) and ``assert replacement`` (a non-empty literal, always
    true). The only statements that could raise were two imports, identical in
    both cases, so the parametrization was decorative and the assertions were
    vacuous. It tested that two symbols import, twice.

    This resolves the *named* replacement against the tree, per parameter.

    **Fires when:** the replacement symbol is gone or renamed (the mapping
    table drifted from `sdk_`), or the retired predicate reappears on
    `_cli_completion`'s public surface (the deletion was reverted).

    ``current_success_counts`` is deliberately absent: it is three questions
    under one name, and one of its three targets -- a schema-shape predicate
    for *"is this a legacy state?"* -- **does not exist**. Asserting a
    replacement for it would assert a fiction.
    """
    import importlib

    replacement_module = importlib.import_module(module)
    assert hasattr(replacement_module, symbol), (
        f"{retired}'s replacement {module}.{symbol} is not reachable; the "
        "mapping table has drifted from the tree"
    )

    from phenotypic._cli import _cli_completion

    assert not hasattr(_cli_completion, retired), (
        f"{retired} is public on _cli_completion again -- the retirement was "
        "reverted, and every consumer guard above passes on a private rename "
        "rather than on a deletion"
    )


def test_one_completion_query_walks_each_image_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit §4's double walk, and the number §9.2 rests on.

    A single completion query must verify each image **once**. Before Task 0
    the CLI's startup block (``phenotypicCLI.py:2497-2516``) called three
    readers -- ``current_success_counts``, ``current_aggregate_is_current``
    and ``valid_run_completion`` -- each doing its own O(N) pass over the
    images. It is now one ``resolve_run_state`` at ``:2508``.

    **Counts ``_verify_image``, NOT ``valid_image_success``.** The plan's
    original instruction named the latter, and after this migration that
    function is not on the reader path at all: ``resolve_run_state``
    re-derives the same judgement through ``_run_state._verify_image``
    (``:682``, called at ``:1424``), and ``_run_state``'s only mentions of
    ``valid_image_success`` are prose. Instrumenting it here would report
    **0 against an expected 6** -- which does not read as a refuted
    measurement, it reads as a broken fixture, and the step would have been
    abandoned rather than corrected.

    **What each other answer means**, because an assertion that cannot tell
    them apart is worth much less than one that names both:

    * **2N** -- the double walk is back. That is the specific regression Task
      0 removed, and it is *silent*: nothing fails, the run is merely twice as
      slow, and it stays invisible until someone re-derives §9.2 by hand.
    * **0** -- ``resolve_run_state`` no longer reaches ``_verify_image``, so
      the query is answering from somewhere else entirely. An assertion
      written as ``<= N`` would call that a pass.

    A permanent test rather than the throwaway counter the plan describes:
    the regression it guards has no failure mode of its own, so a measurement
    taken once and discarded protects nothing after the day it ran.
    """
    from phenotypic.sdk_ import _run_state, resolve_run_state

    from .conftest import _publish_successful_images

    stems = ["a", "b", "c"]
    _publish_successful_images(tmp_path, stems=stems)

    calls: list[str] = []
    real_verify = _run_state._verify_image

    def counting_verify(*args: object, **kwargs: object) -> object:
        calls.append(str(args[:2]))
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(_run_state, "_verify_image", counting_verify)

    state = resolve_run_state(tmp_path, depth="deep")

    assert len(state.images) == len(stems), (
        f"the fixture produced {len(state.images)} images, not {len(stems)}; "
        "the call-count assertion below would be measuring the wrong tree"
    )
    assert len(calls) == len(stems), (
        f"one completion query verified {len(calls)} times over "
        f"{len(stems)} images. {2 * len(stems)} means audit §4's double walk "
        f"is back; 0 means resolve_run_state no longer reaches _verify_image"
    )
