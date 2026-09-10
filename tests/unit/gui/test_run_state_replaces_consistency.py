"""Guards for P6 Task 2: `_output_consistency.py` is gone and stays gone.

Spec §4.3, §11. The 617-line classifier that cross-checked nine evidence
sources against each other is deleted; `resolve_run_state` answers the same
questions from the written authorities alone.

**The fourth state went with it.** ``contradictory`` existed only because
derived counts were checked against each other, and two authorities that
cannot disagree cannot produce it. The guard that stops it returning is
``test_the_completion_verdict_has_exactly_four_states`` in
``tests/unit/cli/test_completion_split.py`` -- it already existed when this
file was written, so it is **not** repeated here. A second copy of one
invariant is the duplication this whole change exists to remove.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

from phenotypic._gui.results_viewer._mutation_guard import (
    OutputMutationBlocked,
    OutputMutationGuard,
)
from phenotypic._gui.results_viewer._output_root import OutputRoot, core_readable
from phenotypic.sdk_ import (
    BundleLayout,
    aggregate_publication_marker_path,
    atomic_write_json,
    gui_launch_owner_path,
    resolve_run_state,
    run_completion_marker_path,
)
from tests._output_layout import build_complete_viewer_run, write_master

#: Every name the deleted module bound. Greping the module name alone is not
#: enough: the package re-exported ``OutputConsistencyReport`` by
#: ``__getattr__``, so a consumer could import the type without ever spelling
#: the module -- which is exactly the form that was live in
#: ``test_output_discovery_contracts.py`` when this task started.
_DELETED_NAMES = (
    "_output_consistency",
    "OutputConsistencyReport",
    "OutputCompletionEvidence",
    "classify_output_consistency",
    "inspect_output_consistency",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _viewer_run(root: Path, *, complete: bool = True) -> Path:
    """A published run the results viewer can bind.

    Not `build_complete_run`: its master is keyed on `Metadata_ImageFile` and
    `OutputRoot.discover` requires `Metadata_ImageName`, so binding it raises.
    """
    import polars as pl

    from phenotypic.schema import IMAGE

    return build_complete_viewer_run(
        root,
        frame=pl.DataFrame(
            {
                "Metadata_Dataset": ["plate", "plate"],
                str(IMAGE.IMAGE_NAME): ["a", "b"],
                "Size_Area": [10.0, 20.0],
            }
        ),
        stems=("a", "b"),
        complete=complete,
    )


def test_the_grep_root_is_the_repository() -> None:
    """The guard below is only as good as the directory it searches.

    An earlier draft of it greped a **relative** ``src/``. Off the repository
    root -- under the sharded regression harness, or any invocation with a
    different working directory -- that path does not exist, ``grep`` returns
    nothing, and "nothing" is the passing condition. The guard would have
    reported success precisely when it could not look.
    """
    assert (_REPO_ROOT / "src" / "phenotypic").is_dir(), _REPO_ROOT
    assert (_REPO_ROOT / "pyproject.toml").is_file(), _REPO_ROOT


def test_no_production_module_still_imports_the_deleted_classifier() -> None:
    """No module under ``src/`` imports the deleted classifier or its types.

    **An AST walk, not a grep, and the difference is not cosmetic.** The
    property is *depends on a deleted name*, which is semantic; ``grep``
    measures spellings. Two modules legitimately name
    ``inspect_output_consistency`` in prose -- ``_gui/_snapshot_status.py``
    records what it stopped doing -- and a text search cannot tell that from a
    live import. A grep-based guard here would either fail on a correct tree
    or have to be weakened until it stopped catching anything.

    It also catches a form a search for the module name misses entirely: the
    package re-exported ``OutputConsistencyReport`` through ``__getattr__``,
    so ``from phenotypic._gui.results_viewer import OutputConsistencyReport``
    depends on the deleted module without spelling it. That form was live in
    ``test_output_discovery_contracts.py`` when this task started.

    **Fires when:** any module under ``src/`` imports any of the five names,
    by any of the three import forms.
    """
    package = "phenotypic._gui.results_viewer"
    offenders: list[str] = []
    for path in sorted((_REPO_ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = {alias.name for alias in node.names}
                if module.endswith("_output_consistency") or (
                    module == package and names & set(_DELETED_NAMES)
                ):
                    offenders.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("._output_consistency"):
                        offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "these modules still import the deleted classifier: "
        + ", ".join(offenders)
    )


def test_the_module_file_is_gone_and_no_longer_importable() -> None:
    """The file itself, not just its references.

    A stale ``.py`` left on disk still imports, so a consumer added later
    would work locally and fail only once the file was noticed and removed.
    """
    module = (
        _REPO_ROOT
        / "src/phenotypic/_gui/results_viewer/_output_consistency.py"
    )
    assert not module.exists(), module
    with pytest.raises(ImportError):
        __import__(
            "phenotypic._gui.results_viewer._output_consistency",
        )


def test_the_package_no_longer_re_exports_the_report_type() -> None:
    """``OutputConsistencyReport`` is gone from the package's public surface.

    It was reachable two ways -- a lazy ``__getattr__`` arm and an ``__all__``
    entry -- and removing only one of them leaves the type importable while
    looking deleted.
    """
    import phenotypic._gui.results_viewer as results_viewer

    assert "OutputConsistencyReport" not in results_viewer.__all__
    with pytest.raises(AttributeError):
        results_viewer.OutputConsistencyReport


# ---------------------------------------------------------------------------
# CAN-17: the two predicates that are NOT `completion` in disguise.
# ---------------------------------------------------------------------------


def test_an_incomplete_output_is_not_mutable(tmp_path: Path) -> None:
    """Mutation requires ``complete``, not merely "not active".

    ``is_read_only`` was ``state != "coherent"``, so an ``incomplete`` output
    prohibited writes. The withdrawn ``completion != "active"`` spelling would
    have granted write access to **every** incomplete output -- a widening
    with no spec authority, since §4.3 says only that ``incomplete`` is "safe
    to read, safe to resume", which is not "safe to write".

    **Fires when:** the mutation gate is relaxed to admit any non-``complete``
    run. Asserted at ``OutputMutationGuard.authorize`` rather than at
    ``output_mutations_disabled``, because the invariant is *the write is
    refused*, not *a button renders disabled* -- a test on the latter stays
    green even if ``authorize`` starts granting the write.
    """
    source = _viewer_run(tmp_path / "run", complete=False)
    output = OutputRoot.discover(source, cache_root=tmp_path / "cache")
    assert output.run_state is not None
    assert output.run_state.completion == "incomplete"

    guard = OutputMutationGuard(output, None)
    with pytest.raises(OutputMutationBlocked, match="incomplete"):
        guard.authorize("curation", presented_generation=None)


def test_an_active_run_with_a_valid_proof_is_still_core_readable(
    tmp_path: Path,
) -> None:
    """``core_readable`` is not a completion test, and its error hides itself.

    An active run whose earlier finalization published a valid aggregate
    proof **is** core-readable: the bytes are authorized, and the fact that
    work is in flight again does not un-authorize them. Any spelling that
    enumerates acceptable completions excludes ``active`` and gets this wrong.

    **Why it matters more than it looks:** this predicate gates a live-run
    ``skipif``. A false ``False`` *skips* the tests that ask it rather than
    failing them, and a skip does not show up in a summary line -- so the
    error is invisible in exactly the runs it breaks.
    """
    source = _viewer_run(tmp_path / "run")
    # The verdict ladder is total and ordered: `complete` OUTRANKS `active`
    # (Q2). So a finished run with a live worker still reads `complete`, and
    # the scenario this test is about -- finalized once, running again -- is
    # reached by removing the RUN proof while leaving the AGGREGATE proof in
    # place. That is also what such a tree actually looks like: the aggregate
    # bytes from the earlier finalization are still on disk and still
    # authorized, which is the whole point of the assertion below.
    run_completion_marker_path(source).unlink()
    # A live pid: `_live_authority` believes a GUI owner record only while the
    # process it names is alive, so a fabricated pid would read as terminal
    # and this test would assert `active` against a run that is not.
    owner = gui_launch_owner_path(source)
    owner.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "active-after-finalize",
            "generation": "generation-2",
            "status": "running",
            "pid": os.getpid(),
        },
    )

    assert resolve_run_state(source, depth="deep").completion == "active"
    assert core_readable(BundleLayout.detect(source)) is True


def test_a_legacy_tree_is_core_readable_with_no_aggregate_proof(
    tmp_path: Path,
) -> None:
    """The first disjunct, which a completion test cannot express at all.

    A legacy tree never published an aggregate proof and never required
    success markers. It is readable on the strength of the second fact alone.

    **Fires when:** ``core_readable`` is rewritten to require a proof
    unconditionally -- which would make every pre-marker output in the wild
    unopenable rather than merely unfinished.
    """
    import polars as pl

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    # `BundleLayout.detect` resolves topology from the master's presence, so
    # a legacy tree still needs one. What it does NOT have is a
    # `processing_state.json` requiring success markers, or any proof.
    write_master(
        legacy,
        pl.DataFrame({"Metadata_Dataset": ["plate"], "Size_Area": [1.0]}),
    )
    assert core_readable(BundleLayout.detect(legacy)) is True


def test_a_marker_authorized_tree_without_a_proof_is_not_core_readable(
    tmp_path: Path,
) -> None:
    """The other side of the disjunction, so neither half is decorative.

    **Fires when:** ``core_readable`` degenerates to ``True`` -- which the
    two tests above cannot catch, because both of them expect ``True``.
    """
    source = _viewer_run(tmp_path / "run")
    assert core_readable(BundleLayout.detect(source)) is True

    proof = aggregate_publication_marker_path(source)
    assert proof.is_file(), proof
    proof.unlink()

    assert core_readable(BundleLayout.detect(source)) is False


def test_the_ledger_row_for_this_feature_still_resolves() -> None:
    """`FEATURES.md`'s renamed row must still name a test that exists.

    ``scripts/check_features_md.py`` asserts a literal ``def <name>(`` for
    every ``✅ shipping`` row. This task rewrote that row's title and
    description while migrating the file its test lives in, and the CI job
    that catches a broken ref runs only on a PR -- so it is asserted here too.
    """
    ledger = (_REPO_ROOT / "src/phenotypic/_gui/FEATURES.md").read_text(
        encoding="utf-8"
    )
    row = next(
        line for line in ledger.splitlines() if "Run completion state" in line
    )
    assert "✅ shipping" in row
    match = re.search(r"(tests/[\w/]+\.py)::([\w]+)", row)
    assert match is not None, row
    target = (_REPO_ROOT / match.group(1)).read_text(encoding="utf-8")
    assert re.search(rf"\bdef\s+{re.escape(match.group(2))}\s*\(", target)
