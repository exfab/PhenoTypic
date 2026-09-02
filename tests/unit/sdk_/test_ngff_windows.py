"""Windows consequences of the store layout, asserted rather than assumed.

Windows is a supported CLI platform for staged runs, and six facts follow from
that (design §3.8 / Global Constraints "Windows"):

1. no directory ``fsync`` -- Windows cannot open a directory handle to flush;
2. the move-aside retries on ``ERROR_SHARING_VIOLATION``;
3. the two-step move-aside is the only path -- ``MOVEFILE_REPLACE_EXISTING``
   cannot name a directory, so there is no single-call replace fallback;
4. store paths are ``\\\\?\\``-prefixed;
5. no case-only collisions among store path segments -- NTFS is
   case-insensitive;
6. per-file antivirus overhead is documented, not mitigated -- **no test**.

**Three of those already have tests, and they are not repeated here.**
``tests/unit/sdk_/test_ngff_promote.py`` covers (2) with
``test_a_transient_rename_failure_is_retried_not_surfaced``, (4) with
``test_long_path_prefixes_on_windows`` / ``test_long_path_is_a_passthrough_on_posix``,
(5) with ``test_store_path_segments_have_no_case_only_collisions``, and (3)
with ``test_bare_os_replace_onto_a_non_empty_directory_still_fails`` plus
``test_promote_replaces_a_non_empty_existing_store``. The Phase-7 plan's draft
of this file restated all four verbatim; duplicating them would give a future
editor two places to change and one place to forget.

What is left, and what lives here:

* the POSIX guard on the **directory** half of ``fsync_tree`` (1) -- nothing
  asserts it, and on Windows a lost guard is an ``OSError`` on every durable
  write, not a subtle degradation;
* ``_is_retryable``'s errno discrimination -- the retry test above proves a
  retryable error retries, but nothing proves a **non**-retryable one does not
  reach the backoff budget from the Windows side of the predicate;
* a deep-path write, which is the actual ``MAX_PATH`` claim rather than a
  string-prefix assertion (Windows-only);
* the **CI lane wiring itself**, which is the only part of this file a Linux
  machine can check on behalf of Windows. See
  ``test_the_windows_nightly_lane_still_collects_the_store_suites``.

Chunk-key nesting is deliberately *not* asserted here. Ledger SIMP-23 replaced
the observational ``rglob("*/0")`` form with the declared-separator assertion in
``tests/_ngff_conformance.py::_assert_reader_level_musts``, which runs against
every store written anywhere in the suite and, under the sharding codec, is the
only thing in the store that turns coordinates into a path segment at all.
"""

from __future__ import annotations

import ast
import errno
import os
import types
from pathlib import Path

import pytest
import yaml

from phenotypic.sdk_ import ngff_

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: The two suites whose whole purpose is to run on Windows. Both must be
#: reachable from a configured ``testpaths`` entry and unmarked.
STORE_SUITES = (
    Path("tests/integration/cli/test_commit_protocol.py"),
    Path("tests/unit/sdk_/test_ngff_windows.py"),
)


def _os_reporting(platform_name: str) -> types.ModuleType:
    """A stand-in for ``os`` that differs from the real one in ``name`` only.

    **Do not patch ``os.name`` globally to simulate Windows.** ``Path.__new__``
    dispatches on it, so every ``Path(...)`` constructed afterwards -- including
    the one on ``fsync_tree``'s first line -- becomes a ``WindowsPath`` whose
    ``rglob`` silently yields nothing on a POSIX filesystem. A test written that
    way asserts "no directory was flushed" against a walk that visited no file
    either, and passes for the wrong reason in both directions. Measured: the
    file flush disappears along with the directory flush.

    Patching the module's own ``os`` reference confines the simulation to the
    code under test and leaves ``pathlib`` on the real platform.
    """
    proxy = types.ModuleType("os")
    proxy.__dict__.update(os.__dict__)
    proxy.name = platform_name
    return proxy


def test_directory_fsync_is_posix_guarded(tmp_path: Path, monkeypatch) -> None:
    """Windows cannot open a directory handle for flushing.

    ``_fsync_path`` is shimmed because it is the seam that records *what* was
    flushed; leaving it real would also route the simulated platform through
    ``long_path`` and build a ``\\\\?\\`` path no POSIX ``os.open`` resolves.

    The POSIX half is asserted in the same test as a control. Without it the
    Windows assertion is satisfied by a ``fsync_tree`` that flushes no
    directory on any platform -- the silent wrong-data mode §3.7 exists to
    close -- and by the broken-walk failure described in :func:`_os_reporting`.

    Coupling this test knowingly accepts: the guard is expressed as
    ``os.name == "posix"``. Re-expressing it as ``sys.platform != "win32"``
    would be behaviour-preserving and would make this test red. That is the
    narrowest instrument available from Linux, and it is recorded here rather
    than discovered later.
    """
    store = tmp_path / "s"
    (store / "nested").mkdir(parents=True)
    (store / "nested" / "0.0").write_bytes(b"x")

    flushed: list[Path] = []
    monkeypatch.setattr(ngff_, "_fsync_path", lambda path: flushed.append(Path(path)))

    monkeypatch.setattr(ngff_, "os", _os_reporting("nt"))
    ngff_.fsync_tree(store)
    assert [path.name for path in flushed] == ["0.0"], (
        f"a directory handle was opened under a Windows os.name: {flushed}"
    )

    flushed.clear()
    monkeypatch.setattr(ngff_, "os", _os_reporting("posix"))
    ngff_.fsync_tree(store)
    assert store in flushed and (store / "nested") in flushed, (
        f"POSIX control: the directory dirents were not flushed: {flushed}"
    )


def test_is_retryable_discriminates_on_errno() -> None:
    """A genuine ENOSPC must fail fast, not burn the whole backoff budget.

    Five attempts of exponential backoff is 3.1 s per image; across 10k images
    that is an hour of sleeping before surfacing an error that was never going
    to clear.

    The ``winerror`` cases are the point of the Windows side: an
    ``ERROR_SHARING_VIOLATION`` arrives as ``OSError(EACCES, ...)`` with
    ``winerror == 32``, and it is the ``winerror`` -- not the errno -- that
    makes it retryable. The bare-EACCES case pins the other half: on POSIX the
    same errno carries no ``winerror`` and must **not** be retried.
    """
    assert ngff_._is_retryable(OSError(errno.ENOTEMPTY, "not empty")) is True
    assert ngff_._is_retryable(OSError(errno.ENOENT, "missing")) is True
    assert ngff_._is_retryable(OSError(errno.EEXIST, "exists")) is True
    assert ngff_._is_retryable(OSError(errno.ENOSPC, "no space")) is False
    assert ngff_._is_retryable(OSError(errno.EACCES, "permission denied")) is False

    for winerror in (32, 33):  # SHARING_VIOLATION, LOCK_VIOLATION
        held = OSError(errno.EACCES, "sharing violation")
        held.winerror = winerror  # type: ignore[attr-defined]
        assert ngff_._is_retryable(held) is True

    not_ours = OSError(errno.EACCES, "access denied")
    not_ours.winerror = 5  # type: ignore[attr-defined]
    assert ngff_._is_retryable(not_ours) is False


@pytest.mark.skipif(os.name != "nt", reason="MAX_PATH is a Windows limit")
def test_a_deep_store_path_still_writes(tmp_path: Path) -> None:
    """The actual MAX_PATH claim: an output root + dataset + stem + internals.

    ``test_long_path_prefixes_on_windows`` asserts the prefix is applied;
    this asserts the thing the prefix exists for.
    """
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    deep = tmp_path.joinpath(*["longish_directory_name_segment"] * 6)
    deep.mkdir(parents=True)
    store = Image(load_synth_yeast_plate()).save2zarr(deep / "p.ome.zarr")
    assert ngff_.valid_staged_store(store) is True


# ---------------------------------------------------------------------------
# The lane wiring. A Linux machine cannot execute a Windows run, but it can
# assert that the Windows run would collect these suites -- which is the only
# part of "Windows is supported" that is checkable from here, and the part that
# a future workflow edit could silently drop.
# ---------------------------------------------------------------------------


def _pytest_invocations(workflow_path: Path, job: str) -> list[str]:
    """Every ``run:`` block in *job* that invokes pytest."""
    document = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = document["jobs"][job]["steps"]
    return [
        step["run"]
        for step in steps
        if isinstance(step.get("run"), str) and "pytest" in step["run"]
    ]


def test_the_windows_nightly_lane_still_collects_the_store_suites() -> None:
    """Commit-protocol coverage on Windows is a nightly lane, by design.

    The spec accepts a one-day latency on a Windows-specific promote
    regression rather than promoting the whole Windows suite to PR time. That
    trade is only honoured if the nightly job actually collects these files,
    and the ways it could stop are all invisible from Linux: a marker filter, an
    ``--ignore``, or an explicit path list that omits them.

    The job runs bare ``pytest`` under an ``addopts`` override, so collection
    comes from ``testpaths`` -- asserted by
    ``test_the_store_suites_are_reachable_from_testpaths``.
    """
    invocations = _pytest_invocations(WORKFLOWS / "run-pytest-full.yml", "tests-windows-full")
    assert invocations, "the Windows nightly job no longer runs pytest at all"
    for command in invocations:
        assert "--ignore" not in command, command
        assert "--deselect" not in command, command
        # A `-m` filter would silently drop these suites if either ever gained
        # a marker. The addopts override deliberately clears pyproject's
        # `-m 'not slow'`, so there must be no replacement.
        assert " -m " not in command, f"a marker filter reached the Windows lane: {command}"


def test_the_pr_lane_runs_the_commit_protocol_tests_on_linux() -> None:
    """The other half of the trade: PR-time signal on Linux."""
    invocations = _pytest_invocations(WORKFLOWS / "run-pytest.yml", "tests-linux")
    assert invocations, "the PR Linux job no longer runs pytest at all"
    for command in invocations:
        assert "--ignore" not in command, command
        assert "--deselect" not in command, command


def test_the_store_suites_are_reachable_from_testpaths() -> None:
    """Both lanes collect by ``testpaths``; a file outside it is never run.

    This is the failure mode ``tests/e2e`` already sits in deliberately, and
    the one that would make every assertion above vacuous.
    """
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    testpaths = config["tool"]["pytest"]["ini_options"]["testpaths"]
    for suite in STORE_SUITES:
        assert (REPO_ROOT / suite).is_file(), suite
        assert any(suite.is_relative_to(Path(entry)) for entry in testpaths), (
            f"{suite} is not under any testpaths entry {testpaths}"
        )


def test_neither_store_suite_is_marked_slow() -> None:
    """``addopts = -m 'not slow'`` is the PR lane's only filter.

    A ``slow`` marker on either file would drop it from every PR run while
    leaving both nightly lanes green -- the regression would surface a day
    late on Windows and never on Linux.
    """
    for suite in STORE_SUITES:
        # Parsed, not grepped: this file names the marker in its own prose, so
        # a substring search reports itself as marked.
        tree = ast.parse((REPO_ROOT / suite).read_text(encoding="utf-8"))
        decorators = [
            ast.unparse(decorator)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for decorator in node.decorator_list
        ]
        module_marks = [
            ast.unparse(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
        ]
        for expression in [*decorators, *module_marks]:
            assert "mark.slow" not in expression, f"{suite}: {expression}"
