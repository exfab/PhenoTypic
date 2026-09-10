"""Integration tests for ``run_console._recent_runs.scan_recent_runs``.

Pre-populate a sandbox with multiple CLI-output dirs (varying status,
mode, recency), then verify:

    * The scanner returns one row per output dir.
    * has_dashboard reflects the filesystem.
    * Status comes from ``resolve_run_state`` and mode from the CLI's
      ``job_metadata.json``. **Neither comes from ``manifest.json`` any more**
      -- spec §4.2 demoted it, and P6 Task 4 removed the last reader. A fixture
      here that writes only a manifest describes no run at all.
    * Rows are sorted newest first.
    * When a registry is supplied, it is rehydrated as a side effect.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest

from phenotypic._gui._config import DELIVERABLES_DIRNAME
from phenotypic._gui.run_console._recent_runs import (
    RecentRunRow,
    scan_recent_runs,
)
from phenotypic._gui.shell._runs_registry import RunRegistry
from phenotypic._gui.shell._sandbox import SandboxRoot
from phenotypic.sdk_ import job_metadata_path, terminal_failures_jsonl_path
from tests._output_layout import build_incomplete_run


def _make_run(
    root: Path,
    name: str,
    *,
    has_dashboard: bool = False,
    mtime_offset_seconds: float = 0.0,
) -> Path:
    """Build the cheapest directory the sidebar classifier calls a CLI output.

    **A DISCOVERY fixture only.** It carries no run state, so every row it
    produces is ``status="unknown"``, ``mode="unknown"`` -- which is the right
    answer for a directory holding no run, and is why the tests below assert
    rel_path, ordering, depth and has_dashboard rather than status.

    It used to write a ``progress/manifest.json`` with ``is_complete`` /
    ``failed`` / ``execution_mode``, and ``test_scan_rows_carry_status_and_mode``
    read its status and mode back out. Spec §4.2 demoted that file and P6 Task 4
    removed the last reader, so those five parameters became inert: they were
    still accepted, still written, and no longer reachable by any assertion.
    They are deleted rather than defaulted, so a caller that wants a *status*
    has to go build a real tree -- see the test below.

    The manifest is not a discovery signal either: ``classify`` reads it only
    on the ``is_process_only_output`` branch, which requires **no** ``results/``
    (`gui/shell/_classifier.py:286-294`), and this fixture creates one.
    """
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    # User-facing deliverables live under ``out/deliverables/``; ``results/``
    # stays at the run root. Together they are what `classify` keys on.
    deliverables = out / DELIVERABLES_DIRNAME
    deliverables.mkdir(exist_ok=True)
    (deliverables / "master_measurements.parquet").write_bytes(b"")
    (out / "results").mkdir(exist_ok=True)
    if has_dashboard:
        (deliverables / "dashboard.html").write_text("<html/>")
    if mtime_offset_seconds:
        new_mtime = time.time() + mtime_offset_seconds
        os.utime(out, (new_mtime, new_mtime))
    return out


def _seed_discoverable(run_root: Path) -> Path:
    """Make a real fixture tree discoverable without disturbing its state.

    ``build_complete_run`` writes a master whose digest the run proof binds, so
    the empty marker ``_make_run`` stamps must not land on top of it -- that
    would leave a tree that is discovered and no longer ``complete``, passing
    the discovery half of a test while silently failing the half under test.
    """
    (run_root / "results").mkdir(exist_ok=True)
    return run_root


def _write_job_metadata(run_root: Path, *, execution_mode: str) -> None:
    """Write the CLI's submission record -- the mode source since P6 Task 4.

    ``job_metadata.json``, not ``manifest.json``: both carry ``execution_mode``,
    and only one of them is the record that owns it.
    """
    path = job_metadata_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"execution_mode": execution_mode}), encoding="utf-8"
    )


def test_scan_returns_one_row_per_output(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_a", has_dashboard=True)
    _make_run(tmp_path, "run_b")
    sandbox = SandboxRoot.from_path(tmp_path)
    rows = scan_recent_runs(sandbox)
    rel_paths = {r.rel_path for r in rows}
    assert rel_paths == {"run_a", "run_b"}


def test_scan_rows_carry_status_and_mode(tmp_path: Path) -> None:
    """The Recent Runs list shows a failed run as failed, and names its mode.

    **This is a user-visible surface, not a schema detail.** P6 Task 4 replaced
    the manifest-count reader behind these two fields and this file was outside
    that task's gate, so for one commit a failed run displayed as ``unknown``
    with an ``unknown`` mode and nothing in ``tests/unit/gui/shell`` could see
    it. Three of this test's four assertions were failing.

    Both trees are built by the real publishers, because status is now a
    verdict over artifacts: a hand-written JSON file cannot produce one.
    """
    failed_run = build_incomplete_run(tmp_path / "rs")
    terminal_failures_jsonl_path(failed_run).parent.mkdir(
        parents=True, exist_ok=True
    )
    terminal_failures_jsonl_path(failed_run).write_text(
        json.dumps({"work_id": "work-b", "exception_type": "OSError"}) + "\n",
        encoding="utf-8",
    )
    _seed_discoverable(failed_run)
    _write_job_metadata(failed_run, execution_mode="local")

    unfinished_run = build_incomplete_run(tmp_path / "rl")
    _seed_discoverable(unfinished_run)
    _write_job_metadata(unfinished_run, execution_mode="slurm")

    sandbox = SandboxRoot.from_path(tmp_path)
    rows = {r.rel_path: r for r in scan_recent_runs(sandbox)}

    assert rows["rs/run"].status == "failed"
    assert rows["rs/run"].mode == "local"
    # Unfinished with no live authority is `incomplete` (O-4), which is a
    # verdict rather than a liveness claim -- it says the run did not finish,
    # not that anyone is working on it. `unknown` now means only "no run of
    # ours lives here", which is what `test_a_manifest_alone_...` pins.
    assert rows["rl/run"].status == "incomplete"
    assert rows["rl/run"].mode == "slurm"


def test_a_manifest_alone_no_longer_carries_a_status(tmp_path: Path) -> None:
    """§4.2, pinned on the surface that regressed.

    The fixture this file used to share wrote exactly this and nothing else,
    and the test above read ``failed`` back out of it. It is a cache of what a
    run reported, never rewritten when the tree beneath it changes, so an
    output whose only evidence is that file now reads ``unknown`` -- and this
    test fails if any reader is reintroduced.
    """
    out = _make_run(tmp_path, "manifest-only")
    progress = out / "progress"
    progress.mkdir(exist_ok=True)
    (progress / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "execution_mode": "local",
                "is_complete": True,
                "completed": 5,
                "failed": 2,
                "total_images": 5,
            }
        ),
        encoding="utf-8",
    )

    rows = {r.rel_path: r for r in scan_recent_runs(SandboxRoot.from_path(tmp_path))}

    assert rows["manifest-only"].status == "unknown"
    assert rows["manifest-only"].mode == "unknown"


def test_scan_has_dashboard_flag_reflects_filesystem(tmp_path: Path) -> None:
    _make_run(tmp_path, "with_dash", has_dashboard=True)
    _make_run(tmp_path, "no_dash", has_dashboard=False)
    sandbox = SandboxRoot.from_path(tmp_path)
    by_name = {r.rel_path: r for r in scan_recent_runs(sandbox)}
    assert by_name["with_dash"].has_dashboard is True
    assert by_name["no_dash"].has_dashboard is False


def test_scan_sorts_newest_first(tmp_path: Path) -> None:
    _make_run(tmp_path, "old", mtime_offset_seconds=-3600)
    _make_run(tmp_path, "new", mtime_offset_seconds=0)
    sandbox = SandboxRoot.from_path(tmp_path)
    rows = scan_recent_runs(sandbox)
    assert [r.rel_path for r in rows] == ["new", "old"]


def test_scan_rehydrates_supplied_registry(tmp_path: Path) -> None:
    _make_run(tmp_path, "x")
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    rows = scan_recent_runs(sandbox, registry=reg)
    assert isinstance(rows[0], RecentRunRow)
    assert reg.get("x") is not None


def test_registry_revision_redraw_does_not_rescan_sandbox(
    tmp_path: Path,
) -> None:
    _make_run(tmp_path, "x")
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()

    with mock.patch.object(
        registry,
        "rehydrate_from_sandbox",
        wraps=registry.rehydrate_from_sandbox,
    ) as rehydrate:
        scan_recent_runs(sandbox, registry=registry)
        scan_recent_runs(sandbox, registry=registry)

    rehydrate.assert_called_once()


def test_scan_with_no_registry_does_not_persist(tmp_path: Path) -> None:
    _make_run(tmp_path, "x")
    sandbox = SandboxRoot.from_path(tmp_path)
    # Without a registry, the function still returns rows but no shared
    # state is left behind for callers.
    rows1 = scan_recent_runs(sandbox)
    rows2 = scan_recent_runs(sandbox)
    assert {r.rel_path for r in rows1} == {r.rel_path for r in rows2}


def test_scan_skips_unreadable_dir(tmp_path: Path) -> None:
    """An OS-level stat failure on a single dir doesn't kill the scan."""
    _make_run(tmp_path, "ok")
    sandbox = SandboxRoot.from_path(tmp_path)
    bad = tmp_path / "bad"
    bad.mkdir()
    from phenotypic._gui.shell import _runs_registry

    real_classify = _runs_registry.classify

    def classify_with_unreadable_entry(path: Path):
        if path == bad:
            raise PermissionError("simulated unreadable run directory")
        return real_classify(path)

    with mock.patch.object(
        _runs_registry,
        "classify",
        side_effect=classify_with_unreadable_entry,
    ):
        rows = scan_recent_runs(sandbox)

    assert {row.rel_path for row in rows} == {"ok"}


def test_scan_returns_empty_for_empty_sandbox(tmp_path: Path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    assert scan_recent_runs(sandbox) == []


@pytest.mark.skipif(
    os.name == "nt",
    reason="rehydrate-with-depth check uses POSIX directory layout",
)
def test_rehydrate_respects_max_depth(tmp_path: Path) -> None:
    nested = tmp_path / "level1" / "level2" / "level3"
    _make_run(nested, "deep")
    sandbox = SandboxRoot.from_path(tmp_path)
    # depth=1 means root + immediate children only — too shallow.
    rows_shallow = scan_recent_runs(sandbox, max_depth=1)
    assert rows_shallow == []
    # depth=4 reaches it.
    rows_deep = scan_recent_runs(sandbox, max_depth=4)
    assert any(r.rel_path.endswith("deep") for r in rows_deep)


def test_scan_ignores_backup_artifacts_at_every_depth_but_keeps_nested_run(
    tmp_path: Path,
) -> None:
    """Recognized backup artifacts are never independent historical runs."""
    outer = _make_run(tmp_path, "run")
    _make_run(outer, "_legacy_metadata_backup")
    _make_run(outer, "nested_run")
    _make_run(tmp_path, "_legacy_experiment_backup")
    nested_container = tmp_path / "container" / "nested"
    _make_run(nested_container, "copied-output-backup")
    _make_run(nested_container, "copied-output.backup")

    rows = scan_recent_runs(SandboxRoot.from_path(tmp_path), max_depth=4)

    assert {row.rel_path for row in rows} == {
        "run",
        "run/nested_run",
    }


def test_scan_prunes_backup_tree_with_invalid_owner_record(
    tmp_path: Path,
) -> None:
    """A corrupt owner artifact cannot turn a backup into a current run."""
    container = tmp_path / "container"
    owner = (
        container
        / ".phenotypic"
        / "progress"
        / "gui_launch_owner.json"
    )
    owner.parent.mkdir(parents=True)
    owner.write_text("{broken", encoding="utf-8")
    _make_run(container, "_legacy_experiment_backup")

    rows = scan_recent_runs(SandboxRoot.from_path(tmp_path), max_depth=4)

    assert rows == []


@pytest.mark.parametrize(
    "name",
    (
        "root-level-backup",
        "root_level_backup",
        "root-level.backup",
        "_legacy_experiment_backup",
    ),
)
def test_scan_excludes_root_level_backup_suffixes(
    tmp_path: Path,
    name: str,
) -> None:
    """All reserved root-level backup suffixes require a valid owner."""
    _make_run(tmp_path, name)

    rows = scan_recent_runs(SandboxRoot.from_path(tmp_path), max_depth=2)

    assert rows == []


def test_scan_keeps_backup_named_run_with_valid_generation_owner(
    tmp_path: Path,
) -> None:
    """A valid generation owner wins over the directory-name heuristic."""
    output = tmp_path / "intentional-backup"
    output.mkdir()
    registry = RunRegistry()
    owned = registry.allocate(
        mode="local",
        output_dir=output,
        rel_path=output.name,
        command_digest="current-generation",
        status="complete",
    )

    rows = scan_recent_runs(SandboxRoot.from_path(tmp_path), max_depth=2)

    assert owned.generation is not None
    assert [row.rel_path for row in rows] == ["intentional-backup"]


def test_scan_excludes_backup_shaped_sandbox_output_root(
    tmp_path: Path,
) -> None:
    """Canonicalizing ``deliverables`` cannot reintroduce a backup as ``.``."""
    output = _make_run(tmp_path, "sandbox-backup")

    rows = scan_recent_runs(SandboxRoot.from_path(output), max_depth=2)

    assert rows == []


def test_scan_keeps_owned_backup_shaped_sandbox_output_root(
    tmp_path: Path,
) -> None:
    """A valid depth-zero generation owner still overrides its backup name."""
    output = _make_run(tmp_path, "sandbox-backup")
    registry = RunRegistry()
    owned = registry.allocate(
        mode="local",
        output_dir=output,
        rel_path=".",
        run_id=".",
        command_digest="owned-root",
        status="complete",
    )

    rows = scan_recent_runs(SandboxRoot.from_path(output), max_depth=2)

    assert owned.generation is not None
    assert [row.rel_path for row in rows] == ["."]


def test_scan_prunes_private_backup_when_sandbox_root_is_output(
    tmp_path: Path,
) -> None:
    """The sandbox root participates in output ancestry classification."""
    output = _make_run(tmp_path, "run")
    _make_run(output, "_legacy_metadata_backup")

    rows = scan_recent_runs(SandboxRoot.from_path(output), max_depth=4)

    assert [row.rel_path for row in rows] == ["."]
