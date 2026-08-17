"""Integration tests for ``run_console._recent_runs.scan_recent_runs``.

Pre-populate a sandbox with multiple CLI-output dirs (varying status,
mode, recency), then verify:

    * The scanner returns one row per output dir.
    * Status / mode / has_dashboard reflect the manifest + filesystem.
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

from phenotypic.gui._config import DELIVERABLES_DIRNAME
from phenotypic.gui.run_console._recent_runs import (
    RecentRunRow,
    scan_recent_runs,
)
from phenotypic.gui.shell._runs_registry import RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot


def _make_run(
    root: Path,
    name: str,
    *,
    is_complete: bool = True,
    failed: int = 0,
    completed: int = 5,
    total: int = 5,
    has_dashboard: bool = False,
    execution_mode: str = "local",
    mtime_offset_seconds: float = 0.0,
) -> Path:
    """Build a fake CLI-output directory under ``root``."""
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    # User-facing deliverables live under ``out/deliverables/``; ``results/``
    # and ``progress/`` stay at the run root.
    deliverables = out / DELIVERABLES_DIRNAME
    deliverables.mkdir(exist_ok=True)
    (deliverables / "master_measurements.parquet").write_bytes(b"")
    (out / "results").mkdir(exist_ok=True)
    progress = out / "progress"
    progress.mkdir(exist_ok=True)
    (progress / "manifest.json").write_text(json.dumps({
        "version": 1,
        "execution_mode": execution_mode,
        "is_complete": is_complete,
        "completed": completed,
        "failed": failed,
        "total_images": total,
    }))
    if has_dashboard:
        (deliverables / "dashboard.html").write_text("<html/>")
    if mtime_offset_seconds:
        new_mtime = time.time() + mtime_offset_seconds
        os.utime(out, (new_mtime, new_mtime))
    return out


def test_scan_returns_one_row_per_output(tmp_path: Path) -> None:
    _make_run(tmp_path, "run_a", has_dashboard=True)
    _make_run(tmp_path, "run_b")
    sandbox = SandboxRoot.from_path(tmp_path)
    rows = scan_recent_runs(sandbox)
    rel_paths = {r.rel_path for r in rows}
    assert rel_paths == {"run_a", "run_b"}


def test_scan_rows_carry_status_and_mode(tmp_path: Path) -> None:
    _make_run(tmp_path, "rs", failed=2, total=5, completed=5)
    _make_run(
        tmp_path, "rl",
        is_complete=False, completed=3, total=10,
        execution_mode="slurm",
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    rows = {r.rel_path: r for r in scan_recent_runs(sandbox)}
    assert rows["rs"].status == "failed"
    assert rows["rs"].mode == "local"
    assert rows["rl"].status == "unknown"
    assert rows["rl"].mode == "slurm"


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
    # The scan implementation moved to phenotypic._services.runs, so ``classify``
    # resolves in that module's globals. Patching gui.shell._runs_registry (now a
    # re-export shim) would no-op and the unreadable dir would never be simulated.
    from phenotypic._services import runs as _runs_registry

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
