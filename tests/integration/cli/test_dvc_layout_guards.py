"""DVC layout guards (#281 spec C1, C13, C14; P2 Task 5).

- **C1:** a run writes nothing at the output root outside the three DVC outs.
- **C14:** the outs exist after the command even when every image fails. DVC 3
  (3.67.1, observed 2026-11-13) fails the stage on a missing declared out --
  ``ERROR: failed to reproduce 'run': output 'out/results' does not exist``,
  exit 255 -- and accepts an empty one with a warning.
- **C13:** a ``dvc checkout`` of another revision restores ``.phenotypic/``
  wholesale, so liveness and ownership records restored from it must neither
  block a run nor make the tree read as in flight. Each test here restores a
  record that **claims** work -- an ``active`` lifecycle fence, a ``running``
  GUI owner, a migration lease and an active migration fence -- whose owner is
  in fact gone, which is the case the guard has to get right.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from phenotypic._cli import _cli_staged_orchestration
from phenotypic._cli._cli_migrate import (
    _migration_attempt_lease,
    migration_attempt_lease_path,
    reconcile_interrupted_migration_attempt,
)
from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    initialize_slurm_lifecycle,
    load_slurm_lifecycle,
)
from phenotypic._cli._cli_staged_orchestration import live_slurm_job_ids
from phenotypic.sdk_ import (
    RunExitCode,
    gui_launch_owner_path,
    resolve_run_state,
    run_completion_marker_path,
)
from tests._tiny_run import run_cli

#: The supported DVC outs (spec C1): every path a run writes at the root.
DVC_OUTS = frozenset({"deliverables", "results", ".phenotypic"})


def _root_entries(out: Path) -> set[str]:
    return {p.name for p in out.iterdir()}


# --- C1: only the three outs --------------------------------------------------


def test_full_and_measure_write_only_the_three_outs(
    tiny_inputs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out)
    assert result.exit_code == RunExitCode.COMPLETE, result.output
    assert _root_entries(out) == DVC_OUTS
    result = run_cli(None, out, mode="measure")
    assert result.exit_code == RunExitCode.COMPLETE, result.output
    assert _root_entries(out) == DVC_OUTS


def test_recompile_writes_only_the_three_outs(finished_tiny_run: Path) -> None:
    result = run_cli(None, finished_tiny_run, mode="recompile")
    assert result.exit_code == RunExitCode.COMPLETE, result.output
    assert _root_entries(finished_tiny_run) == DVC_OUTS


# --- C14: the outs exist even when every image fails -------------------------


def test_outs_exist_when_every_image_fails(all_bad_inputs: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run_cli(all_bad_inputs, out)
    # Not a preflight refusal: both images reach the worker and fail there.
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert all((out / name).is_dir() for name in DVC_OUTS)
    assert _root_entries(out) == DVC_OUTS


def test_process_out_root_exists_when_every_image_fails(
    all_bad_inputs: Path, tmp_path: Path
) -> None:
    """A process stage declares the whole ``<out>`` root (how-to page, spec C9)."""
    out = tmp_path / "out"
    result = run_cli(all_bad_inputs, out, mode="process", extra=["--layer", "gray"])
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert out.is_dir()


# --- C13: restored liveness and ownership state fails safe --------------------


def dead_pid() -> int:
    """The pid of a child process that has already exited (and been reaped)."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def write_owner_record(out: Path, *, pid: int, status: str = "running") -> None:
    """A ``gui_launch_owner.json`` in ``RunRegistry._persist_record_locked``'s shape.

    ``_gui/shell/_runs_registry.py`` is the only writer; this mirrors its keys so
    the reader sees what a GUI left behind when it was killed mid-run.
    """
    path = gui_launch_owner_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "version": 1,
        "run_id": "run-restored",
        "generation": "gui-old",
        "mode": "full",
        "output_dir": str(out),
        "rel_path": out.name,
        "status": status,
        "pid": pid,
        "scheduler_ids": [],
        "primary_scheduler_id": None,
        "log_paths": [],
        "submitted_at": None,
        "terminal_at": None,
        "returncode": None,
        "status_detail": None,
        "command_digest": "0" * 64,
        "lifecycle_epoch": None,
        "record_revision": 1,
        "started_at": 1_700_000_000.0,
        "created_at": "2023-11-14T22:13:20+00:00",
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def _restore_active_fence(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``active`` fence whose one ledgered job the scheduler reports finished."""
    initialize_slurm_lifecycle(out, generation="g-old", mode="ordinary")
    append_lifecycle_entry(
        out, generation="g-old", token="t", role="finalizer",
        status="submitted", job_id="77",
    )
    asked: list[str] = []

    def _finished(job_id: str) -> bool:
        asked.append(job_id)
        return False

    def _no_comment_query(**_kwargs):
        raise AssertionError("a clean ledger must not query scheduler comments")

    monkeypatch.setattr(_cli_staged_orchestration, "scheduler_job_is_active", _finished)
    monkeypatch.setattr(_cli_staged_orchestration, "query_scheduler_comments", _no_comment_query)
    assert load_slurm_lifecycle(out)["active"] is True
    assert not live_slurm_job_ids(out)
    assert asked == ["77"], "the guard must ask the scheduler, not trust the fence"


def test_restored_liveness_state_fails_safe(
    finished_tiny_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = finished_tiny_run
    _restore_active_fence(out, monkeypatch)
    write_owner_record(out, pid=dead_pid())
    assert resolve_run_state(out, depth="deep").completion == "complete"
    result = run_cli(None, out, mode="recompile")
    assert result.exit_code == RunExitCode.COMPLETE, result.output
    result = run_cli(out.parent / "in", out)  # a plain local continuation
    assert result.exit_code == RunExitCode.COMPLETE, result.output
    # The continuation retired the dead fence, leaving no close reason that
    # `phenotypic-wait` could read as g-old's publication.
    fence = load_slurm_lifecycle(out)
    assert fence["generation"] == "g-old" and fence["active"] is False
    assert "closed_by" not in fence


def test_restored_gui_owner_with_a_dead_pid_is_not_believed(finished_tiny_run: Path) -> None:
    """On a tree with no proof, only the owner record could read as ``active``."""
    out = finished_tiny_run
    run_completion_marker_path(out).unlink()
    write_owner_record(out, pid=os.getpid())  # control: a live owner is believed
    assert resolve_run_state(out, depth="deep").completion == "active"
    write_owner_record(out, pid=dead_pid())
    assert resolve_run_state(out, depth="deep").completion != "active"


def test_restored_migration_lease_and_fence_do_not_block(finished_tiny_run: Path) -> None:
    """The lease is an ``flock`` on a file, so a restored file is inert.

    A restored ``active`` migration fence whose local owner ran on another host
    is terminalized by the reconcile every migrate runs first, rather than
    blocking the next migration.
    """
    out = finished_tiny_run
    lease = migration_attempt_lease_path(out)
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.write_bytes(b"")
    with _migration_attempt_lease(out):  # timeout=0: raises if it would block
        pass
    run_completion_marker_path(out).unlink()
    from phenotypic._cli._cli_slurm_lifecycle import lifecycle_state_path

    lifecycle_state_path(out).write_text(
        json.dumps({
            "schema_version": 2,
            "generation": "m-old",
            "mode": "migrate",
            "active": True,
            "restart_epoch": 0,
            "owner_kind": "local",
            "owner_pid": dead_pid(),
            "owner_started_at": 1_700_000_000.0,
            "owner_host": f"not-{socket.gethostname()}",
            "control_root": str((out / ".phenotypic").resolve()),
        }),
        encoding="utf-8",
    )
    assert reconcile_interrupted_migration_attempt(out) is True
    assert load_slurm_lifecycle(out)["active"] is False
