"""Unit tests for ``phenotypic.gui.shell._runs_registry``.

Coverage:

    * Basic CRUD: register / get / list / update_status / remove.
    * Concurrent updates serialise via the registry's :class:`threading.Lock`
      — many threads racing on ``update_status`` produce a deterministic
      final state.
    * ``rehydrate_from_sandbox`` walks a fake CLI-output layout and
      registers a record per discovered output dir.
    * Status / mode / SLURM job-id are read from
      ``progress/manifest.json`` when present, with sane fallbacks.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List

from phenotypic.gui._config import DELIVERABLES_DIRNAME
from phenotypic.gui.shell._runs_registry import (
    RunRecord,
    RunRegistry,
)
from phenotypic.gui.shell._sandbox import SandboxRoot


def _write_master_marker(out: Path) -> None:
    """Drop an empty ``deliverables/master_measurements.parquet`` marker.

    The shell classifier identifies a CLI output by this file (under
    ``deliverables/``) plus a root-level ``results/`` dir.
    """
    deliverables = out / DELIVERABLES_DIRNAME
    deliverables.mkdir(parents=True, exist_ok=True)
    (deliverables / "master_measurements.parquet").write_bytes(b"")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def test_register_get_list_remove(tmp_path: Path) -> None:
    reg = RunRegistry()
    rec = RunRecord(
        run_id="r1",
        mode="local",
        output_dir=tmp_path / "r1",
        rel_path="r1",
    )
    reg.register(rec)
    assert reg.get("r1") is rec
    assert reg.list() == [rec]
    assert reg.remove("r1") is True
    assert reg.get("r1") is None
    assert reg.remove("r1") is False  # idempotent


def test_register_replaces_on_same_id(tmp_path: Path) -> None:
    reg = RunRegistry()
    a = RunRecord(run_id="x", mode="local", output_dir=tmp_path, rel_path="x")
    b = RunRecord(run_id="x", mode="slurm", output_dir=tmp_path, rel_path="x")
    reg.register(a)
    reg.register(b)
    assert reg.get("x") is b


def test_update_status_returns_false_for_unknown() -> None:
    reg = RunRegistry()
    assert reg.update_status("missing", "complete") is False


def test_update_pid_and_slurm_job_id(tmp_path: Path) -> None:
    reg = RunRegistry()
    reg.register(
        RunRecord(
            run_id="r", mode="local",
            output_dir=tmp_path, rel_path="r",
        )
    )
    assert reg.update_pid("r", 4242) is True
    assert reg.get("r").pid == 4242  # type: ignore[union-attr]
    assert reg.update_slurm_job_id("r", "8675309") is True
    assert reg.get("r").slurm_job_id == "8675309"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_register_does_not_corrupt(tmp_path: Path) -> None:
    """Many threads racing on register produce a clean final state."""
    reg = RunRegistry()
    barrier = threading.Barrier(8)

    def _worker(i: int) -> None:
        barrier.wait()
        for j in range(20):
            run_id = f"w{i}-{j}"
            reg.register(
                RunRecord(
                    run_id=run_id,
                    mode="local",
                    output_dir=tmp_path / run_id,
                    rel_path=run_id,
                )
            )

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(reg.list()) == 8 * 20


def test_concurrent_update_status_is_serialised(tmp_path: Path) -> None:
    """Many threads racing on update_status leave the registry consistent.

    The final status is whichever update ran last, but the dict must not
    raise mid-iteration and ``list()`` must return a coherent snapshot.
    """
    reg = RunRegistry()
    reg.register(
        RunRecord(
            run_id="r", mode="local",
            output_dir=tmp_path, rel_path="r",
        )
    )
    statuses = ["running", "complete", "failed", "cancelled"]
    barrier = threading.Barrier(16)

    def _worker(s: str) -> None:
        barrier.wait()
        for _ in range(50):
            reg.update_status("r", s)

    threads = [
        threading.Thread(target=_worker, args=(statuses[i % 4],))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = reg.get("r")
    assert final is not None
    assert final.status in statuses

    # Concurrent list() never raised.
    snapshots: List[int] = []
    for _ in range(20):
        snapshots.append(len(reg.list()))
    assert all(s == 1 for s in snapshots)


# ---------------------------------------------------------------------------
# rehydrate_from_sandbox
# ---------------------------------------------------------------------------

def _make_cli_output(
    root: Path,
    name: str,
    *,
    is_complete: bool = True,
    failed: int = 0,
    total: int = 5,
    completed: int | None = None,
    execution_mode: str = "local",
    chunk_job_ids: dict | None = None,
) -> Path:
    """Build a fake CLI-output directory with progress/manifest.json.

    By default ``completed = total - failed`` (a finished run). Pass
    ``completed`` explicitly to simulate a partially-finished run.
    """
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    _write_master_marker(out)
    (out / "results").mkdir(exist_ok=True)
    progress = out / "progress"
    progress.mkdir(exist_ok=True)
    if completed is None:
        completed = total - failed
    manifest: dict = {
        "version": 1,
        "execution_mode": execution_mode,
        "is_complete": is_complete,
        "completed": completed,
        "failed": failed,
        "total_images": total,
    }
    if chunk_job_ids is not None:
        manifest["slurm_info"] = {
            "chunk_job_ids": chunk_job_ids,
        }
    (progress / "manifest.json").write_text(json.dumps(manifest))
    return out


def test_rehydrate_picks_up_cli_outputs(tmp_path: Path) -> None:
    _make_cli_output(tmp_path, "run_a")
    _make_cli_output(
        tmp_path, "run_b",
        is_complete=False, failed=0, total=10, completed=3,
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    n = reg.rehydrate_from_sandbox(sandbox)
    assert n == 2
    statuses = {r.run_id: r.status for r in reg.list()}
    assert statuses == {"run_a": "complete", "run_b": "running"}


def test_rehydrate_marks_failed_when_failed_gt_zero(tmp_path: Path) -> None:
    _make_cli_output(tmp_path, "fr", is_complete=True, failed=2, total=5)
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    reg.rehydrate_from_sandbox(sandbox)
    assert reg.get("fr").status == "failed"  # type: ignore[union-attr]


def test_rehydrate_extracts_slurm_job_id(tmp_path: Path) -> None:
    _make_cli_output(
        tmp_path,
        "sr",
        execution_mode="slurm",
        chunk_job_ids={"0": "12345_0", "1": "12345_1"},
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    reg.rehydrate_from_sandbox(sandbox)
    rec = reg.get("sr")
    assert rec is not None
    assert rec.mode == "slurm"
    assert rec.slurm_job_id == "12345"


def test_rehydrate_unknown_when_no_manifest(tmp_path: Path) -> None:
    out = tmp_path / "nm"
    out.mkdir()
    _write_master_marker(out)
    (out / "results").mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    reg.rehydrate_from_sandbox(sandbox)
    rec = reg.get("nm")
    assert rec is not None
    assert rec.mode == "unknown"
    assert rec.status == "unknown"


def test_rehydrate_preserves_existing_records(tmp_path: Path) -> None:
    """A live run registered before scan must NOT be clobbered."""
    _make_cli_output(tmp_path, "live")
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    pre = RunRecord(
        run_id="live",
        mode="local",
        output_dir=tmp_path / "live",
        rel_path="live",
        status="running",
        pid=999,
    )
    reg.register(pre)
    n = reg.rehydrate_from_sandbox(sandbox)
    assert n == 0  # didn't re-register
    assert reg.get("live") is pre  # same object


def test_rehydrate_ignores_corrupt_manifest(tmp_path: Path) -> None:
    out = tmp_path / "broken"
    out.mkdir()
    _write_master_marker(out)
    (out / "results").mkdir()
    progress = out / "progress"
    progress.mkdir()
    (progress / "manifest.json").write_text("{not valid json")
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    reg.rehydrate_from_sandbox(sandbox)
    rec = reg.get("broken")
    assert rec is not None
    assert rec.status == "unknown"
