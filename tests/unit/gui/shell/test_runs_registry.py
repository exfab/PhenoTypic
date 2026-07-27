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
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

import pytest

import phenotypic.gui.shell._runs_registry as runs_registry_module
from phenotypic._cli._cli_update_state import append_event
from phenotypic.gui._config import DELIVERABLES_DIRNAME
from phenotypic.gui.shell._runs_registry import (
    RunRecord,
    RunRegistry,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.sdk_ import (
    event_log_path,
    manifest_json_path,
    processing_state_path,
    run_completion_marker_path,
)


def _write_master_marker(out: Path) -> None:
    """Drop an empty ``deliverables/master_measurements.parquet`` marker.

    The shell classifier identifies a CLI output by this file (under
    ``deliverables/``) plus a root-level ``results/`` dir.
    """
    deliverables = out / DELIVERABLES_DIRNAME
    deliverables.mkdir(parents=True, exist_ok=True)
    (deliverables / "master_measurements.parquet").write_bytes(b"")


def _write_local_terminal_manifest(
    output: Path,
    *,
    start_time: float,
    is_complete: bool = True,
    completed: int = 1,
    failed: int = 0,
    total: int = 1,
    execution_mode: str = "local",
) -> None:
    """Write canonical local publication evidence for lifecycle tests."""
    path = manifest_json_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "execution_mode": execution_mode,
                "start_time": datetime.fromtimestamp(
                    start_time
                ).isoformat(timespec="milliseconds"),
                "is_complete": is_complete,
                "completed": completed,
                "failed": failed,
                "total_images": total,
            }
        ),
        encoding="utf-8",
    )


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


def test_allocate_persists_durable_generation_owner(tmp_path: Path) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="sha256:abc",
        status="running",
    )

    owner_path = (
        output / ".phenotypic" / "progress" / "gui_launch_owner.json"
    )
    payload = json.loads(owner_path.read_text(encoding="utf-8"))
    assert payload["generation"] == str(record.generation)
    assert payload["run_id"] == "run"
    assert payload["rel_path"] == "run"
    assert payload["command_digest"] == "sha256:abc"
    assert payload["lifecycle_epoch"] == str(record.generation)
    assert payload["status"] == "running"
    assert reg.revision == 1


def test_allocate_rejects_second_nonterminal_generation(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    first = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="one",
        status="running",
    )

    try:
        with pytest.raises(RuntimeError, match="nonterminal"):
            reg.allocate(
                mode="slurm",
                output_dir=output,
                rel_path="run",
                command_digest="two",
            )
    finally:
        reg.compare_and_set(
            "run",
            first.generation,  # type: ignore[arg-type]
            status="cancelled",
        )


def test_allocate_rejects_durable_owner_from_another_registry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    first_registry = RunRegistry()
    first_registry.allocate(
        mode="slurm",
        output_dir=output,
        rel_path="run",
        command_digest="one",
        status="queued",
    )

    second_registry = RunRegistry()
    with pytest.raises(RuntimeError, match="durable nonterminal"):
        second_registry.allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="two",
        )
    assert second_registry.list() == []


def test_allocate_refuses_invalid_existing_owner(tmp_path: Path) -> None:
    output = tmp_path / "run"
    owner_path = (
        output / ".phenotypic" / "progress" / "gui_launch_owner.json"
    )
    owner_path.parent.mkdir(parents=True)
    owner_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid generation owner"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_two_registries_atomically_compete_for_one_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    registries = (RunRegistry(), RunRegistry())
    barrier = threading.Barrier(2)
    successes: list[RunRecord] = []
    failures: list[BaseException] = []
    result_lock = threading.Lock()

    def _allocate(registry: RunRegistry, digest: str) -> None:
        barrier.wait()
        try:
            record = registry.allocate(
                mode="local",
                output_dir=output,
                rel_path="run",
                command_digest=digest,
                status="running",
            )
            with result_lock:
                successes.append(record)
        except BaseException as exc:
            with result_lock:
                failures.append(exc)

    threads = [
        threading.Thread(target=_allocate, args=(registry, f"digest-{index}"))
        for index, registry in enumerate(registries)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    payload = json.loads(
        (
            output
            / ".phenotypic"
            / "progress"
            / "gui_launch_owner.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["generation"] == str(successes[0].generation)


def _write_processing_inventory(
    output: Path,
    *,
    images: list[str],
) -> None:
    state_path = processing_state_path(output)
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "execution_mode": "local",
                "datasets": {
                    "plate": {
                        "initial_images": images,
                        "completed": [],
                        "failed": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_publication_manifest(
    output: Path,
    *,
    completed: int,
    failed: int,
    total: int,
    is_complete: bool,
) -> None:
    path = manifest_json_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "is_complete": is_complete,
                "completed": completed,
                "failed": failed,
                "total_images": total,
            }
        ),
        encoding="utf-8",
    )


def test_allocate_rejects_nonterminal_non_gui_processing_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_processing_inventory(
        output,
        images=["a.tif", "b.tif"],
    )
    append_event(event_log_path(output), "plate", "a.tif", "started")
    append_event(event_log_path(output), "plate", "a.tif", "completed")
    append_event(event_log_path(output), "plate", "b.tif", "started")
    _write_publication_manifest(
        output,
        completed=1,
        failed=0,
        total=2,
        is_complete=False,
    )

    with pytest.raises(RuntimeError, match="non-GUI processing state"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_allocate_accepts_completed_event_log_with_terminal_publication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_processing_inventory(
        output,
        images=["a.tif", "b.tif"],
    )
    for image in ("a.tif", "b.tif"):
        append_event(event_log_path(output), "plate", image, "started")
        append_event(event_log_path(output), "plate", image, "completed")
    _write_publication_manifest(
        output,
        completed=2,
        failed=0,
        total=2,
        is_complete=True,
    )

    record = RunRegistry().allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
    )
    assert record.generation is not None


def test_allocate_rejects_failed_event_log_even_when_manifest_is_terminal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_processing_inventory(
        output,
        images=["a.tif", "b.tif"],
    )
    append_event(event_log_path(output), "plate", "a.tif", "completed")
    append_event(
        event_log_path(output),
        "plate",
        "b.tif",
        "failed",
        error_msg="segmentation failed",
    )
    _write_publication_manifest(
        output,
        completed=1,
        failed=1,
        total=2,
        is_complete=True,
    )

    with pytest.raises(RuntimeError, match="failed non-GUI processing state"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_allocate_rejects_completed_events_without_publication_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_processing_inventory(output, images=["a.tif"])
    append_event(event_log_path(output), "plate", "a.tif", "completed")

    with pytest.raises(RuntimeError, match="terminal publication evidence"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_allocate_rejects_completed_then_restarted_image_as_active(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_processing_inventory(output, images=["a.tif"])
    append_event(event_log_path(output), "plate", "a.tif", "completed")
    append_event(event_log_path(output), "plate", "a.tif", "started")
    _write_publication_manifest(
        output,
        completed=1,
        failed=0,
        total=1,
        is_complete=True,
    )

    with pytest.raises(RuntimeError, match="unfinished image"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_allocate_rejects_stale_self_consistent_manifest_inventory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_processing_inventory(
        output,
        images=["a.tif", "b.tif"],
    )
    append_event(event_log_path(output), "plate", "a.tif", "completed")
    append_event(event_log_path(output), "plate", "b.tif", "completed")
    _write_publication_manifest(
        output,
        completed=1,
        failed=0,
        total=1,
        is_complete=True,
    )

    with pytest.raises(RuntimeError, match="inventory does not match"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_allocate_rejects_nonterminal_non_gui_orchestration_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    state_path = (
        output
        / ".phenotypic"
        / "progress"
        / "staged_orchestration.json"
    )
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"epoch": "other", "phase": "stage2"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="staged orchestration"):
        RunRegistry().allocate(
            mode="slurm",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


@pytest.mark.parametrize("phase", ["failed", "cancelled"])
def test_allocate_rejects_unsuccessful_terminal_orchestration(
    tmp_path: Path,
    phase: str,
) -> None:
    output = tmp_path / "run"
    state_path = (
        output
        / ".phenotypic"
        / "progress"
        / "staged_orchestration.json"
    )
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"epoch": "old", "phase": phase}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsuccessful"):
        RunRegistry().allocate(
            mode="slurm",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_allocate_rejects_complete_orchestration_with_mismatched_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    progress = output / ".phenotypic" / "progress"
    progress.mkdir(parents=True)
    (progress / "staged_orchestration.json").write_text(
        json.dumps({"epoch": "current", "phase": "complete"}),
        encoding="utf-8",
    )
    (progress / "staged_finalization_complete.json").write_text(
        json.dumps({"epoch": "other"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no matching successful"):
        RunRegistry().allocate(
            mode="slurm",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_allocate_accepts_successful_matching_orchestration(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    progress = output / ".phenotypic" / "progress"
    progress.mkdir(parents=True)
    (progress / "staged_orchestration.json").write_text(
        json.dumps({"epoch": "old", "phase": "complete"}),
        encoding="utf-8",
    )
    (progress / "staged_finalization_complete.json").write_text(
        json.dumps({"epoch": "old"}),
        encoding="utf-8",
    )
    record = RunRegistry().allocate(
        mode="slurm",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
    )
    assert record.generation is not None


def test_allocate_replaces_terminal_generation(tmp_path: Path) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    first = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="one",
        status="complete",
    )
    second = reg.allocate(
        mode="slurm",
        output_dir=output,
        rel_path="run",
        command_digest="two",
    )
    assert second.generation != first.generation
    assert reg.get("run") is second


def test_compare_and_set_rejects_stale_generation_without_revision_bump(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    revision = reg.revision
    assert (
        reg.compare_and_set(
            "run",
            uuid4(),
            status="failed",
            returncode=9,
        )
        is False
    )
    assert reg.revision == revision
    assert record.status == "running"
    assert record.returncode is None


def test_compare_and_set_updates_generalized_fields_and_aliases(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="slurm",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
    )
    submitted = datetime.now(timezone.utc)
    log_paths = (tmp_path / "submit.log", tmp_path / "slurm.log")
    assert record.generation is not None
    assert reg.compare_and_set(
        "run",
        record.generation,
        expected_statuses={"submitting"},
        expected_record_revision=0,
        status="queued",
        scheduler_ids=("22", "11", "22"),
        primary_scheduler_id="11",
        log_paths=log_paths,
        submitted_at=submitted,
        status_detail="waiting for resources",
    )
    updated = reg.get("run")
    assert updated is not None
    assert updated is not record
    assert record.status == "submitting"
    assert record.record_revision == 0
    assert updated.status == "queued"
    assert updated.scheduler_ids == ("22", "11")
    assert updated.primary_scheduler_id == "11"
    assert updated.slurm_job_id == "11"
    assert updated.log_paths == log_paths
    assert updated.log_path == log_paths[0]
    assert updated.submitted_at == submitted
    assert updated.record_revision == 1
    assert reg.revision == 2

    payload = json.loads(
        (
            updated.output_dir
            / ".phenotypic"
            / "progress"
            / "gui_launch_owner.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["scheduler_ids"] == ["22", "11"]
    assert payload["primary_scheduler_id"] == "11"
    assert payload["record_revision"] == 1


def test_compare_and_set_write_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    owner_path = (
        record.output_dir
        / ".phenotypic"
        / "progress"
        / "gui_launch_owner.json"
    )
    owner_before = owner_path.read_bytes()
    registry_revision = reg.revision

    def _fail_write(*_args, **_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        runs_registry_module,
        "atomic_write_json",
        _fail_write,
    )
    with pytest.raises(OSError, match="disk full"):
        reg.compare_and_set(
            "run",
            record.generation,
            status="failed",
            returncode=7,
        )

    assert reg.get("run") is record
    assert record.status == "running"
    assert record.returncode is None
    assert record.record_revision == 0
    assert reg.revision == registry_revision
    assert owner_path.read_bytes() == owner_before


def test_compare_and_set_rejects_durable_revision_changed_by_other_registry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    first_registry = RunRegistry()
    first = first_registry.allocate(
        mode="slurm",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert first.generation is not None

    stale_registry = RunRegistry()
    stale_registry.rehydrate_from_sandbox(
        SandboxRoot.from_path(tmp_path)
    )
    stale = stale_registry.get("run")
    assert stale is not None
    assert stale.record_revision == 0

    assert first_registry.compare_and_set(
        "run",
        first.generation,
        status="complete",
    )
    assert (
        stale_registry.compare_and_set(
            "run",
            first.generation,
            status="failed",
        )
        is False
    )
    assert stale_registry.get("run") is stale
    assert stale.status == "running"
    payload = json.loads(
        (
            output
            / ".phenotypic"
            / "progress"
            / "gui_launch_owner.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "complete"
    assert payload["record_revision"] == 1


def test_compare_and_set_honors_expected_record_revision(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
    )
    assert record.generation is not None
    assert (
        reg.compare_and_set(
            "run",
            record.generation,
            expected_record_revision=7,
            status="running",
        )
        is False
    )
    assert record.status == "submitting"
    assert record.record_revision == 0


def test_observe_local_exit_maps_nonzero_and_rejects_stale_generation(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    assert reg.observe_local_exit("run", uuid4(), 0) is False
    assert record.status == "running"
    assert reg.observe_local_exit("run", record.generation, 3) is True
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert updated.returncode == 3
    assert updated.terminal_at is not None
    assert updated.status_detail == "local process exited with status 3"


def test_observe_validate_zero_exit_completes_without_publication(
    tmp_path: Path,
) -> None:
    """A dry-run validates configuration and intentionally publishes no output."""
    reg = RunRegistry()
    record = reg.allocate(
        mode="validate",
        output_dir=tmp_path / "validate",
        rel_path="validate",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None

    assert reg.observe_local_exit("validate", record.generation, 0)
    updated = reg.get("validate")
    assert updated is not None
    assert updated.status == "complete"
    assert updated.returncode == 0
    assert updated.status_detail is None


def test_observe_local_zero_exit_fails_without_canonical_manifest(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert updated.returncode == 0
    assert "no canonical terminal publication evidence" in (
        updated.status_detail or ""
    )


def test_observe_local_zero_exit_ignores_legacy_shadow_manifest(
    tmp_path: Path,
) -> None:
    """A legacy manifest cannot shadow missing current-generation evidence."""
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    legacy_path = output / "progress" / "manifest.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "execution_mode": "local",
                "start_time": datetime.fromtimestamp(
                    record.started_at + 1.0
                ).isoformat(timespec="milliseconds"),
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            }
        ),
        encoding="utf-8",
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "no canonical terminal publication evidence" in (
        updated.status_detail or ""
    )


def test_observe_local_zero_exit_rejects_incomplete_manifest(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
        is_complete=False,
        completed=0,
        total=1,
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "publication is incomplete" in (updated.status_detail or "")


def test_observe_local_zero_exit_rejects_preexisting_complete_manifest(
    tmp_path: Path,
) -> None:
    """A prior run's complete manifest cannot terminalize a new generation."""
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at - 60.0,
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "predates the current launch generation" in (
        updated.status_detail or ""
    )


def test_observe_local_zero_exit_rejects_mismatched_completion_marker(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
    )
    marker = run_completion_marker_path(output)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "generation": str(uuid4()),
                "status": "complete",
                "finalizer_succeeded": True,
            }
        ),
        encoding="utf-8",
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "different launch generation" in (updated.status_detail or "")


def test_observe_local_zero_exit_accepts_current_canonical_manifest(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "complete"
    assert updated.returncode == 0
    assert updated.status_detail is None


def test_observe_local_zero_exit_accepts_matching_completion_marker(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
    )
    marker = run_completion_marker_path(output)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "generation": str(record.generation),
                "status": "complete",
                "finalizer_succeeded": True,
            }
        ),
        encoding="utf-8",
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "complete"
    assert updated.returncode == 0
    assert updated.status_detail is None


def test_stale_local_exit_cannot_terminalize_replacement_generation(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    first = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="first",
        status="running",
    )
    assert first.generation is not None
    assert reg.compare_and_set(
        "run",
        first.generation,
        expected_statuses={"running"},
        status="failed",
    )
    replacement = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="replacement",
        status="running",
    )
    assert replacement.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=replacement.started_at + 1.0,
    )

    assert reg.observe_local_exit("run", first.generation, 0) is False
    current = reg.get("run")
    assert current is not None
    assert current.generation == replacement.generation
    assert current.status == "running"

    assert reg.observe_local_exit("run", replacement.generation, 0)
    completed = reg.get("run")
    assert completed is not None
    assert completed.generation == replacement.generation
    assert completed.status == "complete"


def test_observe_local_exit_preserves_cancelled_status(tmp_path: Path) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="cancelled",
    )
    assert record.generation is not None
    assert reg.observe_local_exit("run", record.generation, -15)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "cancelled"
    assert updated.returncode == -15


def test_observe_local_exit_finishes_cancelling_as_cancelled(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="cancelling",
    )
    assert record.generation is not None
    assert reg.observe_local_exit("run", record.generation, -15)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "cancelled"
    assert updated.status_detail is None


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
    assert statuses == {"run_a": "complete", "run_b": "unknown"}


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


def test_rehydrate_restores_persisted_generation_and_terminal_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "owned"
    source = RunRegistry()
    original = source.allocate(
        mode="local",
        output_dir=output,
        rel_path="owned",
        command_digest="digest",
        status="running",
    )
    assert original.generation is not None
    assert source.compare_and_set(
        "owned",
        original.generation,
        status="complete",
        returncode=0,
        log_paths=(output / ".gui_log" / "stdout.log",),
    )

    restored = RunRegistry()
    count = restored.rehydrate_from_sandbox(
        SandboxRoot.from_path(tmp_path)
    )
    assert count == 1
    record = restored.get("owned")
    assert record is not None
    assert record.generation == original.generation
    assert record.status == "complete"
    assert record.returncode == 0
    assert record.log_path == output / ".gui_log" / "stdout.log"
    assert record.record_revision == 1


def test_rehydrate_downgrades_unobserved_local_liveness(
    tmp_path: Path,
) -> None:
    output = tmp_path / "owned"
    source = RunRegistry()
    original = source.allocate(
        mode="local",
        output_dir=output,
        rel_path="owned",
        command_digest="digest",
        status="running",
    )
    assert original.generation is not None
    assert source.compare_and_set(
        "owned",
        original.generation,
        pid=12345,
    )

    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))
    record = restored.get("owned")
    assert record is not None
    assert record.generation == original.generation
    assert record.status == "unknown"
    assert record.pid is None
    assert "restarted" in (record.status_detail or "")


def test_rehydrate_legacy_manifest_does_not_invent_generation(
    tmp_path: Path,
) -> None:
    _make_cli_output(tmp_path, "legacy")
    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))
    record = restored.get("legacy")
    assert record is not None
    assert record.generation is None
    assert record.status == "complete"


def test_rehydrate_incomplete_legacy_manifest_does_not_invent_liveness(
    tmp_path: Path,
) -> None:
    _make_cli_output(
        tmp_path,
        "legacy",
        is_complete=False,
        completed=3,
        total=10,
    )
    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = restored.get("legacy")
    assert record is not None
    assert record.generation is None
    assert record.status == "unknown"
    assert record.mode == "local"
    assert "no observable nonterminal owner" in (
        record.status_detail or ""
    )


def test_explicit_incomplete_manifest_counts_are_not_terminal(
    tmp_path: Path,
) -> None:
    _make_cli_output(
        tmp_path,
        "legacy",
        is_complete=False,
        completed=10,
        total=10,
    )
    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = restored.get("legacy")
    assert record is not None
    assert record.status == "unknown"


def test_missing_completion_flag_preserves_legacy_count_fallback(
    tmp_path: Path,
) -> None:
    output = _make_cli_output(
        tmp_path,
        "legacy",
        is_complete=False,
        completed=10,
        total=10,
    )
    manifest_path = output / "progress" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["is_complete"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = restored.get("legacy")
    assert record is not None
    assert record.status == "complete"


def test_rehydrate_invalid_owner_does_not_invent_generation(
    tmp_path: Path,
) -> None:
    output = _make_cli_output(tmp_path, "invalid-owner")
    owner_path = (
        output / ".phenotypic" / "progress" / "gui_launch_owner.json"
    )
    owner_path.parent.mkdir(parents=True)
    owner_path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_id": "invalid-owner",
                "rel_path": "invalid-owner",
                "generation": "not-a-uuid",
            }
        ),
        encoding="utf-8",
    )

    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))
    record = restored.get("invalid-owner")
    assert record is not None
    assert record.generation is None
