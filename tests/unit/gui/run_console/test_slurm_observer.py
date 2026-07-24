"""Deterministic tests for the Dash-free SLURM lifecycle observer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    initialize_slurm_lifecycle,
    lifecycle_state_path,
)
from phenotypic.gui.run_console._slurm_observer import (
    IncrementalLogReader,
    SchedulerQueryResult,
    SlurmLifecycleObserver,
)
from phenotypic.gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.sdk_ import (
    atomic_write_json,
    job_metadata_path,
    resolve_manifest_json_path,
    run_completion_marker_path,
)


class FakeScheduler:
    """Mutable scheduler state used to model controller transitions."""

    def __init__(
        self,
        states: Mapping[str, str] | None = None,
        *,
        available: bool = True,
    ) -> None:
        self.states = dict(states or {})
        self.available = available
        self.queries: list[tuple[str, ...]] = []

    def query(self, job_ids: Sequence[str]) -> SchedulerQueryResult:
        self.queries.append(tuple(job_ids))
        return SchedulerQueryResult(
            states={
                job_id: self.states[job_id]
                for job_id in job_ids
                if job_id in self.states
            },
            available=self.available,
            detail=None if self.available else "fake scheduler unavailable",
        )


def _registered_slurm_run(tmp_path: Path) -> tuple[RunRegistry, RunRecord]:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    registry = RunRegistry()
    record = RunRecord(
        run_id="out",
        generation=__import__("uuid").uuid4(),
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        status="submitting",
    )
    registry.register(record)
    initialize_slurm_lifecycle(
        output_dir,
        generation=record.generation.hex,
        mode="ordinary",
    )
    return registry, record


def _write_jobs(
    record: RunRecord, jobs: Mapping[str, tuple[str, str]]
) -> None:
    assert record.generation is not None
    metadata_path = job_metadata_path(record.output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slurm_generation": record.generation.hex,
        "chunk_job_ids": {},
        "slurm_job_ids": {
            token: {
                "job_id": job_id,
                "role": role,
                "generation": record.generation.hex,
            }
            for token, (job_id, role) in jobs.items()
        },
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    for token, (job_id, role) in jobs.items():
        append_lifecycle_entry(
            record.output_dir,
            generation=record.generation.hex,
            token=token,
            role=role,
            status="submitted",
            job_id=job_id,
        )


def test_controller_only_submission_is_queued(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"controller-initial": ("101", "controller-initial")})
    scheduler = FakeScheduler({"101": "PENDING"})

    assert SlurmLifecycleObserver(registry, scheduler).observe_once() == 1

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "queued"
    assert updated.primary_scheduler_id == "101"


def test_unchanged_observation_does_not_bump_registry_revision(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"controller-initial": ("111", "controller-initial")})
    observer = SlurmLifecycleObserver(
        registry, FakeScheduler({"111": "PENDING"})
    )

    assert observer.observe_once() == 1
    revision = registry.revision
    assert observer.observe_once() == 0
    assert registry.revision == revision


def test_one_completed_job_does_not_hide_concurrent_running_job(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(
        record,
        {
            "chunk-0": ("201", "chunk"),
            "finalizer": ("202", "finalizer"),
        },
    )
    scheduler = FakeScheduler({"201": "FAILED", "202": "RUNNING"})

    SlurmLifecycleObserver(registry, scheduler).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "running"


def test_one_failed_job_does_not_terminalize_unresolved_peer(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(
        record,
        {
            "chunk-0": ("211", "chunk"),
            "finalizer": ("212", "finalizer"),
        },
    )

    SlurmLifecycleObserver(
        registry, FakeScheduler({"211": "FAILED"})
    ).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"


def test_scheduler_unavailable_is_unknown_not_failed(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"chunk-0": ("301", "chunk")})

    SlurmLifecycleObserver(
        registry, FakeScheduler(available=False)
    ).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"
    assert "unavailable" in (updated.status_detail or "")


def test_cancellation_waits_for_every_recovered_id(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(
        record,
        {
            "chunk-0": ("401", "chunk"),
            "dispatcher-1": ("402", "dispatcher"),
        },
    )
    state = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    state["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), state)
    scheduler = FakeScheduler({"401": "CANCELLED", "402": "RUNNING"})
    observer = SlurmLifecycleObserver(registry, scheduler)

    observer.observe_once()
    assert registry.get(record.run_id).status == "cancelling"  # type: ignore[union-attr]

    scheduler.states["402"] = "CANCELLED"
    observer.observe_once()
    assert registry.get(record.run_id).status == "cancelled"  # type: ignore[union-attr]


def test_completed_jobs_enter_grace_before_failed(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"chunk-0": ("501", "chunk")})
    now = [10.0]
    observer = SlurmLifecycleObserver(
        registry,
        FakeScheduler({"501": "COMPLETED"}),
        reconciliation_grace_seconds=5.0,
        monotonic=lambda: now[0],
    )

    observer.observe_once()
    assert registry.get(record.run_id).status == "reconciling"  # type: ignore[union-attr]

    now[0] = 16.0
    observer.observe_once()
    assert registry.get(record.run_id).status == "failed"  # type: ignore[union-attr]


def test_generation_marker_manifest_and_finalizer_complete_run(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "chunk-0": ("601", "chunk"),
            "finalizer": ("602", "finalizer"),
        },
    )
    atomic_write_json(
        run_completion_marker_path(record.output_dir),
        {
            "generation": record.generation.hex,
            "status": "complete",
            "finalizer_succeeded": True,
        },
    )
    atomic_write_json(
        resolve_manifest_json_path(record.output_dir),
        {
            "is_complete": True,
            "failed": 0,
            "completed": 3,
            "total_images": 3,
        },
    )

    SlurmLifecycleObserver(
        registry,
        FakeScheduler({"601": "COMPLETED", "602": "COMPLETED"}),
    ).observe_once()

    assert registry.get(record.run_id).status == "complete"  # type: ignore[union-attr]


def test_incremental_logs_handle_multiple_sources_and_rotation(
    tmp_path: Path,
) -> None:
    gui_log = tmp_path / "submitter.log"
    scheduler_log = tmp_path / "scheduler.log"
    gui_log.write_text("submitted\n", encoding="utf-8")
    scheduler_log.write_text("queued\n", encoding="utf-8")
    reader = IncrementalLogReader(byte_budget=64, line_budget=4)

    first = reader.read({"GUI submitter": gui_log, "SLURM": scheduler_log})
    assert "== GUI submitter ==" in first.text
    assert "submitted" in first.text
    assert "== SLURM ==" in first.text
    assert "queued" in first.text
    assert first.bytes_read <= 64
    assert first.lines_read <= 4
    assert reader.read({"GUI submitter": gui_log, "SLURM": scheduler_log}).text == ""

    gui_log.write_text("rotated\n", encoding="utf-8")
    rotated = reader.read({"GUI submitter": gui_log})
    assert gui_log in rotated.reset_paths
    assert "cursor reset" in rotated.text
    assert "rotated" in rotated.text
