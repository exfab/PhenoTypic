"""Deterministic tests for the Dash-free SLURM lifecycle observer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    initialize_slurm_lifecycle,
    lifecycle_state_path,
    read_lifecycle_ledger,
)
from phenotypic._cli._cli_staged_orchestration import (
    orchestration_state_path,
)
from phenotypic.gui.run_console._slurm_observer import (
    IncrementalLogReader,
    SchedulerCommentQueryResult,
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
        comment_matches: Mapping[str, tuple[str, ...]] | None = None,
        comments_available: bool = True,
    ) -> None:
        self.states = dict(states or {})
        self.available = available
        self.comment_matches = dict(comment_matches or {})
        self.comments_available = comments_available
        self.queries: list[tuple[str, ...]] = []
        self.comment_queries: list[tuple[object, tuple[str, ...]]] = []

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

    def find_by_comments(
        self,
        generation: object,
        tokens: Sequence[str],
    ) -> SchedulerCommentQueryResult:
        self.comment_queries.append((generation, tuple(tokens)))
        return SchedulerCommentQueryResult(
            matches={
                token: self.comment_matches[token]
                for token in tokens
                if token in self.comment_matches
            },
            available=self.comments_available,
            detail=(
                None
                if self.comments_available
                else "fake comment query unavailable"
            ),
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


def _bound_observer(
    registry: RunRegistry,
    record: RunRecord,
    scheduler: FakeScheduler,
    **kwargs: Any,
) -> SlurmLifecycleObserver:
    """Construct an observer with the explicit lifecycle binding S3 supplies."""
    assert record.generation is not None
    observer = SlurmLifecycleObserver(registry, scheduler, **kwargs)
    observer.bind_generation(
        run_id=record.run_id,
        record_generation=record.generation,
        scheduler_generation=record.generation,
    )
    return observer


def test_unbound_observer_refuses_to_infer_scheduler_generation(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"chunk-0": ("91", "chunk")})
    scheduler = FakeScheduler({"91": "RUNNING"})

    SlurmLifecycleObserver(registry, scheduler).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"
    assert "not explicitly bound" in (updated.status_detail or "")
    assert scheduler.queries == []


def test_binding_rejects_mismatched_scheduler_epoch(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    observer = SlurmLifecycleObserver(registry, FakeScheduler())

    with pytest.raises(ValueError, match="lifecycle epoch"):
        observer.bind_generation(
            run_id=record.run_id,
            record_generation=record.generation,
            scheduler_generation=uuid4(),
        )


def test_marker_must_match_explicit_scheduler_generation_not_gui_generation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    gui_generation = uuid4()
    scheduler_generation = uuid4()
    registry = RunRegistry()
    record = RunRecord(
        run_id="out",
        generation=gui_generation,
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        status="submitting",
        scheduler_ids=("88",),
        primary_scheduler_id="88",
    )
    registry.register(record)
    initialize_slurm_lifecycle(
        output_dir,
        generation=scheduler_generation.hex,
        mode="ordinary",
    )
    metadata_path = job_metadata_path(output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        metadata_path,
        {
            "slurm_generation": scheduler_generation.hex,
            "slurm_job_ids": {
                "finalizer": {
                    "job_id": "99",
                    "role": "finalizer",
                    "generation": scheduler_generation.hex,
                }
            },
        },
    )
    append_lifecycle_entry(
        output_dir,
        generation=scheduler_generation.hex,
        token="finalizer",
        role="finalizer",
        status="submitted",
        job_id="99",
    )
    atomic_write_json(
        resolve_manifest_json_path(output_dir),
        {
            "is_complete": True,
            "failed": 0,
            "completed": 1,
            "total_images": 1,
        },
    )
    observer = SlurmLifecycleObserver(
        registry,
        FakeScheduler({"88": "RUNNING", "99": "COMPLETED"}),
    )
    observer.bind_generation(
        run_id=record.run_id,
        record_generation=gui_generation,
        scheduler_generation=scheduler_generation,
    )

    atomic_write_json(
        run_completion_marker_path(output_dir),
        {
            "generation": gui_generation.hex,
            "status": "complete",
            "finalizer_succeeded": True,
        },
    )
    observer.observe_once()
    assert registry.get(record.run_id).status == "reconciling"  # type: ignore[union-attr]

    atomic_write_json(
        run_completion_marker_path(output_dir),
        {
            "generation": scheduler_generation.hex,
            "status": "complete",
            "finalizer_succeeded": True,
        },
    )
    observer.observe_once()
    assert registry.get(record.run_id).status == "complete"  # type: ignore[union-attr]
    scheduler = observer.scheduler
    assert isinstance(scheduler, FakeScheduler)
    assert scheduler.queries == [("99",), ("99",)]


def test_observer_recovers_unresolved_intent_by_exact_comment(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    append_lifecycle_entry(
        record.output_dir,
        generation=record.generation.hex,
        token="chunk-0",
        role="chunk",
        status="intent",
    )
    scheduler = FakeScheduler(
        {"701": "PENDING"},
        comment_matches={"chunk-0": ("701",)},
    )

    _bound_observer(registry, record, scheduler).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "queued"
    assert updated.scheduler_ids == ("701",)
    rows = read_lifecycle_ledger(record.output_dir)
    assert rows[-1]["status"] == "recovered"
    assert rows[-1]["job_id"] == "701"


def test_empty_comment_query_retains_submitting(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    append_lifecycle_entry(
        record.output_dir,
        generation=record.generation.hex,
        token="chunk-0",
        role="chunk",
        status="intent",
    )

    _bound_observer(registry, record, FakeScheduler()).observe_once()

    assert registry.get(record.run_id).status == "submitting"  # type: ignore[union-attr]


def test_unavailable_comment_query_retains_unknown(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    append_lifecycle_entry(
        record.output_dir,
        generation=record.generation.hex,
        token="chunk-0",
        role="chunk",
        status="intent",
    )

    _bound_observer(
        registry,
        record,
        FakeScheduler(comments_available=False),
    ).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"
    assert "comment query unavailable" in (updated.status_detail or "")


def test_controller_only_submission_is_queued(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"controller-initial": ("101", "controller-initial")})
    scheduler = FakeScheduler({"101": "PENDING"})

    assert _bound_observer(registry, record, scheduler).observe_once() == 1

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "queued"
    assert updated.primary_scheduler_id == "101"


def test_unchanged_observation_does_not_bump_registry_revision(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"controller-initial": ("111", "controller-initial")})
    observer = _bound_observer(
        registry, record, FakeScheduler({"111": "PENDING"})
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

    _bound_observer(registry, record, scheduler).observe_once()

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

    _bound_observer(
        registry, record, FakeScheduler({"211": "FAILED"})
    ).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"


def test_scheduler_unavailable_is_unknown_not_failed(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"chunk-0": ("301", "chunk")})

    _bound_observer(
        registry, record, FakeScheduler(available=False)
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
    observer = _bound_observer(registry, record, scheduler)

    observer.observe_once()
    assert registry.get(record.run_id).status == "cancelling"  # type: ignore[union-attr]

    scheduler.states["402"] = "CANCELLED"
    observer.observe_once()
    assert registry.get(record.run_id).status == "cancelled"  # type: ignore[union-attr]


def test_completed_jobs_enter_grace_before_failed(tmp_path: Path) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    _write_jobs(record, {"chunk-0": ("501", "chunk")})
    now = [10.0]
    observer = _bound_observer(
        registry,
        record,
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

    _bound_observer(
        registry,
        record,
        FakeScheduler({"601": "COMPLETED", "602": "COMPLETED"}),
    ).observe_once()

    assert registry.get(record.run_id).status == "complete"  # type: ignore[union-attr]


def test_ordinary_marker_missing_manifest_fails_after_grace(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(record, {"finalizer": ("611", "finalizer")})
    atomic_write_json(
        run_completion_marker_path(record.output_dir),
        {
            "generation": record.generation.hex,
            "status": "complete",
            "finalizer_succeeded": True,
        },
    )
    now = [10.0]
    observer = _bound_observer(
        registry,
        record,
        FakeScheduler({"611": "COMPLETED"}),
        reconciliation_grace_seconds=5.0,
        monotonic=lambda: now[0],
    )

    observer.observe_once()
    assert registry.get(record.run_id).status == "reconciling"  # type: ignore[union-attr]
    now[0] = 16.0
    observer.observe_once()
    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "failed"
    assert "grace expired" in (updated.status_detail or "")


def test_staged_missing_publication_markers_fails_after_grace(
    tmp_path: Path,
) -> None:
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    orchestration_path = orchestration_state_path(record.output_dir)
    orchestration_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        orchestration_path,
        {
            "epoch": record.generation.hex,
            "phase": "complete",
        },
    )
    now = [20.0]
    observer = _bound_observer(
        registry,
        record,
        FakeScheduler(),
        reconciliation_grace_seconds=5.0,
        monotonic=lambda: now[0],
    )

    observer.observe_once()
    assert registry.get(record.run_id).status == "reconciling"  # type: ignore[union-attr]
    now[0] = 26.0
    observer.observe_once()
    assert registry.get(record.run_id).status == "failed"  # type: ignore[union-attr]


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
