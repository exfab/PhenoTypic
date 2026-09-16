"""Deterministic tests for the Dash-free SLURM lifecycle observer."""

from __future__ import annotations

import ast
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
    staged_completion_path,
)
from phenotypic._cli._cli_staged_resume import write_stage3_completion_marker
from phenotypic._gui.run_console._slurm_observer import (
    _all_stage3_markers_exist,
    discover_log_files,
    IncrementalLogReader,
    SchedulerCommentQueryResult,
    SchedulerQueryResult,
    SlurmLifecycleObserver,
)
from phenotypic._gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.sdk_ import (
    atomic_write_json,
    job_metadata_path,
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


def _publish_run_proof(output_dir: Path, *, generation: str) -> None:
    """Re-publish the run proof so its ``generation`` is *generation*.

    The observer matches ``marker["generation"]`` against the bound scheduler
    generation, and ``build_complete_run`` publishes with the epoch
    ``"local"``. The existing proof is **unlinked first**: the publisher's
    idempotence check compares only the stable digests, none of which change
    with the epoch, so publishing over it would return the old generation
    untouched and the test would silently assert the wrong thing.
    """
    from phenotypic._cli._cli_completion import (
        publish_run_completion_evidence,
    )

    run_completion_marker_path(output_dir).unlink(missing_ok=True)
    publish_run_completion_evidence(output_dir, execution_epoch=generation)


def _registered_complete_slurm_run(
    tmp_path: Path,
    *,
    stems: Sequence[str] = ("a", "b"),
) -> tuple[RunRegistry, RunRecord]:
    """Register a SLURM record over a tree whose shallow verdict is complete.

    ``_run_marker_observation``'s publication half now asks
    ``resolve_run_state``, which wants an accepted inventory in
    ``processing_state.json``, a per-image record for every entry in it, an
    aggregate proof, and a run proof binding to both. A hand-written marker
    beside a hand-written ``manifest.json`` -- what these tests used while
    ``_manifest_is_complete`` was the fallback -- reads ``incomplete``.

    ``build_complete_run`` publishes all of it through the **real**
    publishers, so the fixture cannot keep passing after the proof format
    changes underneath it.
    """
    from tests._output_layout import build_complete_run

    output_dir = build_complete_run(tmp_path, stems=stems)
    registry = RunRegistry()
    record = RunRecord(
        run_id=output_dir.name,
        generation=uuid4(),
        mode="slurm",
        output_dir=output_dir,
        rel_path=output_dir.name,
        status="submitting",
    )
    registry.register(record)
    assert record.generation is not None
    initialize_slurm_lifecycle(
        output_dir,
        generation=record.generation.hex,
        mode="ordinary",
    )
    _publish_run_proof(output_dir, generation=record.generation.hex)
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
    assert record.generation is not None
    registry.compare_and_set(
        record.run_id,
        record.generation,
        status="running",
    )
    record = registry.get(record.run_id)
    assert record is not None
    _write_jobs(record, {"chunk-0": ("91", "chunk")})
    scheduler = FakeScheduler({"91": "RUNNING"})

    SlurmLifecycleObserver(registry, scheduler).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"
    assert "not explicitly bound" in (updated.status_detail or "")
    assert scheduler.queries == []


def test_unbound_observer_does_not_race_active_submitter(
    tmp_path: Path,
) -> None:
    """The submit future owns status until it can publish the exact binding."""
    registry, record = _registered_slurm_run(tmp_path)
    scheduler = FakeScheduler()

    SlurmLifecycleObserver(registry, scheduler).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "submitting"
    assert "awaiting explicit" in (updated.status_detail or "")
    assert scheduler.queries == []


def test_restart_binding_retries_when_lifecycle_appears_late(
    tmp_path: Path,
) -> None:
    """A restart window heals only after all durable identities agree."""
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
    )
    registry.register(record)
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            "gui_record_generation": gui_generation.hex,
            "slurm_generation": scheduler_generation.hex,
            "slurm_job_ids": {},
        },
    )
    observer = SlurmLifecycleObserver(registry, FakeScheduler())

    assert observer.reconcile_durable_bindings() == 0
    initialize_slurm_lifecycle(
        output_dir,
        generation=scheduler_generation.hex,
        mode="ordinary",
    )
    assert observer.reconcile_durable_bindings() == 1

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.lifecycle_epoch == scheduler_generation.hex
    assert (record.run_id, gui_generation) in observer._bindings  # noqa: SLF001


def test_restart_binding_rejects_stale_old_lifecycle(
    tmp_path: Path,
) -> None:
    """A prior launch cannot be attached to a newer GUI owner record."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    old_gui_generation = uuid4()
    current_gui_generation = uuid4()
    old_scheduler_generation = uuid4()
    registry = RunRegistry()
    record = RunRecord(
        run_id="out",
        generation=current_gui_generation,
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        status="unknown",
    )
    registry.register(record)
    initialize_slurm_lifecycle(
        output_dir,
        generation=old_scheduler_generation.hex,
        mode="ordinary",
    )
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            "gui_record_generation": old_gui_generation.hex,
            "slurm_generation": old_scheduler_generation.hex,
            "slurm_job_ids": {},
        },
    )
    observer = SlurmLifecycleObserver(registry, FakeScheduler())

    assert observer.reconcile_durable_bindings() == 0
    observer.observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"
    assert (record.run_id, current_gui_generation) not in observer._bindings  # noqa: SLF001


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
    from tests._output_layout import build_complete_run

    output_dir = build_complete_run(tmp_path)
    gui_generation = uuid4()
    scheduler_generation = uuid4()
    registry = RunRegistry()
    record = RunRecord(
        run_id=output_dir.name,
        generation=gui_generation,
        mode="slurm",
        output_dir=output_dir,
        rel_path=output_dir.name,
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
    observer = SlurmLifecycleObserver(
        registry,
        FakeScheduler({"88": "RUNNING", "99": "COMPLETED"}),
    )
    observer.bind_generation(
        run_id=record.run_id,
        record_generation=gui_generation,
        scheduler_generation=scheduler_generation,
    )

    _publish_run_proof(output_dir, generation=gui_generation.hex)
    observer.observe_once()
    assert registry.get(record.run_id).status == "reconciling"  # type: ignore[union-attr]

    _publish_run_proof(output_dir, generation=scheduler_generation.hex)
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
    assert observer.tracked_generation_counts == (0, 0)


def test_generation_marker_and_current_evidence_complete_run(
    tmp_path: Path,
) -> None:
    """A matched proof over a tree that still verifies completes the run."""
    registry, record = _registered_complete_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "chunk-0": ("601", "chunk"),
            "finalizer": ("602", "finalizer"),
        },
    )
    lifecycle = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    lifecycle["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), lifecycle)

    _bound_observer(
        registry,
        record,
        FakeScheduler({"601": "COMPLETED", "602": "COMPLETED"}),
    ).observe_once()

    assert registry.get(record.run_id).status == "complete"  # type: ignore[union-attr]


def test_inactive_fence_with_published_ordinary_run_reconciles_finalizer(
    tmp_path: Path,
) -> None:
    """Publication before finalizer exit is not mistaken for cancellation."""
    registry, record = _registered_complete_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "chunk-0": ("613", "chunk"),
            "finalizer": ("614", "finalizer"),
        },
    )
    lifecycle = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    lifecycle["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), lifecycle)
    scheduler = FakeScheduler({"613": "COMPLETED", "614": "RUNNING"})
    now = [10.0]
    observer = _bound_observer(
        registry,
        record,
        scheduler,
        reconciliation_grace_seconds=5.0,
        monotonic=lambda: now[0],
    )

    observer.observe_once()
    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "reconciling"
    assert "awaiting terminal jobs and finalizer" in (
        updated.status_detail or ""
    )

    now[0] = 20.0
    observer.observe_once()
    assert registry.get(record.run_id).status == "reconciling"  # type: ignore[union-attr]

    scheduler.states["614"] = "COMPLETED"
    observer.observe_once()
    assert registry.get(record.run_id).status == "complete"  # type: ignore[union-attr]


def test_explicit_cancellation_precedes_visible_publication(
    tmp_path: Path,
) -> None:
    """A user cancellation remains authoritative during publication races."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "chunk-0": ("615", "chunk"),
            "finalizer": ("616", "finalizer"),
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
    lifecycle = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    lifecycle["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), lifecycle)
    assert registry.compare_and_set(
        record.run_id,
        record.generation,
        status="cancelling",
    )

    _bound_observer(
        registry,
        record,
        FakeScheduler({"615": "COMPLETED", "616": "RUNNING"}),
    ).observe_once()

    assert registry.get(record.run_id).status == "cancelling"  # type: ignore[union-attr]


def test_explicit_cancellation_survives_active_lifecycle_window(
    tmp_path: Path,
) -> None:
    """The observer cannot overwrite cancellation before fence deactivation."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(record, {"chunk-0": ("617", "chunk")})
    assert registry.compare_and_set(
        record.run_id,
        record.generation,
        status="cancelling",
    )

    _bound_observer(
        registry,
        record,
        FakeScheduler({"617": "RUNNING"}),
    ).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "cancelling"
    assert "awaiting inactive fence" in (updated.status_detail or "")


def test_missing_finalizer_scheduler_row_does_not_expire_grace(
    tmp_path: Path,
) -> None:
    """Absent accounting evidence remains reconciliation, not timeout failure."""
    registry, record = _registered_complete_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "chunk-0": ("618", "chunk"),
            "finalizer": ("619", "finalizer"),
        },
    )
    lifecycle = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    lifecycle["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), lifecycle)
    scheduler = FakeScheduler({"618": "COMPLETED"})
    now = [10.0]
    observer = _bound_observer(
        registry,
        record,
        scheduler,
        reconciliation_grace_seconds=5.0,
        monotonic=lambda: now[0],
    )

    observer.observe_once()
    now[0] = 20.0
    observer.observe_once()
    assert registry.get(record.run_id).status == "reconciling"  # type: ignore[union-attr]

    scheduler.states["619"] = "COMPLETED"
    observer.observe_once()
    assert registry.get(record.run_id).status == "complete"  # type: ignore[union-attr]


def test_inactive_fence_does_not_hide_failed_ordinary_finalizer(
    tmp_path: Path,
) -> None:
    """Exact finalizer failure outranks an ambiguous inactive lifecycle fence."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "chunk-0": ("620", "chunk"),
            "finalizer": ("621", "finalizer"),
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
    lifecycle = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    lifecycle["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), lifecycle)

    _bound_observer(
        registry,
        record,
        FakeScheduler({"620": "COMPLETED", "621": "FAILED"}),
    ).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "failed"
    assert "621=FAILED" in (updated.status_detail or "")


def test_visible_marker_cannot_hide_failed_finalizer_window(
    tmp_path: Path,
) -> None:
    """A marker written before finalizer exit cannot publish false success."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "chunk-0": ("603", "chunk"),
            "finalizer": ("604", "finalizer"),
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
    scheduler = FakeScheduler({"603": "COMPLETED", "604": "RUNNING"})
    observer = _bound_observer(registry, record, scheduler)

    observer.observe_once()
    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "reconciling"

    scheduler.states["604"] = "FAILED"
    observer.observe_once()
    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "failed"
    assert "604=FAILED" in (updated.status_detail or "")


def test_ordinary_marker_without_current_evidence_fails_after_grace(
    tmp_path: Path,
) -> None:
    """A generation-matched marker over a tree that cannot be verified.

    Renamed from ``..._missing_manifest_...``: ``manifest.json`` is no longer
    read here (spec §4.2 demotes it), so the thing this tree is missing is
    what ``resolve_run_state`` needs -- an accepted inventory, per-image
    records, an aggregate proof. Absent all of it the verdict is
    ``incomplete``, and a marker alone must not publish success.
    """
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


def _hashes_during_one_publication_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stems: Sequence[str],
) -> int:
    """Count ``sha256`` constructions in one warm observation of a run.

    The warm-up call is what a previous tick would have been: it is the
    verification cache's cold pass, and counting it would measure the cache
    miss rather than the steady state the observer actually lives in.
    """
    import hashlib

    from phenotypic.sdk_ import clear_verification_cache, resolve_run_state

    clear_verification_cache()
    registry, record = _registered_complete_slurm_run(tmp_path, stems=stems)
    _write_jobs(record, {"finalizer": ("701", "finalizer")})
    observer = _bound_observer(
        registry, record, FakeScheduler({"701": "COMPLETED"})
    )
    resolve_run_state(record.output_dir, depth="shallow")

    calls = 0
    real_sha256 = hashlib.sha256

    def _counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", _counted)
    observer.observe_once()
    monkeypatch.undo()

    updated = registry.get(record.run_id)
    assert updated is not None
    # Without this the whole measurement is vacuous: an observation that
    # never reached the publication check hashes nothing, and two zeroes
    # compare equal.
    assert updated.status == "complete", updated.status_detail
    return calls


def test_the_publication_check_hashes_a_constant_number_of_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit §4: the tick's hashing must not grow with the image count.

    What this replaced walked every accepted image on **every** tick --
    ``valid_run_completion`` -> ``_all_accepted_images_succeeded`` ->
    ``_walk_current_success`` -> ``valid_image_success``, which re-hashes each
    image's declared artifacts -- and then re-validated the aggregate on top.
    ``resolve_run_state(depth="shallow")`` re-stats a warm cache entry
    instead, so the only hashing left is over the run-level aggregate's
    deliverables: a fixed handful, whatever N is.

    **Two tree sizes rather than one bound.** A fixed bound on a single tree
    cannot tell "constant" from "small", which is why
    ``build_complete_run`` takes ``stems`` at all. Equality is the assertion:
    ``<=`` would also pass for a run that hashed nothing because
    ``resolve_run_state`` was never reached, and the non-zero assertion below
    closes the same hole from the other side. The shape is P1 Task 6's --
    ``tests/unit/sdk_/test_run_state.py::
    test_shallow_reuse_is_independent_of_the_image_count`` makes the same
    claim about the reader; this one makes it about the **tick**, which is
    where the repeated cost actually lands.

    **No absolute figure is asserted, and none should be.** The plan's
    original ``<= 8`` came from a per-``stat`` cost that was measured to be
    wrong by two-and-a-half orders of magnitude and, worse, was the cost of a
    different operation. What survives that refutation is the shape --
    per-image hashing removed from a repeating poll -- and the shape is what
    this test pins.
    """
    small = _hashes_during_one_publication_check(
        tmp_path / "small", monkeypatch, stems=("a", "b")
    )
    large = _hashes_during_one_publication_check(
        tmp_path / "large", monkeypatch, stems=("a", "b", "c", "d", "e")
    )

    assert small > 0, (
        "the observation hashed nothing at all, so it never reached the "
        "run-level proof -- the count is not measuring what it claims"
    )
    assert small == large, (
        f"a 2-image tree hashed {small} files and a 5-image tree {large}: "
        "the publication check is linear in the image count again"
    )


def _resolve_run_state_call_sites() -> list[tuple[str, ast.Call]]:
    """Return ``(enclosing function, call)`` for every ``resolve_run_state``.

    Derived from the module's AST, never from ``grep``: a text search
    measures spellings, so it counts the import line and any mention in a
    comment or docstring, and it cannot say which function a call sits in.
    """
    from phenotypic._gui.run_console import _slurm_observer

    source = Path(_slurm_observer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    sites: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = (
            func.id
            if isinstance(func, ast.Name)
            else getattr(func, "attr", None)
        )
        if called != "resolve_run_state":
            continue
        # The innermost enclosing `def`, so a call inside a nested function
        # is attributed to that function rather than to both.
        enclosing = [
            candidate
            for candidate in functions
            if candidate.lineno
            <= node.lineno
            <= (candidate.end_lineno or candidate.lineno)
        ]
        enclosing.sort(key=lambda candidate: candidate.lineno)
        sites.append(
            (enclosing[-1].name if enclosing else "<module>", node)
        )
    return sites


def test_the_observer_asks_resolve_run_state_once_and_shallowly() -> None:
    """Spec §11's observer row, and §2.2 / DEFERRED D-1's scope fence.

    The row reads "1 x ``resolve_run_state(shallow)``". This fires when:

    * a second call appears anywhere in the module -- including one added to
      ``_staged_terminal_observation`` or to the poll loop, which is the
      scope creep §2.2 and D-1 fence off;
    * the one call leaves ``_run_marker_observation``, the only site where
      the run proof rule 1 requires is present by construction;
    * ``depth="shallow"`` is dropped or becomes ``"deep"``, which would put a
      full re-verification of every image on the 2-second tick.
    """
    sites = _resolve_run_state_call_sites()

    assert [name for name, _ in sites] == ["_run_marker_observation"]
    depths = [
        keyword.value.value
        for _, call in sites
        for keyword in call.keywords
        if keyword.arg == "depth"
    ]
    assert depths == ["shallow"]


def test_the_decision_tree_and_grace_window_are_untouched() -> None:
    """Spec §2.2, DEFERRED D-1: scope creep should fail CI, not review.

    The observer's decision tree, its reconciliation grace window and its
    ``squeue``/``sacct`` state ranking are out of scope for the consumer
    migration, and ``manifest.json`` is out of the evidence set (§4.2).

    Each assertion below fails on a real edit rather than reading as one:
    the module-level check fails **today**, before this task deletes
    ``_manifest_is_complete``, and the two grace assertions fail if the
    publication grace is dropped from the decision tree -- which is what
    "just return the observation directly" looks like when someone
    simplifies this method.
    """
    import inspect

    from phenotypic._gui.run_console import _slurm_observer

    assert not hasattr(_slurm_observer, "_manifest_is_complete")

    source = inspect.getsource(
        _slurm_observer.SlurmLifecycleObserver._observe_record
    )
    assert "_apply_publication_grace" in source
    assert "_clear_grace" in source


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


def test_observer_prunes_superseded_terminal_generation_state(
    tmp_path: Path,
) -> None:
    """A replacement registry owner evicts the prior generation next cycle."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    observer = _bound_observer(registry, record, FakeScheduler())
    old_key = (record.run_id, record.generation)
    observer._reconciling_since[old_key] = 1.0  # noqa: SLF001
    assert observer.tracked_generation_counts == (1, 1)
    assert registry.compare_and_set(
        record.run_id,
        record.generation,
        status="failed",
    )
    replacement = RunRecord(
        run_id=record.run_id,
        generation=uuid4(),
        mode="local",
        output_dir=record.output_dir,
        rel_path=record.rel_path,
        status="running",
    )
    registry.register(replacement, persist=False)

    observer.observe_once()

    assert observer.tracked_generation_counts == (0, 0)


def test_exact_staged_completion_precedes_inactive_fence(
    tmp_path: Path,
) -> None:
    """A completed staged publication is not relabelled as cancellation."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    atomic_write_json(
        job_metadata_path(record.output_dir),
        {
            "slurm_generation": record.generation.hex,
            "datasets": {
                "plate": {"total": 1, "images": ["image.tif"]},
            },
            "slurm_job_ids": {},
        },
    )
    atomic_write_json(
        orchestration_state_path(record.output_dir),
        {
            "epoch": record.generation.hex,
            "phase": "complete",
        },
    )
    write_stage3_completion_marker(
        record.output_dir,
        "plate",
        "image.tif",
        "image",
    )
    atomic_write_json(
        staged_completion_path(record.output_dir),
        {"epoch": record.generation.hex},
    )
    state = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    state["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), state)

    _bound_observer(registry, record, FakeScheduler()).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "complete"
    assert (
        updated.status_detail
        == "staged orchestration and publication completed"
    )


def test_stage3_inventory_uses_canonical_direct_store_stem(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    atomic_write_json(
        job_metadata_path(output),
        {
            "datasets": {
                "single_image": {
                    "total": 1,
                    "images": ["p01.ome.zarr"],
                }
            }
        },
    )
    write_stage3_completion_marker(
        output,
        "single_image",
        "p01.ome.zarr",
        "p01",
    )

    assert _all_stage3_markers_exist(output)


def test_inactive_fence_with_staged_publication_reconciles_finalizer(
    tmp_path: Path,
) -> None:
    """Staged publication waits for the scheduler without false cancellation."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "stage3-0": ("801", "stage3"),
            "finalizer": ("802", "finalizer"),
        },
    )
    metadata = json.loads(
        job_metadata_path(record.output_dir).read_text(encoding="utf-8")
    )
    metadata["datasets"] = {
        "plate": {"total": 1, "images": ["image.tif"]},
    }
    atomic_write_json(job_metadata_path(record.output_dir), metadata)
    atomic_write_json(
        orchestration_state_path(record.output_dir),
        {
            "epoch": record.generation.hex,
            "phase": "complete",
        },
    )
    write_stage3_completion_marker(
        record.output_dir,
        "plate",
        "image.tif",
        "image",
    )
    atomic_write_json(
        staged_completion_path(record.output_dir),
        {"epoch": record.generation.hex},
    )
    lifecycle = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    lifecycle["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), lifecycle)
    scheduler = FakeScheduler({"801": "COMPLETED", "802": "RUNNING"})
    observer = _bound_observer(registry, record, scheduler)

    observer.observe_once()
    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "reconciling"
    assert "staged publication is visible" in (updated.status_detail or "")

    scheduler.states["802"] = "COMPLETED"
    observer.observe_once()
    assert registry.get(record.run_id).status == "complete"  # type: ignore[union-attr]


def test_inactive_fence_does_not_hide_failed_staged_finalizer(
    tmp_path: Path,
) -> None:
    """Exact staged finalizer failure outranks an inactive lifecycle fence."""
    registry, record = _registered_slurm_run(tmp_path)
    assert record.generation is not None
    _write_jobs(
        record,
        {
            "stage3-0": ("803", "stage3"),
            "finalizer": ("804", "finalizer"),
        },
    )
    metadata = json.loads(
        job_metadata_path(record.output_dir).read_text(encoding="utf-8")
    )
    metadata["datasets"] = {
        "plate": {"total": 1, "images": ["image.tif"]},
    }
    atomic_write_json(job_metadata_path(record.output_dir), metadata)
    atomic_write_json(
        orchestration_state_path(record.output_dir),
        {
            "epoch": record.generation.hex,
            "phase": "complete",
        },
    )
    write_stage3_completion_marker(
        record.output_dir,
        "plate",
        "image.tif",
        "image",
    )
    atomic_write_json(
        staged_completion_path(record.output_dir),
        {"epoch": record.generation.hex},
    )
    lifecycle = json.loads(
        lifecycle_state_path(record.output_dir).read_text(encoding="utf-8")
    )
    lifecycle["active"] = False
    atomic_write_json(lifecycle_state_path(record.output_dir), lifecycle)

    _bound_observer(
        registry,
        record,
        FakeScheduler({"803": "COMPLETED", "804": "FAILED"}),
    ).observe_once()

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "failed"
    assert "804=FAILED" in (updated.status_detail or "")


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


def test_log_discovery_fences_generation_scoped_submitter_logs(
    tmp_path: Path,
) -> None:
    current = uuid4()
    previous = uuid4()
    gui_dir = tmp_path / ".phenotypic" / "logs" / "gui"
    gui_dir.mkdir(parents=True)
    current_log = gui_dir / f"submitter.{current.hex}.stdout.log"
    previous_log = gui_dir / f"submitter.{previous.hex}.stdout.log"
    legacy_log = gui_dir / "submitter.stdout.log"
    for path in (current_log, previous_log, legacy_log):
        path.write_text(path.name, encoding="utf-8")

    discovered = discover_log_files(
        tmp_path,
        record_generation=current,
    )

    assert current_log in discovered
    assert legacy_log in discovered
    assert previous_log not in discovered
