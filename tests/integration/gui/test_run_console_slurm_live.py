"""Opt-in live SLURM acceptance tests for the GUI Run Console.

The neutral safety harness in :mod:`tests._support.live_slurm` requires an
exact clean SHA, a shared canonical root, independently labeled protected
latest/active paths (or a validated no-active-output inspection record), one
96 x 96 TIFF, bounded CPU-only resources, and fail-closed scheduler cleanup.
Completed cases remain at their exact paths with fd-bound cleanup evidence.
After external scheduler and evidence verification, the dedicated per-run
root may be moved to trash manually.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import Future
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from phenotypic.gui.run_console._callbacks import (
    _cancel_bound_generation,
    _complete_slurm_submission,
)
from phenotypic.gui.run_console._slurm import (
    SlurmSubmitResult,
    SubmittedJobSet,
    submit_slurm,
)
from phenotypic.gui.run_console._slurm_observer import SlurmLifecycleObserver
from phenotypic.gui.run_console._state import RunConsoleState
from phenotypic.gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.sdk_ import (
    job_metadata_path,
    resolve_manifest_json_path,
    run_completion_marker_path,
)
from tests._support.live_slurm import (
    ACCOUNT_ENV,
    active_generation_comment_ids,
    cleanup_case,
    jobs_by_role,
    prepared_case,
    require_live_environment,
    validate_retained_case_evidence,
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PHENOTYPIC_RUN_LIVE_SLURM") != "1",
        reason="set PHENOTYPIC_RUN_LIVE_SLURM=1 for real scheduler tests",
    ),
    pytest.mark.skipif(
        any(
            shutil.which(tool) is None
            for tool in ("sbatch", "squeue", "sacct", "scancel")
        ),
        reason="requires sbatch, squeue, sacct, and scancel",
    ),
]

_POLL_SECONDS = 2.0
_TERMINAL_TIMEOUT_SECONDS = 10 * 60.0
_CANCELLATION_TIMEOUT_SECONDS = 60.0


def _state(
    *,
    pipeline_path: Path,
    input_dir: Path,
    output_dir: Path,
    partition: str,
    hold: bool = False,
) -> RunConsoleState:
    """Return one bounded CPU-only Run Console request."""
    extra: dict[str, str] = {}
    account = os.environ.get(ACCOUNT_ENV, "").strip()
    if account:
        extra["slurm_account"] = account
    if hold:
        extra["slurm_begin"] = "now+2minutes"
    return RunConsoleState(
        pipeline_path=str(pipeline_path),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        mode="slurm",
        advanced_args={"image_type": "Image", "workers": 1},
        slurm_args={
            "partition": partition,
            "time": "00:10:00",
            "mem": "4G",
            "cpus_per_task": 1,
            "extra": extra,
        },
        gpu_slurm_args=(),
        gpu_shards=1,
    )


def _submit_from_gui(
    state: RunConsoleState,
    *,
    sandbox_root: Path,
) -> tuple[
    RunRegistry,
    RunRecord,
    SlurmSubmitResult,
    SlurmLifecycleObserver,
]:
    """Exercise production GUI future completion and durable binding."""
    assert state.output_dir is not None
    output_dir = Path(state.output_dir)
    registry = RunRegistry()
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path=output_dir.name,
        command_digest=f"live-{uuid4().hex}",
        status="submitting",
    )
    assert record.generation is not None
    observer = SlurmLifecycleObserver(
        registry,
        poll_interval_seconds=_POLL_SECONDS,
        reconciliation_grace_seconds=45.0,
    )
    result = submit_slurm(
        state,
        sandbox_root=sandbox_root,
        record_generation=record.generation,
        timeout=120.0,
    )
    jobs = _require_job_set(result)
    future: Future[SlurmSubmitResult] = Future()
    future.set_result(result)
    # This production completion seam invokes
    # _persist_and_bind_scheduler_generation before queuing the GUI record.
    _complete_slurm_submission(
        future,
        registry=registry,
        run_id=record.run_id,
        generation=record.generation,
        observer=observer,
    )
    record = registry.get(record.run_id)
    assert record is not None
    assert record.status == "queued", record.status_detail
    assert record.lifecycle_epoch == jobs.generation.hex
    assert record.scheduler_ids == jobs.all_ids
    binding = observer.proven_binding(record)
    assert binding is not None
    assert binding.scheduler_generation == jobs.generation
    return registry, record, result, observer


def _require_job_set(result: SlurmSubmitResult) -> SubmittedJobSet:
    """Return the typed scheduler identities or fail with submitter evidence."""
    jobs = result.submitted_jobs
    assert jobs is not None, (
        f"submission {result.job_id} did not expose typed scheduler metadata"
    )
    assert jobs.all_ids
    assert all(job_id.isdigit() for job_id in jobs.all_ids)
    return jobs


def _wait_for_terminal(
    observer: SlurmLifecycleObserver,
    registry: RunRegistry,
    run_id: str,
    *,
    timeout: float = _TERMINAL_TIMEOUT_SECONDS,
) -> RunRecord:
    """Poll the Dash-free observer until the run reaches a terminal state."""
    deadline = time.monotonic() + timeout
    last_status: str | None = None
    while time.monotonic() < deadline:
        observer.observe_once(run_id)
        record = registry.get(run_id)
        assert record is not None
        if record.status != last_status:
            print(
                "LIVE_SLURM_STATUS "
                f"run={run_id} status={record.status} "
                f"jobs={','.join(record.scheduler_ids)}"
            )
            last_status = record.status
        if record.status in {"complete", "failed", "cancelled"}:
            return record
        time.sleep(_POLL_SECONDS)
    pytest.fail(f"SLURM run {run_id} did not terminalize within {timeout}s")


def test_live_gui_ordinary_slurm_completion_has_terminal_evidence() -> None:
    """A one-image GUI submission completes only with generation evidence."""
    root, partition, forbidden = require_live_environment()
    with prepared_case(root, forbidden) as (
        case_root,
        pipeline_path,
        output_dir,
    ):
        submitted_ids: tuple[str, ...] = ()
        scheduler_generation: UUID | None = None
        try:
            state = _state(
                pipeline_path=pipeline_path,
                input_dir=case_root / "input",
                output_dir=output_dir,
                partition=partition,
            )
            registry, record, result, observer = _submit_from_gui(
                state,
                sandbox_root=root,
            )
            jobs = _require_job_set(result)
            scheduler_generation = jobs.generation
            submitted_ids = jobs.all_ids
            terminal = _wait_for_terminal(observer, registry, record.run_id)

            roles = jobs_by_role(output_dir, jobs.generation)
            marker = json.loads(
                run_completion_marker_path(output_dir).read_text(encoding="utf-8")
            )
            manifest = json.loads(
                resolve_manifest_json_path(output_dir).read_text(encoding="utf-8")
            )
            metadata = json.loads(
                job_metadata_path(output_dir).read_text(encoding="utf-8")
            )
            assert terminal.status == "complete", terminal.status_detail
            assert set(roles) >= {"chunk", "finalizer"}
            assert marker["generation"] == jobs.generation.hex
            assert marker["status"] == "complete"
            assert marker["finalizer_succeeded"] is True
            assert marker["schema_version"] == 1
            assert marker["completed_at"]
            assert manifest["is_complete"] is True
            assert manifest["completed"] == 1
            assert manifest["failed"] == 0
            assert metadata["gui_record_generation"] == str(record.generation)
            assert metadata["slurm_generation"] == jobs.generation.hex
            assert len(tuple((case_root / "input").glob("*.tiff"))) == 1
            print(
                "LIVE_SLURM_COMPLETION "
                f"generation={jobs.generation.hex} "
                f"output={output_dir} "
                f"roles={json.dumps(roles, sort_keys=True)}"
            )
        finally:
            evidence_name = cleanup_case(
                case_root,
                output_dir,
                scheduler_generation,
                iter(submitted_ids),
                forbidden=forbidden,
            )
            validate_retained_case_evidence(
                case_root,
                evidence_name,
                scheduler_generation=scheduler_generation,
            )


def test_live_gui_cancellation_fences_dependent_finalizer() -> None:
    """Production cancellation removes a queued chunk and its continuation."""
    root, partition, forbidden = require_live_environment()
    with prepared_case(root, forbidden) as (
        case_root,
        pipeline_path,
        output_dir,
    ):
        submitted_ids: tuple[str, ...] = ()
        scheduler_generation: UUID | None = None
        try:
            state = _state(
                pipeline_path=pipeline_path,
                input_dir=case_root / "input",
                output_dir=output_dir,
                partition=partition,
                hold=True,
            )
            registry, record, result, observer = _submit_from_gui(
                state,
                sandbox_root=root,
            )
            jobs = _require_job_set(result)
            scheduler_generation = jobs.generation
            submitted_ids = jobs.all_ids
            roles_before = jobs_by_role(output_dir, jobs.generation)
            assert set(roles_before) >= {"chunk", "finalizer"}
            assert roles_before["finalizer"], "dependent finalizer was not submitted"
            assert active_generation_comment_ids(jobs.generation)

            assert record.generation is not None
            assert registry.compare_and_set(
                record.run_id,
                record.generation,
                expected_statuses={"queued", "running", "unknown"},
                status="cancelling",
            )
            cancelling_record = registry.get(record.run_id)
            assert cancelling_record is not None
            binding = observer.proven_binding(cancelling_record)
            assert binding is not None
            # This is the same production orchestration helper used by Cancel.
            cancelled_ids = _cancel_bound_generation(
                cancelling_record,
                binding,
            )
            assert set(cancelled_ids) >= set(jobs.all_ids)
            terminal = _wait_for_terminal(
                observer,
                registry,
                record.run_id,
                timeout=_CANCELLATION_TIMEOUT_SECONDS,
            )
            assert terminal.status == "cancelled", terminal.status_detail
            assert not active_generation_comment_ids(jobs.generation)
            time.sleep(5.0)
            assert not active_generation_comment_ids(jobs.generation)
            assert not run_completion_marker_path(output_dir).exists()
            print(
                "LIVE_SLURM_CANCELLATION "
                f"generation={jobs.generation.hex} "
                f"output={output_dir} "
                f"roles={json.dumps(roles_before, sort_keys=True)} "
                f"cancelled={','.join(cancelled_ids)}"
            )
        finally:
            evidence_name = cleanup_case(
                case_root,
                output_dir,
                scheduler_generation,
                iter(submitted_ids),
                forbidden=forbidden,
            )
            validate_retained_case_evidence(
                case_root,
                evidence_name,
                scheduler_generation=scheduler_generation,
            )
