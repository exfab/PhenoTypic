"""Pure integration coverage for GUI scheduler startup wiring.

These tests inject both scheduler-facing dependencies. They never call
``sbatch``, ``squeue``, ``sacct``, or ``scancel``.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

import phenotypic.gui.builder as builder_package
import phenotypic.gui.run_console as run_console_package
from phenotypic._cli import _cli_preload
from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    initialize_slurm_lifecycle,
)
from phenotypic.gui._config import CFG_RUN_REGISTRY
from phenotypic.gui.run_console._app import SLURM_OBSERVER_EXTENSION
from phenotypic.gui.run_console._request_safety import (
    build_metadata_preflight,
    confirm_output_target,
)
from phenotypic.gui.run_console._slurm import (
    SlurmSubmitResult,
    SubmittedJobSet,
)
from phenotypic.gui.run_console._slurm_observer import (
    SchedulerCommentQueryResult,
    SchedulerQueryResult,
    SlurmLifecycleObserver,
)
from phenotypic.gui.run_console._state import RunConsoleState
from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.sdk_ import JobMetadataKey, atomic_write_json, job_metadata_path


class InMemoryScheduler:
    """Scheduler client whose observations are ordinary Python mappings."""

    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.queries: list[tuple[str, ...]] = []

    def query(self, job_ids: Sequence[str]) -> SchedulerQueryResult:
        """Return configured states without starting a subprocess."""
        normalized = tuple(str(item) for item in job_ids)
        self.queries.append(normalized)
        return SchedulerQueryResult(
            states={
                job_id: self.states[job_id]
                for job_id in normalized
                if job_id in self.states
            }
        )

    def find_by_comments(
        self,
        scheduler_epoch: str,
        tokens: Sequence[str],
    ) -> SchedulerCommentQueryResult:
        """Return no unresolved comments for the bounded fixture."""
        del scheduler_epoch, tokens
        return SchedulerCommentQueryResult(matches={})


def _callback_by_name(app: Any, name: str) -> Any:
    """Return one unwrapped Dash callback by Python function name."""
    return next(
        callback.__wrapped__
        for spec in app.callback_map.values()
        if (callback := spec.get("callback")) is not None
        and callback.__wrapped__.__name__ == name
    )


def _slurm_action_controls(
    sandbox: SandboxRoot,
    *,
    pipeline: Path,
    images: Path,
    output: Path,
) -> tuple[object, ...]:
    """Build the exact authoritative Run callback state tuple."""
    preflight = build_metadata_preflight(
        sandbox,
        str(images),
        None,
    )
    confirmation = confirm_output_target(sandbox, str(output))
    return (
        str(pipeline),
        str(images),
        str(output),
        "slurm",
        [],
        None,
        None,
        None,
        None,
        None,
        None,
        "test-partition",
        "00:05:00",
        "2G",
        1,
        0,
        None,
        None,
        1,
        None,
        "omit",
        [],
        preflight.to_json(),
        confirmation.to_json(),
    )


def _write_mock_submission(
    state: RunConsoleState,
    *,
    record_generation: UUID,
    lifecycle_mode: str,
    job_id: str,
) -> SlurmSubmitResult:
    """Publish scheduler-shaped durable evidence without scheduler commands."""
    assert state.output_dir is not None
    output_dir = Path(state.output_dir)
    scheduler_generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=scheduler_generation.hex,
        mode=lifecycle_mode,
    )
    role = "controller-initial" if lifecycle_mode == "staged" else "chunk"
    metadata: dict[str, object] = {
        JobMetadataKey.GUI_RECORD_GENERATION: str(record_generation),
        "slurm_generation": scheduler_generation.hex,
        "slurm_job_ids": {
            role: {
                "job_id": job_id,
                "role": role,
                "generation": scheduler_generation.hex,
            }
        },
        "chunk_job_ids": (
            {} if lifecycle_mode == "staged" else {"0": job_id}
        ),
    }
    if lifecycle_mode == "staged":
        metadata[JobMetadataKey.ORCHESTRATION_EPOCH] = (
            scheduler_generation.hex
        )
    atomic_write_json(job_metadata_path(output_dir), metadata)
    append_lifecycle_entry(
        output_dir,
        generation=scheduler_generation.hex,
        token=role,
        role=role,
        status="submitted",
        job_id=job_id,
    )
    jobs = SubmittedJobSet(
        primary_id=job_id,
        all_ids=(job_id,),
        roles={role: (job_id,)},
        generation=scheduler_generation,
    )
    return SlurmSubmitResult(
        job_id=job_id,
        output_dir=output_dir,
        stdout="",
        stderr="",
        returncode=0,
        submitted_jobs=jobs,
    )


@pytest.mark.parametrize(
    ("lifecycle_mode", "job_id"),
    (("ordinary", "8101"), ("staged", "8201")),
)
def test_hub_preloads_and_wires_generation_fenced_scheduler_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_mode: str,
    job_id: str,
) -> None:
    """The real hub seam shares one injected submitter and observer."""
    sandbox = SandboxRoot.from_path(tmp_path)
    pipeline = tmp_path / "pipeline.json"
    images = tmp_path / "images"
    output = tmp_path / f"{lifecycle_mode}-output"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    images.mkdir()
    (images / "plate.tif").write_bytes(b"one-image")
    output.mkdir()

    startup_events: list[str] = []
    command_invocations: list[Sequence[str]] = []
    submitted_record_generations: list[UUID] = []
    captured_run_apps: list[Any] = []
    scheduler = InMemoryScheduler()

    def trap_scheduler_subprocess(
        command: Sequence[str],
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        command_invocations.append(command)
        raise AssertionError(f"unexpected scheduler subprocess: {command!r}")

    def preload() -> None:
        startup_events.append("preload")

    original_builder_create = builder_package.create_app

    def create_builder_after_preload(*args: object, **kwargs: object) -> Any:
        assert startup_events == ["preload"]
        return original_builder_create(*args, **kwargs)

    original_run_create = run_console_package.create_app

    def capture_run_app(*args: object, **kwargs: object) -> Any:
        app = original_run_create(*args, **kwargs)
        captured_run_apps.append(app)
        return app

    def mock_submitter(
        state: RunConsoleState,
        *,
        sandbox_root: Path,
        record_generation: UUID,
    ) -> SlurmSubmitResult:
        assert sandbox_root == sandbox.root
        submitted_record_generations.append(record_generation)
        return _write_mock_submission(
            state,
            record_generation=record_generation,
            lifecycle_mode=lifecycle_mode,
            job_id=job_id,
        )

    monkeypatch.setattr(
        _cli_preload,
        "preload_custom_operation_modules",
        preload,
    )
    monkeypatch.setattr(
        builder_package,
        "create_app",
        create_builder_after_preload,
    )
    monkeypatch.setattr(
        run_console_package,
        "create_app",
        capture_run_app,
    )
    monkeypatch.setattr(subprocess, "run", trap_scheduler_subprocess)

    shell_app, _viewer = compose_hub(
        sandbox,
        start_idle_thread=False,
        start_slurm_observer=False,
        slurm_scheduler=scheduler,
        slurm_submitter=mock_submitter,
    )

    assert startup_events == ["preload"]
    assert len(captured_run_apps) == 1
    registry = shell_app.server.config[CFG_RUN_REGISTRY]
    observer = shell_app.server.extensions[SLURM_OBSERVER_EXTENSION]
    assert isinstance(registry, RunRegistry)
    assert isinstance(observer, SlurmLifecycleObserver)
    assert observer.registry is registry
    assert observer.scheduler is scheduler

    click_action = _callback_by_name(captured_run_apps[0], "click_action")
    response = click_action(
        0,
        1,
        *_slurm_action_controls(
            sandbox,
            pipeline=pipeline,
            images=images,
            output=output,
        ),
        0,
    )
    run_id = response[4]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        record = registry.get(run_id)
        if record is not None and record.status == "queued":
            break
        time.sleep(0.01)
    else:
        pytest.fail("injected submitter did not publish a queued generation")

    assert isinstance(record, RunRecord)
    assert record.generation is not None
    assert submitted_record_generations == [record.generation]
    assert response[6]["generation"] == str(record.generation)
    binding = observer.proven_binding(record)
    assert binding is not None
    assert binding.record_generation == record.generation
    assert binding.scheduler_generation != record.generation
    assert record.lifecycle_epoch == binding.scheduler_epoch

    scheduler.states[job_id] = "RUNNING"
    assert observer.observe_once(run_id) == 1
    running = registry.get(run_id)
    assert running is not None
    assert running.generation == record.generation
    assert running.status == "running"
    assert running.scheduler_ids == (job_id,)
    assert scheduler.queries == [(job_id,)]
    assert command_invocations == []
