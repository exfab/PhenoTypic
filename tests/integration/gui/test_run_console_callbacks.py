"""Phase 6 regression tests for the Run console callback wiring.

Targets the bugs found in the Phase 6 post-impl review:

* H3: Local re-run on the same output dir would `RuntimeError` because
  the prior handle was never reaped (`runner.reap` is caller-driven).
  The Run callback now reaps before starting.
* C2: Validate runs registered with ``mode="local"`` blocked subsequent
  Local runs via the concurrency cap. They now register as
  ``mode="validate"`` and ``_local_run_active`` excludes them.
* H4: ``refresh_recents`` previously took ``RC_INTERVAL_LOG.n_intervals``
  as an Input, walking the sandbox every second. It now subscribes
  only to ``RC_STORE_RECENTS_REFRESH``.

These tests poke the implementation directly (not through Dash's
``/_dash-update-component`` round trip) where possible, since Dash's
callback dispatch isn't the seam that breaks. The aim is fast unit-ish
coverage for the post-review fixes.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from dash import no_update

import phenotypic.gui.run_console._app as app_module
import phenotypic.gui.run_console._callbacks as callbacks_module
from phenotypic.gui.run_console._callbacks import (
    _SlurmLogTailCache,
    _action_control_states,
    _local_run_active,
    _resolved_output_identity,
    _run_receipt,
    _state_from_action_controls,
    _track_pending_slurm,
)
from phenotypic.gui.run_console._state import RunConsoleState
from phenotypic.gui.run_console._slurm import SlurmSubmitResult
from phenotypic.gui.run_console._slurm import (
    SlurmSubmitPending,
    SubmittedJobSet,
)
from phenotypic.gui.run_console._slurm_observer import SlurmLifecycleObserver
from phenotypic.gui.run_console._runner import LocalRunner
from phenotypic.gui.run_console._app import create_app
from phenotypic.gui.run_console._request_safety import (
    build_metadata_preflight,
    confirm_output_target,
)
from phenotypic.gui.shell._runs_registry import RunRecord, RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic._cli._cli_slurm_lifecycle import (
    CancellationResult,
    append_lifecycle_entry,
    initialize_slurm_lifecycle,
)
from phenotypic.sdk_ import atomic_write_json, job_metadata_path


def _durable_submission(
    output_dir: Path,
    *,
    job_id: str = "701",
    role: str = "controller-initial",
) -> SubmittedJobSet:
    """Write one exact lifecycle generation and return its typed job set."""
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=generation.hex,
        mode="ordinary",
    )
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            "slurm_generation": generation.hex,
            "slurm_job_ids": {
                role: {
                    "job_id": job_id,
                    "role": role,
                    "generation": generation.hex,
                }
            },
            "chunk_job_ids": {},
        },
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token=role,
        role=role,
        status="submitted",
        job_id=job_id,
    )
    return SubmittedJobSet(
        primary_id=job_id,
        all_ids=(job_id,),
        roles={role: (job_id,)},
        generation=generation,
    )


@pytest.fixture()
def runner() -> LocalRunner:
    return LocalRunner()


@pytest.fixture()
def registry() -> RunRegistry:
    return RunRegistry()


def test_resolved_output_identity_uses_posix_registry_path(
    tmp_path: Path,
) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    output_dir = tmp_path / "nested" / "run"

    resolved, rel_path = _resolved_output_identity(
        RunConsoleState(output_dir=str(output_dir)),
        sandbox=sandbox,
    )

    assert resolved == output_dir
    assert rel_path == "nested/run"


def _callback_by_name(app: Any, name: str) -> Any:
    """Return one unwrapped Dash callback by its Python function name."""
    return next(
        callback.__wrapped__
        for spec in app.callback_map.values()
        if (callback := spec.get("callback")) is not None
        and callback.__wrapped__.__name__ == name
    )


def _guard_action_controls(
    sandbox: SandboxRoot,
    controls: tuple[Any, ...],
    *,
    metadata_choice: str = "omit",
    acknowledgement: list[str] | None = None,
) -> tuple[Any, ...]:
    """Add current output and metadata receipts to the legacy control prefix."""
    assert len(controls) == 20
    preflight = build_metadata_preflight(
        sandbox,
        controls[1],
        controls[19],
    )
    confirmation = confirm_output_target(sandbox, controls[2])
    return (
        *controls,
        metadata_choice,
        acknowledgement or [],
        preflight.to_json(),
        confirmation.to_json(),
    )


# ---------------------------------------------------------------------------
# H3: rerun the same output dir
# ---------------------------------------------------------------------------


def test_local_runner_reap_unblocks_rerun(
    runner: LocalRunner,
    tmp_path: Path,
) -> None:
    """Reaping a completed handle drops it so the same run_id can re-start.

    Reap must present the retained generation, so a stale caller cannot drop
    a later same-id handle before the replacement starts.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    argv = [sys.executable, "-c", "print('first')"]
    first_generation = uuid4()
    handle = runner.start(
        "plate-0",
        argv,
        output_dir=output_dir,
        generation=first_generation,
    )
    handle.process.wait(timeout=5.0)

    # Without reap, start() would refuse the second invocation.
    with pytest.raises(RuntimeError):
        runner.start(
            "plate-0",
            argv,
            output_dir=output_dir,
            generation=first_generation,
        )

    # Reaping drops the prior handle.
    rc = runner.reap("plate-0", generation=first_generation)
    assert rc == 0
    second_generation = uuid4()
    handle2 = runner.start(
        "plate-0",
        argv,
        output_dir=output_dir,
        generation=second_generation,
    )
    handle2.process.wait(timeout=5.0)
    runner.reap("plate-0", generation=second_generation)


# ---------------------------------------------------------------------------
# C2: validate runs do not block Local runs
# ---------------------------------------------------------------------------


def test_local_run_active_excludes_validate_records(
    runner: LocalRunner,
    registry: RunRegistry,
    tmp_path: Path,
) -> None:
    """``_local_run_active`` ignores ``mode="validate"`` records.

    Previously a long-running dry-run probe registered as
    ``mode="local"`` would block the Run button via the concurrency cap.
    The fix tags validate records distinctly so the cap only considers
    real Local runs.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    validate_generation = uuid4()
    handle = runner.start(
        "validate-1",
        [sys.executable, "-c", "import time; time.sleep(2)"],
        output_dir=output_dir,
        generation=validate_generation,
    )
    registry.register(
        RunRecord(
            run_id="validate-1",
            mode="validate",
            output_dir=output_dir,
            rel_path="out",
            generation=validate_generation,
            status="running",
        )
    )
    local_generation = uuid4()
    try:
        assert (
            runner.is_running(
                "validate-1",
                generation=validate_generation,
            )
            is True
        )
        assert _local_run_active(runner, registry) is False
        registry.register(
            RunRecord(
                run_id="local-1",
                mode="local",
                output_dir=output_dir,
                rel_path="out",
                generation=local_generation,
                status="running",
            )
        )
        assert _local_run_active(runner, registry) is False
        runner.start(
            "local-1",
            [sys.executable, "-c", "import time; time.sleep(2)"],
            output_dir=output_dir,
            generation=local_generation,
        )
        assert _local_run_active(runner, registry) is True
    finally:
        runner.stop(
            "validate-1",
            generation=validate_generation,
            grace_seconds=0.1,
        )
        runner.stop(
            "local-1",
            generation=local_generation,
            grace_seconds=0.1,
        )
        runner.reap("validate-1", generation=validate_generation)
        runner.reap("local-1", generation=local_generation)
        del handle


# ---------------------------------------------------------------------------
# Async SLURM submit infrastructure
# ---------------------------------------------------------------------------


def test_slurm_future_terminalizes_stable_record_without_browser_poll(
    tmp_path: Path,
) -> None:
    """The future callback updates its allocated generation immediately."""
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="submitting",
    )
    assert record.generation is not None
    jobs = _durable_submission(output_dir)
    observer = SlurmLifecycleObserver(registry)
    future: Future[SlurmSubmitResult] = Future()
    _track_pending_slurm(
        record.run_id,
        record.generation,
        future,
        registry=registry,
        observer=observer,
    )

    future.set_result(
        SlurmSubmitResult(
            job_id="701",
            output_dir=output_dir,
            stdout="",
            stderr="",
            returncode=0,
            submitted_jobs=jobs,
        )
    )

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "queued"
    assert updated.scheduler_ids == ("701",)
    assert updated.primary_scheduler_id == "701"


def test_slurm_future_cannot_update_replaced_generation(
    tmp_path: Path,
) -> None:
    """A late callback cannot write through a stale launch generation."""
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    current = RunRecord(
        run_id="out",
        generation=uuid4(),
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        status="submitting",
        command_digest="current",
    )
    registry.register(current)
    stale_generation = uuid4()
    jobs = _durable_submission(output_dir, job_id="702")
    observer = SlurmLifecycleObserver(registry)
    future: Future[SlurmSubmitResult] = Future()
    _track_pending_slurm(
        "out",
        stale_generation,
        future,
        registry=registry,
        observer=observer,
    )

    future.set_result(
        SlurmSubmitResult(
            "702",
            output_dir,
            "",
            "",
            0,
            submitted_jobs=jobs,
        )
    )

    unchanged = registry.get("out")
    assert unchanged is not None
    assert unchanged.generation == current.generation
    assert unchanged.status == "submitting"
    assert unchanged.scheduler_ids == ()


def test_slurm_future_failure_preserves_diagnostic_record(
    tmp_path: Path,
) -> None:
    """Submission failures terminalize instead of deleting their run row."""
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="submitting",
    )
    assert record.generation is not None
    observer = SlurmLifecycleObserver(registry)
    future: Future[SlurmSubmitResult] = Future()
    _track_pending_slurm(
        record.run_id,
        record.generation,
        future,
        registry=registry,
        observer=observer,
    )

    future.set_exception(RuntimeError("scheduler unavailable"))

    failed = registry.get(record.run_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.status_detail == "RuntimeError: scheduler unavailable"


def test_pending_timeout_binds_generation_and_remains_nonterminal(
    tmp_path: Path,
) -> None:
    """An unresolved timeout is observer-owned, not a submission failure."""
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="submitting",
    )
    assert record.generation is not None
    scheduler_generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=scheduler_generation.hex,
        mode="ordinary",
    )
    append_lifecycle_entry(
        output_dir,
        generation=scheduler_generation.hex,
        token="controller-initial",
        role="controller-initial",
        status="intent",
    )
    observer = SlurmLifecycleObserver(registry)
    future: Future[SlurmSubmitResult] = Future()
    _track_pending_slurm(
        record.run_id,
        record.generation,
        future,
        registry=registry,
        observer=observer,
    )

    future.set_exception(
        SlurmSubmitPending(
            output_dir=output_dir,
            generation=scheduler_generation,
            unresolved_tokens=("controller-initial",),
            submitted_jobs=None,
            scheduler_available=False,
            returncode=-1,
        )
    )

    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "unknown"
    assert updated.terminal_at is None
    assert "remains recoverable" in (updated.status_detail or "")
    assert (record.run_id, record.generation) in observer._bindings  # noqa: SLF001


@pytest.mark.parametrize("lifecycle_mode", ["ordinary", "staged"])
def test_slurm_cancel_fences_exact_epoch_and_stays_cancelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_mode: str,
) -> None:
    """Ordinary and staged Cancel share the observer-owned semantics."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    scheduler_generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=str(scheduler_generation),
        mode=lifecycle_mode,
    )
    observer = SlurmLifecycleObserver(registry)
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            "gui_record_generation": str(record.generation),
            "slurm_generation": str(scheduler_generation),
            "slurm_job_ids": {},
            "chunk_job_ids": {},
        },
    )
    assert observer.reconcile_durable_binding(record) is True
    calls: list[tuple[Path, str]] = []

    def fake_cancel(output: Path, generation: str, **_kwargs: object) -> Any:
        calls.append((output, generation))
        return CancellationResult(("801",), (), False)

    monkeypatch.setattr(
        "phenotypic._cli._cli_slurm_lifecycle.cancel_generation",
        fake_cancel,
    )
    app = create_app(
        sandbox,
        registry=registry,
        slurm_observer=observer,
        start_slurm_observer=False,
    )

    response = _callback_by_name(app, "click_cancel")(
        1,
        _run_receipt(record),
        0,
    )

    assert calls == [(output_dir, str(scheduler_generation))]
    updated = registry.get(record.run_id)
    assert updated is not None
    assert updated.status == "cancelling"
    assert updated.terminal_at is None
    assert response[-1] == 1


def test_cancel_before_new_epoch_never_adopts_stale_output_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early Cancel waits for the submitter's exact new durable generation."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    stale_generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=stale_generation.hex,
        mode="ordinary",
    )
    stale_state_path = (
        output_dir / ".phenotypic" / "progress" / "slurm_lifecycle.json"
    )
    stale_state = __import__("json").loads(
        stale_state_path.read_text(encoding="utf-8")
    )
    stale_state["active"] = False
    atomic_write_json(stale_state_path, stale_state)
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            "gui_record_generation": str(uuid4()),
            "slurm_generation": stale_generation.hex,
            "slurm_job_ids": {},
            "chunk_job_ids": {},
        },
    )
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="new-command",
        status="submitting",
    )
    assert record.generation is not None
    observer = SlurmLifecycleObserver(registry)
    future: Future[Any] = Future()
    assert future.set_running_or_notify_cancel() is True
    _track_pending_slurm(
        record.run_id,
        record.generation,
        future,
        registry=registry,
        observer=observer,
    )
    calls: list[str] = []

    def fake_cancel(
        _output: Path,
        generation: str,
        **_kwargs: object,
    ) -> CancellationResult:
        calls.append(generation)
        return CancellationResult(("9901", "9902"), (), False)

    monkeypatch.setattr(
        "phenotypic._cli._cli_slurm_lifecycle.cancel_generation",
        fake_cancel,
    )
    app = create_app(
        sandbox,
        registry=registry,
        slurm_observer=observer,
        start_slurm_observer=False,
    )

    response = _callback_by_name(app, "click_cancel")(
        1,
        _run_receipt(record),
        0,
    )

    assert calls == []
    assert response[-1] == 1
    cancelling = registry.get(record.run_id)
    assert cancelling is not None and cancelling.status == "cancelling"
    assert cancelling.lifecycle_epoch == str(record.generation)
    assert cancelling.lifecycle_epoch != stale_generation.hex

    scheduler_generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir,
        generation=scheduler_generation.hex,
        mode="ordinary",
    )
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            "gui_record_generation": str(record.generation),
            "slurm_generation": scheduler_generation.hex,
            "slurm_job_ids": {
                "chunk-0": {
                    "job_id": "9901",
                    "role": "chunk",
                    "generation": scheduler_generation.hex,
                },
                "finalizer": {
                    "job_id": "9902",
                    "role": "finalizer",
                    "generation": scheduler_generation.hex,
                },
            },
            "chunk_job_ids": {"0": "9901"},
        },
    )
    jobs = SubmittedJobSet(
        primary_id="9901",
        all_ids=("9901", "9902"),
        roles={"chunk": ("9901",), "finalizer": ("9902",)},
        generation=scheduler_generation,
    )
    future.set_result(
        SlurmSubmitResult(
            job_id="9901",
            output_dir=output_dir,
            stdout="",
            stderr="",
            returncode=0,
            submitted_jobs=jobs,
        )
    )

    assert calls == [scheduler_generation.hex]
    assert stale_generation.hex not in calls
    bound = registry.get(record.run_id)
    assert bound is not None
    assert bound.status == "cancelling"
    assert bound.lifecycle_epoch == scheduler_generation.hex


def test_stale_cancel_receipt_cannot_stop_replacement_generation(
    tmp_path: Path,
) -> None:
    """A stale browser receipt cannot stop a newer run with the same run id."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    runner = LocalRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    predecessor = registry.allocate(
        mode="local",
        output_dir=output_dir,
        rel_path="out",
        command_digest="first",
        status="complete",
    )
    stale_receipt = _run_receipt(predecessor)
    replacement = registry.allocate(
        mode="local",
        output_dir=output_dir,
        rel_path="out",
        command_digest="second",
        status="queued",
    )
    assert replacement.generation is not None
    handle = runner.start(
        replacement.run_id,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        output_dir=output_dir,
        generation=replacement.generation,
    )
    registry.compare_and_set(
        replacement.run_id,
        replacement.generation,
        expected_statuses={"queued"},
        status="running",
        pid=handle.process.pid,
    )
    app = create_app(
        sandbox,
        registry=registry,
        runner=runner,
        start_slurm_observer=False,
    )

    try:
        response = _callback_by_name(app, "click_cancel")(
            1,
            stale_receipt,
            0,
        )

        assert "no longer current" in response[1]
        assert runner.is_running(
            replacement.run_id,
            generation=replacement.generation,
        )
        current = registry.get(replacement.run_id)
        assert current is not None and current.status == "running"
    finally:
        runner.stop(
            replacement.run_id,
            generation=replacement.generation,
            grace_seconds=0.1,
        )


def test_cancel_disabled_tracks_exact_generation_terminal_state(
    tmp_path: Path,
) -> None:
    """Terminal publication disables Cancel without clearing the receipt."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    record = registry.allocate(
        mode="local",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    receipt = _run_receipt(record)
    app = create_app(
        sandbox,
        registry=registry,
        start_slurm_observer=False,
    )
    callback = _callback_by_name(app, "update_cancel_disabled")

    assert callback(0, receipt) is False
    registry.compare_and_set(
        record.run_id,
        record.generation,
        expected_statuses={"running"},
        status="complete",
    )
    assert callback(1, receipt) is True


def test_run_app_owns_one_startable_observer_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app starts, stores, and registers cleanup for the exact observer."""

    class ObserverSpy:
        def __init__(self) -> None:
            self.starts = 0
            self.stops = 0

        def start(self) -> None:
            self.starts += 1

        def stop(self, *, timeout: float = 5.0) -> None:
            del timeout
            self.stops += 1

        def bind_generation(self, **_kwargs: object) -> None:
            return None

        def reconcile_durable_bindings(self) -> int:
            return 0

    observer = ObserverSpy()
    cleanup_callbacks: list[Any] = []
    monkeypatch.setattr(
        app_module.atexit,
        "register",
        lambda callback: cleanup_callbacks.append(callback),
    )
    app = create_app(
        SandboxRoot.from_path(tmp_path),
        slurm_observer=observer,  # type: ignore[arg-type]
        start_slurm_observer=True,
    )

    assert app.server.extensions["phenotypic_slurm_observer"] is observer
    assert observer.starts == 1
    assert cleanup_callbacks == [observer.stop]
    cleanup_callbacks[0]()
    assert observer.stops == 1


def test_staged_gpu_controls_follow_pipeline_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GPU-stage profile is visible only for SLURM GPU pipelines."""
    app = create_app(SandboxRoot.from_path(tmp_path))
    monkeypatch.setattr(
        callbacks_module,
        "_pipeline_uses_staged_gpu",
        lambda path: path == "/gpu.json",
    )
    callback = _callback_by_name(app, "show_staged_gpu_controls")

    assert callback("/gpu.json", "slurm") == {"display": "block"}
    assert callback("/cpu.json", "slurm") == {"display": "none"}
    assert callback("/gpu.json", "local") == {"display": "none"}


def test_terminal_no_dashboard_surfaces_detail_and_manual_refresh(
    tmp_path: Path,
) -> None:
    """Terminal output remains diagnosable and can be checked on demand."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="failed",
    )
    assert record.generation is not None
    registry.compare_and_set(
        record.run_id,
        record.generation,
        status_detail="finalizer exited before dashboard publication",
    )
    app = create_app(sandbox, registry=registry)
    callback = _callback_by_name(app, "poll_dashboard")

    receipt = _run_receipt(record)
    missing = callback(1, 0, "out", receipt)

    assert missing[1] == {"display": "none"}
    assert "finalizer exited" in missing[3]
    assert missing[4] is True

    dashboard = output_dir / "deliverables" / "dashboard.html"
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text("<html></html>", encoding="utf-8")
    refreshed = callback(1, 1, "out", receipt)

    assert refreshed[0].endswith("/runs/out/deliverables/dashboard.html")
    assert refreshed[1] == {"display": "block"}
    assert refreshed[4] is True


def test_terminal_slurm_generation_evicts_reader_and_retains_polling(
    tmp_path: Path,
) -> None:
    """Terminal transitions evict readers while the UI keeps its receipt."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    log_path = output_dir / "worker.log"
    log_path.write_text("running\n", encoding="utf-8")
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    registry.compare_and_set(
        record.run_id,
        record.generation,
        log_paths=(log_path,),
    )
    record = registry.get(record.run_id)
    assert record is not None
    cache = _SlurmLogTailCache()

    assert "running" in cache.read(record)
    assert cache.tracked_generations == 1

    registry.compare_and_set(
        record.run_id,
        record.generation,
        status="complete",
    )
    terminal = registry.get(record.run_id)
    assert terminal is not None
    assert "running" in cache.read(terminal)
    assert cache.tracked_generations == 0

    app = create_app(
        sandbox,
        registry=registry,
        start_slurm_observer=False,
    )
    response = _callback_by_name(app, "resolve_pending_slurm")(
        1,
        _run_receipt(record),
        0,
    )
    assert response == (no_update,) * 5


def test_slurm_log_cache_is_globally_bounded_and_prunes_other_runs(
    tmp_path: Path,
) -> None:
    """Reading one run evicts terminal and superseded generations globally."""
    registry = RunRegistry()
    records: list[RunRecord] = []
    for index in range(3):
        output_dir = tmp_path / f"out-{index}"
        output_dir.mkdir()
        log_path = output_dir / "worker.log"
        log_path.write_text(f"run-{index}\n", encoding="utf-8")
        record = RunRecord(
            run_id=f"run-{index}",
            generation=uuid4(),
            mode="slurm",
            output_dir=output_dir,
            rel_path=output_dir.name,
            status="running",
            log_paths=(log_path,),
        )
        registry.register(record, persist=False)
        records.append(record)

    cache = _SlurmLogTailCache(registry, max_generations=2)
    for record in records:
        assert record.run_id in cache.read(record)
    assert cache.tracked_generations == 2
    assert (records[0].run_id, records[0].generation) not in cache._readers  # noqa: SLF001

    records[1].status = "complete"
    registry.register(records[1], persist=False)
    replacement = RunRecord(
        run_id=records[2].run_id,
        generation=uuid4(),
        mode="slurm",
        output_dir=records[2].output_dir,
        rel_path=records[2].rel_path,
        status="running",
        log_paths=records[2].log_paths,
    )
    registry.register(replacement, persist=False)

    assert records[0].run_id in cache.read(records[0])
    assert cache.tracked_generations == 1
    assert set(cache._readers) == {  # noqa: SLF001
        (records[0].run_id, records[0].generation)
    }


def test_slurm_log_callback_reads_incrementally(
    tmp_path: Path,
) -> None:
    """Scheduler log polling appends new bytes without rereading old content."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    log_path = output_dir / ".phenotypic" / "logs" / "gui" / "stdout.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("alpha\n", encoding="utf-8")
    record = registry.allocate(
        mode="slurm",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="submitting",
    )
    assert record.generation is not None
    app = create_app(sandbox, registry=registry)
    callback = _callback_by_name(app, "update_log_tail")

    receipt = _run_receipt(record)
    first, _banner = callback(1, receipt)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("beta\n")
    second, _banner = callback(2, receipt)

    assert "alpha" in first
    assert second.count("alpha") == 1
    assert "beta" in second


def test_recents_redraw_uses_registry_revision_without_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle changes redraw cached registry rows without a sandbox walk."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    app = create_app(sandbox, registry=registry)
    publish = _callback_by_name(app, "publish_registry_revision")
    refresh = _callback_by_name(app, "refresh_recents")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    registry.allocate(
        mode="local",
        output_dir=output_dir,
        rel_path="out",
        command_digest="digest",
        status="running",
    )
    monkeypatch.setattr(
        registry,
        "rehydrate_from_sandbox",
        lambda *_args, **_kwargs: pytest.fail("unexpected sandbox rescan"),
    )

    revision = publish(1, 0)
    rows, _presets = refresh(revision)

    assert revision == registry.revision
    assert rows


def test_action_callbacks_capture_raw_controls_not_aggregate_store(
    tmp_path: Path,
) -> None:
    """Run, Validate, and Save Preset have the full raw control contract."""
    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(sandbox)
    expected = [
        dependency.component_id for dependency in _action_control_states()
    ]
    for action_id in (
        "rc-btn-run",
        "rc-btn-validate",
        "rc-btn-save-preset",
    ):
        callback = next(
            spec
            for spec in app.callback_map.values()
            if spec.get("callback") is not None
            if any(item["id"] == action_id for item in spec["inputs"])
        )
        state_ids = [item["id"] for item in callback["state"]]
        assert "rc-store-form-state" not in state_ids
        start = state_ids.index(expected[0])
        assert state_ids[start : start + len(expected)] == expected


def test_preset_round_trip_restores_all_controls_for_raw_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save and Load preserve every control consumed by the Run callback."""
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
    )
    from phenotypic.schema import IMAGE

    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    app = create_app(sandbox, registry=registry)
    pipeline = tmp_path / "pipeline.json"
    images = tmp_path / "images"
    output = tmp_path / "output"
    metadata = tmp_path / "metadata.csv"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    images.mkdir()
    (images / "plate_a.tif").write_bytes(b"one-image")
    output.mkdir()
    metadata.write_text(
        f"{IMAGE.IMAGE_NAME},Treatment\nplate_a,control\n",
        encoding="utf-8",
    )
    metadata_payload = metadata_payload_from_path(sandbox, metadata)
    control_prefix = (
        str(pipeline),
        str(images),
        str(output),
        "slurm",
        ["dry_run"],
        7,
        8,
        12,
        "GridImage",
        3,
        "DEBUG",
        "compute",
        "02:30:00",
        "32G",
        8,
        2,
        "account=lab\nqos=normal",
        "slurm_partition=gpu\nslurm_account=lab",
        4,
        metadata_payload,
    )
    controls = _guard_action_controls(
        sandbox,
        control_prefix,
        metadata_choice="include",
    )

    save_callback = next(
        spec["callback"].__wrapped__
        for spec in app.callback_map.values()
        if spec["inputs"] and spec["inputs"][0]["id"] == "rc-btn-save-preset"
    )
    load_callback = next(
        spec["callback"].__wrapped__
        for spec in app.callback_map.values()
        if spec["inputs"]
        and spec["inputs"][0]["id"] == "rc-dropdown-load-preset"
    )
    run_callback = _callback_by_name(app, "click_action")

    save_response = save_callback(1, "full-slurm", *controls)
    assert save_response[1] == "Saved preset full-slurm"

    preset = tmp_path / ".phenotypic-gui" / "presets" / "full-slurm.json"
    load_response = load_callback(str(preset))
    loaded_controls = tuple(load_response[:22])

    assert loaded_controls[:19] == control_prefix[:19]
    assert loaded_controls[19] == "omit"
    assert loaded_controls[20:] == (None, None)

    captured_states = []

    def _capture_submit(*args: object, **kwargs: object) -> Future[Any]:
        state = args[1]
        captured_states.append(state)
        future: Future[Any] = Future()
        future.set_result(
            SlurmSubmitResult(
                job_id="901",
                output_dir=Path(state.output_dir),
                stdout="",
                stderr="",
                returncode=0,
            )
        )
        return future

    monkeypatch.setattr(
        callbacks_module._SLURM_EXECUTOR,
        "submit",
        _capture_submit,
    )
    reauthorized = _guard_action_controls(
        sandbox,
        (*loaded_controls[:19], metadata_payload),
        metadata_choice="include",
    )
    run_response = run_callback(0, 1, *reauthorized, 0)

    assert "SLURM submitting" in run_response[1]
    assert len(captured_states) == 1
    raw_run_state = captured_states[0]
    assert raw_run_state.advanced_args == {
        "sample": 7,
        "nrows": 8,
        "ncols": 12,
        "image_type": "GridImage",
        "workers": 3,
        "log_level": "DEBUG",
    }
    assert raw_run_state.slurm_args == {
        "partition": "compute",
        "time": "02:30:00",
        "mem": "32G",
        "cpus_per_task": 8,
        "gpus": 2,
        "extra": {"account": "lab", "qos": "normal"},
    }
    assert raw_run_state.gpu_slurm_args == (
        "slurm_partition=gpu",
        "slurm_account=lab",
    )
    assert raw_run_state.gpu_shards == 4
    assert raw_run_state.metadata_csv == str(metadata)


def test_load_legacy_preset_uses_visible_control_defaults(
    tmp_path: Path,
) -> None:
    """A preset missing newer fields clears them to layout-safe defaults."""
    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(sandbox)
    preset = tmp_path / "legacy.json"
    preset.write_text("{}", encoding="utf-8")
    load_callback = next(
        spec["callback"].__wrapped__
        for spec in app.callback_map.values()
        if spec["inputs"]
        and spec["inputs"][0]["id"] == "rc-dropdown-load-preset"
    )

    response = load_callback(str(preset))

    assert response[:5] == (None, None, None, "local", [])
    assert response[5:18] == (None,) * 13
    assert response[18] == 1
    assert response[19] == "omit"
    assert response[20:22] == (None, None)
    assert response[23] == "Loaded preset legacy"


def test_raw_mode_at_action_time_is_authoritative(tmp_path: Path) -> None:
    """Derived-store lag cannot change the execution mode selected at click."""
    sandbox = SandboxRoot.from_path(tmp_path)
    pipeline = tmp_path / "pipeline.json"
    images = tmp_path / "images"
    output = tmp_path / "output"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    images.mkdir()
    output.mkdir()
    base = (
        str(pipeline),
        str(images),
        str(output),
        "local",
        [],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1,
        None,
    )

    local = _state_from_action_controls(
        _guard_action_controls(sandbox, base),
        sandbox=sandbox,
    )
    slurm_values = list(base)
    slurm_values[3] = "slurm"
    slurm_values[11] = "compute"
    slurm = _state_from_action_controls(
        _guard_action_controls(sandbox, tuple(slurm_values)),
        sandbox=sandbox,
    )

    assert local.mode == "local"
    assert slurm.mode == "slurm"
    assert slurm.slurm_args["partition"] == "compute"


def test_run_action_shows_stale_output_confirmation_error(
    tmp_path: Path,
) -> None:
    """A changed typed output is rejected visibly before durable allocation."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    app = create_app(sandbox, registry=registry)
    pipeline = tmp_path / "pipeline.json"
    images = tmp_path / "images"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    images.mkdir()
    (images / "plate_a.tif").write_bytes(b"one-image")
    controls = _guard_action_controls(
        sandbox,
        (
            str(pipeline),
            str(images),
            str(tmp_path / "confirmed-output"),
            "local",
            [],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
    )
    stale_controls = list(controls)
    stale_controls[2] = str(tmp_path / "changed-after-confirm")
    callback = _callback_by_name(app, "click_action")

    response = callback(0, 1, *stale_controls, 0)

    assert "stale" in response[1]
    assert registry.list() == []


def test_validate_action_shows_changed_source_fingerprint_error(
    tmp_path: Path,
) -> None:
    """A changed source snapshot is rejected visibly before validation spawn."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    app = create_app(sandbox, registry=registry)
    pipeline = tmp_path / "pipeline.json"
    images = tmp_path / "images"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    images.mkdir()
    image = images / "plate_a.tif"
    image.write_bytes(b"one-image")
    controls = _guard_action_controls(
        sandbox,
        (
            str(pipeline),
            str(images),
            str(tmp_path / "validation-output"),
            "local",
            [],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
    )
    image.write_bytes(b"changed-source")
    callback = _callback_by_name(app, "click_action")

    response = callback(1, 0, *controls, 0)

    assert "changed after preflight" in response[1]
    assert registry.list() == []


def test_empty_slurm_profile_never_invokes_submitter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SLURM validation fails before executor submission or durable claim."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    app = create_app(sandbox, registry=registry)
    pipeline = tmp_path / "pipeline.json"
    images = tmp_path / "images"
    output = tmp_path / "output"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    images.mkdir()
    output.mkdir()
    controls = _guard_action_controls(
        sandbox,
        (
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
    )
    monkeypatch.setattr(
        callbacks_module._SLURM_EXECUTOR,
        "submit",
        lambda *args, **kwargs: pytest.fail("submitter was invoked"),
    )
    callback = _callback_by_name(app, "click_action")

    response = callback(0, 1, *controls, 0)

    assert "nonempty CPU SLURM profile" in response[1]
    assert registry.list() == []


def test_immediate_local_exit_terminalizes_allocated_generation(
    tmp_path: Path,
) -> None:
    """A process that exits before the Dash response still becomes terminal."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    runner = LocalRunner()
    app = create_app(sandbox, registry=registry, runner=runner)
    pipeline = tmp_path / "invalid-pipeline.json"
    images = tmp_path / "images"
    output = tmp_path / "output"
    pipeline.write_text('{"operations": "invalid"}', encoding="utf-8")
    images.mkdir()
    output.mkdir()
    controls = _guard_action_controls(
        sandbox,
        (
            str(pipeline),
            str(images),
            str(output),
            "local",
            [],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
    )
    callback = _callback_by_name(app, "click_action")

    response = callback(0, 1, *controls, 0)
    run_id = response[4]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        record = registry.get(run_id)
        if record is not None and record.status in {"complete", "failed"}:
            break
        time.sleep(0.02)
    else:
        pytest.fail("local exit observer did not terminalize the run")

    assert record is not None
    assert record.generation is not None
    assert record.status == "failed"
    assert record.returncode not in {None, 0}


def test_local_record_is_durable_before_spawn_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Popen-boundary failures retain a failed generation owner."""
    sandbox = SandboxRoot.from_path(tmp_path)
    registry = RunRegistry()
    runner = LocalRunner()
    app = create_app(sandbox, registry=registry, runner=runner)
    pipeline = tmp_path / "pipeline.json"
    images = tmp_path / "images"
    output = tmp_path / "output"
    pipeline.write_text('{"operations": []}', encoding="utf-8")
    images.mkdir()
    output.mkdir()
    controls = _guard_action_controls(
        sandbox,
        (
            str(pipeline),
            str(images),
            str(output),
            "local",
            [],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            None,
        ),
    )

    def fail_after_claim(*args, **kwargs):
        claimed = registry.get("output")
        assert claimed is not None
        assert claimed.status == "queued"
        assert claimed.generation is not None
        raise OSError("Popen failed")

    monkeypatch.setattr(runner, "start", fail_after_claim)
    callback = _callback_by_name(app, "click_action")

    callback(0, 1, *controls, 0)

    failed = registry.get("output")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.status_detail == "OSError: Popen failed"


# ---------------------------------------------------------------------------
# Shared descriptor consumption
# ---------------------------------------------------------------------------


def test_run_console_has_no_passive_shared_authority_writer(
    tmp_path: Path,
) -> None:
    """Run may consume shell descriptors but never publish either authority."""
    from phenotypic.gui.shell._ids import (
        SHELL_METADATA_CSV_STORE,
        SHELL_SOURCE_IMAGE_ROOT_STORE,
    )

    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(sandbox)
    outputs = [str(spec["output"]) for spec in app.callback_map.values()]

    assert not any(
        SHELL_SOURCE_IMAGE_ROOT_STORE in output for output in outputs
    )
    assert not any(SHELL_METADATA_CSV_STORE in output for output in outputs)


def test_shared_source_initializes_empty_run_input(tmp_path: Path) -> None:
    from phenotypic.gui.run_console._callbacks import (
        _input_dir_from_shared_source,
    )
    from phenotypic.gui.shell._source_context import source_payload_from_path

    plates = tmp_path / "plates"
    plates.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, plates, source="manual")

    assert _input_dir_from_shared_source(sandbox, payload, None) == str(
        plates.resolve()
    )


def test_shared_source_does_not_overwrite_non_empty_run_input(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.run_console._callbacks import (
        _input_dir_from_shared_source,
    )
    from phenotypic.gui.shell._source_context import source_payload_from_path

    plates = tmp_path / "plates"
    existing = tmp_path / "existing"
    plates.mkdir()
    existing.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = source_payload_from_path(sandbox, plates, source="manual")

    assert (
        _input_dir_from_shared_source(sandbox, payload, str(existing)) is None
    )


def test_form_state_omits_ambient_metadata_until_explicit_include(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.run_console._callbacks import _form_inputs_to_state
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
    )
    from phenotypic.schema import IMAGE

    csv_path = tmp_path / "layout.csv"
    csv_path.write_text(f"{IMAGE.IMAGE_NAME},Treatment\nplate_a,control\n")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)

    omitted = _form_inputs_to_state(
        "/p.json",
        "/in",
        "/out",
        "local",
        [],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        metadata_payload=payload,
        sandbox=sandbox,
    )
    included = _form_inputs_to_state(
        "/p.json",
        "/in",
        "/out",
        "local",
        [],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        metadata_payload=payload,
        metadata_choice="include",
        sandbox=sandbox,
    )

    assert omitted["metadata_csv"] is None
    assert included["metadata_csv"] == str(csv_path.resolve())
