"""Race and recovery tests for the shared SLURM lifecycle."""

from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import pytest

import phenotypic._cli._cli_slurm_lifecycle as lifecycle
from phenotypic._cli._cli_slurm_lifecycle import (
    SlurmDependencyKind,
    append_lifecycle_entry,
    cancel_generation,
    initialize_slurm_lifecycle,
    load_slurm_lifecycle,
    mirror_job_to_metadata,
    read_lifecycle_ledger,
    submit_with_lifecycle,
)
from phenotypic.sdk_ import (
    JobMetadataKey,
    atomic_write_json,
    job_metadata_path,
)


class FakeScheduler:
    """Small scheduler fake keyed by deterministic submission comment."""

    def __init__(self) -> None:
        self.jobs: dict[str, str] = {}
        self.cancelled: set[str] = set()
        self.sbatch_commands: list[list[str]] = []
        self.next_id = 700
        self.timeout_after_accept = False

    def __call__(self, command, **kwargs):
        executable = command[0]
        if executable == "sbatch":
            self.sbatch_commands.append(list(command))
            comment = command[command.index("--comment") + 1]
            self.next_id += 1
            job_id = str(self.next_id)
            self.jobs[comment] = job_id
            if self.timeout_after_accept:
                self.timeout_after_accept = False
                raise subprocess.TimeoutExpired(command, 30)
            return subprocess.CompletedProcess(command, 0, f"{job_id}\n", "")
        if executable in {"squeue", "sacct"}:
            lines = [
                f"{job_id}|{comment}"
                for comment, job_id in self.jobs.items()
                if job_id not in self.cancelled
            ]
            return subprocess.CompletedProcess(
                command, 0, "\n".join(lines), ""
            )
        if executable == "scancel":
            self.cancelled.update(command[1:])
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)


def _metadata_skeleton(output_dir: Path) -> None:
    path = job_metadata_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            JobMetadataKey.CHUNK_JOB_IDS: {},
            JobMetadataKey.SLURM_JOB_IDS: {},
        },
    )


def test_generation_publication_guard_uses_five_minute_lock_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Concurrent worker completion receives the 300-second lock budget."""
    generation = "generation-lock-timeout"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="recompile"
    )
    acquisitions: list[tuple[Path, float]] = []

    def record_lock(path: Path, *, timeout: float):
        acquisitions.append((path, timeout))
        return nullcontext()

    monkeypatch.setattr(lifecycle, "exclusive_path_lock", record_lock)

    with lifecycle.generation_publication_guard(tmp_path, generation):
        pass

    assert acquisitions == [(lifecycle.lifecycle_lock_path(tmp_path), 300.0)]


def test_intent_precedes_sbatch_and_job_record(monkeypatch, tmp_path) -> None:
    generation = "generation-1"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    _metadata_skeleton(tmp_path)
    observed_statuses: list[list[str]] = []
    observed_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        observed_commands.append(list(command))
        observed_statuses.append(
            [str(row["status"]) for row in read_lifecycle_ledger(tmp_path)]
        )
        return subprocess.CompletedProcess(command, 0, "701\n", "")

    job_id = submit_with_lifecycle(
        tmp_path,
        generation=generation,
        token="chunk-0",
        role="chunk",
        script_path=tmp_path / "chunk.sh",
        run_command=fake_run,
        discover=lambda comment: None,
    )

    assert job_id == "701"
    assert observed_statuses == [["intent"]]
    assert observed_commands == [
        [
            "sbatch",
            "--parsable",
            "--export=ALL",
            "--comment",
            "phenotypic:generation-1:chunk-0",
            str(tmp_path / "chunk.sh"),
        ]
    ]
    assert [row["status"] for row in read_lifecycle_ledger(tmp_path)] == [
        "intent",
        "submitted",
    ]


def test_submission_snapshots_pythonpath_for_export_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The sbatch process should receive a namespaced Python path snapshot."""
    from phenotypic.sdk_.slurm import SLURM_PYTHONPATH_ENV_VAR

    generation = "generation-pythonpath"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    monkeypatch.setenv(
        "PYTHONPATH", "/exact/reviewed/src:/exact/reviewed/tests"
    )
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "704\n", "")

    submit_with_lifecycle(
        tmp_path,
        generation=generation,
        token="chunk-0",
        role="chunk",
        script_path=tmp_path / "chunk.sh",
        run_command=fake_run,
        discover=lambda comment: None,
    )

    assert "--export=ALL" in observed["command"]
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"] == (
        "/exact/reviewed/src:/exact/reviewed/tests"
    )
    assert environment[SLURM_PYTHONPATH_ENV_VAR] == environment["PYTHONPATH"]


def test_initialize_rejects_conflicting_active_generation(tmp_path) -> None:
    first = initialize_slurm_lifecycle(
        tmp_path, generation="generation-1", mode="ordinary"
    )

    with pytest.raises(RuntimeError, match="active SLURM generation"):
        initialize_slurm_lifecycle(
            tmp_path, generation="generation-2", mode="ordinary"
        )

    assert load_slurm_lifecycle(tmp_path) == first


def test_initialize_is_idempotent_for_same_active_generation(tmp_path) -> None:
    first = initialize_slurm_lifecycle(
        tmp_path, generation="generation-1", mode="ordinary"
    )
    second = initialize_slurm_lifecycle(
        tmp_path, generation="generation-1", mode="ordinary"
    )

    assert second == first


def test_timeout_after_accept_recovers_by_generation_comment(tmp_path) -> None:
    generation = "generation-2"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    _metadata_skeleton(tmp_path)
    scheduler = FakeScheduler()
    scheduler.timeout_after_accept = True

    job_id = submit_with_lifecycle(
        tmp_path,
        generation=generation,
        token="chunk-0",
        role="chunk",
        script_path=tmp_path / "chunk.sh",
        dependencies=("601", "602"),
        run_command=scheduler,
    )

    assert job_id == "701"
    assert scheduler.sbatch_commands == [
        [
            "sbatch",
            "--parsable",
            "--export=ALL",
            "--comment",
            "phenotypic:generation-2:chunk-0",
            "--dependency",
            "afterany:601:602",
            str(tmp_path / "chunk.sh"),
        ]
    ]
    rows = read_lifecycle_ledger(tmp_path)
    assert [row["status"] for row in rows] == ["intent", "submitted"]
    assert rows[-1]["dependencies"] == ["601", "602"]
    assert rows[-1]["dependency_kind"] == "afterany"
    metadata = json.loads(
        job_metadata_path(tmp_path).read_text(encoding="utf-8")
    )
    assert metadata[JobMetadataKey.SLURM_JOB_IDS]["chunk-0"] == {
        "job_id": "701",
        "role": "chunk",
        "generation": generation,
    }
    assert metadata[JobMetadataKey.CHUNK_JOB_IDS] == {"0": "701"}


def test_afterok_dependency_is_emitted_and_persisted(tmp_path: Path) -> None:
    """A success-only continuation uses the exact requested SLURM condition."""
    generation = "generation-afterok"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    scheduler = FakeScheduler()

    submit_with_lifecycle(
        tmp_path,
        generation=generation,
        token="dispatcher-1",
        role="dispatcher",
        script_path=tmp_path / "dispatcher.sh",
        dependencies=("601",),
        dependency_kind="afterok",
        run_command=scheduler,
    )

    assert scheduler.sbatch_commands[0][-3:] == [
        "--dependency",
        "afterok:601",
        str(tmp_path / "dispatcher.sh"),
    ]
    rows = read_lifecycle_ledger(tmp_path)
    assert [row["dependency_kind"] for row in rows] == [
        "afterok",
        "afterok",
    ]


def test_invalid_dependency_kind_fails_before_submission_or_files(
    tmp_path: Path,
) -> None:
    """Submission validation precedes locks, intents, and scheduler calls."""
    output_dir = tmp_path / "output"
    scheduler = FakeScheduler()

    with pytest.raises(ValueError, match="dependency_kind"):
        submit_with_lifecycle(
            output_dir,
            generation="generation-invalid",
            token="dispatcher-1",
            role="dispatcher",
            script_path=tmp_path / "dispatcher.sh",
            dependency_kind=cast(
                SlurmDependencyKind, "afterinvalid"
            ),
            active_check=lambda: True,
            run_command=scheduler,
        )

    assert scheduler.sbatch_commands == []
    assert not output_dir.exists()


def test_invalid_continuation_kind_submits_no_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bad continuation edge is rejected before its chunk submission."""
    calls: list[dict[str, object]] = []

    def fake_submit(_output_dir: Path, **kwargs: object) -> str:
        calls.append(kwargs)
        return "701"

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)

    with pytest.raises(ValueError, match="dependency_kind"):
        lifecycle.dispatch_continuation(
            tmp_path,
            generation="generation-invalid",
            chunk_index=1,
            chunk_script=tmp_path / "chunk.sh",
            dispatcher_script=tmp_path / "dispatcher.sh",
            dependency_kind=cast(
                SlurmDependencyKind, "afterinvalid"
            ),
        )

    assert calls == []


def test_retry_replays_journaled_afterok_dependency(tmp_path: Path) -> None:
    """A retry cannot weaken an existing success-only submission intent."""
    generation = "generation-afterok-replay"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    append_lifecycle_entry(
        tmp_path,
        generation=generation,
        token="dispatcher-1",
        role="dispatcher",
        status="intent",
        dependencies=("601",),
        dependency_kind="afterok",
    )
    scheduler = FakeScheduler()

    submit_with_lifecycle(
        tmp_path,
        generation=generation,
        token="dispatcher-1",
        role="dispatcher",
        script_path=tmp_path / "dispatcher.sh",
        dependencies=("601",),
        run_command=scheduler,
    )

    assert scheduler.sbatch_commands == [
        [
            "sbatch",
            "--parsable",
            "--export=ALL",
            "--comment",
            "phenotypic:generation-afterok-replay:dispatcher-1",
            "--dependency",
            "afterok:601",
            str(tmp_path / "dispatcher.sh"),
        ]
    ]
    assert read_lifecycle_ledger(tmp_path)[-1]["dependency_kind"] == (
        "afterok"
    )


def test_dispatch_cli_threads_afterok_to_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generated dispatcher CLI arguments retain the selected edge kind."""
    observed: dict[str, object] = {}

    def fake_dispatch(output_dir: Path, **kwargs: object) -> tuple[str, None]:
        observed["output_dir"] = output_dir
        observed.update(kwargs)
        return "701", None

    monkeypatch.setattr(lifecycle, "dispatch_continuation", fake_dispatch)

    exit_code = lifecycle._dispatch_from_argv(
        [
            "--output",
            str(tmp_path),
            "--generation",
            "generation-cli",
            "--chunk-index",
            "2",
            "--chunk-script",
            str(tmp_path / "chunk.sh"),
            "--dispatcher-script",
            str(tmp_path / "dispatcher.sh"),
            "--dependency-kind",
            "afterok",
        ]
    )

    assert exit_code == 0
    assert observed["dependency_kind"] == "afterok"


def test_next_invocation_recovers_accept_before_ledger_crash(
    monkeypatch,
    tmp_path,
) -> None:
    generation = "generation-crash"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    _metadata_skeleton(tmp_path)
    scheduler = FakeScheduler()
    real_record = lifecycle._record_submission

    def crash_before_record(*args, **kwargs):
        raise SystemExit(19)

    monkeypatch.setattr(
        lifecycle,
        "_record_submission",
        crash_before_record,
    )

    with pytest.raises(SystemExit):
        submit_with_lifecycle(
            tmp_path,
            generation=generation,
            token="chunk-0",
            role="chunk",
            script_path=tmp_path / "chunk.sh",
            run_command=scheduler,
        )

    assert [row["status"] for row in read_lifecycle_ledger(tmp_path)] == [
        "intent"
    ]
    monkeypatch.setattr(lifecycle, "_record_submission", real_record)
    recovered = submit_with_lifecycle(
        tmp_path,
        generation=generation,
        token="chunk-0",
        role="chunk",
        script_path=tmp_path / "chunk.sh",
        run_command=scheduler,
    )
    assert recovered == "701"
    assert scheduler.next_id == 701


def test_timeout_without_visible_job_does_not_duplicate_submit(
    tmp_path,
) -> None:
    generation = "generation-timeout"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    sbatch_calls = 0

    def timeout_scheduler(command, **kwargs):
        nonlocal sbatch_calls
        if command[0] == "sbatch":
            sbatch_calls += 1
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="Ambiguous"):
        submit_with_lifecycle(
            tmp_path,
            generation=generation,
            token="chunk-0",
            role="chunk",
            script_path=tmp_path / "chunk.sh",
            run_command=timeout_scheduler,
        )

    assert sbatch_calls == 1
    assert read_lifecycle_ledger(tmp_path)[-1]["status"] == "blocked"


def test_recovery_rejects_multiple_jobs_for_one_comment(tmp_path) -> None:
    generation = "generation-ambiguous"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    append_lifecycle_entry(
        tmp_path,
        generation=generation,
        token="chunk-0",
        role="chunk",
        status="intent",
    )

    def duplicate_scheduler(command, **kwargs):
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command,
                0,
                "\n".join(
                    [
                        "701|phenotypic:generation-ambiguous:chunk-0",
                        "702|phenotypic:generation-ambiguous:chunk-0",
                    ]
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="matched jobs 701, 702"):
        submit_with_lifecycle(
            tmp_path,
            generation=generation,
            token="chunk-0",
            role="chunk",
            script_path=tmp_path / "chunk.sh",
            run_command=duplicate_scheduler,
        )


def test_cancel_fences_before_scancel_and_rejects_continuation(
    tmp_path,
) -> None:
    generation = "generation-3"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    scheduler = FakeScheduler()
    submit_with_lifecycle(
        tmp_path,
        generation=generation,
        token="chunk-0",
        role="chunk",
        script_path=tmp_path / "chunk.sh",
        run_command=scheduler,
    )

    result = cancel_generation(
        tmp_path, generation, run_command=scheduler, max_rescans=2
    )

    assert result.job_ids == ("701",)
    assert "701" in scheduler.cancelled
    with pytest.raises(RuntimeError, match="inactive"):
        submit_with_lifecycle(
            tmp_path,
            generation=generation,
            token="chunk-1",
            role="chunk",
            script_path=tmp_path / "chunk-1.sh",
            run_command=scheduler,
        )


def test_cancel_resolves_intent_when_scheduler_proves_no_job(tmp_path) -> None:
    generation = "generation-no-job"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="ordinary"
    )
    scheduler = FakeScheduler()
    append_lifecycle_entry(
        tmp_path,
        generation=generation,
        token="chunk-0",
        role="chunk",
        status="intent",
    )

    result = cancel_generation(tmp_path, generation, run_command=scheduler)

    assert result.quiescent is True
    assert result.unresolved_tokens == ()
    assert read_lifecycle_ledger(tmp_path)[-1]["status"] == (
        "reconciled-no-job"
    )


def test_preversioned_metadata_is_upgraded_without_losing_unknown_id(
    tmp_path,
) -> None:
    _metadata_skeleton(tmp_path)
    path = job_metadata_path(tmp_path)
    atomic_write_json(
        path,
        {
            JobMetadataKey.CHUNK_JOB_IDS: {"0": "111"},
            JobMetadataKey.SLURM_JOB_IDS: {"legacy": "111"},
        },
    )

    mirror_job_to_metadata(
        tmp_path,
        generation="generation-4",
        token="finalizer",
        role="finalizer",
        job_id="222",
    )

    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata[JobMetadataKey.SLURM_JOB_IDS]["legacy"] == "111"
    assert metadata[JobMetadataKey.SLURM_JOB_IDS]["finalizer"]["role"] == (
        "finalizer"
    )


def test_later_continuation_failure_fences_submitted_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dynamic dispatcher cannot orphan its newly submitted chunk."""
    generation = "later-dispatch-failure"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="recompile"
    )
    submissions: list[str] = []

    def fake_submit(*_args: object, **kwargs: object) -> str:
        token = str(kwargs["token"])
        submissions.append(token)
        if token.startswith("dispatcher-"):
            raise ValueError("scheduler adapter failed unexpectedly")
        return "701"

    cancellations: list[tuple[Path, str]] = []

    def fake_cancel(output_dir: Path, found_generation: str) -> object:
        cancellations.append((output_dir, found_generation))
        lifecycle.deactivate_generation(output_dir, found_generation)
        return lifecycle.CancellationResult(("701",), (), True)

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)
    monkeypatch.setattr(lifecycle, "cancel_generation", fake_cancel)

    with pytest.raises(RuntimeError, match="fenced and reconciled"):
        lifecycle.dispatch_continuation(
            tmp_path,
            generation=generation,
            chunk_index=2,
            chunk_script=tmp_path / "chunk-2.sh",
            dispatcher_script=tmp_path / "dispatch-3.sh",
        )

    assert submissions == ["chunk-2", "dispatcher-3"]
    assert cancellations == [(tmp_path, generation)]
    state = load_slurm_lifecycle(tmp_path)
    assert state["active"] is False
    assert state["terminal_status"] == "failed"
    assert "Continuation submission failed" in state["terminal_error"]


def test_next_chunk_submission_failure_fences_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dynamic chunk submission error enters the same cancellation path."""
    generation = "next-chunk-failure"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="recompile"
    )

    def fake_submit(*_args: object, **_kwargs: object) -> str:
        raise OSError("chunk scheduler transport failed")

    cancellations: list[tuple[Path, str]] = []

    def fake_cancel(output_dir: Path, found_generation: str) -> object:
        cancellations.append((output_dir, found_generation))
        lifecycle.deactivate_generation(output_dir, found_generation)
        return lifecycle.CancellationResult((), (), True)

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)
    monkeypatch.setattr(lifecycle, "cancel_generation", fake_cancel)

    with pytest.raises(RuntimeError, match="Next chunk submission failed"):
        lifecycle.dispatch_continuation(
            tmp_path,
            generation=generation,
            chunk_index=2,
            chunk_script=tmp_path / "chunk-2.sh",
            dispatcher_script=tmp_path / "dispatch-3.sh",
        )

    assert cancellations == [(tmp_path, generation)]
    state = load_slurm_lifecycle(tmp_path)
    assert state["active"] is False
    assert state["terminal_status"] == "failed"
    assert "Next chunk submission failed" in state["terminal_error"]


def test_continuation_cancel_failure_still_deactivates_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local fence survives an unexpected reconciliation failure."""
    generation = "later-cancel-failure"
    initialize_slurm_lifecycle(
        tmp_path, generation=generation, mode="recompile"
    )

    def fake_submit(*_args: object, **kwargs: object) -> str:
        if str(kwargs["token"]).startswith("dispatcher-"):
            raise OSError("scheduler transport failed")
        return "701"

    def fake_cancel(*_args: object, **_kwargs: object) -> object:
        raise TimeoutError("reconciliation lock timed out")

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)
    monkeypatch.setattr(lifecycle, "cancel_generation", fake_cancel)

    with pytest.raises(RuntimeError, match="reconciliation failed"):
        lifecycle.dispatch_continuation(
            tmp_path,
            generation=generation,
            chunk_index=2,
            chunk_script=tmp_path / "chunk-2.sh",
            dispatcher_script=tmp_path / "dispatch-3.sh",
        )

    state = load_slurm_lifecycle(tmp_path)
    assert state["active"] is False
    assert state["terminal_status"] == "failed"
    assert "reconciliation failed" in state["terminal_error"]
