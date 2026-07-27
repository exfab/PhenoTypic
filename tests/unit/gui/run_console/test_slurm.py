"""Unit tests for ``phenotypic.gui.run_console._slurm``.

The SLURM submitter is a thin shell-out around the CLI, so the
"interesting" behaviour is everything that happens around the streamed
``Popen`` wrapper:

* argv assembly from :class:`RunConsoleState`.
* Reading ``progress/job_metadata.json`` for the array job id.
* Error mapping for non-zero exit, timeout, missing/empty metadata.

We mock the streamed wrapper (and use a real metadata file on tmp_path)
so the tests are fast and don't require a real CLI install.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock
from uuid import uuid4

import pytest

from phenotypic._cli._cli_slurm_lifecycle import (
    append_lifecycle_entry,
    initialize_slurm_lifecycle,
    lifecycle_state_path,
)
from phenotypic.gui.run_console._slurm import (
    SlurmSubmitError,
    SlurmSubmitPending,
    SlurmSubmitResult,
    _StreamedProcessResult,
    _build_subprocess_argv,
    _run_submitter_streamed,
    read_submitted_job_set,
    submit_slurm,
    wait_for_job_id,
)
from phenotypic.gui.run_console._state import RunConsoleState
from phenotypic.sdk_ import atomic_write_json, job_metadata_path

_STREAM_TARGET = (
    "phenotypic.gui.run_console._slurm._run_submitter_streamed"
)


def _state_for(output_dir: Path) -> RunConsoleState:
    return RunConsoleState(
        pipeline_path="/p/pipeline.json",
        input_dir="/p/in",
        output_dir=str(output_dir),
        mode="slurm",
        slurm_args={"partition": "compute", "mem": "16G"},
    )


def _write_manifest(
    output_dir: Path,
    chunk_job_ids: dict[str, str],
) -> None:
    metadata_path = job_metadata_path(output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps({"chunk_job_ids": chunk_job_ids}),
        encoding="utf-8",
    )


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(
        args=["python", "-m", "phenotypic"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _streamed(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    *,
    timed_out: bool = False,
    stream_error: str | None = None,
) -> _StreamedProcessResult:
    return _StreamedProcessResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
        stream_error=stream_error,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_submit_slurm_returns_array_job_id(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {"0": "45678901_0", "1": "45678901_1"})

    state = _state_for(output_dir)
    record_generation = uuid4()
    with mock.patch(
        _STREAM_TARGET, return_value=_streamed()
    ) as run_mock:
        result = submit_slurm(
            state,
            sandbox_root=tmp_path,
            record_generation=record_generation,
            timeout=5.0,
        )

    assert isinstance(result, SlurmSubmitResult)
    assert result.job_id == "45678901"
    assert result.output_dir == output_dir
    assert result.returncode == 0
    # Subprocess called exactly once with the expected argv tail.
    run_mock.assert_called_once()
    argv = run_mock.call_args.args[0]
    assert argv[0].endswith("python") or argv[0].endswith("python3") or "python" in argv[0]
    assert argv[1:3] == ["-m", "phenotypic"]
    assert argv[3:11] == [
        "--mode",
        "full",
        "--pipeline",
        "/p/pipeline.json",
        "--input",
        "/p/in",
        "--output",
        str(output_dir),
    ]
    assert "--slurm" in argv
    # SLURM kwargs forwarded as ``--slurm key=value`` repeats.
    pairs = [argv[i + 1] for i, t in enumerate(argv) if t == "--slurm"]
    assert "partition=compute" in pairs
    assert "mem=16G" in pairs
    assert (
        run_mock.call_args.kwargs["env"][
            "PHENOTYPIC_GUI_RECORD_GENERATION"
        ]
        == str(record_generation)
    )
    assert run_mock.call_args.kwargs["env"]["PYTHONUNBUFFERED"] == "1"


def test_streamed_submitter_publishes_log_before_child_exit(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    results: list[_StreamedProcessResult] = []

    def _invoke() -> None:
        results.append(
            _run_submitter_streamed(
                [
                    sys.executable,
                    "-c",
                    (
                        "import time; "
                        "print('Generating scripts...', flush=True); "
                        "time.sleep(0.5); "
                        "print('Submitted job 123', flush=True)"
                    ),
                ],
                output_dir=output_dir,
                log_generation=generation,
                cwd=tmp_path,
                env={"PYTHONUNBUFFERED": "1"},
                timeout=5.0,
            )
        )

    thread = threading.Thread(target=_invoke)
    thread.start()
    stdout_path = (
        output_dir
        / ".phenotypic"
        / "logs"
        / "gui"
        / f"submitter.{generation.hex}.stdout.log"
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if (
            stdout_path.is_file()
            and "Generating scripts..." in stdout_path.read_text()
        ):
            break
        time.sleep(0.01)

    assert thread.is_alive()
    assert "Generating scripts..." in stdout_path.read_text()
    thread.join(timeout=5.0)
    assert results[0].returncode == 0
    assert "Submitted job 123" in results[0].stdout


def test_streamed_submitter_keeps_bounded_tail_and_full_disk_log(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()

    result = _run_submitter_streamed(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 200000); sys.stdout.flush()",
        ],
        output_dir=output_dir,
        log_generation=generation,
        cwd=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        timeout=5.0,
    )

    stdout_path = (
        output_dir
        / ".phenotypic"
        / "logs"
        / "gui"
        / f"submitter.{generation.hex}.stdout.log"
    )
    assert len(result.stdout) <= 128 * 1024
    assert stdout_path.stat().st_size == 200000


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX process-group timeout semantics",
)
def test_streamed_submitter_timeout_stops_pipe_inheriting_descendant(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    descendant_pid_path = tmp_path / "descendant.pid"
    descendant_code = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(descendant_pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    started = time.monotonic()

    result = _run_submitter_streamed(
        [
            sys.executable,
            "-c",
            (
                "import os, subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); "
                f"pid_path = {str(descendant_pid_path)!r}; "
                "deadline = time.monotonic() + 2; "
                "exec(\"while not os.path.exists(pid_path) and "
                "time.monotonic() < deadline:\\n time.sleep(0.01)\"); "
                "print('spawned descendant', flush=True); "
                "time.sleep(60)"
            ),
        ],
        output_dir=output_dir,
        log_generation=generation,
        cwd=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        timeout=0.2,
    )

    assert result.timed_out is True
    assert time.monotonic() - started < 5.0
    descendant_pid = int(descendant_pid_path.read_text())
    descendant_deadline = time.monotonic() + 2.0
    while time.monotonic() < descendant_deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)
    thread_suffix = generation.hex[:8]
    assert not any(
        thread.is_alive()
        and thread.name.startswith("phenotypic-submit-")
        and thread.name.endswith(thread_suffix)
        for thread in threading.enumerate()
    )


def test_streamed_submitter_reports_log_open_failure(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    stdout_path = (
        output_dir
        / ".phenotypic"
        / "logs"
        / "gui"
        / f"submitter.{generation.hex}.stdout.log"
    )
    stdout_path.mkdir(parents=True)

    result = _run_submitter_streamed(
        [sys.executable, "-c", "print('submitted', flush=True)"],
        output_dir=output_dir,
        log_generation=generation,
        cwd=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        timeout=5.0,
    )

    assert result.returncode == 0
    assert result.stream_error is not None
    assert stdout_path.name in result.stream_error


def test_submit_slurm_uses_sandbox_root_as_cwd(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {"0": "12345_0"})
    state = _state_for(output_dir)
    with mock.patch(
        _STREAM_TARGET, return_value=_streamed()
    ) as run_mock:
        submit_slurm(state, sandbox_root=tmp_path)
    assert run_mock.call_args.kwargs["cwd"] == tmp_path


def test_submit_slurm_includes_extra_pairs(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {"0": "1_0"})

    state = RunConsoleState(
        pipeline_path="/p/pipeline.json",
        input_dir="/p/in",
        output_dir=str(output_dir),
        mode="slurm",
        slurm_args={"extra": {"qos": "bench", "account": "lab"}},
    )
    with mock.patch(
        _STREAM_TARGET, return_value=_streamed()
    ) as run_mock:
        submit_slurm(state, sandbox_root=tmp_path)
    argv = run_mock.call_args.args[0]
    pairs = [argv[i + 1] for i, t in enumerate(argv) if t == "--slurm"]
    assert "qos=bench" in pairs
    assert "account=lab" in pairs


def test_submit_slurm_forwards_gpu_stage_profile_and_shards(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {"0": "1_0"})
    state = _state_for(output_dir)
    state.gpu_slurm_args = (
        "slurm_partition=gpu",
        "slurm_account=lab",
    )
    state.gpu_shards = 4

    with mock.patch(
        _STREAM_TARGET, return_value=_streamed()
    ) as run_mock:
        submit_slurm(state, sandbox_root=tmp_path)

    argv = run_mock.call_args.args[0]
    gpu_pairs = [
        argv[index + 1]
        for index, token in enumerate(argv)
        if token == "--gpu-slurm"
    ]
    assert gpu_pairs == [
        "slurm_partition=gpu",
        "slurm_account=lab",
    ]
    shards_index = argv.index("--gpu-shards")
    assert argv[shards_index + 1] == "4"


def test_live_cancellation_hold_profile_reaches_cli_argv(
    tmp_path: Path,
) -> None:
    """The live-test hold is a bounded CPU profile, not an implicit GPU job."""
    output_dir = tmp_path / "out"
    state = RunConsoleState(
        pipeline_path="/p/pipeline.json",
        input_dir="/p/in",
        output_dir=str(output_dir),
        mode="slurm",
        advanced_args={"image_type": "Image", "workers": 1},
        slurm_args={
            "partition": "short",
            "time": "00:10:00",
            "mem": "4G",
            "cpus_per_task": 1,
            "extra": {"slurm_begin": "now+2minutes"},
        },
        gpu_slurm_args=(),
        gpu_shards=1,
    )

    argv = _build_subprocess_argv(state)
    slurm_pairs = [
        argv[index + 1]
        for index, token in enumerate(argv)
        if token == "--slurm"
    ]

    assert slurm_pairs == [
        "partition=short",
        "time=00:10:00",
        "mem=4G",
        "cpus_per_task=1",
        "slurm_begin=now+2minutes",
    ]
    assert argv[argv.index("--njobs") + 1] == "1"
    assert "--gpu-slurm" not in argv
    assert "--gpu-shards" not in argv


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_submit_slurm_raises_on_non_zero_exit(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    state = _state_for(output_dir)
    failing = _streamed(returncode=2, stderr="sbatch: error: nope")
    with mock.patch(_STREAM_TARGET, return_value=failing):
        with pytest.raises(SlurmSubmitError) as exc:
            submit_slurm(state, sandbox_root=tmp_path)
    assert "exited with code 2" in str(exc.value)
    assert "sbatch: error: nope" in str(exc.value)


def test_submit_slurm_raises_on_timeout(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    state = _state_for(output_dir)

    with mock.patch(
        _STREAM_TARGET,
        return_value=_streamed(returncode=-1, timed_out=True),
    ):
        with pytest.raises(SlurmSubmitError) as exc:
            submit_slurm(state, sandbox_root=tmp_path, timeout=0.5)
    assert "timed out" in str(exc.value).lower()


def test_submit_slurm_reports_stream_failure_without_durable_job(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    state = _state_for(output_dir)

    with mock.patch(
        _STREAM_TARGET,
        return_value=_streamed(stream_error="stdout log: permission denied"),
    ):
        with pytest.raises(SlurmSubmitError, match="logging failed"):
            submit_slurm(state, sandbox_root=tmp_path)


def test_submit_slurm_reconciles_stream_failure_with_durable_job(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {"0": "12345_0"})

    with mock.patch(
        _STREAM_TARGET,
        return_value=_streamed(stream_error="stdout log: permission denied"),
    ):
        result = submit_slurm(
            _state_for(output_dir),
            sandbox_root=tmp_path,
        )

    assert result.job_id == "12345"
    assert result.reconciled is True
    assert "Submitter log streaming failed" in result.stderr


def test_submit_slurm_raises_on_missing_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()  # No progress/job_metadata.json written.
    state = _state_for(output_dir)
    with mock.patch(_STREAM_TARGET, return_value=_streamed()):
        with pytest.raises(SlurmSubmitError) as exc:
            submit_slurm(state, sandbox_root=tmp_path)
    assert "missing or" in str(exc.value).lower() or "job_metadata.json" in str(exc.value)


def test_submit_slurm_raises_on_empty_chunk_job_ids(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {})
    state = _state_for(output_dir)
    with mock.patch(_STREAM_TARGET, return_value=_streamed()):
        with pytest.raises(SlurmSubmitError):
            submit_slurm(state, sandbox_root=tmp_path)


def test_submit_slurm_raises_when_state_missing_output_dir(tmp_path: Path) -> None:
    state = RunConsoleState(
        pipeline_path="/p.json",
        input_dir="/in",
        output_dir=None,  # missing
        mode="slurm",
    )
    with pytest.raises(SlurmSubmitError):
        submit_slurm(state, sandbox_root=tmp_path)


def test_submit_slurm_raises_when_state_missing_pipeline(tmp_path: Path) -> None:
    """to_argv() raises ValueError; submit_slurm wraps it as SlurmSubmitError."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    state = RunConsoleState(
        pipeline_path=None,
        input_dir="/in",
        output_dir=str(output_dir),
        mode="slurm",
    )
    with pytest.raises(SlurmSubmitError):
        submit_slurm(state, sandbox_root=tmp_path)


def test_submit_slurm_rejects_non_slurm_mode_before_cli(tmp_path: Path) -> None:
    state = _state_for(tmp_path / "out")
    state.mode = "local"
    with mock.patch(_STREAM_TARGET) as run_mock:
        with pytest.raises(SlurmSubmitError, match="mode='slurm'"):
            submit_slurm(state, sandbox_root=tmp_path)
    run_mock.assert_not_called()


def test_submit_slurm_rejects_empty_slurm_config_before_cli(
    tmp_path: Path,
) -> None:
    state = _state_for(tmp_path / "out")
    state.slurm_args = {}
    with mock.patch(_STREAM_TARGET) as run_mock:
        with pytest.raises(SlurmSubmitError, match="non-empty"):
            submit_slurm(state, sandbox_root=tmp_path)
    run_mock.assert_not_called()


def test_submit_slurm_rejects_semantically_empty_slurm_config(
    tmp_path: Path,
) -> None:
    state = _state_for(tmp_path / "out")
    state.slurm_args = {"partition": "", "extra": {}}
    with mock.patch(_STREAM_TARGET) as run_mock:
        with pytest.raises(SlurmSubmitError, match="non-empty"):
            submit_slurm(state, sandbox_root=tmp_path)
    run_mock.assert_not_called()


def test_reader_accepts_controller_only_staged_submission(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    metadata_path = job_metadata_path(output_dir)
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "slurm_generation": generation.hex,
                "chunk_job_ids": {},
                "slurm_job_ids": {
                    "finalizer": {
                        "job_id": "9002",
                        "role": "finalizer",
                        "generation": generation.hex,
                    },
                    "controller-initial": {
                        "job_id": "9001",
                        "role": "controller-initial",
                        "generation": generation.hex,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    jobs = read_submitted_job_set(output_dir)

    assert jobs is not None
    assert jobs.primary_id == "9001"
    assert jobs.all_ids == ("9001", "9002")
    assert jobs.roles["controller-initial"] == ("9001",)
    assert jobs.generation == generation


def test_reader_merges_preversioned_metadata_with_ledger(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir, generation=generation.hex, mode="ordinary"
    )
    _write_manifest(output_dir, {"1": "7002_1", "0": "7001_0"})
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="controller-initial",
        role="controller-initial",
        status="submitted",
        job_id="7003",
    )

    jobs = read_submitted_job_set(output_dir)

    assert jobs is not None
    assert jobs.primary_id == "7003"
    assert set(jobs.all_ids) == {"7001", "7002", "7003"}
    assert jobs.roles["controller-initial"] == ("7003",)


def test_reader_fences_stale_versioned_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    current_generation = uuid4()
    stale_generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir, generation=current_generation.hex, mode="ordinary"
    )
    metadata_path = job_metadata_path(output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "slurm_generation": stale_generation.hex,
                "chunk_job_ids": {"0": "7100_0"},
                "slurm_job_ids": {
                    "chunk-0": {
                        "job_id": "7100",
                        "role": "chunk",
                        "generation": stale_generation.hex,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    append_lifecycle_entry(
        output_dir,
        generation=current_generation.hex,
        token="chunk-0",
        role="chunk",
        status="submitted",
        job_id="7200",
    )

    jobs = read_submitted_job_set(
        output_dir, expected_generation=current_generation
    )

    assert jobs is not None
    assert jobs.all_ids == ("7200",)
    assert jobs.generation == current_generation


def test_timeout_attaches_to_durable_submission(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir, generation=generation.hex, mode="ordinary"
    )
    metadata_path = job_metadata_path(output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "slurm_generation": generation.hex,
                "slurm_job_ids": {
                    "chunk-0": {
                        "job_id": "8123",
                        "role": "chunk",
                        "generation": generation.hex,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    state = _state_for(output_dir)

    with mock.patch(
        _STREAM_TARGET,
        return_value=_streamed(returncode=-1, timed_out=True),
    ):
        result = submit_slurm(state, sandbox_root=tmp_path, timeout=0.5)

    assert result.job_id == "8123"
    assert result.reconciled is True
    assert result.returncode == -1


def test_timeout_recovers_incomplete_intent_from_scheduler_comment(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir, generation=generation.hex, mode="ordinary"
    )
    metadata_path = job_metadata_path(output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "slurm_generation": generation.hex,
                "slurm_job_ids": {},
                "chunk_job_ids": {},
            }
        ),
        encoding="utf-8",
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="chunk-0",
        role="chunk",
        status="intent",
    )

    def run_command(argv: list[str], **_kwargs: Any) -> Any:
        if argv[0] == "squeue":
            return _completed(
                stdout=f"9123|phenotypic:{generation.hex}:chunk-0\n"
            )
        return _completed()

    with (
        mock.patch(
            _STREAM_TARGET,
            return_value=_streamed(returncode=-1, timed_out=True),
        ),
        mock.patch("subprocess.run", side_effect=run_command),
    ):
        result = submit_slurm(
            _state_for(output_dir), sandbox_root=tmp_path, timeout=0.5
        )

    assert result.reconciled is True
    assert result.job_id == "9123"
    jobs = read_submitted_job_set(output_dir)
    assert jobs is not None
    assert jobs.roles["chunk"] == ("9123",)


@pytest.mark.parametrize("comments_available", [True, False])
def test_timeout_with_unresolved_intent_remains_recoverable(
    tmp_path: Path,
    comments_available: bool,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    initialize_slurm_lifecycle(
        output_dir, generation=generation.hex, mode="ordinary"
    )
    append_lifecycle_entry(
        output_dir,
        generation=generation.hex,
        token="chunk-0",
        role="chunk",
        status="intent",
    )

    def run_command(argv: list[str], **_kwargs: Any) -> Any:
        if not comments_available:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=30)
        return _completed()

    with (
        mock.patch(
            _STREAM_TARGET,
            return_value=_streamed(returncode=-1, timed_out=True),
        ),
        mock.patch("subprocess.run", side_effect=run_command),
    ):
        with pytest.raises(SlurmSubmitPending) as exc:
            submit_slurm(
                _state_for(output_dir),
                sandbox_root=tmp_path,
                timeout=0.5,
            )

    assert exc.value.generation == generation
    assert exc.value.unresolved_tokens == ("chunk-0",)
    assert exc.value.scheduler_available is comments_available
    assert exc.value.returncode == -1


def test_abnormal_exit_cancels_recovered_inactive_generation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    generation = uuid4()
    state_path = initialize_slurm_lifecycle(
        output_dir, generation=generation.hex, mode="ordinary"
    )
    state_path["active"] = False
    atomic_write_json(lifecycle_state_path(output_dir), state_path)
    _write_manifest(output_dir, {"0": "9911_0"})

    with (
        mock.patch(
            _STREAM_TARGET,
            return_value=_streamed(returncode=2),
        ),
        mock.patch(
            "phenotypic.gui.run_console._slurm.cancel_generation"
        ) as cancel,
    ):
        with pytest.raises(SlurmSubmitError, match="exited with code 2"):
            submit_slurm(_state_for(output_dir), sandbox_root=tmp_path)

    cancel.assert_called_once_with(output_dir, generation.hex)


# ---------------------------------------------------------------------------
# wait_for_job_id
# ---------------------------------------------------------------------------

def test_wait_for_job_id_returns_immediately_if_present(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _write_manifest(out, {"0": "999_0"})
    job = wait_for_job_id(out, timeout=0.5, poll_interval=0.05)
    assert job == "999"


def test_wait_for_job_id_returns_none_on_timeout(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()  # no manifest
    job = wait_for_job_id(out, timeout=0.2, poll_interval=0.05)
    assert job is None
