"""Unit tests for ``phenotypic.gui.run_console._slurm``.

The SLURM submitter is a thin shell-out around the CLI, so the
"interesting" behaviour is everything that happens around
:func:`subprocess.run`:

* argv assembly from :class:`RunConsoleState`.
* Reading ``progress/job_metadata.json`` for the array job id.
* Error mapping for non-zero exit, timeout, missing/empty metadata.

We mock :func:`subprocess.run` (and use a real metadata file on tmp_path)
so the tests are fast and don't require a real CLI install.
"""
from __future__ import annotations

import json
import subprocess
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
    SlurmSubmitResult,
    read_submitted_job_set,
    submit_slurm,
    wait_for_job_id,
)
from phenotypic.gui.run_console._state import RunConsoleState
from phenotypic.sdk_ import atomic_write_json, job_metadata_path


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


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_submit_slurm_returns_array_job_id(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {"0": "45678901_0", "1": "45678901_1"})

    state = _state_for(output_dir)
    with mock.patch("subprocess.run", return_value=_completed()) as run_mock:
        result = submit_slurm(state, sandbox_root=tmp_path, timeout=5.0)

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


def test_submit_slurm_uses_sandbox_root_as_cwd(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {"0": "12345_0"})
    state = _state_for(output_dir)
    with mock.patch("subprocess.run", return_value=_completed()) as run_mock:
        submit_slurm(state, sandbox_root=tmp_path)
    assert run_mock.call_args.kwargs["cwd"] == str(tmp_path)


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
    with mock.patch("subprocess.run", return_value=_completed()) as run_mock:
        submit_slurm(state, sandbox_root=tmp_path)
    argv = run_mock.call_args.args[0]
    pairs = [argv[i + 1] for i, t in enumerate(argv) if t == "--slurm"]
    assert "qos=bench" in pairs
    assert "account=lab" in pairs


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_submit_slurm_raises_on_non_zero_exit(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    state = _state_for(output_dir)
    failing = _completed(returncode=2, stderr="sbatch: error: nope")
    with mock.patch("subprocess.run", return_value=failing):
        with pytest.raises(SlurmSubmitError) as exc:
            submit_slurm(state, sandbox_root=tmp_path)
    assert "exited with code 2" in str(exc.value)
    assert "sbatch: error: nope" in str(exc.value)


def test_submit_slurm_raises_on_timeout(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    state = _state_for(output_dir)

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="python", timeout=0.5)

    with mock.patch("subprocess.run", side_effect=_raise):
        with pytest.raises(SlurmSubmitError) as exc:
            submit_slurm(state, sandbox_root=tmp_path, timeout=0.5)
    assert "timed out" in str(exc.value).lower()


def test_submit_slurm_raises_on_missing_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()  # No progress/job_metadata.json written.
    state = _state_for(output_dir)
    with mock.patch("subprocess.run", return_value=_completed()):
        with pytest.raises(SlurmSubmitError) as exc:
            submit_slurm(state, sandbox_root=tmp_path)
    assert "missing or" in str(exc.value).lower() or "job_metadata.json" in str(exc.value)


def test_submit_slurm_raises_on_empty_chunk_job_ids(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_manifest(output_dir, {})
    state = _state_for(output_dir)
    with mock.patch("subprocess.run", return_value=_completed()):
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
    with mock.patch("subprocess.run") as run_mock:
        with pytest.raises(SlurmSubmitError, match="mode='slurm'"):
            submit_slurm(state, sandbox_root=tmp_path)
    run_mock.assert_not_called()


def test_submit_slurm_rejects_empty_slurm_config_before_cli(
    tmp_path: Path,
) -> None:
    state = _state_for(tmp_path / "out")
    state.slurm_args = {}
    with mock.patch("subprocess.run") as run_mock:
        with pytest.raises(SlurmSubmitError, match="non-empty"):
            submit_slurm(state, sandbox_root=tmp_path)
    run_mock.assert_not_called()


def test_submit_slurm_rejects_semantically_empty_slurm_config(
    tmp_path: Path,
) -> None:
    state = _state_for(tmp_path / "out")
    state.slurm_args = {"partition": "", "extra": {}}
    with mock.patch("subprocess.run") as run_mock:
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
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="python", timeout=0.5),
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
        if "-m" in argv:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=0.5)
        if argv[0] == "squeue":
            return _completed(
                stdout=f"9123|phenotypic:{generation.hex}:chunk-0\n"
            )
        return _completed()

    with mock.patch("subprocess.run", side_effect=run_command):
        result = submit_slurm(
            _state_for(output_dir), sandbox_root=tmp_path, timeout=0.5
        )

    assert result.reconciled is True
    assert result.job_id == "9123"
    jobs = read_submitted_job_set(output_dir)
    assert jobs is not None
    assert jobs.roles["chunk"] == ("9123",)


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
        mock.patch("subprocess.run", return_value=_completed(returncode=2)),
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
