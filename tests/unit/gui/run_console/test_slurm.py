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

import pytest

from phenotypic.gui.run_console._slurm import (
    SlurmSubmitError,
    SlurmSubmitResult,
    submit_slurm,
    wait_for_job_id,
)
from phenotypic.gui.run_console._state import RunConsoleState


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
    progress = output_dir / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "job_metadata.json").write_text(
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
    assert argv[3:9] == [
        "--pipeline",
        "/p/pipeline.json",
        "--input",
        "/p/in",
        "-o",
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
