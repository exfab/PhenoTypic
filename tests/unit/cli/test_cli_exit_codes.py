"""The CLI's exit codes are a contract (#281 spec C10, plan P2 T1).

An orchestrator acts on the exit code without parsing logs, so every path a
run can end on is pinned here: a clean local run, a run with a terminal image
failure (and its re-run), a process run, every SLURM submission (forward,
process, recompile, and the forward run whose only remaining work is a SLURM
recompile), and Click's own usage refusal.

The migrate submission (and a SLURM migrate ``--dry-run``, which submits its
validation chain) is pinned by ``test_cli_migrate_slurm.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from phenotypic.sdk_ import RunExitCode
from tests._tiny_run import run_cli

_PROCESS_ZARR = ["--layer", "gray", "--process-format", "zarr"]


def test_the_codes_are_the_documented_values() -> None:
    assert {code.name: int(code) for code in RunExitCode} == {
        "COMPLETE": 0,
        "ERROR": 1,
        "COMPLETE_WITH_FAILURES": 3,
        "SUBMITTED": 4,
        "ENDED_UNPUBLISHED": 5,
        "WAIT_TIMEOUT": 6,
    }
    assert 2 not in {int(code) for code in RunExitCode}  # Click's own


def test_clean_local_run_exits_0(tiny_inputs: Path, tmp_path: Path) -> None:
    result = run_cli(tiny_inputs, tmp_path / "out")
    assert result.exit_code == RunExitCode.COMPLETE, result.output


def _terminal_failure_stems(output_dir: Path) -> set[str]:
    from phenotypic._cli._cli_failure_tracker import terminal_failure_index

    return {
        Path(record.relative_image_path).stem
        for record in terminal_failure_index(output_dir).values()
    }


def test_the_truncated_plate_is_a_terminal_failure(
    tiny_inputs_one_bad: Path, tmp_path: Path
) -> None:
    """The fixture's premise: the bad plate is journalled, the others are not."""
    out = tmp_path / "out"
    run_cli(tiny_inputs_one_bad, out)
    assert _terminal_failure_stems(out) == {"plate_bad"}


def test_local_run_with_a_terminal_failure_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = run_cli(tiny_inputs_one_bad, out)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert (out / "deliverables" / "measurements.tsv").is_file()


def test_rerunning_a_run_that_exited_3_still_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path
) -> None:
    """The re-run skips the failed plate; skipped is not succeeded.

    It is not a "nothing left" no-op either: a run without a run proof
    re-finalizes, which used to reach the GUI-completion publication and
    exit ``1`` with "Cannot publish GUI local completion".
    """
    out = tmp_path / "out"
    run_cli(tiny_inputs_one_bad, out)
    result = run_cli(tiny_inputs_one_bad, out)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output


def test_rerunning_with_retry_failures_retries_and_still_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path
) -> None:
    """Retried, the plate fails again: a real failure this time, not a skip."""
    out = tmp_path / "out"
    run_cli(tiny_inputs_one_bad, out)
    result = run_cli(tiny_inputs_one_bad, out, extra=["--retry-failures"])
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output


def test_rerunning_a_clean_run_exits_0(
    finished_tiny_run: Path, tiny_inputs: Path
) -> None:
    result = run_cli(tiny_inputs, finished_tiny_run)
    assert result.exit_code == RunExitCode.COMPLETE, result.output
    assert "All images already processed" in result.output


def test_process_local_run_with_a_failure_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path
) -> None:
    result = run_cli(
        tiny_inputs_one_bad, tmp_path / "out", mode="process", extra=_PROCESS_ZARR
    )
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output


def test_rerunning_a_process_run_with_a_failure_still_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path
) -> None:
    """Process mode has no run proof to refresh: this is the "nothing left" exit."""
    out = tmp_path / "out"
    run_cli(tiny_inputs_one_bad, out, mode="process", extra=_PROCESS_ZARR)
    result = run_cli(tiny_inputs_one_bad, out, mode="process", extra=_PROCESS_ZARR)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert "recorded failure(s) skipped" in result.output


def test_clean_process_local_run_exits_0(tiny_inputs: Path, tmp_path: Path) -> None:
    result = run_cli(tiny_inputs, tmp_path / "out", mode="process", extra=_PROCESS_ZARR)
    assert result.exit_code == RunExitCode.COMPLETE, result.output


@pytest.mark.skipif(sys.platform == "win32", reason="SLURM submission is POSIX-only")
@pytest.mark.parametrize(
    ("mode", "extra"), [("full", []), ("process", _PROCESS_ZARR)], ids=["full", "process"]
)
def test_every_slurm_submission_exits_4(
    tiny_inputs: Path,
    tmp_path: Path,
    submissions: list[str],
    mode: str,
    extra: list[str],
) -> None:
    result = run_cli(tiny_inputs, tmp_path / "out", mode=mode, slurm=True, extra=extra)
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert submissions == ["ordinary"]


@pytest.mark.skipif(sys.platform == "win32", reason="SLURM submission is POSIX-only")
def test_a_slurm_recompile_exits_4(
    finished_tiny_run: Path, submissions: list[str]
) -> None:
    result = run_cli(None, finished_tiny_run, mode="recompile", slurm=True)
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert submissions == ["recompile"]


def test_a_local_recompile_exits_0(finished_tiny_run: Path) -> None:
    result = run_cli(None, finished_tiny_run, mode="recompile")
    assert result.exit_code == RunExitCode.COMPLETE, result.output


@pytest.mark.skipif(sys.platform == "win32", reason="SLURM submission is POSIX-only")
def test_a_forward_slurm_run_with_nothing_left_exits_4(
    finished_tiny_run: Path,
    tiny_inputs: Path,
    submissions: list[str],
    metadata_bytes: dict[str, bytes],
    tmp_path: Path,
) -> None:
    """A metadata file the run has not snapshotted routes it to a SLURM recompile."""
    meta = tmp_path / "meta.csv"
    meta.write_bytes(metadata_bytes[".csv"])
    result = run_cli(
        tiny_inputs, finished_tiny_run, slurm=True, extra=["--metadata", str(meta)]
    )
    assert result.exit_code == RunExitCode.SUBMITTED, result.output
    assert submissions == ["recompile"]


def test_a_usage_refusal_still_exits_2(tiny_inputs: Path, tmp_path: Path) -> None:
    """Click's own code: ``--revert`` belongs to ``--mode migrate`` only."""
    result = run_cli(tiny_inputs, tmp_path / "out", extra=["--revert"])
    assert result.exit_code == 2, result.output
