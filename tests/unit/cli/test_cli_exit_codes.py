"""The CLI's exit codes are a contract (#281 spec C10, plan P2 T1).

An orchestrator acts on the exit code without parsing logs, so every path a
run can end on is pinned here: a clean local run, a run with a terminal image
failure (and its re-run), a process run, every SLURM submission (forward,
process, recompile, and the forward run whose only remaining work is a SLURM
recompile), and Click's own usage refusal.

The migrate submission (and a SLURM migrate ``--dry-run``, which submits its
validation chain) is pinned by ``test_cli_migrate_mode.py`` and
``test_cli_provenance_migration_slurm.py``; ``test_cli_migrate_slurm.py``
drives only the migration worker CLI, whose ``0``/``1`` are unchanged.
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


# ---- #281 P2 review I2 / fork F4: a recompile reports the run's failures ----


def test_a_local_recompile_of_a_run_with_a_failure_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path
) -> None:
    """Review I2: on SLURM the waiter returns 3 for this run, so locally it is 3 too."""
    out = tmp_path / "out"
    first = run_cli(tiny_inputs_one_bad, out)
    assert first.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, first.output
    result = run_cli(None, out, mode="recompile")
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output


@pytest.mark.skipif(sys.platform == "win32", reason="SLURM submission is POSIX-only")
@pytest.mark.parametrize("fallback", ["no-measurement-tasks", "no-scripts"])
def test_a_slurm_recompile_run_locally_reports_the_runs_failures(
    tiny_inputs_one_bad: Path,
    tmp_path: Path,
    submissions: list[str],
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
) -> None:
    """Review I2: both ``_handle_recompile_slurm`` local fallbacks exit like a local recompile."""
    from phenotypic import phenotypicCLI

    out = tmp_path / "out"
    first = run_cli(tiny_inputs_one_bad, out)
    assert first.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, first.output
    if fallback == "no-measurement-tasks":
        monkeypatch.setattr(phenotypicCLI, "build_recompile_tasks", lambda **_kw: [])
        expected = "No measurement sources require SLURM recompile"
    else:
        monkeypatch.setattr(
            phenotypicCLI, "generate_recompile_slurm_scripts", lambda **_kw: []
        )
        expected = "running local recompile instead"
    result = run_cli(None, out, mode="recompile", slurm=True)
    assert expected in " ".join(result.output.split()), result.output  # rich wraps
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert submissions == []


# ---- #281 P2 review I3 / fork F3: a staged local run with a terminal failure ----


def _staged_pipeline_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-``FakeGpuDetector`` pipeline, so the run takes the local staged engine."""
    import phenotypic
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureSize
    from tests._fakes.fake_gpu_detector import FakeGpuDetector

    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)
    path = tmp_path / "staged.pht-pipe.json"
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)], meas={"size": MeasureSize()}
    )
    path.write_text(pipe.to_json() or "", encoding="utf-8")
    return path


def test_a_local_staged_run_with_a_terminal_failure_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review I3: every unfinished image is a terminal failure, so the run finished."""
    from phenotypic._cli._cli_staged_orchestration import staged_completion_path

    out = tmp_path / "out"
    pipeline = _staged_pipeline_file(tmp_path, monkeypatch)
    result = run_cli(tiny_inputs_one_bad, out, pipeline=pipeline)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert _terminal_failure_stems(out) == {"plate_bad"}
    assert "STAGED FINALIZATION FAILED" not in result.output
    assert "Run the same command again" not in result.output
    assert (out / "deliverables" / "measurements.tsv").is_file()
    # As the SLURM finalizer's `terminal_incomplete` close: not staged-complete.
    assert not staged_completion_path(out).is_file()


def test_rerunning_a_local_staged_run_with_a_terminal_failure_still_exits_3(
    tiny_inputs_one_bad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review I3: the re-run used to exit 1 forever, telling the user to re-run."""
    out = tmp_path / "out"
    pipeline = _staged_pipeline_file(tmp_path, monkeypatch)
    run_cli(tiny_inputs_one_bad, out, pipeline=pipeline)
    result = run_cli(tiny_inputs_one_bad, out, pipeline=pipeline)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert "Run the same command again" not in result.output


def test_a_local_staged_publication_failure_still_exits_1(
    tiny_inputs_one_bad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review I3's boundary: a real publication failure stays ``ERROR``.

    The run also carries a terminal failure, so this pins that the new ``3``
    arm requires the publication to have succeeded.
    """
    from phenotypic._cli._cli_readme_generator import READMEGenerator

    out = tmp_path / "out"
    pipeline = _staged_pipeline_file(tmp_path, monkeypatch)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("README publication failed")

    monkeypatch.setattr(READMEGenerator, "generate", _fail)
    result = run_cli(tiny_inputs_one_bad, out, pipeline=pipeline)
    assert result.exit_code == RunExitCode.ERROR, result.output
    assert "STAGED FINALIZATION FAILED" in result.output
    assert "Run the same command again" in result.output


# ---- #281 P2 review M1: an option value checked after parsing is a usage error ----


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        (["--slurm", "partition"], "KEY=VALUE"),
        (["--slurm", "time=banana"], "banana"),
        (["--gpu-slurm", "partition"], "--gpu-slurm must be KEY=VALUE"),
        (["--gpu-slurm", "time=banana"], "banana"),
        (["--ext", ".banana"], ".banana"),
    ],
    ids=[
        "slurm-not-key-value",
        "slurm-time",
        "gpu-slurm-not-key-value",
        "gpu-slurm-time",
        "ext",
    ],
)
def test_a_malformed_option_value_exits_2(
    tiny_inputs: Path,
    tiny_pipeline_file: Path,
    tmp_path: Path,
    extra: list[str],
    named: str,
) -> None:
    """Review M1: ``2`` is "a refused option", whether Click or the CLI checks the value."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    args = [
        "--input", str(tiny_inputs), "--output", str(tmp_path / "out"),
        "--pipeline", str(tiny_pipeline_file), "--dry-run", *extra,
    ]  # fmt: skip
    result = CliRunner().invoke(phenotypic_cli, args)
    assert result.exit_code == 2, result.output
    assert named in result.output
    assert "Traceback" not in result.output
