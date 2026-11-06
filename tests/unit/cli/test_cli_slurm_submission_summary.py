"""Every SLURM run returns at submission (0.21.0 removed ``--wait``).

An ordinary, staged or process ``--slurm`` run prints a ``SUBMITTED`` summary
that points at the strategy's "Monitor progress" lines, writes the README (not
for a process run, which has none), writes no ``processing_report.html``, and
exits 0. Its dependent finalizer publishes the run's outputs.

These tests drive the **real** strategies. Every scheduler seam a submission
reaches is patched, and any other ``sbatch``/``scontrol``/``squeue``/``sacct``/
``sinfo`` process is refused outright: on the HPCC an unpatched test queries
the real scheduler.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="SLURM submission is POSIX-only"
)

from phenotypic.phenotypicCLI import phenotypic_cli  # noqa: E402
from phenotypic.sdk_ import (  # noqa: E402
    processing_report_html_path,
    readme_md_path,
)

#: Scheduler clients a test must never start.
_SCHEDULER_COMMANDS = frozenset(
    {"sbatch", "scontrol", "squeue", "sacct", "sacctmgr", "sinfo", "scancel", "srun"}
)

#: The strategies' "how to follow the run" lines
#: (``echo_slurm_monitor_commands``). The summary points at them; it must not
#: print them a second time.
_MONITOR_HEADER = "Jobs submitted. Monitor progress with:"
_SQUEUE_LINE = "squeue -u $USER --array"
_POINTER = 'Follow the jobs with the "Monitor progress" commands above.'


def _command_name(args: Any) -> str:
    """The executable a ``Popen`` call would start, without its directory."""
    if isinstance(args, (str, bytes, os.PathLike)):
        words = os.fsdecode(args).split()
        first = words[0] if words else ""
    else:
        argv = list(args)
        first = os.fsdecode(argv[0]) if argv else ""
    return Path(first).name


@pytest.fixture
def submissions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch both SLURM strategies' scheduler seams; refuse any other.

    Returns the list each (fake) submission appends its kind to, so a test can
    prove the real strategy reached submission.
    """
    import phenotypic.sdk_.slurm as sdk_slurm
    from phenotypic._cli import _cli_execution_strategies as strategies
    from phenotypic._cli import _cli_slurm_config, _cli_staged_slurm
    from phenotypic._cli._cli_slurm_submission import SLURMScriptChainSubmission

    seen: list[str] = []

    def _submit_chain(*, flat_chunk_scripts, **_kwargs) -> SLURMScriptChainSubmission:
        seen.append("ordinary")
        return SLURMScriptChainSubmission(
            job_ids=["4242", "4243"],
            warning=None,
            flat_scripts=list(flat_chunk_scripts),
            dispatcher_scripts=[],
        )

    def _submit_controller(*_args: Any, **_kwargs: Any) -> str:
        seen.append("staged")
        return "4244"

    # `submit_slurm_script_chain` is imported into the strategies module, so it
    # is patched there; patching `phenotypicCLI` only reaches recompile.
    monkeypatch.setattr(strategies, "submit_slurm_script_chain", _submit_chain)
    monkeypatch.setattr(strategies, "get_slurm_array_limit", lambda: 1000)
    monkeypatch.setattr(_cli_staged_slurm, "submit_with_intent", _submit_controller)
    monkeypatch.setattr(_cli_staged_slurm, "get_slurm_array_limit", lambda: 1000)
    monkeypatch.setattr(_cli_staged_slurm, "get_slurm_max_submit_jobs", lambda: 100)
    # The max-submit and array-limit probes, wherever else they are resolved.
    monkeypatch.setattr(sdk_slurm, "get_slurm_array_limit", lambda: 1000)
    monkeypatch.setattr(sdk_slurm, "get_slurm_max_submit_jobs", lambda: 100)
    # The checkpoint-interval estimate runs `sinfo` for the partition; the
    # array-script writer imports it from here at call time.
    monkeypatch.setattr(
        _cli_slurm_config, "estimate_concurrent_capacity", lambda **_kwargs: 16
    )

    real_popen = subprocess.Popen

    def _refuse_scheduler(args: Any, *popen_args: Any, **popen_kwargs: Any):
        if _command_name(args) in _SCHEDULER_COMMANDS:
            raise AssertionError(f"a unit test started a scheduler client: {args!r}")
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(subprocess, "Popen", _refuse_scheduler)
    return seen


def _slurm_args(pipeline: Path, input_root: Path, output: Path) -> list[str]:
    return [
        "--pipeline",
        str(pipeline),
        "--input",
        str(input_root),
        "--output",
        str(output),
        "--slurm",
        "slurm_partition=short",
        "--skip-validation",
    ]


def _assert_returned_at_submission(output: str, title: str) -> None:
    assert title in output
    assert "COMPLETE" not in output
    assert "Success rate" not in output
    assert "Completed:" not in output
    # Points at the strategy's monitor lines rather than repeating them.
    assert output.count(_MONITOR_HEADER) == 1
    assert output.count(_SQUEUE_LINE) == 1
    assert output.index(_MONITOR_HEADER) < output.index(title)
    assert _POINTER in output.split(title, 1)[1]


def test_an_ordinary_slurm_run_returns_at_submission(
    tmp_path: Path,
    synth_one_level_input: Path,
    simple_pipeline_json: Path,
    submissions: list[str],
) -> None:
    output = tmp_path / "out"

    result = CliRunner().invoke(
        phenotypic_cli, _slurm_args(simple_pipeline_json, synth_one_level_input, output)
    )

    assert result.exit_code == 0, result.output
    assert submissions == ["ordinary"]
    _assert_returned_at_submission(result.output, "PROCESSING SUBMITTED")
    assert "Submitted: 1 image(s)" in result.output
    # The finalizer is the sole publisher: nothing is aggregated here.
    assert "Aggregating measurements" not in result.output
    assert readme_md_path(output).is_file()
    assert not processing_report_html_path(output).exists()


def test_a_process_slurm_run_returns_at_submission(
    tmp_path: Path,
    synth_one_level_input: Path,
    simple_pipeline_json: Path,
    submissions: list[str],
) -> None:
    output = tmp_path / "out"

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            *_slurm_args(simple_pipeline_json, synth_one_level_input, output),
            "--mode",
            "process",
            "--layer",
            "rgb",
        ],
    )

    assert result.exit_code == 0, result.output
    assert submissions == ["ordinary"]
    _assert_returned_at_submission(result.output, "PROCESS-ONLY SUBMITTED")
    assert "Layer to export: rgb" in result.output
    # Process runs build no dashboard, so the monitor lines offer none.
    assert "dashboard.html" not in result.output.split(_MONITOR_HEADER, 1)[1]
    assert not processing_report_html_path(output).exists()


def test_a_staged_slurm_run_returns_at_submission(
    tmp_path: Path,
    synth_one_level_input: Path,
    submissions: list[str],
) -> None:
    """A GPU pipeline takes ``StagedSlurmStrategy``; it ends the same way."""
    from tests.unit.cli.conftest import write_stageable_pipeline

    output = tmp_path / "out"
    pipeline = write_stageable_pipeline(tmp_path / "staged_pipeline.json")

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            *_slurm_args(pipeline, synth_one_level_input, output),
            "--image-type",
            "Image",
        ],
    )

    assert result.exit_code == 0, result.output
    assert submissions == ["staged"]
    _assert_returned_at_submission(result.output, "PROCESSING SUBMITTED")
    # The staged path builds no dashboard at submission.
    assert "dashboard.html" not in result.output.split(_MONITOR_HEADER, 1)[1]
    assert readme_md_path(output).is_file()
    assert not processing_report_html_path(output).exists()


@pytest.mark.parametrize(
    "mode_args",
    [
        [],
        ["--mode", "process", "--layer", "rgb"],
        ["--mode", "measure"],
        ["--mode", "recompile"],
        ["--mode", "migrate"],
    ],
    ids=["full", "process", "measure", "recompile", "migrate"],
)
def test_wait_is_an_unknown_option(tmp_path: Path, mode_args: list[str]) -> None:
    result = CliRunner().invoke(
        phenotypic_cli,
        [*mode_args, "--output", str(tmp_path / "out"), "--wait"],
    )

    assert result.exit_code == 2, result.output
    assert "No such option: --wait" in result.output
