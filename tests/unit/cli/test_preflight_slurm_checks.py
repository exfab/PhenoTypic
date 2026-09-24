"""Cluster checks of the run preflight (spec 2026-09-24-cli-preflight §6).

Findings F16-F19, F26; review R11, R15. No Slurm is needed: every scheduler
call goes through ``_cli_preflight._run_scheduler_command``, which these tests
replace with a fake that records the command and returns a scripted result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_preflight
from phenotypic._cli._cli_preflight import (
    SBATCH_COMMUNICATION_PATTERNS,
    check_gpu_partition,
    check_partition_time,
    check_slurm_profiles,
    check_staged_slurm_limits,
    parse_slurm_duration_minutes,
)
from phenotypic._cli._cli_staged_slurm import staged_slurm_limit_errors
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli
from tests._fakes.fake_gpu_detector import FakeGpuDetector
from tests.unit.cli._preflight_support import make_context


class FakeScheduler:
    """Scripted replacement for ``_run_scheduler_command``."""

    def __init__(self, responses: dict[str, tuple[int, str, str]] | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []

    def __call__(self, command, *, input=None, timeout, env=None):
        self.calls.append({"command": list(command), "input": input, "timeout": timeout, "env": env})
        key = " ".join(command[:3])
        for prefix, (code, out, err) in self.responses.items():
            if key.startswith(prefix):
                return subprocess.CompletedProcess(command, code, out, err)
        return subprocess.CompletedProcess(command, 0, "", "sbatch: Job 1 to start at now")


@pytest.fixture
def scheduler(monkeypatch) -> FakeScheduler:
    fake = FakeScheduler()
    monkeypatch.setattr(_cli_preflight, "_run_scheduler_command", fake)
    monkeypatch.setattr(_cli_preflight, "_scheduler_available", lambda name: True)
    return fake


def _cpu_pipeline() -> ImagePipeline:
    return ImagePipeline(ops={"d": OtsuDetector()}, meas={"s": MeasureSize()})


def _slurm(pipeline=None, **slurm):
    return make_context(
        pipeline or _cpu_pipeline(),
        slurm_args={"slurm_partition": "short", "slurm_time": "01:00:00", **slurm},
        force_local=False,
    )


# --- --gpu-slurm time (F16) --------------------------------------------------------


def test_gpu_slurm_time_is_parsed_at_startup_even_with_skip_validation(tmp_path: Path) -> None:
    pipeline = tmp_path / "p.json"
    pipeline.write_text(_cpu_pipeline().to_json(), encoding="utf-8")
    (tmp_path / "in").mkdir()

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tmp_path / "in"),
         "--output", str(tmp_path / "out"), "--gpu-slurm", "time=banana",
         "--skip-validation", "--dry-run"],
    )

    assert result.exit_code != 0
    assert "banana" in result.output


# --- sbatch --test-only (F17, R15) ---------------------------------------------------


def test_the_test_script_is_well_formed_and_uses_the_submission_environment(
    scheduler, monkeypatch
) -> None:
    # The snapshot exists only when PYTHONPATH is set; pin it rather than
    # inherit the runner's environment.
    monkeypatch.setenv("PYTHONPATH", "/site/custom_ops")

    assert check_slurm_profiles(_slurm()) == []

    (call,) = scheduler.calls
    assert call["command"] == ["sbatch", "--test-only"]
    assert call["input"].startswith("#!/bin/bash\n")
    assert "--output=/dev/null" in call["input"] and "--error=/dev/null" in call["input"]
    assert "--partition=short" in call["input"]
    assert call["timeout"] == 30
    assert call["env"]["PHENOTYPIC_SLURM_PYTHONPATH"] == "/site/custom_ops"


def test_a_rejected_profile_is_an_error_carrying_stderr(scheduler) -> None:
    scheduler.responses["sbatch --test-only"] = (
        1, "", "sbatch: error: invalid partition specified: nope"
    )

    (finding,) = check_slurm_profiles(_slurm(slurm_partition="nope"))

    assert finding.code == "PF-SBATCH-REJECTED" and finding.severity == "error"
    assert "invalid partition specified" in finding.message


@pytest.mark.parametrize("stderr", [f"sbatch: error: {p}" for p in SBATCH_COMMUNICATION_PATTERNS])
def test_a_controller_fault_is_only_a_warning(scheduler, stderr: str) -> None:
    scheduler.responses["sbatch --test-only"] = (1, "", stderr)

    (finding,) = check_slurm_profiles(_slurm())

    assert finding.code == "PF-SBATCH-UNAVAILABLE" and finding.severity == "warning"


def test_missing_sbatch_and_a_timeout_are_warnings(monkeypatch) -> None:
    monkeypatch.setattr(_cli_preflight, "_scheduler_available", lambda name: False)
    (missing,) = check_slurm_profiles(_slurm())
    assert missing.code == "PF-SBATCH-UNAVAILABLE"

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired("sbatch", 30)

    monkeypatch.setattr(_cli_preflight, "_scheduler_available", lambda name: True)
    monkeypatch.setattr(_cli_preflight, "_run_scheduler_command", hang)
    (timed_out,) = check_slurm_profiles(_slurm())
    assert timed_out.code == "PF-SBATCH-UNAVAILABLE"


def test_a_staged_gpu_run_tests_both_profiles(scheduler) -> None:
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()})

    check_slurm_profiles(_slurm(pipeline))

    scripts = [call["input"] for call in scheduler.calls]
    assert len(scripts) == 2
    assert sum("--gpus-per-node=1" in script for script in scripts) == 1


def test_a_local_run_checks_nothing(scheduler) -> None:
    assert check_slurm_profiles(make_context(_cpu_pipeline())) == []
    assert scheduler.calls == []


# --- partition time (F19, R11) --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "minutes"),
    [("UNLIMITED", None), ("2-00:00:00", 2880), ("1-12:00:00", 2160), ("04:30:00", 270),
     ("45:00", 45), ("30", 30), ("infinite", None)],
)
def test_slurm_durations_parse(value: str, minutes) -> None:
    assert parse_slurm_duration_minutes(value) == minutes


def _partition_info(max_time: str, default: str = "NO", name: str = "short") -> str:
    return f"PartitionName={name}\n   AllowGroups=ALL Default={default}\n   MaxTime={max_time} MinNodes=0\n"


def test_time_over_the_partition_limit_warns_with_the_live_setting(scheduler) -> None:
    scheduler.responses["scontrol show partition"] = (0, _partition_info("02:00:00"), "")
    scheduler.responses["scontrol show config"] = (0, "EnforcePartLimits       = NO\n", "")

    (finding,) = check_partition_time(_slurm(slurm_time="03:00:00"))

    assert finding.code == "PF-TIME-OVER-PARTITION" and finding.severity == "warning"
    assert "pend" in finding.message


def test_enforced_limits_change_the_wording(scheduler) -> None:
    scheduler.responses["scontrol show partition"] = (0, _partition_info("02:00:00"), "")
    scheduler.responses["scontrol show config"] = (0, "EnforcePartLimits       = ALL\n", "")

    (finding,) = check_partition_time(_slurm(slurm_time="03:00:00"))

    assert "rejected" in finding.message


def test_time_within_or_unlimited_is_fine(scheduler) -> None:
    scheduler.responses["scontrol show partition"] = (0, _partition_info("UNLIMITED"), "")

    assert check_partition_time(_slurm(slurm_time="30-00:00:00")) == []


def test_a_partition_list_uses_the_tightest_limit(monkeypatch) -> None:
    limits = {"a": "10:00:00", "b": "02:00:00"}

    def run(command, *, input=None, timeout, env=None):
        if command[:3] == ["scontrol", "show", "partition"]:
            name = command[3]
            return subprocess.CompletedProcess(command, 0, _partition_info(limits[name], name=name), "")
        return subprocess.CompletedProcess(command, 0, "EnforcePartLimits = NO\n", "")

    monkeypatch.setattr(_cli_preflight, "_run_scheduler_command", run)
    monkeypatch.setattr(_cli_preflight, "_scheduler_available", lambda name: True)

    (finding,) = check_partition_time(_slurm(slurm_partition="a,b", slurm_time="05:00:00"))

    assert "02:00:00" in finding.message and "partition b" in finding.message


def test_no_partition_uses_the_default_partition(scheduler) -> None:
    listing = _partition_info("01:00:00", default="NO", name="gpu") + _partition_info(
        "00:30:00", default="YES", name="short"
    )
    scheduler.responses["scontrol show partition"] = (0, listing, "")

    (finding,) = check_partition_time(
        make_context(_cpu_pipeline(), slurm_args={"slurm_time": "01:00:00"}, force_local=False)
    )

    assert "short" in finding.message


# --- staged limits and GRES (F18) -------------------------------------------------------


def test_staged_limit_errors_keep_the_existing_messages() -> None:
    assert staged_slurm_limit_errors(2, 1000, 1) == [
        "SLURM MaxSubmitJobs must be at least 3 for staged GPU orchestration "
        "(controller, array, recovery controller)."
    ]
    assert staged_slurm_limit_errors(10, 4, 8) == [
        "--gpu-shards (8) exceeds the SLURM chunk limit (4); reduce the shard count."
    ]
    assert staged_slurm_limit_errors(None, 1000, 1) == []


def test_the_staged_limits_are_checked_before_submission(monkeypatch) -> None:
    monkeypatch.setattr(_cli_preflight, "_slurm_submission_limits", lambda: (2, 1000))
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()})

    (finding,) = check_staged_slurm_limits(_slurm(pipeline))

    assert finding.code == "PF-SLURM-LIMIT" and finding.severity == "error"


def test_an_unknown_partition_is_not_reported_as_having_no_gpus(scheduler) -> None:
    scheduler.responses["sinfo -p nope"] = (1, "", "sinfo: error: Invalid partition name")
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()})

    (finding,) = check_gpu_partition(_slurm(pipeline, slurm_partition="nope"))

    assert finding.code == "PF-GPU-PARTITION"
    assert "no GPUs" not in finding.message
    assert "Invalid partition name" in finding.message


def test_a_cpu_partition_for_the_gpu_stage_is_an_error(scheduler) -> None:
    scheduler.responses["sinfo -p short"] = (0, "(null)\n", "")
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()})

    (finding,) = check_gpu_partition(_slurm(pipeline))

    assert finding.severity == "error" and "no GPUs" in finding.message


# --- dry-run preview (F26) --------------------------------------------------------------


def test_the_dry_run_preview_prints_the_real_directives(capsys) -> None:
    from phenotypic._cli._cli_interactive import _display_slurm_config

    _display_slurm_config({"slurm_partition": "short", "slurm_time": 90, "mem_gb": 8})
    printed = capsys.readouterr().out

    assert "#SBATCH --partition=short" in printed
    assert "#SBATCH --time=01:30:00" in printed
    assert "#SBATCH --mem=8G" in printed
