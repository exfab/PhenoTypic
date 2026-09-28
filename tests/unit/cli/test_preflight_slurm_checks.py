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


# Slurm's own messages (slurm_errno.c, SchedMD/slurm master, read 2026-09-25);
# ``sbatch --test-only`` prints them as "allocation failure: <message>".
REJECTIONS = [
    "allocation failure: Invalid partition name specified",
    "allocation failure: Invalid account or account/partition combination specified",
    "allocation failure: Invalid qos specification",
    "allocation failure: Invalid generic resource (gres) specification",
    "sbatch: unrecognized option '--memm=4G'",
]
NOT_REJECTIONS = [
    # will-run ignores DOWN and DRAINED nodes; a real submission queues (review E1)
    "allocation failure: Requested node configuration is not available",
    "allocation failure: Requested partition configuration not available now",
    "allocation failure: Zero Bytes were transmitted or received",
    "allocation failure: Communication connection failure",
    "allocation failure: Invalid authentication credential",
    "allocation failure: MaxJobCount limit reached",
    "allocation failure: Resource temporarily unavailable",
    "sbatch: error: job_submit plugin says: some site-specific text",
]


@pytest.mark.parametrize("stderr", REJECTIONS)
def test_a_configuration_fault_is_an_error_carrying_stderr(scheduler, stderr: str) -> None:
    scheduler.responses["sbatch --test-only"] = (1, "", stderr)

    (finding,) = check_slurm_profiles(_slurm(slurm_partition="nope"))

    assert finding.code == "PF-SBATCH-REJECTED" and finding.severity == "error"
    assert stderr in finding.message


@pytest.mark.parametrize("stderr", NOT_REJECTIONS)
def test_a_failure_a_real_submission_may_queue_through_is_a_warning(scheduler, stderr: str) -> None:
    """Review E1: a drained partition fails --test-only, yet the real job queues."""
    scheduler.responses["sbatch --test-only"] = (1, "", stderr)

    (finding,) = check_slurm_profiles(_slurm())

    assert finding.code == "PF-SBATCH-UNAVAILABLE" and finding.severity == "warning"
    assert stderr in finding.message


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


def test_a_process_mode_gpu_run_tests_the_profile_it_submits(scheduler) -> None:
    """Review E2: the non-staged strategy adds a GPU request; so must the test."""
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()})
    context = make_context(
        pipeline, "process", slurm_args={"slurm_partition": "gpu"}, force_local=False
    )
    scheduler.responses["sinfo -p gpu"] = (0, "(null)\n", "")

    check_slurm_profiles(context)
    (finding,) = check_gpu_partition(context)

    (script,) = [call["input"] for call in scheduler.calls if call["command"][0] == "sbatch"]
    assert "--gpus-per-node=1" in script
    assert finding.code == "PF-GPU-PARTITION" and finding.severity == "error"


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


@pytest.mark.parametrize(
    ("enforce", "consequence"),
    [
        ("NO", "the job would be accepted and then pend indefinitely (EnforcePartLimits=NO)"),
        ("ALL", "the job would be rejected at submission (EnforcePartLimits=ALL)"),
        (None, "the job would pend or be rejected (EnforcePartLimits could not be read)"),
    ],
)
def test_time_over_the_partition_limit_warns_with_the_live_setting(
    scheduler, enforce, consequence
) -> None:
    """The whole sentence, so the wording cannot garble again (review E7)."""
    scheduler.responses["scontrol show partition"] = (0, _partition_info("02:00:00"), "")
    scheduler.responses["scontrol show config"] = (
        (0, f"EnforcePartLimits       = {enforce}\n", "") if enforce else (1, "", "")
    )

    (finding,) = check_partition_time(_slurm(slurm_time="03:00:00"))

    assert finding.code == "PF-TIME-OVER-PARTITION" and finding.severity == "warning"
    assert finding.message == (
        "the CPU profile (--slurm) requests 03:00:00, above MaxTime 02:00:00 of "
        f"partition short; {consequence}"
    )


def test_the_limit_itself_is_within_it(scheduler) -> None:
    """Review E10 (M13): a request equal to MaxTime fits."""
    scheduler.responses["scontrol show partition"] = (0, _partition_info("02:00:00"), "")

    assert check_partition_time(_slurm(slurm_time="02:00:00")) == []


def test_time_within_or_unlimited_is_fine(scheduler) -> None:
    scheduler.responses["scontrol show partition"] = (0, _partition_info("UNLIMITED"), "")

    assert check_partition_time(_slurm(slurm_time="30-00:00:00")) == []


@pytest.mark.parametrize(
    ("enforce", "requested", "warned_about"),
    [
        ("ALL", "05:00:00", "partition b"),  # must satisfy every partition: tightest decides
        ("ANY", "05:00:00", None),           # satisfies partition a: accepted
        ("NO", "05:00:00", None),            # pends only when over every partition
        ("NO", "12:00:00", "partition a"),   # over both: the loosest is named
    ],
)
def test_a_partition_list_follows_enforce_part_limits(
    monkeypatch, enforce, requested, warned_about
) -> None:
    """Review E7: slurm.conf.5's EnforcePartLimits rule for a partition list."""
    limits = {"a": "10:00:00", "b": "02:00:00"}

    def run(command, *, input=None, timeout, env=None):
        if command[:3] == ["scontrol", "show", "partition"]:
            name = command[3]
            return subprocess.CompletedProcess(command, 0, _partition_info(limits[name], name=name), "")
        return subprocess.CompletedProcess(command, 0, f"EnforcePartLimits = {enforce}\n", "")

    monkeypatch.setattr(_cli_preflight, "_run_scheduler_command", run)
    monkeypatch.setattr(_cli_preflight, "_scheduler_available", lambda name: True)

    findings = check_partition_time(_slurm(slurm_partition="a,b", slurm_time=requested))

    if warned_about is None:
        assert findings == []
    else:
        (finding,) = findings
        assert warned_about in finding.message


def test_no_partition_uses_the_default_partition(scheduler) -> None:
    """The default is the LOOSER one, so dropping the filter fails (review E10, M9)."""
    listing = _partition_info("10:00:00", default="NO", name="huge") + _partition_info(
        "02:00:00", default="YES", name="long"
    )
    scheduler.responses["scontrol show partition"] = (0, listing, "")

    (finding,) = check_partition_time(
        make_context(_cpu_pipeline(), slurm_args={"slurm_time": "03:00:00"}, force_local=False)
    )

    assert "partition long" in finding.message


def test_the_gui_spelling_of_partition_and_time_is_read(scheduler) -> None:
    """Review E3: the GUI sends partition=/time=, which render the same directives."""
    scheduler.responses["scontrol show partition"] = (
        0, _partition_info("04:00:00", name="gpu-short"), ""
    )

    (finding,) = check_partition_time(make_context(
        _cpu_pipeline(), slurm_args={"partition": "gpu-short", "time": "1-00:00:00"},
        force_local=False,
    ))

    assert "partition gpu-short" in finding.message
    assert ["scontrol", "show", "partition", "gpu-short"] in [c["command"] for c in scheduler.calls]


def test_sbatch_partition_in_the_environment_overrides_the_script(scheduler, monkeypatch) -> None:
    """Review E7: SBATCH_* variables override #SBATCH lines (sbatch.1, NOTE)."""
    monkeypatch.setenv("SBATCH_PARTITION", "override")
    scheduler.responses["scontrol show partition"] = (
        0, _partition_info("01:00:00", name="override"), ""
    )

    (finding,) = check_partition_time(_slurm(slurm_time="03:00:00"))

    assert "partition override" in finding.message
    assert ["scontrol", "show", "partition", "override"] in [c["command"] for c in scheduler.calls]


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


@pytest.mark.parametrize(
    "response",
    [
        (0, "", ""),  # what sinfo prints for an unknown or hidden partition (review E6)
        (1, "", "slurm_load_partitions: Unable to contact slurm controller"),
    ],
)
def test_an_unknown_or_unreadable_partition_is_not_reported_as_having_no_gpus(
    scheduler, response
) -> None:
    """sbatch --test-only reports an unknown partition; sinfo cannot tell it apart."""
    scheduler.responses["sinfo -p nope"] = response
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()})

    assert check_gpu_partition(_slurm(pipeline, slurm_partition="nope")) == []


def test_a_gpu_gres_after_a_long_entry_is_seen(scheduler) -> None:
    """Review E5: --Format=gres truncated this at 20 characters; -o %G does not."""
    scheduler.responses["sinfo -p short"] = (0, "shard:a100:64(S:0-1),gpu:a100:4(S:0-1)\n", "")
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()})

    assert check_gpu_partition(_slurm(pipeline)) == []
    (sinfo,) = [c["command"] for c in scheduler.calls if c["command"][0] == "sinfo"]
    assert sinfo[-2:] == ["-o", "%G"]


def test_the_gpu_stage_is_checked_in_the_partition_it_runs_in(scheduler) -> None:
    """Review E3: slurm_partition=batch with --gpu-slurm partition=gpu runs in gpu."""
    scheduler.responses["sinfo -p gpu"] = (0, "gpu:a100:4\n", "")
    scheduler.responses["sinfo -p batch"] = (0, "(null)\n", "")
    pipeline = ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()})
    context = make_context(
        pipeline, slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"partition": "gpu"}, force_local=False,
    )

    assert check_gpu_partition(context) == []


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


def test_the_dry_run_preview_shows_every_submitted_profile(tmp_path: Path, monkeypatch) -> None:
    """Review E9: a staged GPU run also submits the GPU stage's profile."""
    import phenotypic
    from phenotypic._cli._cli_interactive import _submitted_slurm_profiles

    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)
    from tests.unit.cli._preflight_support import make_config

    pipeline = tmp_path / "gpu.json"
    pipeline.write_text(
        ImagePipeline(ops={"gpu": FakeGpuDetector()}, meas={"s": MeasureSize()}).to_json(),
        encoding="utf-8",
    )
    config = make_config(
        pipeline_json=pipeline, slurm_args={"slurm_partition": "cpu"},
        gpu_slurm_args={"slurm_partition": "gpu"}, force_local=False,
    )

    labels = [label for label, _ in _submitted_slurm_profiles(config)]
    (gpu,) = [profile for label, profile in _submitted_slurm_profiles(config) if "GPU" in label]

    assert len(labels) == 2
    assert gpu["slurm_partition"] == "gpu" and gpu["slurm_gpus_per_node"] == 1


def _fake_scheduler_bin(tmp_path: Path, sbatch_stderr: str) -> Path:
    """sbatch/scontrol/sinfo stand-ins: sbatch --test-only fails with *sbatch_stderr*."""
    import stat
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "sbatch": (
            "import sys\n"
            "sys.stdin.read()\n"
            f"sys.stderr.write({sbatch_stderr!r} + '\\n')\n"
            "sys.exit(1)\n"
        ),
        "scontrol": "print('')\n",
        "sinfo": "print('')\n",
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bin_dir


@pytest.mark.parametrize(
    ("stderr", "exit_code", "code"),
    [
        ("allocation failure: Invalid partition name specified", 1, "PF-SBATCH-REJECTED"),
        ("allocation failure: Requested node configuration is not available", 0, "PF-SBATCH-UNAVAILABLE"),
    ],
)
def test_the_cli_dry_run_reports_the_real_sbatch_verdict(
    tmp_path: Path, monkeypatch, stderr: str, exit_code: int, code: str
) -> None:
    """Review E10: through the real subprocess path, a rejection refuses and a drain warns."""
    import os

    import numpy as np
    import tifffile

    bin_dir = _fake_scheduler_bin(tmp_path, stderr)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    pipeline = tmp_path / "p.json"
    pipeline.write_text(_cpu_pipeline().to_json(), encoding="utf-8")
    (tmp_path / "in" / "plate1").mkdir(parents=True)
    tifffile.imwrite(tmp_path / "in" / "plate1" / "img001.tiff", np.zeros((16, 16, 3), np.uint8))

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tmp_path / "in"),
         "--output", str(tmp_path / "out"), "--image-type", "Image",
         "--slurm", "slurm_partition=nope", "--dry-run"],
    )

    assert f"[{code}]" in result.output, result.output
    assert result.exit_code == exit_code, result.output
    assert not (tmp_path / "out").exists()
