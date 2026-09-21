"""``scheduler_job_is_active`` must not read an aged-out job ID as "unknown".

slurmctld forgets finished jobs after ``MinJobAge`` (300 s here), after which
``squeue --jobs <id>`` exits non-zero with ``Invalid job id specified``. The
staged controller blocks on any answer that is not ``False`` (INV-ONEWRITER
fail-safe), so mapping that error to ``None`` made a finished stage-1 array look
permanently live: the controller resubmitted itself forever and stage 2 never
started. The accounting database (``sacct``) still holds the job, so it is the
source of truth once the controller has forgotten it -- but only a *definitive*
all-terminal answer may unblock; anything else stays ``None``.
"""

from __future__ import annotations

import subprocess

import pytest

from phenotypic._cli import _cli_staged_orchestration as orch

INVALID = "slurm_load_jobs error: Invalid job id specified\n"


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        [], returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture
def fake_slurm(monkeypatch):
    """Route ``squeue``/``sacct`` calls to canned results; record each argv."""
    calls: list[list[str]] = []
    answers: dict[str, subprocess.CompletedProcess | BaseException] = {}

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        answer = answers[cmd[0]]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    return answers, calls


def test_queued_job_is_active(fake_slurm):
    answers, _ = fake_slurm
    answers["squeue"] = _completed(0, stdout="RUNNING\n")
    assert orch.scheduler_job_is_active("1") is True


def test_job_absent_from_healthy_queue_is_inactive(fake_slurm):
    answers, _ = fake_slurm
    answers["squeue"] = _completed(0, stdout="")
    assert orch.scheduler_job_is_active("1") is False


def test_aged_out_job_finished_per_sacct_is_inactive(fake_slurm):
    answers, _ = fake_slurm
    answers["squeue"] = _completed(1, stderr=INVALID)
    answers["sacct"] = _completed(
        0, stdout="COMPLETED\nCOMPLETED\nFAILED\nCANCELLED by 123\n"
    )
    assert orch.scheduler_job_is_active("1") is False


@pytest.mark.parametrize(
    "sacct",
    [
        _completed(0, stdout=""),  # sacct has no record either
        _completed(0, stdout="COMPLETED\nRUNNING\n"),  # one task still live
        _completed(0, stdout="COMPLETED\nPENDING\n"),
        _completed(
            0, stdout="COMPLETED\nSUSPENDED\n"
        ),  # unrecognised -> unsafe
        # Non-zero exit must win even when partial rows were printed
        # (multi-cluster / federation partial failure).
        _completed(
            1,
            stdout="COMPLETED\n",
            stderr="sacct: error: Problem talking to the database\n",
        ),
    ],
)
def test_aged_out_job_without_definitive_sacct_answer_stays_unknown(
    fake_slurm, sacct
):
    answers, _ = fake_slurm
    answers["squeue"] = _completed(1, stderr=INVALID)
    answers["sacct"] = sacct
    assert orch.scheduler_job_is_active("1") is None


def test_blank_sacct_rows_are_ignored_not_read_as_states(fake_slurm):
    """Padding must neither block a terminal answer nor fake one from nothing."""
    answers, _ = fake_slurm
    answers["squeue"] = _completed(1, stderr=INVALID)
    answers["sacct"] = _completed(0, stdout="\nCOMPLETED\n  \n COMPLETED \n\n")
    assert orch.scheduler_job_is_active("1") is False

    answers["sacct"] = _completed(0, stdout="\n   \n\n")
    assert orch.scheduler_job_is_active("1") is None


def test_sacct_is_called_with_the_flags_the_parser_depends_on(fake_slurm):
    """Each flag changes sacct's real output shape, which the parser assumes.

    ``-X`` one row per task (no ``.batch``/``.extern`` steps that can read
    RUNNING), ``-n`` no ``State`` header row, ``-P`` untruncated states,
    ``--format=State`` a single column. The canned stdout above cannot notice
    any of these being dropped, so pin the argv.
    """
    answers, calls = fake_slurm
    answers["squeue"] = _completed(1, stderr=INVALID)
    answers["sacct"] = _completed(0, stdout="COMPLETED\n")
    orch.scheduler_job_is_active("1")
    assert ["sacct", "-j", "1", "-X", "-n", "-P", "--format=State"] in calls


def test_sacct_header_row_is_not_read_as_a_terminal_state(fake_slurm):
    """A ``State`` header (``-n`` dropped) must fail safe, not unblock."""
    answers, _ = fake_slurm
    answers["squeue"] = _completed(1, stderr=INVALID)
    answers["sacct"] = _completed(0, stdout="State\nCOMPLETED\n")
    assert orch.scheduler_job_is_active("1") is None


@pytest.mark.parametrize(
    "boom",
    [
        subprocess.TimeoutExpired(["sacct"], 30),
        FileNotFoundError("sacct"),
    ],
)
def test_unusable_sacct_stays_unknown(fake_slurm, boom):
    answers, _ = fake_slurm
    answers["squeue"] = _completed(1, stderr=INVALID)
    answers["sacct"] = boom
    assert orch.scheduler_job_is_active("1") is None


def test_unreachable_controller_stays_unknown_and_skips_sacct(fake_slurm):
    answers, calls = fake_slurm
    answers["squeue"] = _completed(
        1, stderr="squeue: error: Unable to contact slurm controller\n"
    )
    assert orch.scheduler_job_is_active("1") is None
    assert all(argv[0] != "sacct" for argv in calls)
