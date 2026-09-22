"""``scheduler_job_is_active`` must not read an aged-out job ID as "unknown".

slurmctld forgets finished jobs after ``MinJobAge`` (300 s here), after which
``squeue --jobs <id>`` exits non-zero with ``Invalid job id specified``. The
staged controller blocks on any answer that is not ``False`` (INV-ONEWRITER
fail-safe), so mapping that error to ``None`` made a finished stage-1 array look
permanently live: the controller resubmitted itself forever and stage 2 never
started. The accounting database (``sacct``) still holds the job, so it is the
source of truth once the controller has forgotten it -- but only a *definitive*
answer may unblock; anything else stays ``None``.

A requeued array task has one accounting row per attempt. An earlier attempt
that was never closed (its node died) keeps reading ``RUNNING`` forever, so only
each task's *latest* attempt counts.
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


def _rows(*rows: tuple[str, str, str]) -> str:
    """Render ``(jobid, state, start)`` triples as ``sacct -nP`` output."""
    return "".join(f"{job}|{state}|{start}\n" for job, state, start in rows)


T1, T2, T3 = (f"2026-09-19T0{i}:00:00" for i in (1, 2, 3))


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


def _aged_out(fake_slurm, sacct):
    """Make squeue report the id forgotten and sacct answer ``sacct``."""
    answers, _ = fake_slurm
    answers["squeue"] = _completed(1, stderr=INVALID)
    answers["sacct"] = sacct


def test_queued_job_is_active(fake_slurm):
    answers, _ = fake_slurm
    answers["squeue"] = _completed(0, stdout="RUNNING\n")
    assert orch.scheduler_job_is_active("1") is True


def test_job_absent_from_healthy_queue_is_inactive(fake_slurm):
    answers, _ = fake_slurm
    answers["squeue"] = _completed(0, stdout="")
    assert orch.scheduler_job_is_active("1") is False


def test_aged_out_job_finished_per_sacct_is_inactive(fake_slurm):
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout=_rows(
                ("1_0", "COMPLETED", T1),
                ("1_1", "COMPLETED", T1),
                ("1_2", "FAILED", T1),
                ("1_3", "CANCELLED by 123", T1),
            ),
        ),
    )
    assert orch.scheduler_job_is_active("1") is False


@pytest.mark.parametrize(
    "sacct",
    [
        _completed(0, stdout=""),  # sacct has no record either
        # one task still live
        _completed(
            0,
            stdout=_rows(("1_0", "COMPLETED", T1), ("1_1", "RUNNING", T1)),
        ),
        # a pending task has no start time yet
        _completed(
            0,
            stdout=_rows(
                ("1_0", "COMPLETED", T1), ("1_1", "PENDING", "Unknown")
            ),
        ),
        # unrecognised -> unsafe
        _completed(
            0,
            stdout=_rows(("1_0", "COMPLETED", T1), ("1_1", "SUSPENDED", T1)),
        ),
        # Non-zero exit must win even when partial rows were printed
        # (multi-cluster / federation partial failure).
        _completed(
            1,
            stdout=_rows(("1_0", "COMPLETED", T1)),
            stderr="sacct: error: Problem talking to the database\n",
        ),
        # A row that is not jobid|state|start is not an answer.
        _completed(0, stdout="COMPLETED\n"),
        _completed(0, stdout=f"1_0||{T1}\n"),  # empty state field
    ],
)
def test_aged_out_job_without_definitive_sacct_answer_stays_unknown(
    fake_slurm, sacct
):
    _aged_out(fake_slurm, sacct)
    assert orch.scheduler_job_is_active("1") is None


def test_superseded_unclosed_attempt_does_not_keep_the_job_alive(fake_slurm):
    """The real failure: task 1605 was requeued after its node died.

    Attempt one (start T1) was never closed and still reads ``RUNNING``; the
    requeued attempt two (start T2) completed. Only the latest attempt counts.
    """
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout=_rows(
                ("1_5", "RUNNING", T1),
                ("1_5", "COMPLETED", T2),
                ("1_6", "COMPLETED", T1),
            ),
        ),
    )
    assert orch.scheduler_job_is_active("1") is False


def test_latest_attempt_still_running_keeps_the_job_alive(fake_slurm):
    """An earlier ``COMPLETED``/``NODE_FAIL`` row must not mask a live retry."""
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout=_rows(("1_5", "NODE_FAIL", T1), ("1_5", "RUNNING", T2)),
        ),
    )
    assert orch.scheduler_job_is_active("1") is None


@pytest.mark.parametrize("placeholder", ["Unknown", "None"])
def test_requeued_attempt_not_yet_started_keeps_the_job_alive(
    fake_slurm, placeholder
):
    """A requeued task waiting to restart has no start time: it is the latest.

    This cluster prints ``Unknown`` for PENDING and ``None`` for a task that
    was cancelled before it started; either placeholder on a live state wins.
    """
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout=_rows(
                ("1_5", "COMPLETED", T1), ("1_5", "PENDING", placeholder)
            ),
        ),
    )
    assert orch.scheduler_job_is_active("1") is None


def test_cancelled_before_start_row_does_not_mask_a_live_attempt(fake_slurm):
    """``CANCELLED|None`` never ran, so it must not outrank a real attempt."""
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout=_rows(
                ("1_5", "CANCELLED by 5188", "None"), ("1_5", "RUNNING", T1)
            ),
        ),
    )
    assert orch.scheduler_job_is_active("1") is None


def test_cancelled_before_start_alone_is_finished(fake_slurm):
    _aged_out(
        fake_slurm,
        _completed(0, stdout=_rows(("1_5", "CANCELLED by 5188", "None"))),
    )
    assert orch.scheduler_job_is_active("1") is False


@pytest.mark.parametrize("live_first", [True, False])
def test_tied_start_prefers_the_non_terminal_row(fake_slurm, live_first):
    """Equal start times: never let print order decide -- the live row wins."""
    rows = [("1_5", "RUNNING", T1), ("1_5", "COMPLETED", T1)]
    _aged_out(
        fake_slurm,
        _completed(
            0, stdout=_rows(*(rows if live_first else list(reversed(rows))))
        ),
    )
    assert orch.scheduler_job_is_active("1") is None


def test_ghost_row_printed_last_is_still_superseded(fake_slurm):
    """Latest-by-start, not last-printed: the ghost may come out in any order."""
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout=_rows(("1_5", "COMPLETED", T2), ("1_5", "RUNNING", T1)),
        ),
    )
    assert orch.scheduler_job_is_active("1") is False


def test_blank_sacct_rows_are_ignored_not_read_as_states(fake_slurm):
    """Padding must neither block a terminal answer nor fake one from nothing."""
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout=f"\n1_0|COMPLETED|{T1}\n  \n 1_1|COMPLETED|{T1} \n\n",
        ),
    )
    assert orch.scheduler_job_is_active("1") is False

    _aged_out(fake_slurm, _completed(0, stdout="\n   \n\n"))
    assert orch.scheduler_job_is_active("1") is None


def test_sacct_is_called_with_the_flags_the_parser_depends_on(fake_slurm):
    """Each flag changes sacct's real output shape, which the parser assumes.

    ``-X`` one row per task attempt (no ``.batch``/``.extern`` steps that can
    read RUNNING), ``-n`` no header row, ``-P`` untruncated fields,
    ``--format=JobID,State,Start`` the three columns the latest-attempt rule
    reads. The canned stdout above cannot notice any of these being dropped, so
    pin the argv.
    """
    _aged_out(
        fake_slurm, _completed(0, stdout=_rows(("1_0", "COMPLETED", T1)))
    )
    _, calls = fake_slurm
    orch.scheduler_job_is_active("1")
    assert [
        "sacct",
        "-j",
        "1",
        "-X",
        "-n",
        "-P",
        "--format=JobID,State,Start",
    ] in calls


def test_sacct_header_row_is_not_read_as_a_terminal_state(fake_slurm):
    """A header (``-n`` dropped) must fail safe, not unblock."""
    _aged_out(
        fake_slurm,
        _completed(
            0,
            stdout="JobID|State|Start\n" + _rows(("1_0", "COMPLETED", T1)),
        ),
    )
    assert orch.scheduler_job_is_active("1") is None


@pytest.mark.parametrize(
    "boom",
    [
        subprocess.TimeoutExpired(["sacct"], 30),
        FileNotFoundError("sacct"),
    ],
)
def test_unusable_sacct_stays_unknown(fake_slurm, boom):
    _aged_out(fake_slurm, boom)
    assert orch.scheduler_job_is_active("1") is None


def test_unreachable_controller_stays_unknown_and_skips_sacct(fake_slurm):
    answers, calls = fake_slurm
    answers["squeue"] = _completed(
        1, stderr="squeue: error: Unable to contact slurm controller\n"
    )
    assert orch.scheduler_job_is_active("1") is None
    assert all(argv[0] != "sacct" for argv in calls)
