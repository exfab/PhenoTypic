"""``phenotypic-wait``: the read-only SLURM run waiter (#281 spec C3, P2 T2).

The decision is a pure function (:func:`decide_wait_outcome`) so its table is
pinned without a scheduler. The loop (:func:`wait_for_run`) is driven through
the module-level seams ``_completion``/``_live_jobs``/``_fence_generation``/
``_published``/``_is_process_run``, and the seams themselves are pinned against
real lifecycle files written by the real lifecycle writers.
"""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from phenotypic._cli import (
    _cli_slurm_lifecycle,
    _cli_staged_orchestration,
    _cli_wait,
)
from phenotypic._cli._cli_slurm_lifecycle import (
    FENCE_CLOSED_BY_FINALIZER,
    SchedulerQueryUnavailable,
    append_lifecycle_entry,
    cancel_generation,
    deactivate_generation,
    initialize_slurm_lifecycle,
    load_slurm_lifecycle,
    mark_generation_failed,
)
from phenotypic._cli._cli_staged_orchestration import LiveSlurmJobs
from phenotypic._cli._cli_wait import decide_wait_outcome, main, wait_for_run
from phenotypic.sdk_ import (
    RunExitCode,
    aggregate_publication_marker_path,
    run_completion_marker_path,
)

NONE = LiveSlurmJobs()
LIVE = LiveSlurmJobs(live=("123",))


# ---------------------------------------------------------------------------
# The decision table
# ---------------------------------------------------------------------------


def test_a_live_job_keeps_waiting_even_when_complete():
    # a measure resubmission over a completed run: the old proof is still current
    assert (
        decide_wait_outcome(
            "complete", LIVE, published=False, process_run=False
        )
        is None
    )


def test_a_live_job_keeps_waiting_even_when_published():
    assert (
        decide_wait_outcome(
            "complete", LIVE, published=True, process_run=False
        )
        is None
    )


def test_unqueryable_scheduler_keeps_waiting():
    assert (
        decide_wait_outcome(
            "active",
            LiveSlurmJobs(unresolved=("phenotypic:g:t",)),
            published=False,
            process_run=False,
        )
        is None
    )


def test_an_unverified_job_keeps_waiting():
    assert (
        decide_wait_outcome(
            "failed",
            LiveSlurmJobs(unverified=("77",)),
            published=True,
            process_run=False,
        )
        is None
    )


def test_complete_and_published_is_0():
    assert (
        decide_wait_outcome(
            "complete", NONE, published=True, process_run=False
        )
        is RunExitCode.COMPLETE
    )


def test_failed_and_published_is_3():
    assert (
        decide_wait_outcome("failed", NONE, published=True, process_run=False)
        is RunExitCode.COMPLETE_WITH_FAILURES
    )


def test_a_stale_proof_with_a_dead_finalizer_is_5():
    assert (
        decide_wait_outcome("failed", NONE, published=False, process_run=False)
        is RunExitCode.ENDED_UNPUBLISHED
    )


def test_complete_but_unpublished_is_5():
    assert (
        decide_wait_outcome(
            "complete", NONE, published=False, process_run=False
        )
        is RunExitCode.ENDED_UNPUBLISHED
    )


def test_a_process_run_with_one_failure_is_3():
    assert (
        decide_wait_outcome("failed", NONE, published=False, process_run=True)
        is RunExitCode.COMPLETE_WITH_FAILURES
    )


def test_a_process_run_complete_but_unpublished_is_still_5():
    # The process carve-out is for `failed` only: a process run that reads
    # `complete` has a run proof, and this submission must have published it.
    assert (
        decide_wait_outcome(
            "complete", NONE, published=False, process_run=True
        )
        is RunExitCode.ENDED_UNPUBLISHED
    )


def test_dead_finalizer_with_active_fence_ends_unpublished():
    # Review Focus 4: the fence still says active, but no job is alive
    assert (
        decide_wait_outcome("active", NONE, published=False, process_run=False)
        is RunExitCode.ENDED_UNPUBLISHED
    )


@pytest.mark.parametrize("completion", ["active", "incomplete"])
def test_neither_complete_nor_failed_is_5_even_when_published(completion):
    assert (
        decide_wait_outcome(completion, NONE, published=True, process_run=True)
        is RunExitCode.ENDED_UNPUBLISHED
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _patch_seams(
    monkeypatch,
    *,
    completion,
    jobs,
    generations=("g",),
    published=None,
    process_run=False,
):
    completions = iter(completion) if isinstance(completion, list) else None
    monkeypatch.setattr(
        _cli_wait,
        "_completion",
        (lambda d: next(completions))
        if completions
        else (lambda d: completion),
    )
    job_iter = iter(jobs)
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: next(job_iter))
    generation_iter = iter(generations)
    monkeypatch.setattr(
        _cli_wait, "_fence_generation", lambda d: next(generation_iter)
    )
    if published is not None:
        monkeypatch.setattr(_cli_wait, "_published", published)
    monkeypatch.setattr(_cli_wait, "_is_process_run", lambda d: process_run)


def test_measure_resubmission_waits_then_binds_to_its_own_generation(
    monkeypatch, tmp_path
):
    jobs = iter([LIVE, LIVE, NONE])
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: "complete")
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: next(jobs))
    generations = iter(["g-measure", "g-later"])
    monkeypatch.setattr(
        _cli_wait, "_fence_generation", lambda d: next(generations)
    )
    seen = []
    monkeypatch.setattr(
        _cli_wait, "_published", lambda d, g: seen.append(g) or True
    )
    monkeypatch.setattr(_cli_wait, "_is_process_run", lambda d: False)
    sleeps = []
    assert (
        wait_for_run(
            tmp_path, poll_seconds=5, sleep=sleeps.append, echo=lambda *_: None
        )
        is RunExitCode.COMPLETE
    )
    assert sleeps == [5, 5] and set(seen) == {
        "g-measure"
    }  # read once, at the first poll


def test_jobs_are_read_before_completion_each_poll(monkeypatch, tmp_path):
    # A finalizer that ends between the two reads must not be judged on the
    # completion it saw before it published: liveness is read first.
    order = []
    monkeypatch.setattr(
        _cli_wait, "_live_jobs", lambda d: order.append("jobs") or NONE
    )
    monkeypatch.setattr(
        _cli_wait,
        "_completion",
        lambda d: order.append("completion") or "complete",
    )
    monkeypatch.setattr(
        _cli_wait, "_fence_generation", lambda d: order.append("fence") or "g"
    )
    monkeypatch.setattr(
        _cli_wait, "_published", lambda d, g: order.append("published") or True
    )
    monkeypatch.setattr(_cli_wait, "_is_process_run", lambda d: False)
    assert (
        wait_for_run(tmp_path, sleep=lambda s: None, echo=lambda *_: None)
        is RunExitCode.COMPLETE
    )
    assert order[:3] == ["fence", "jobs", "completion"]


def test_each_poll_prints_one_line_naming_completion_jobs_and_elapsed(
    monkeypatch, tmp_path
):
    _patch_seams(
        monkeypatch,
        completion="active",
        jobs=[
            LiveSlurmJobs(live=("123",), unresolved=("phenotypic:g:t",)),
            NONE,
        ],
        published=lambda d, g: False,
    )
    lines = []
    clock = iter([0.0, 0.0, 60.0])
    result = wait_for_run(
        tmp_path,
        poll_seconds=60,
        sleep=lambda s: None,
        clock=lambda: next(clock),
        echo=lines.append,
    )
    assert result is RunExitCode.ENDED_UNPUBLISHED
    poll_lines = [
        line for line in lines if line.startswith("phenotypic-wait: poll")
    ]
    assert len(poll_lines) == 2
    assert "completion=active" in poll_lines[0]
    assert "123" in poll_lines[0] and "phenotypic:g:t" in poll_lines[0]
    assert "elapsed=0s" in poll_lines[0] and "elapsed=60s" in poll_lines[1]
    assert any("could not be queried" in line for line in lines)


def test_timeout_returns_6_and_cancels_nothing(monkeypatch, tmp_path):
    cancelled = []
    monkeypatch.setattr(
        _cli_slurm_lifecycle,
        "cancel_generation",
        lambda *a, **k: cancelled.append(a),
    )
    clock = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: "active")
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: LIVE)
    monkeypatch.setattr(_cli_wait, "_fence_generation", lambda d: "g")
    assert (
        wait_for_run(
            tmp_path,
            poll_seconds=10,
            timeout_seconds=15,
            clock=lambda: next(clock),
            sleep=lambda s: None,
            echo=lambda *_: None,
        )
        is RunExitCode.WAIT_TIMEOUT
    )
    assert cancelled == []


def test_the_last_sleep_never_overshoots_the_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: "active")
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: LIVE)
    monkeypatch.setattr(_cli_wait, "_fence_generation", lambda d: "g")
    clock = iter([0.0, 0.0, 50.0, 70.0])
    sleeps = []
    assert (
        wait_for_run(
            tmp_path,
            poll_seconds=50,
            timeout_seconds=70,
            clock=lambda: next(clock),
            sleep=sleeps.append,
            echo=lambda *_: None,
        )
        is RunExitCode.WAIT_TIMEOUT
    )
    assert sleeps == [50, 20]


# ---------------------------------------------------------------------------
# The seams, against real lifecycle files
# ---------------------------------------------------------------------------


def test_an_intent_only_ledger_row_keeps_waiting(monkeypatch, tmp_path):
    initialize_slurm_lifecycle(tmp_path, generation="g", mode="full")
    append_lifecycle_entry(
        tmp_path, generation="g", token="t", role="finalizer", status="intent"
    )
    monkeypatch.setattr(
        _cli_staged_orchestration,
        "query_scheduler_comments",
        lambda **_: (_ for _ in ()).throw(SchedulerQueryUnavailable("down")),
    )
    assert (
        decide_wait_outcome(
            "active",
            _cli_wait._live_jobs(tmp_path),
            published=False,
            process_run=False,
        )
        is None
    )


def test_fence_generation_is_none_without_a_fence(tmp_path):
    assert _cli_wait._fence_generation(tmp_path) is None
    initialize_slurm_lifecycle(tmp_path, generation="g1", mode="full")
    assert _cli_wait._fence_generation(tmp_path) == "g1"


def test_a_fence_closed_by_its_finalizer_is_published(tmp_path):
    initialize_slurm_lifecycle(tmp_path, generation="g", mode="full")
    assert not _cli_wait._published(tmp_path, "g")
    deactivate_generation(tmp_path, "g", closed_by=FENCE_CLOSED_BY_FINALIZER)
    assert _cli_wait._published(tmp_path, "g")


def test_a_fence_closed_without_a_reason_is_not_published(tmp_path):
    # Every cancel path closes the fence with no `closed_by`.
    initialize_slurm_lifecycle(tmp_path, generation="g", mode="full")
    deactivate_generation(tmp_path, "g")
    assert not _cli_wait._published(tmp_path, "g")


def test_a_fence_marked_failed_is_not_published(tmp_path):
    initialize_slurm_lifecycle(tmp_path, generation="g", mode="full")
    mark_generation_failed(tmp_path, "g", "boom")
    assert not _cli_wait._published(tmp_path, "g")


def test_a_closed_fence_of_another_generation_is_not_published(tmp_path):
    initialize_slurm_lifecycle(tmp_path, generation="g-old", mode="full")
    deactivate_generation(
        tmp_path, "g-old", closed_by=FENCE_CLOSED_BY_FINALIZER
    )
    assert not _cli_wait._published(tmp_path, "g-new")


def test_a_run_proof_from_this_generation_is_published(tmp_path):
    initialize_slurm_lifecycle(tmp_path, generation="g", mode="full")
    marker = run_completion_marker_path(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "finalizer_succeeded": True,
        "execution_epoch": "g-other",
    }
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert not _cli_wait._published(tmp_path, "g")
    payload["execution_epoch"] = "g"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert _cli_wait._published(tmp_path, "g")


def test_with_no_fence_at_the_first_poll_the_run_state_decides(tmp_path):
    # The run before it was local: there is no submission to bind to.
    assert _cli_wait._published(tmp_path, None)


def _aggregate_proof(out):
    return json.loads(
        aggregate_publication_marker_path(out).read_text(encoding="utf-8")
    )


def _resubmit(out, generation, *, finalizer_job="999"):
    """Fence *out* for a new SLURM generation whose finalizer job is ledgered."""
    initialize_slurm_lifecycle(out, generation=generation, mode="measure")
    append_lifecycle_entry(
        out,
        generation=generation,
        token="fin",
        role="finalizer",
        status="intent",
    )
    append_lifecycle_entry(
        out,
        generation=generation,
        token="fin",
        role="finalizer",
        status="submitted",
        job_id=finalizer_job,
    )


def test_an_aggregate_proof_stamped_with_this_generation_is_published(
    finished_tiny_run,
):
    from phenotypic._cli._cli_output_manager import aggregate_measurements

    out = finished_tiny_run
    _resubmit(out, "g")
    assert not _cli_wait._published(out, "g")
    aggregate_measurements(
        output_dir=out, dataset_names=["ds"], execution_epoch="g"
    )
    assert _aggregate_proof(out)["execution_epoch"] == "g"
    assert _cli_wait._published(out, "g")
    assert not _cli_wait._published(out, "g-later")


def test_an_aggregate_proof_without_the_stamp_is_not_published(
    finished_tiny_run,
):
    # A proof written before the stamp existed never counts for a submission.
    from phenotypic.sdk_ import aggregate_proof_is_current

    out = finished_tiny_run
    _resubmit(out, "g")
    path = aggregate_publication_marker_path(out)
    proof = _aggregate_proof(out)
    proof.pop("execution_epoch", None)
    path.write_text(json.dumps(proof), encoding="utf-8")
    assert aggregate_proof_is_current(out)  # still byte-current
    assert not _cli_wait._published(out, "g")


def test_a_local_run_stamps_no_generation(finished_tiny_run):
    assert _aggregate_proof(finished_tiny_run)["execution_epoch"] is None


def test_dead_finalizer_over_an_active_fence_exits_5_end_to_end(
    monkeypatch, tmp_path
):
    # Review Focus 4 through the real seams: the finalizer was OOM-killed, so
    # the fence stays active and the only ledgered job is gone from SLURM.
    _resubmit(tmp_path, "g")
    monkeypatch.setattr(
        _cli_staged_orchestration, "scheduler_job_is_active", lambda job: False
    )
    assert (
        wait_for_run(tmp_path, sleep=lambda s: None, echo=lambda *_: None)
        is RunExitCode.ENDED_UNPUBLISHED
    )


def test_measure_resubmission_whose_finalizer_died_exits_5(
    monkeypatch, finished_tiny_run
):
    # A measure resubmission keeps no run state, so the earlier run's proofs
    # still match it and the run still reads `complete`. Only the generation
    # binding tells the dead finalizer apart from a finished one.
    out = finished_tiny_run
    _resubmit(out, "g-measure")
    monkeypatch.setattr(
        _cli_staged_orchestration, "scheduler_job_is_active", lambda job: False
    )
    lines = []
    assert wait_for_run(out, sleep=lambda s: None, echo=lines.append) is (
        RunExitCode.ENDED_UNPUBLISHED
    )
    assert "completion=complete" in lines[0]


def test_cancelled_resubmission_exits_5(monkeypatch, finished_tiny_run):
    import subprocess

    out = finished_tiny_run
    _resubmit(out, "g-cancel")
    scancelled = []

    def fake_scheduler(command, **_kwargs):
        if command[0] == "scancel":
            scancelled.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    cancel_generation(out, "g-cancel", run_command=fake_scheduler)
    lifecycle = load_slurm_lifecycle(out)
    assert lifecycle["active"] is False and "closed_by" not in lifecycle
    lines = []
    assert wait_for_run(out, sleep=lambda s: None, echo=lines.append) is (
        RunExitCode.ENDED_UNPUBLISHED
    )
    assert "completion=complete" in lines[0]


def test_a_finished_resubmission_exits_0(monkeypatch, finished_tiny_run):
    out = finished_tiny_run
    _resubmit(out, "g-done")
    deactivate_generation(out, "g-done", closed_by=FENCE_CLOSED_BY_FINALIZER)
    assert (
        wait_for_run(out, sleep=lambda s: None, echo=lambda *_: None)
        is RunExitCode.COMPLETE
    )


def test_a_stamped_aggregate_proof_alone_exits_0(
    monkeypatch, finished_tiny_run
):
    # The finalizer published, then died before closing the fence: the
    # generation stamp in the aggregate proof is what still binds it.
    from phenotypic._cli._cli_output_manager import aggregate_measurements

    out = finished_tiny_run
    _resubmit(out, "g-stamped")
    aggregate_measurements(
        output_dir=out, dataset_names=["ds"], execution_epoch="g-stamped"
    )
    monkeypatch.setattr(
        _cli_staged_orchestration, "scheduler_job_is_active", lambda job: False
    )
    assert load_slurm_lifecycle(out)["active"] is True
    assert (
        wait_for_run(out, sleep=lambda s: None, echo=lambda *_: None)
        is RunExitCode.COMPLETE
    )


def test_a_finalized_resubmission_with_a_failure_exits_3(
    tiny_inputs_one_bad, tmp_path
):
    from tests._tiny_run import run_cli

    out = tmp_path / "out"
    assert (
        run_cli(tiny_inputs_one_bad, out).exit_code
        == RunExitCode.COMPLETE_WITH_FAILURES
    )
    _resubmit(out, "g-failed")
    deactivate_generation(out, "g-failed", closed_by=FENCE_CLOSED_BY_FINALIZER)
    lines = []
    assert wait_for_run(out, sleep=lambda s: None, echo=lines.append) is (
        RunExitCode.COMPLETE_WITH_FAILURES
    )
    assert "completion=failed" in lines[0]


def test_is_process_run_is_false_without_run_state(tmp_path):
    assert not _cli_wait._is_process_run(tmp_path)


def test_is_process_run_true_for_a_process_tree(tmp_path):
    from tests._output_layout import build_complete_run

    assert _cli_wait._is_process_run(
        build_complete_run(tmp_path, process_only_layer="gray")
    )
    assert not _cli_wait._is_process_run(build_complete_run(tmp_path / "full"))


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_cli_exit_code_matches(finished_tiny_run):
    assert (
        CliRunner()
        .invoke(main, ["--output", str(finished_tiny_run)])
        .exit_code
        == 0
    )


def test_cli_refuses_a_missing_output(tmp_path):
    result = CliRunner().invoke(main, ["--output", str(tmp_path / "absent")])
    assert result.exit_code == 2


def test_cli_rejects_a_non_positive_poll(tmp_path):
    result = CliRunner().invoke(
        main, ["-o", str(tmp_path), "--poll-seconds", "0"]
    )
    assert result.exit_code == 2


def test_cli_timeout_exits_6(monkeypatch, tmp_path):
    monkeypatch.setattr(_cli_wait, "_completion", lambda d: "active")
    monkeypatch.setattr(_cli_wait, "_live_jobs", lambda d: LIVE)
    monkeypatch.setattr(_cli_wait, "_fence_generation", lambda d: "g")
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path), "--poll-seconds", "1", "--timeout-seconds", "0"],
    )
    assert result.exit_code == RunExitCode.WAIT_TIMEOUT, result.output


# ---------------------------------------------------------------------------
# The launch form: phenotypic-wait --output <out> -- <submit> (spec C3, P2 T2a)
# ---------------------------------------------------------------------------


def _refuse_launch(monkeypatch):
    monkeypatch.setattr(
        _cli_wait, "_run_submit", lambda argv: pytest.fail("launched")
    )
    monkeypatch.setattr(
        _cli_wait, "wait_for_run", lambda *a, **k: pytest.fail("polled")
    )


def test_submit_exit_4_then_waits(monkeypatch, tmp_path):
    calls, waits = [], []
    monkeypatch.setattr(
        _cli_wait, "_run_submit", lambda argv: calls.append(list(argv)) or 4
    )
    monkeypatch.setattr(
        _cli_wait,
        "wait_for_run",
        lambda d, **k: waits.append((d, k))
        or RunExitCode.COMPLETE_WITH_FAILURES,
    )
    r = CliRunner().invoke(
        main,
        [
            "--output",
            str(tmp_path),
            "--timeout-seconds",
            "9",
            "--",
            "python",
            "-m",
            "phenotypic",
            "-o",
            str(tmp_path),
            "--slurm",
            "partition=intel",
        ],
    )
    assert r.exit_code == 3, r.output
    assert calls == [
        [
            "python",
            "-m",
            "phenotypic",
            "-o",
            str(tmp_path),
            "--slurm",
            "partition=intel",
        ]
    ]
    # --timeout-seconds bounds the wait, never the submit
    assert len(waits) == 1 and waits[0][1]["timeout_seconds"] == 9


@pytest.mark.parametrize("code", [0, 1, 2, 3, 130])
def test_any_other_submit_exit_is_returned_without_polling(
    monkeypatch, tmp_path, code
):
    monkeypatch.setattr(_cli_wait, "_run_submit", lambda argv: code)
    monkeypatch.setattr(
        _cli_wait, "wait_for_run", lambda *a, **k: pytest.fail("polled")
    )
    assert (
        CliRunner().invoke(main, ["-o", str(tmp_path), "--", "true"]).exit_code
        == code
    )


@pytest.mark.parametrize(
    "flag", ["--output={}", "-o {}", "--output {}", "-o{}"]
)
def test_a_child_naming_a_different_output_is_refused_before_launch(
    monkeypatch, tmp_path, flag
):
    _refuse_launch(monkeypatch)
    child = flag.format(tmp_path / "b").split(" ")
    r = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path / "a"),
            "--",
            "python",
            "-m",
            "phenotypic",
            *child,
        ],
    )
    assert r.exit_code == 2
    assert str(tmp_path / "b") in r.output


def test_a_relative_child_output_naming_the_same_dir_is_accepted(
    monkeypatch, tmp_path
):
    (tmp_path / "out").mkdir()
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(
        _cli_wait, "_run_submit", lambda argv: calls.append(argv) or 0
    )
    r = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path / "out"),
            "--",
            "python",
            "-m",
            "phenotypic",
            "-o",
            "out",
        ],
    )
    assert r.exit_code == 0, r.output
    assert len(calls) == 1


def test_a_child_naming_no_output_is_not_checked(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        _cli_wait, "_run_submit", lambda argv: calls.append(argv) or 0
    )
    r = CliRunner().invoke(main, ["-o", str(tmp_path), "--", "./submit.sh"])
    assert r.exit_code == 0, r.output
    assert calls == [("./submit.sh",)] or calls == [["./submit.sh"]]


@pytest.mark.parametrize(
    "mode",
    [
        ["--mode", "migrate"],
        ["--mode=migrate"],
        ["-m", "migrate"],
        ["-mmigrate"],
    ],
)
def test_a_migrate_child_is_refused(monkeypatch, tmp_path, mode):
    _refuse_launch(monkeypatch)
    r = CliRunner().invoke(
        main,
        [
            "-o",
            str(tmp_path),
            "--",
            "python",
            "-m",
            "phenotypic",
            *mode,
            "-o",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 2
    assert "migrate" in r.output


def test_an_empty_command_after_dashdash_is_refused(monkeypatch, tmp_path):
    _refuse_launch(monkeypatch)
    r = CliRunner().invoke(main, ["-o", str(tmp_path), "--"])
    assert r.exit_code == 2
    assert "--" in r.output


def test_a_command_not_introduced_by_dashdash_is_refused(
    monkeypatch, tmp_path
):
    _refuse_launch(monkeypatch)
    r = CliRunner().invoke(
        main, ["-o", str(tmp_path), "python", "-m", "phenotypic"]
    )
    assert r.exit_code == 2


def test_child_options_after_dashdash_are_not_the_waiters(
    monkeypatch, tmp_path
):
    # the child's own --poll-seconds/--timeout-seconds/-h never reach the waiter
    calls, waits = [], []
    monkeypatch.setattr(
        _cli_wait, "_run_submit", lambda argv: calls.append(list(argv)) or 4
    )
    monkeypatch.setattr(
        _cli_wait,
        "wait_for_run",
        lambda d, **k: waits.append(k) or RunExitCode.COMPLETE,
    )
    r = CliRunner().invoke(
        main,
        ["-o", str(tmp_path), "--", "./x", "--timeout-seconds", "1", "-h"],
    )
    assert r.exit_code == 0, r.output
    assert calls == [["./x", "--timeout-seconds", "1", "-h"]]
    assert waits[0]["timeout_seconds"] is None


def test_the_launch_form_accepts_an_output_the_submit_will_create(
    monkeypatch, tmp_path
):
    out = tmp_path / "out"
    monkeypatch.setattr(
        _cli_wait, "_run_submit", lambda argv: out.mkdir() or 4
    )
    monkeypatch.setattr(
        _cli_wait, "wait_for_run", lambda d, **k: RunExitCode.COMPLETE
    )
    r = CliRunner().invoke(
        main,
        ["-o", str(out), "--", "python", "-m", "phenotypic", "-o", str(out)],
    )
    assert r.exit_code == 0, r.output


def test_without_dashdash_it_only_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _cli_wait, "_run_submit", lambda argv: pytest.fail("launched")
    )
    monkeypatch.setattr(
        _cli_wait, "wait_for_run", lambda d, **k: RunExitCode.COMPLETE
    )
    assert CliRunner().invoke(main, ["-o", str(tmp_path)]).exit_code == 0


def test_run_submit_maps_a_signal_to_128_plus_n():
    assert (
        _cli_wait._run_submit(
            [
                sys.executable,
                "-c",
                "import os,signal; os.kill(os.getpid(), signal.SIGTERM)",
            ]
        )
        == 128 + 15
    )


def test_run_submit_passes_through_a_real_exit_code():
    assert (
        _cli_wait._run_submit([sys.executable, "-c", "raise SystemExit(4)"])
        == 4
    )


def test_run_submit_reports_a_missing_program_as_127(tmp_path):
    assert _cli_wait._run_submit([str(tmp_path / "no-such-program")]) == 127


def test_check_submit_argv_refuses_and_passes(tmp_path):
    import click

    _cli_wait.check_submit_argv(
        ["python", "-m", "phenotypic", "-o", str(tmp_path)], tmp_path
    )
    _cli_wait.check_submit_argv(["./wrapper.sh"], tmp_path)
    for bad in (
        [],
        ["x", "--output", str(tmp_path / "z")],
        ["x", "--mode", "migrate"],
    ):
        with pytest.raises(click.UsageError):
            _cli_wait.check_submit_argv(bad, tmp_path)
