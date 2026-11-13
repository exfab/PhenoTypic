"""``phenotypic-wait``: block until a SLURM run ends, then exit with its code.

A SLURM run returns ``SUBMITTED`` (``4``) the moment it is queued. An
orchestrator that hashes outputs when a command exits, such as a DVC stage,
needs a second command that returns only once the run is over. This module is
that command (#281 spec C3). Its exit code is the one the run would have
returned locally (spec C10):

* ``0`` (``COMPLETE``) once this submission has published and no image failed
  terminally;
* ``3`` (``COMPLETE_WITH_FAILURES``) once it has published, some images failed
  terminally, and **every accepted image is resolved** (a success or a
  terminal failure record). ``--accept-failures`` maps a final ``3`` to ``0``;
* ``5`` (``ENDED_UNPUBLISHED``) when no job is alive and the run ended without
  finishing: this submission published nothing (a finalizer killed for running
  out of memory), or some images were never attempted. Re-run the same
  command;
* ``6`` (``WAIT_TIMEOUT``) after ``--timeout-seconds``;
* ``1`` (``ERROR``) when polling itself kept failing, and ``130`` when the
  waiter is interrupted. Neither cancels anything.

**It is read-only.** It never aggregates, publishes or cancels. Its only write
is the verification cache that :func:`resolve_run_state` already refreshes, so
the double publication of the old ``--wait`` cannot come back.

**The launch form** (``phenotypic-wait --output out -- <submit...>``) runs the
submit command itself, because DVC aborts a stage whose command exits
non-zero and every SLURM submission exits ``4``. The child's ``4`` starts the
wait; any other child code is returned without polling (spec C3, "Launch").
That child is the only thing the waiter ever runs. When a job of the output is
already live, the waiter **attaches** to that run instead of launching (spec
C3, P2 review F2): that is the recovery after the job running ``dvc repro``
hits its walltime.

**Why it lives in ``_cli``.** :func:`resolve_run_state` takes an ``active``
lifecycle fence at face value and never asks the scheduler, so a dead finalizer
would read ``active`` forever. The scheduler query that tells a running run
from a dead one is :func:`live_slurm_job_ids`, which is in ``_cli``, and
``sdk_`` may not import ``_cli``.

**It does not serve ``--mode migrate``.** Migration keeps its lifecycle under
an external control root, so :func:`live_slurm_job_ids` sees none of its jobs.

Every scheduler and state read goes through a module-level wrapper
(``_run_snapshot``, ``_live_jobs``, ``_fence_generation``, ``_published``,
``_cancelled``, ``_generation_failures``, ``_is_process_run``). Each imports
lazily, so ``phenotypic-wait --help`` loads no heavy module, and each is a
seam a test can replace.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import click

from phenotypic.sdk_ import RunExitCode

if TYPE_CHECKING:
    from ._cli_staged_orchestration import LiveSlurmJobs

__all__ = [
    "RunSnapshot",
    "check_submit_argv",
    "decide_wait_outcome",
    "main",
    "wait_for_run",
]

#: The default interval between polls, in seconds (spec C3).
DEFAULT_POLL_SECONDS = 60.0

#: Consecutive polls that may raise before the waiter exits ``ERROR``
#: (P2 review M5). A success resets the count.
MAX_CONSECUTIVE_POLL_ERRORS = 5

#: The exit code of an interrupted waiter, the one ``python -m phenotypic``
#: uses for an interrupt.
EXIT_INTERRUPTED = 130


class RunSnapshot(NamedTuple):
    """One poll's view of the run state.

    Attributes:
        completion: ``resolve_run_state(...).completion``.
        accepted: Accepted images in the run's inventory.
        verified: Of those, images with a valid success record.
        failed: Of those, images whose current outcome is a terminal failure.
    """

    completion: str
    accepted: int = 0
    verified: int = 0
    failed: int = 0

    @property
    def unresolved(self) -> int:
        """Accepted images with neither a success nor a terminal failure."""
        return self.accepted - self.verified - self.failed


# ---------------------------------------------------------------------------
# Seams: each one reads, imports lazily, and is what tests replace
# ---------------------------------------------------------------------------


def _run_snapshot(output_dir: Path) -> RunSnapshot:
    """Return the completion verdict and the image counts of one shallow pass.

    One :func:`resolve_run_state` call per poll, so the verdict and the counts
    describe the same instant.
    """
    from phenotypic.sdk_ import resolve_run_state

    state = resolve_run_state(output_dir, depth="shallow")
    return RunSnapshot(
        completion=state.completion,
        accepted=state.diagnostics.accepted,
        verified=state.diagnostics.verified,
        failed=state.diagnostics.failed,
    )


class _InactiveJobMemo:
    """``scheduler_job_is_active``, remembering every ``False`` answer.

    A job SLURM reports ended never runs again under the same id, so that
    answer is final. Ordinary chunks and dispatchers never get a terminal
    ledger row, so without the memo every poll of a long drip-feed chain asks
    ``squeue`` (and ``sacct``, for a purged job) about every job the chain
    ever ran (P2 review M6). Only ``False`` is kept; ``True`` and ``None``
    are asked again. It lives for one waiter process, writes nothing and
    changes no verdict.
    """

    def __init__(self) -> None:
        self._inactive: set[str] = set()

    def __call__(self, job_id: str) -> bool | None:
        if job_id in self._inactive:
            return False
        from . import _cli_staged_orchestration

        state = _cli_staged_orchestration.scheduler_job_is_active(job_id)
        if state is False:
            self._inactive.add(job_id)
        return state

    def clear(self) -> None:
        """Forget every remembered answer."""
        self._inactive.clear()


#: The waiter process's one memo of ended jobs.
_INACTIVE_JOBS = _InactiveJobMemo()


def _live_jobs(output_dir: Path) -> "LiveSlurmJobs":
    """Return the jobs that may still own *output_dir*, from the scheduler."""
    from ._cli_staged_orchestration import live_slurm_job_ids

    return live_slurm_job_ids(output_dir, is_active=_INACTIVE_JOBS)


def _fence_generation(output_dir: Path) -> str | None:
    """Return the lifecycle fence's generation, or ``None`` with no fence."""
    from ._cli_slurm_lifecycle import load_slurm_lifecycle

    lifecycle = load_slurm_lifecycle(output_dir)
    return None if lifecycle is None else str(lifecycle["generation"])


def _published(output_dir: Path, generation: str | None) -> bool:
    """Return whether the submission fenced by *generation* published.

    **This is the one place the "published by this submission" rule lives**
    (spec C3). With no fence at the first poll there is no SLURM submission
    to bind to (the run before it was local), so the run state alone decides
    and this returns ``True``. Otherwise any one of these is enough, and each
    names *generation* itself, so no clock is compared:

    * that generation's fence was closed by its finalizer: same generation,
      ``active: false``, ``closed_by`` is
      :data:`~phenotypic._cli._cli_slurm_lifecycle.FENCE_CLOSED_BY_FINALIZER`
      and no ``terminal_status``. A cancel leaves ``closed_by`` unset, so a
      cancelled resubmission ends unpublished;
    * the run completion marker is a finished one whose ``execution_epoch`` is
      that generation;
    * the aggregate proof is current and its ``execution_epoch`` is that
      generation. A proof written before the stamp existed has no
      ``execution_epoch`` and never counts.

    Matching state alone is deliberately not enough: a ``--mode measure``
    resubmission keeps no run state, so an earlier run's aggregate proof
    would still match it after the new finalizer died.
    """
    if generation is None:
        return True
    from phenotypic.sdk_ import run_completion_marker_path

    from ._cli_completion import valid_aggregate_snapshot
    from ._cli_slurm_lifecycle import (
        FENCE_CLOSED_BY_FINALIZER,
        load_slurm_lifecycle,
    )

    lifecycle = load_slurm_lifecycle(output_dir)
    if (
        lifecycle is not None
        and lifecycle.get("generation") == generation
        and lifecycle.get("active") is False
        and lifecycle.get("closed_by") == FENCE_CLOSED_BY_FINALIZER
        and not lifecycle.get("terminal_status")
    ):
        return True
    try:
        marker = json.loads(
            run_completion_marker_path(output_dir).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        marker = None
    if (
        isinstance(marker, dict)
        and marker.get("status") == "complete"
        and marker.get("finalizer_succeeded") is True
        and marker.get("execution_epoch") == generation
    ):
        return True
    aggregate = valid_aggregate_snapshot(output_dir)
    return (
        aggregate is not None
        and aggregate.get("execution_epoch") == generation
    )


def _cancelled(output_dir: Path, generation: str | None) -> bool:
    """Return whether *generation*'s fence was closed other than by its finalizer.

    That is a cancel (``cancel_generation``, ``deactivate_orchestration``), a
    failure teardown, or a retirement by a later run: each closes the fence
    and leaves ``closed_by`` unset. ``False`` with no generation and while the
    fence is still active.
    """
    if generation is None:
        return False
    from ._cli_slurm_lifecycle import (
        FENCE_CLOSED_BY_FINALIZER,
        load_slurm_lifecycle,
    )

    lifecycle = load_slurm_lifecycle(output_dir)
    return (
        lifecycle is not None
        and lifecycle.get("generation") == generation
        and lifecycle.get("active") is False
        and lifecycle.get("closed_by") != FENCE_CLOSED_BY_FINALIZER
    )


def _generation_failures(output_dir: Path, generation: str | None) -> int:
    """Return how many images failed terminally under *generation*.

    Read from the terminal-failure journal. A SLURM worker records the
    generation it ran under in each row's ``lifecycle_epoch``
    (``PHENOTYPIC_SLURM_GENERATION``, the fence's own generation), so this is
    the SLURM twin of a local run's "failures this invocation recorded". It
    is also the only count that sees a ``--mode measure`` resubmission's
    failures: their work ids carry mode ``measure``, outside the forward
    inventory that :func:`resolve_run_state` counts (P2 review I5). Distinct
    work ids, so a retried image counts once. ``0`` with no generation.
    """
    if generation is None:
        return 0
    from ._cli_failure_tracker import read_terminal_failures

    return len(
        {
            record.work_id
            for record in read_terminal_failures(output_dir)
            if record.lifecycle_epoch == generation
        }
    )


def _is_process_run(output_dir: Path) -> bool:
    """Return whether the run config records a ``--mode process`` layer."""
    from phenotypic.sdk_ import resolve_processing_state_path

    try:
        raw = json.loads(
            resolve_processing_state_path(output_dir).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return False
    config = raw.get("config") if isinstance(raw, dict) else None
    return isinstance(config, dict) and bool(config.get("process_only_layer"))


# ---------------------------------------------------------------------------
# The decision and the loop
# ---------------------------------------------------------------------------


def decide_wait_outcome(
    completion: str,
    live: "LiveSlurmJobs",
    *,
    published: bool,
    process_run: bool,
    unresolved: int = 0,
    generation_failures: int = 0,
    cancelled: bool = False,
) -> RunExitCode | None:
    """Return the exit code for one poll, or ``None`` to keep waiting.

    Applied top to bottom (spec C3):

    1. Any ``live``, ``unverified`` or ``unresolved`` job keeps it waiting,
       **before** any completion check. A measure or same-metadata recompile
       resubmission leaves the previous run's proof current, so ``complete``
       alone does not mean this submission finished. A scheduler that cannot
       answer is never read as "nothing running".
    2. ``complete`` and published: ``COMPLETE``, or ``COMPLETE_WITH_FAILURES``
       when this generation journalled terminal failures. Only a
       ``--mode measure`` resubmission gets there: the forward run state
       cannot see its failures (P2 review I5).
    3. ``failed``, every accepted image resolved, and published:
       ``COMPLETE_WITH_FAILURES``. An image with neither a success nor a
       failure record was never attempted (an array task killed before it
       wrote one), so such a run did not finish (P2 review I1, F5).
    4. The same in a process run that was not cancelled:
       ``COMPLETE_WITH_FAILURES``. Process mode publishes no aggregate proof
       and no run proof over a failure, so there is nothing more to wait for.
    5. Anything else: ``ENDED_UNPUBLISHED``, meaning "ended without
       finishing; re-run the same command".

    Args:
        completion: ``resolve_run_state(...).completion``.
        live: The scheduler's answer from :func:`live_slurm_job_ids`.
        published: Whether this submission published (see ``_published``).
        process_run: Whether the run is a ``--mode process`` run.
        unresolved: Accepted images with neither a success nor a terminal
            failure record (:attr:`RunSnapshot.unresolved`).
        generation_failures: Images journalled as terminal failures under
            this submission's generation (see ``_generation_failures``).
        cancelled: Whether this submission's fence was closed by anything
            other than its finalizer (see ``_cancelled``).
    """
    if live:
        return None
    if published and completion == "complete":
        if generation_failures:
            return RunExitCode.COMPLETE_WITH_FAILURES
        return RunExitCode.COMPLETE
    if (
        completion == "failed"
        and unresolved == 0
        and (published or (process_run and not cancelled))
    ):
        return RunExitCode.COMPLETE_WITH_FAILURES
    return RunExitCode.ENDED_UNPUBLISHED


def _describe_jobs(live: "LiveSlurmJobs") -> str:
    """Return the live-job groups and why each keeps the waiter waiting."""
    if not live:
        return "no live SLURM job"
    parts = []
    if live.live:
        parts.append(f"active: {', '.join(live.live)}")
    if live.unverified:
        parts.append(
            f"unverified (the scheduler could not answer for them): "
            f"{', '.join(live.unverified)}"
        )
    if live.unresolved:
        parts.append(
            "unresolved (the scheduler could not be queried for these "
            f"submissions): {', '.join(live.unresolved)}"
        )
    return "; ".join(parts)


def _accepted_failures_line(count: int | None) -> str:
    """The line ``--accept-failures`` prints when it turns a ``3`` into ``0``."""
    counted = (
        f"{count} image(s) failed terminally"
        if count
        else "some images failed terminally"
    )
    return (
        f"phenotypic-wait: --accept-failures: {counted}; exiting "
        f"{int(RunExitCode.COMPLETE)} instead of "
        f"{int(RunExitCode.COMPLETE_WITH_FAILURES)}. The terminal-failure "
        "journal still records them."
    )


def _report_outcome(
    outcome: RunExitCode,
    snapshot: RunSnapshot,
    generation_failures: int,
    *,
    accept_failures: bool,
    echo: Callable[[str], object],
) -> RunExitCode:
    """Print why the wait ended, apply ``--accept-failures``, return the code."""
    if outcome is RunExitCode.COMPLETE_WITH_FAILURES:
        count = (
            snapshot.failed
            if snapshot.completion == "failed"
            else generation_failures
        )
        echo(f"phenotypic-wait: {count} image(s) failed terminally.")
        if accept_failures:
            echo(_accepted_failures_line(count))
            outcome = RunExitCode.COMPLETE
    elif outcome is RunExitCode.ENDED_UNPUBLISHED and snapshot.unresolved > 0:
        echo(
            f"phenotypic-wait: {snapshot.unresolved} accepted image(s) have "
            "neither a success nor a terminal failure record, so the run "
            "did not finish. Re-run the same command to process them."
        )
    echo(f"phenotypic-wait: {outcome.name} (exit {int(outcome)})")
    return outcome


def wait_for_run(
    output_dir: Path,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float | None = None,
    accept_failures: bool = False,
    sleep: Callable[[float], object] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    echo: Callable[[str], object] = click.echo,
) -> RunExitCode:
    """Poll *output_dir* until its run ends, and return the run's exit code.

    The fence's generation is read **once, at the first poll**, and every
    later "did this submission publish?" question is asked about that
    generation. A fence that a later invocation replaces cannot then be
    mistaken for this one.

    Each poll reads the scheduler **before** the run state. If no job is
    alive at the first read, the state read after it already reflects
    everything those jobs wrote. In the other order, a finalizer that ends
    between the two reads would be judged on the state from before it
    published.

    Each poll prints one line: the completion verdict, the live-job groups
    and the elapsed time. A poll that raises prints its reason instead and is
    retried at the next interval. :data:`MAX_CONSECUTIVE_POLL_ERRORS` in a
    row end the wait with ``ERROR``, naming the last reason, so a broken
    read does not end days of waiting with a bare traceback (P2 review M5).

    Args:
        output_dir: The run's ``--output`` root.
        poll_seconds: Seconds between polls.
        timeout_seconds: Give up with ``WAIT_TIMEOUT`` after this long, or
            never when ``None``. Nothing is cancelled.
        accept_failures: Return ``COMPLETE`` where the run ended
            ``COMPLETE_WITH_FAILURES``, after printing the failure count
            (spec C3, P2 review F1).
        sleep: Injected for tests.
        clock: A monotonic clock, injected for tests.
        echo: Where poll lines go, injected for tests.

    Returns:
        The :class:`RunExitCode` the run ended with.
    """
    output_dir = Path(output_dir)
    start = clock()
    generation: str | None = None
    bound = False
    errors = 0
    while True:
        outcome = None
        try:
            if not bound:
                generation = _fence_generation(output_dir)
                bound = True
            live = _live_jobs(output_dir)
            snapshot = _run_snapshot(output_dir)
            failures = 0
            if not live:
                failures = _generation_failures(output_dir, generation)
                outcome = decide_wait_outcome(
                    snapshot.completion,
                    live,
                    published=_published(output_dir, generation),
                    process_run=_is_process_run(output_dir),
                    unresolved=snapshot.unresolved,
                    generation_failures=failures,
                    cancelled=_cancelled(output_dir, generation),
                )
        except Exception as exc:  # noqa: BLE001 -- every poll error is retried
            errors += 1
            elapsed = clock() - start
            reason = f"{type(exc).__name__}: {exc}"
            if errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                echo(
                    f"phenotypic-wait: {errors} consecutive polls failed; "
                    f"giving up after {elapsed:.0f}s. Last reason: {reason}. "
                    "Nothing was cancelled, and the run may still be going. "
                    f"{RunExitCode.ERROR.name} (exit {int(RunExitCode.ERROR)})"
                )
                return RunExitCode.ERROR
            echo(
                f"phenotypic-wait: poll failed ({reason}); retrying "
                f"({errors} of {MAX_CONSECUTIVE_POLL_ERRORS}); "
                f"elapsed={elapsed:.0f}s"
            )
        else:
            errors = 0
            elapsed = clock() - start
            echo(
                f"phenotypic-wait: poll completion={snapshot.completion}; "
                f"{_describe_jobs(live)}; elapsed={elapsed:.0f}s"
            )
            if outcome is not None:
                return _report_outcome(
                    outcome,
                    snapshot,
                    failures,
                    accept_failures=accept_failures,
                    echo=echo,
                )
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            echo(
                f"phenotypic-wait: gave up after {elapsed:.0f}s "
                f"(--timeout-seconds {timeout_seconds:g}); nothing was "
                f"cancelled. {RunExitCode.WAIT_TIMEOUT.name} "
                f"(exit {int(RunExitCode.WAIT_TIMEOUT)})"
            )
            return RunExitCode.WAIT_TIMEOUT
        interval = poll_seconds
        if timeout_seconds is not None:
            interval = min(interval, timeout_seconds - elapsed)
        sleep(interval)


# ---------------------------------------------------------------------------
# The launch form: phenotypic-wait --output <out> -- <submit...>
# ---------------------------------------------------------------------------

#: The shell's codes for a command that cannot be found or cannot be run.
_EXIT_NOT_FOUND = 127
_EXIT_NOT_EXECUTABLE = 126


class SubmitInterrupted(Exception):
    """The waiter was interrupted while the submit ran; ``code`` is the submit's."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _run_submit(argv: Sequence[str]) -> int:
    """Run the submit command and return its exit code.

    The command runs as an argv list, never through a shell, with the
    waiter's working directory, environment and standard streams. A child
    killed by signal N returns ``128 + N``, the shell's convention, and a
    program that cannot be found or executed returns ``127`` or ``126``.

    **An interrupt waits for the child** (P2 review M4). Ctrl-C reaches the
    child too (it shares the terminal's process group), and the child's own
    interrupt path is what records a submission it has already made.
    ``subprocess.run`` would kill the child a quarter of a second after the
    interrupt, possibly between its ``sbatch`` and its ledger row, so the
    child is waited for, however many interrupts arrive, and its code is
    raised as :class:`SubmitInterrupted` so the caller returns it without
    starting a wait.

    Raises:
        SubmitInterrupted: The waiter was interrupted; carries the child's
            code.
    """
    try:
        child = subprocess.Popen(list(argv))
    except FileNotFoundError as exc:
        click.echo(f"phenotypic-wait: cannot run the submit: {exc}", err=True)
        return _EXIT_NOT_FOUND
    except OSError as exc:
        click.echo(f"phenotypic-wait: cannot run the submit: {exc}", err=True)
        return _EXIT_NOT_EXECUTABLE
    interrupted = False
    while True:
        try:
            code = child.wait()
            break
        except KeyboardInterrupt:
            interrupted = True
            click.echo(
                "phenotypic-wait: interrupted; waiting for the submit to "
                "exit, so it can record what it submitted.",
                err=True,
            )
    code = 128 - code if code < 0 else code
    if interrupted:
        raise SubmitInterrupted(code)
    return code


#: Program names whose following tokens are PhenoTypic CLI arguments.
_PHENOTYPIC_PROGRAMS = frozenset({"phenotypic", "phenotypic.exe"})


def _phenotypic_arguments(argv: Sequence[str]) -> list[str] | None:
    """Return the tokens after the program ``phenotypic`` in *argv*.

    The program is the first ``-m phenotypic`` (or ``-mphenotypic``), or the
    first token whose file name is ``phenotypic`` (the console script, bare
    or by path). Everything before it belongs to a wrapper (``srun -o log``,
    ``uv run``, ``python``), whose options are not the run's (P2 review M3).

    Returns:
        The tokens after the program, or ``None`` when *argv* names none (a
        wrapper script), which is then not checked.
    """
    for index, token in enumerate(argv):
        if token == "-m" and index + 1 < len(argv):
            if argv[index + 1] == "phenotypic":
                return list(argv[index + 2 :])
        elif token == "-mphenotypic" or (
            not token.startswith("-")
            and Path(token).name in _PHENOTYPIC_PROGRAMS
        ):
            return list(argv[index + 1 :])
    return None


def _option_values(tokens: Sequence[str], long: str, short: str) -> list[str]:
    """Return every value *tokens* pass to *long* / *short*.

    Recognises ``--long X``, ``--long=X``, ``-s X`` and ``-sX``. Scanning
    stops at a ``--``, after which nothing is an option.
    """
    values = []
    for i, token in enumerate(tokens):
        if token == "--":
            break
        if token in (long, short):
            if i + 1 < len(tokens):
                values.append(tokens[i + 1])
        elif token.startswith(long + "="):
            values.append(token[len(long) + 1 :])
        elif token.startswith(short) and not token.startswith("--"):
            values.append(token[len(short) :])
    return values


def check_submit_argv(argv: Sequence[str], output_dir: Path) -> None:
    """Refuse a submit command the waiter could not wait for.

    Raised before anything runs, as a usage error (exit ``2``), for:

    * an empty command after ``--``;
    * a child running ``--mode migrate`` (``--mode=migrate``, ``-m migrate``,
      ``-mmigrate``), whose lifecycle the waiter cannot see;
    * a child naming an output (``-o X``, ``--output X``, ``--output=X``,
      ``-oX``) that resolves to a different path from *output_dir*. A
      relative path resolves against the current directory, which the child
      inherits.

    Only the tokens after the program ``phenotypic`` (or ``-m phenotypic``)
    are read, so a wrapper's own ``-o`` (``srun -o log python -m phenotypic
    ...``) is not taken as the run's output. A command that runs no
    ``phenotypic`` program by name, such as a wrapper script, is not checked.

    Args:
        argv: The submit command, everything after ``--``.
        output_dir: The waiter's own ``--output``.

    Raises:
        click.UsageError: For any of the cases above.
    """
    if not argv:
        raise click.UsageError(
            "nothing follows '--': give the submit command after it, e.g. "
            "'phenotypic-wait --output out -- python -m phenotypic ... "
            "-o out --slurm ...'."
        )
    tokens = _phenotypic_arguments(argv)
    if tokens is None:
        return
    if "migrate" in _option_values(tokens, "--mode", "-m"):
        raise click.UsageError(
            "phenotypic-wait does not serve --mode migrate: its lifecycle "
            "lives under an external control root the waiter cannot see."
        )
    expected = output_dir.resolve()
    for value in _option_values(tokens, "--output", "-o"):
        if Path(value).resolve() != expected:
            raise click.UsageError(
                f"the submit command writes to {value!r} "
                f"({Path(value).resolve()}), but --output is {expected}; "
                "the waiter would watch the wrong run."
            )


def _live_run_to_attach(output_dir: Path) -> str | None:
    """Describe a run of *output_dir* the scheduler reports live, or ``None``.

    Only jobs the scheduler reports **active** count (``LiveSlurmJobs.live``).
    With only ``unverified`` or ``unresolved`` jobs the submit is launched
    as usual and its own live-run guard refuses with the remedy, rather than
    the waiter attaching, possibly forever, to a run whose history SLURM has
    purged. A check that raises is reported and the submit is launched: the
    submit's guard asks the same question.
    """
    if not output_dir.is_dir():
        return None
    try:
        live = _live_jobs(output_dir)
        if not live.live:
            return None
        generation = _fence_generation(output_dir)
    except Exception as exc:  # noqa: BLE001 -- the submit's guard decides
        click.echo(
            f"phenotypic-wait: could not check for a live run "
            f"({type(exc).__name__}: {exc}); launching the submit, whose own "
            "guard refuses a live run."
        )
        return None
    label = generation if generation is not None else "unknown (no fence)"
    return f"generation {label}; active jobs {', '.join(live.live)}"


def _child_failure_count(output_dir: Path) -> int | None:
    """The run state's failure count after a local child exited ``3``."""
    try:
        return _run_snapshot(output_dir).failed
    except Exception:  # noqa: BLE001 -- the count is only printed
        return None


#: The ``ctx.meta`` key under which :class:`_WaitCommand` keeps the raw args.
_RAW_ARGS_KEY = "phenotypic_wait.raw_args"


class _WaitCommand(click.Command):
    """A command that keeps its raw arguments for the ``--`` check.

    Click consumes the ``--`` separator, so without the raw arguments the
    waiter could not tell ``-- cmd`` from a stray positional ``cmd``.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta[_RAW_ARGS_KEY] = list(args)
        return super().parse_args(ctx, args)


def _introduced_by_dashdash(raw_args: Sequence[str], n_submit: int) -> bool:
    """Return whether the trailing *n_submit* raw args follow a ``--``."""
    index = len(raw_args) - n_submit - 1
    return index >= 0 and raw_args[index] == "--"


def _launch_or_attach(
    submit_argv: Sequence[str], output_dir: Path, *, accept_failures: bool
) -> None:
    """Attach to a live run, or run the submit; return only to start waiting.

    Exits (``SystemExit``) with the submit's code whenever the submit did
    not exit ``4``, after mapping a ``3`` under ``--accept-failures``.
    """
    attach = _live_run_to_attach(output_dir)
    if attach is not None:
        click.echo(
            f"phenotypic-wait: a run of {output_dir} is already live "
            f"({attach}); attaching to it instead of launching the submit. "
            "The waiter cannot check that it runs the same pipeline."
        )
        return
    click.echo(f"phenotypic-wait: launching: {shlex.join(submit_argv)}")
    try:
        submitted = _run_submit(submit_argv)
    except SubmitInterrupted as exc:
        click.echo(
            f"phenotypic-wait: interrupted; the submit exited {exc.code}. "
            "Not waiting. Run the same command again to attach to a run it "
            "submitted."
        )
        raise SystemExit(exc.code) from None
    if submitted == RunExitCode.SUBMITTED:
        return
    if submitted == RunExitCode.COMPLETE_WITH_FAILURES and accept_failures:
        click.echo(_accepted_failures_line(_child_failure_count(output_dir)))
        raise SystemExit(int(RunExitCode.COMPLETE))
    click.echo(
        f"phenotypic-wait: the submit exited {submitted}, not "
        f"{int(RunExitCode.SUBMITTED)} (submitted); returning its "
        "code without waiting."
    )
    raise SystemExit(submitted)


@click.command(
    name="phenotypic-wait",
    cls=_WaitCommand,
    context_settings={
        "help_option_names": ["-h", "--help"],
        # The first positional ends the waiter's options, so a child's -o
        # or -h is never parsed as the waiter's own.
        "allow_interspersed_args": False,
    },
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="The run's --output directory. Without '--' it must exist; with "
    "'--' the submit may create it.",
)
@click.option(
    "--poll-seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_POLL_SECONDS,
    show_default=True,
    help="Seconds between polls.",
)
@click.option(
    "--timeout-seconds",
    type=click.FloatRange(min=0),
    default=None,
    help="Give up after this many seconds (exit 6). Nothing is cancelled. "
    "Default: wait indefinitely.",
)
@click.option(
    "--accept-failures",
    is_flag=True,
    default=False,
    help="Exit 0 instead of 3 when the run finished with terminal image "
    "failures, after printing how many. Applies to a waited run and to a "
    "submit that ran locally; other codes are unchanged.",
)
@click.argument(
    "submit_argv",
    nargs=-1,
    type=click.UNPROCESSED,
    metavar="[-- SUBMIT_COMMAND...]",
)
def main(
    output_dir: Path,
    poll_seconds: float,
    timeout_seconds: float | None,
    accept_failures: bool,
    submit_argv: tuple[str, ...],
) -> None:
    """Wait for a SLURM PhenoTypic run to end, and exit with its code.

    \b
    Launch and wait (one command, e.g. a DVC stage's cmd):
      phenotypic-wait --output out -- python -m phenotypic -i data -o out
          -p pipeline.pht-pipe.json --slurm slurm_partition=intel
    Wait for a run submitted separately:
      phenotypic-wait --output out

    Everything after -- is the submit command, run without a shell. If it
    exits 4 (submitted), the waiter waits; any other exit code is returned
    as-is, without waiting (a signal N gives 128+N). If a job of --output is
    already running, the submit is skipped and the waiter attaches to that
    run. Refused with exit 2 before launching: nothing after --, a submit
    with --mode migrate, or a submit whose -o/--output (read after the
    program 'phenotypic') is a different directory from --output.

    Exit codes once the run ends: 0 complete; 3 complete with terminal image
    failures (0 with --accept-failures); 5 the run ended without finishing
    (nothing published, or images never attempted): re-run the same command;
    6 --timeout-seconds elapsed (it bounds the wait, never the submit);
    1 polling kept failing; 130 interrupted.

    It never aggregates, publishes or cancels, and submits only the command
    after --. It does not serve --mode migrate.
    """
    raw_args = click.get_current_context().meta.get(_RAW_ARGS_KEY, [])
    if submit_argv or _introduced_by_dashdash(raw_args, 0):
        if not _introduced_by_dashdash(raw_args, len(submit_argv)):
            raise click.UsageError(
                f"unexpected argument {submit_argv[0]!r}; put the submit "
                "command after '--'."
            )
        check_submit_argv(submit_argv, output_dir)
        _launch_or_attach(
            submit_argv, output_dir, accept_failures=accept_failures
        )
    elif not output_dir.is_dir():
        raise click.BadParameter(
            f"Directory '{output_dir}' does not exist.",
            param_hint="'--output' / '-o'",
        )
    try:
        code = wait_for_run(
            output_dir,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            accept_failures=accept_failures,
        )
    except KeyboardInterrupt:
        click.echo(
            "phenotypic-wait: interrupted. Nothing was cancelled; the run "
            "goes on. Run the same command again to wait for it."
        )
        raise SystemExit(EXIT_INTERRUPTED) from None
    raise SystemExit(int(code))
