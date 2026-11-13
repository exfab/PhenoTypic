"""``phenotypic-wait``: block until a SLURM run ends, then exit with its code.

A SLURM run returns ``SUBMITTED`` (``4``) the moment it is queued. An
orchestrator that hashes outputs when a command exits, such as a DVC stage,
needs a second command that returns only once the run is over. This module is
that command (#281 spec C3). Its exit code is the one the run would have
returned locally (spec C10):

* ``0`` (``COMPLETE``) or ``3`` (``COMPLETE_WITH_FAILURES``) once this
  submission has published;
* ``5`` (``ENDED_UNPUBLISHED``) when no job is alive and this submission
  published nothing, for example a finalizer killed for running out of memory;
* ``6`` (``WAIT_TIMEOUT``) after ``--timeout-seconds``.

**It is read-only.** It never aggregates, publishes, submits or cancels. Its
only write is the verification cache that :func:`resolve_run_state` already
refreshes, so the double publication of the old ``--wait`` cannot come back.

**Why it lives in ``_cli``.** :func:`resolve_run_state` takes an ``active``
lifecycle fence at face value and never asks the scheduler, so a dead finalizer
would read ``active`` forever. The scheduler query that tells a running run
from a dead one is :func:`live_slurm_job_ids`, which is in ``_cli``, and
``sdk_`` may not import ``_cli``.

**It does not serve ``--mode migrate``.** Migration keeps its lifecycle under
an external control root, so :func:`live_slurm_job_ids` sees none of its jobs.

Every scheduler and state read goes through a module-level wrapper
(``_completion``, ``_live_jobs``, ``_fence_generation``, ``_published``,
``_is_process_run``). Each imports lazily, so ``phenotypic-wait --help`` loads
no heavy module, and each is a seam a test can replace.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import click

from phenotypic.sdk_ import RunExitCode

if TYPE_CHECKING:
    from ._cli_staged_orchestration import LiveSlurmJobs

__all__ = ["decide_wait_outcome", "main", "wait_for_run"]

#: The default interval between polls, in seconds (spec C3).
DEFAULT_POLL_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Seams: each one reads, imports lazily, and is what tests replace
# ---------------------------------------------------------------------------


def _completion(output_dir: Path) -> str:
    """Return the run's completion verdict from a shallow run-state pass."""
    from phenotypic.sdk_ import resolve_run_state

    return resolve_run_state(output_dir, depth="shallow").completion


def _live_jobs(output_dir: Path) -> "LiveSlurmJobs":
    """Return the jobs that may still own *output_dir*, from the scheduler."""
    from ._cli_staged_orchestration import live_slurm_job_ids

    return live_slurm_job_ids(output_dir)


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
) -> RunExitCode | None:
    """Return the exit code for one poll, or ``None`` to keep waiting.

    Applied top to bottom (spec C3):

    1. Any ``live``, ``unverified`` or ``unresolved`` job keeps it waiting,
       **before** any completion check. A measure or same-metadata recompile
       resubmission leaves the previous run's proof current, so ``complete``
       alone does not mean this submission finished. A scheduler that cannot
       answer is never read as "nothing running".
    2. ``complete`` and published: ``COMPLETE``.
    3. ``failed`` and published: ``COMPLETE_WITH_FAILURES``.
    4. ``failed`` in a process run: ``COMPLETE_WITH_FAILURES``. Process mode
       publishes no aggregate proof and no run proof over a failure, so
       there is nothing more to wait for.
    5. Anything else: ``ENDED_UNPUBLISHED``.

    Args:
        completion: ``resolve_run_state(...).completion``.
        live: The scheduler's answer from :func:`live_slurm_job_ids`.
        published: Whether this submission published (see ``_published``).
        process_run: Whether the run is a ``--mode process`` run.
    """
    if live:
        return None
    if published and completion == "complete":
        return RunExitCode.COMPLETE
    if completion == "failed" and (published or process_run):
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


def wait_for_run(
    output_dir: Path,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float | None = None,
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
    and the elapsed time.

    Args:
        output_dir: The run's ``--output`` root.
        poll_seconds: Seconds between polls.
        timeout_seconds: Give up with ``WAIT_TIMEOUT`` after this long, or
            never when ``None``. Nothing is cancelled.
        sleep: Injected for tests.
        clock: A monotonic clock, injected for tests.
        echo: Where poll lines go, injected for tests.

    Returns:
        The :class:`RunExitCode` the run ended with.
    """
    output_dir = Path(output_dir)
    start = clock()
    generation = _fence_generation(output_dir)
    while True:
        live = _live_jobs(output_dir)
        completion = _completion(output_dir)
        elapsed = clock() - start
        echo(
            f"phenotypic-wait: poll completion={completion}; "
            f"{_describe_jobs(live)}; elapsed={elapsed:.0f}s"
        )
        if live:
            outcome = None
        else:
            outcome = decide_wait_outcome(
                completion,
                live,
                published=_published(output_dir, generation),
                process_run=_is_process_run(output_dir),
            )
        if outcome is not None:
            echo(f"phenotypic-wait: {outcome.name} (exit {int(outcome)})")
            return outcome
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


@click.command(
    name="phenotypic-wait",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The run's --output directory.",
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
def main(
    output_dir: Path, poll_seconds: float, timeout_seconds: float | None
) -> None:
    """Wait for a SLURM PhenoTypic run to end, and exit with its code.

    Run this after `python -m phenotypic ... --slurm ...`, which returns 4
    (submitted) as soon as the run is queued. Exit codes: 0 complete;
    3 complete with terminal image failures; 5 no job is alive and this
    submission published nothing; 6 --timeout-seconds elapsed.

    Read-only: it never aggregates, publishes, submits or cancels. It does
    not serve --mode migrate.
    """
    code = wait_for_run(
        output_dir, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds
    )
    raise SystemExit(int(code))
