"""Process exit codes of a PhenoTypic CLI run (spec #281, C10).

The codes are a contract an orchestrator (a DVC stage, a shell script, a
workflow engine) reads instead of parsing logs, so their values never change.
Code ``2`` is deliberately absent: it is Click's own code for a usage error,
and a refusal raised as a :class:`click.UsageError` keeps it.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["RunExitCode"]


class RunExitCode(IntEnum):
    """What a ``python -m phenotypic`` (or ``phenotypic-wait``) exit means.

    Attributes:
        COMPLETE: The run finished and no image failed terminally.
        ERROR: A runtime or configuration error stopped the run.
        COMPLETE_WITH_FAILURES: The run finished, but at least one image
            failed terminally and was left out of the published outputs
            (or, in process mode, has no exported layer).
        SUBMITTED: The run was submitted to SLURM and is not finished. Every
            SLURM run returns this at submission, including a SLURM
            ``--mode recompile``, a submitted ``--mode migrate`` and a SLURM
            ``--mode migrate --dry-run``, which submits its validation chain.
        ENDED_UNPUBLISHED: ``phenotypic-wait`` found no live job, and the
            submission it waited on published nothing.
        WAIT_TIMEOUT: ``phenotypic-wait`` gave up after ``--timeout-seconds``
            without cancelling anything.

    Examples:
        >>> from phenotypic.sdk_ import RunExitCode
        >>> int(RunExitCode.SUBMITTED)
        4
        >>> RunExitCode(3).name
        'COMPLETE_WITH_FAILURES'
    """

    COMPLETE = 0
    ERROR = 1
    COMPLETE_WITH_FAILURES = 3
    SUBMITTED = 4
    ENDED_UNPUBLISHED = 5
    WAIT_TIMEOUT = 6
