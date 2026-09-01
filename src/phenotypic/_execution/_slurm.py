"""``SlurmExecutor`` — a distributed tune-worker fleet over SLURM arrays.

Reuses the **forward CLI's drip-feed submission layer** (``format_sbatch_directives``
/ ``generate_dispatcher_chain`` / ``submit_drip_feed_start`` in
``phenotypic.sdk_.slurm``) — the queue-throttle plumbing that keeps occupancy at
~one array + one dispatcher — but emits a **fresh tune-specific array body** with
**no image-chunk sentinels**. One array task = one worker running
``python -m phenotypic.tune._tune_cli._worker``, an ask→evaluate→tell loop bound to
the **shared** Optuna ``study.db`` / Postgres URL until the shared budget exhausts
(optuna-integration §7).

The shared study URL (``--storage-url`` / ``$PHENOTYPIC_TUNE_STORAGE_URL``) is a
plain Optuna storage URL — SQLite-WAL for a local single node, or a Postgres
``postgresql+psycopg://USER@HOST:PORT/DB`` for a distributed fleet. PhenoTypic is
backend-agnostic and does not parse any server's handshake files: for a
password-less Postgres URL the password is resolved by **libpq** from
``~/.pgpass`` / ``PGPASSWORD`` (a shared-home ``~/.pgpass`` is read per worker),
so the secret never enters the worker script. A dead/orphaned worker is
re-enqueued **once** (a single bounded retry, never an unbounded resubmit loop).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar, cast

from phenotypic.sdk_ import logs_dir, slurm_scripts_dir
from phenotypic.sdk_.slurm import (
    SLURM_PYTHONPATH_BOOTSTRAP_BASH,
    SlurmArrayScriptSpec,
    format_sbatch_directives,
    generate_dispatcher_chain,
    generation_script_key,
    submit_drip_feed_start,
    submit_script,
    write_slurm_array_script,
)

T = TypeVar("T")
R = TypeVar("R")

#: Characters allowed in a SLURM ``--job-name`` (and our ``# Study:`` comment);
#: anything else in a user-chosen ``study_name`` is replaced with ``-`` so a
#: name with a space / slash / shell metacharacter can't corrupt the directive.
_JOB_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_job_name(study_name: str) -> str:
    """Replace any character outside ``[A-Za-z0-9._-]`` in ``study_name`` with ``-``.

    The study name is user-chosen and flows into the ``#SBATCH --job-name``
    directive and the ``# Study:`` script comment. A space, slash, or shell
    metacharacter there would split the directive or break the comment line, so
    we coerce it to a SLURM-safe token first.

    Args:
        study_name: The (possibly unsafe) study name.

    Returns:
        The sanitized token (safe for a SLURM job name).
    """
    return _JOB_NAME_SAFE.sub("-", study_name)


#: The tune worker module one array task runs (NOT the forward single-image
#: worker — a fresh body with no image-chunk sentinels).
_WORKER_MODULE = "phenotypic.tune._tune_cli._worker"

#: Script filenames written under the generation-scoped Tune script directory
#: for lifecycle-owned runs, or the legacy scripts directory otherwise.
_WORKER_SCRIPT_NAME = "tune_workers.sh"
_FINALIZER_SCRIPT_NAME = "tune_finalizer.sh"


class SlurmExecutor:
    """Submit a fleet of distributed tune workers over a SLURM array.

    Satisfies the :class:`~phenotypic._execution.Executor` Protocol: :meth:`run`
    launches the worker fleet (the ``work``/``items`` arguments parametrize the
    fleet size for Protocol parity — each worker independently pulls trials from
    the shared study, so there is no per-item fan-out).

    Args:
        output_dir: The run directory (scripts under
            ``.phenotypic/slurm_scripts/``, logs under
            ``.phenotypic/logs/slurm/``).
        spec_path: Path to the ``tuning_spec.json`` each worker loads.
        images_dir: The calibration image directory each worker scans.
        split_path: Path to the persisted held-out split each worker applies.
        study_name: The shared Optuna study name (same across the fleet).
        n_workers: The number of array tasks (workers) to launch.
        slurm_args: SLURM parameters (partition, mem, time, …) forwarded to
            ``format_sbatch_directives`` and the drip-feed dispatchers.
        storage_url: The shared Optuna storage URL every worker opens — a plain
            ``sqlite:///…`` or ``postgresql+psycopg://…`` (recommended
            password-less: libpq resolves the secret from ``~/.pgpass`` /
            ``PGPASSWORD`` per worker).
        python_command: The interpreter prefix the worker is launched with
            (default ``["python"]``); override for a non-default launcher.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        spec_path: Path,
        images_dir: Path,
        split_path: Path,
        study_name: str,
        n_workers: int,
        slurm_args: dict[str, Any],
        storage_url: str,
        python_command: Optional[Sequence[str]] = None,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        lifecycle_generation: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.spec_path = Path(spec_path)
        self.images_dir = Path(images_dir)
        self.split_path = Path(split_path)
        self.study_name = study_name
        self.n_workers = int(n_workers)
        self.slurm_args = dict(slurm_args)
        self.python_command = (
            list(python_command) if python_command else ["python"]
        )
        self.storage_url = storage_url
        #: Optional fixed calibration grid forwarded to each worker's
        #: ``GridImage.imread`` (``--nrows``/``--ncols``) so grid-cell-aware
        #: scoring sees a known uniform grid instead of a per-image fitted one.
        self.nrows = nrows
        self.ncols = ncols
        self.lifecycle_generation = lifecycle_generation

        #: Worker indices already re-enqueued once (bounds the retry to a single
        #: resubmit per worker — no unbounded loop).
        self._reenqueued: set[int] = set()

    def _script_dir(self) -> Path:
        """Return the generation-scoped Tune script directory."""
        root = slurm_scripts_dir(self.output_dir)
        if self.lifecycle_generation is None:
            return root
        return root / "tune" / generation_script_key(self.lifecycle_generation)

    # -- script generation ----------------------------------------------------

    def _worker_command(self) -> str:
        """The ``python -m …_worker`` command line bound to the shared study."""
        parts = [
            *self.python_command,
            "-m",
            _WORKER_MODULE,
            "--spec",
            shlex.quote(str(self.spec_path.absolute())),
            "--images",
            shlex.quote(str(self.images_dir.absolute())),
            "--split",
            shlex.quote(str(self.split_path.absolute())),
            "--study-name",
            shlex.quote(self.study_name),
            "--storage-url",
            shlex.quote(self.storage_url),
        ]
        if self.nrows is not None and self.ncols is not None:
            parts += ["--nrows", str(self.nrows), "--ncols", str(self.ncols)]
        return " ".join(parts)

    def generate_worker_array_script(self) -> Path:
        """Generate the array worker script (one array task = one worker).

        The ``--array`` directive is sized to :attr:`n_workers` (NOT an image
        count), and the body has **no** checkpoint/manifest/finalizer sentinels —
        every task runs the same ask→evaluate→tell worker against the shared
        study.

        Returns:
            The path to the generated, executable script.
        """
        script_dir = self._script_dir()
        script_dir.mkdir(parents=True, exist_ok=True)
        log_dir = logs_dir(self.output_dir) / "slurm"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "tune_worker_%A_%a.log"

        safe_study = _sanitize_job_name(self.study_name)
        worker_cmd = self._worker_command()

        script_path = script_dir / _WORKER_SCRIPT_NAME
        return write_slurm_array_script(
            script_path,
            SlurmArrayScriptSpec(
                job_name=f"pht-tune-{safe_study}",
                slurm_args=self.slurm_args,
                log_path=log_path,
                task_indices=list(range(self.n_workers)),
                body=worker_cmd,
                comments=[
                    (
                        "# Auto-generated by PhenoTypic tune SlurmExecutor "
                        "(distributed worker fleet)"
                    ),
                    f"# Study: {safe_study}",
                    (
                        "# One array task = one ask->evaluate->tell worker "
                        "bound to the shared study."
                    ),
                    "# No image-chunk sentinels: the study IS the shared state.",
                ],
            ),
        )

    def generate_finalizer_script(self) -> Path:
        """Generate the one-task publisher with the worker compute profile."""
        if self.lifecycle_generation is None:
            raise ValueError("a terminal finalizer requires a lifecycle generation")
        script_dir = self._script_dir()
        script_dir.mkdir(parents=True, exist_ok=True)
        log_dir = logs_dir(self.output_dir) / "slurm"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "tune_finalizer_%j.log"
        directives = format_sbatch_directives(
            job_name=f"pht-tune-finalize-{_sanitize_job_name(self.study_name)}",
            slurm_args=self.slurm_args,
            output_log=log_path,
            error_log=log_path,
        )
        command = " ".join(
            [
                *(shlex.quote(part) for part in self.python_command),
                "-m",
                "phenotypic.tune._tune_cli._finalize",
                "--output",
                shlex.quote(str(self.output_dir.absolute())),
                "--generation",
                shlex.quote(self.lifecycle_generation),
            ]
        )
        script_path = script_dir / _FINALIZER_SCRIPT_NAME
        script_path.write_text(
            f"""#!/bin/bash
{directives}

# Standalone Tune terminal finalizer; intentionally not a SLURM array.
set -e
set -u

{SLURM_PYTHONPATH_BOOTSTRAP_BASH}

{command}
""",
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        return script_path

    # -- Executor protocol ----------------------------------------------------

    def run(self, work: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """Submit the worker fleet via the drip-feed chain (Executor parity).

        Generates the worker array script, builds the drip-feed dispatcher chain
        (here a single array, so the chain is empty), and starts it via
        ``submit_drip_feed_start``. The ``work`` / ``items`` arguments satisfy the
        :class:`Executor` Protocol signature but are not fanned out — each worker
        independently pulls trials from the shared study.

        Args:
            work: Accepted for Protocol parity (unused — the worker body is the
                array script).
            items: Accepted for Protocol parity (unused — the fleet size is
                :attr:`n_workers`).

        Returns:
            The submitted SLURM job IDs (cast to the Protocol's ``list[R]``).
        """
        script = self.generate_worker_array_script()
        finalizer = (
            self.generate_finalizer_script()
            if self.lifecycle_generation is not None
            else None
        )
        log_dir = logs_dir(self.output_dir) / "slurm"
        # A single array job → the chain is empty; submit_drip_feed_start still
        # submits the one array script (and no dispatcher).
        dispatchers = generate_dispatcher_chain(
            chunk_scripts=[script],
            output_dir=self.output_dir,
            slurm_args=self.slurm_args,
            log_dir=log_dir,
            finalizer_script=finalizer,
            generation=self.lifecycle_generation,
            lifecycle_output_dir=self.output_dir,
        )
        submission_kwargs: dict[str, Any] = {}
        if finalizer is not None:
            submission_kwargs = {
                "finalizer_script": finalizer,
                "continuation_dependency_kind": "afterany",
                "output_dir": self.output_dir,
                "generation": self.lifecycle_generation,
            }
        job_ids, _warning = submit_drip_feed_start(
            chunk_scripts=[script],
            dispatcher_scripts=dispatchers,
            **submission_kwargs,
        )
        # The job-id strings ARE the results; cast satisfies the generic Protocol.
        return cast("list[R]", list(job_ids))

    # -- fault tolerance ------------------------------------------------------

    def reenqueue_dead_worker(self, *, worker_index: int) -> Optional[str]:
        """Resubmit a dead/orphaned worker **once**; ``None`` on a repeat (§8).

        A worker that died (node failure / timeout) leaves its in-flight Optuna
        trial in a non-terminal state; resubmitting the worker script lets a fresh
        worker pick up where the fleet left off (the study is the shared state).
        The retry is bounded to one resubmit per worker index — a second call for
        the same index is a no-op returning ``None``.

        Args:
            worker_index: The array-task index of the dead worker.

        Returns:
            The new SLURM job ID on the first re-enqueue; ``None`` if this worker
            was already re-enqueued once.
        """
        if not 0 <= worker_index < self.n_workers:
            raise ValueError(
                f"worker_index {worker_index} outside array range 0..{self.n_workers - 1}"
            )
        if worker_index in self._reenqueued:
            return None
        self._reenqueued.add(worker_index)
        script_path = self._script_dir() / _WORKER_SCRIPT_NAME
        if not script_path.exists():
            script_path = self.generate_worker_array_script()
        return submit_script(script_path, array_index=worker_index)
