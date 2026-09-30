"""The SLURM finalizer as a chain of dependent jobs.

**Why a chain.** The terminal finalizer used to be one job, or index K of a
``0-K`` array, and it ran all of the finalization under one ``--time``: it
waited for its own aggregation shards, built the master, joined metadata,
wrote the mirror, rendered every plot, fitted the named analysis, rebuilt QC,
wrote the per-feature and per-category splits, published the aggregate proof,
rebuilt the display manifest and dashboard, and published the completion
marker. On a large run that exceeds the walltime of any one job. Each stage
now gets its own job and its own walltime. The cost is one Python start-up
per job, which is accepted.

The chain, in order, every edge ``afterany``::

    prepare    1 task    wait for / reconcile images, then submit the rest
    shards     K tasks   aggregate embedded tables into shards (fan-out only)
    master     1 task    master, metadata join, post ops, mirror, REMBI
    outputs    2 tasks   0: plots + named analysis   1: splits + error re-emit
    qc         1 task    QC rebuild + QC plots (after ``outputs``: QC plots
                         may read a named analysis table)
    publish    1 task    aggregate proof, manifest, dashboard, completion

**Every job is submitted after the image arrays are terminal**, by the
``prepare`` job, which the existing submission slot (the drip-feed
``finalizer_script``, or the staged controller's ``"finalizer"`` token)
submits in place of the old single finalizer. No finalizer job therefore runs
beside an active ordinary array, which is the scheduler-sidecar rule in the
project ``CLAUDE.md``. Every submission goes through
:func:`~phenotypic._cli._cli_slurm_lifecycle.submit_with_lifecycle`: ledgered,
exactly once per token, and fenced by the lifecycle generation, so
``--restart`` and cancellation see every chain job.

**Failure semantics.** Each task writes a status file; a task killed at its
walltime writes none, and a missing status reads as a failure. ``publish``
always runs (``afterany``) and always closes the lifecycle. If any earlier
task failed it certifies nothing: no aggregate proof, no completion marker,
the lifecycle closed as terminal-incomplete, and the job exits non-zero with a
message naming the failed task and its log. Re-running the same command is the
recovery, as it is for any incomplete run.

**What crosses a job boundary, and how it is checked.** The in-memory frames
do not: ``outputs`` and ``qc`` read the master and the mirror back from disk.
``master`` records their SHA-256 in ``handoff.json``, and every later job
refuses to proceed on bytes that no longer match -- so a mirror rewritten
between two jobs cannot be certified by a proof that describes another one.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import click

from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command, logged_step

logger = logging.getLogger(__name__)

STAGE_PREPARE: Final[str] = "prepare"
STAGE_SHARDS: Final[str] = "shards"
STAGE_MASTER: Final[str] = "master"
STAGE_OUTPUTS: Final[str] = "outputs"
STAGE_QC: Final[str] = "qc"
STAGE_PUBLISH: Final[str] = "publish"

#: The jobs ``prepare`` submits, in dependency order.
SUBMITTED_STAGES: Final[tuple[str, ...]] = (
    STAGE_SHARDS,
    STAGE_MASTER,
    STAGE_OUTPUTS,
    STAGE_QC,
    STAGE_PUBLISH,
)

#: Ordinary SLURM forward run (``full`` or ``measure``).
MODE_ORDINARY: Final[str] = "ordinary"
#: Staged GPU run; the lifecycle generation is the orchestration epoch.
MODE_STAGED: Final[str] = "staged"
#: SLURM recompile; the lifecycle generation is the attempt id.
MODE_RECOMPILE: Final[str] = "recompile"
MODES: Final[tuple[str, ...]] = (MODE_ORDINARY, MODE_STAGED, MODE_RECOMPILE)

STATUS_COMPLETED: Final[str] = "completed"
STATUS_FAILED: Final[str] = "failed"
STATUS_SKIPPED: Final[str] = "skipped"

#: ``handoff.json`` status when there was nothing to aggregate.
HANDOFF_EMPTY: Final[str] = "empty"

CHAIN_SPEC_VERSION: Final[int] = 1

#: Seconds ``master`` waits for shard statuses. The shard array is terminal
#: before ``master`` starts (``afterany``), so this only absorbs shared
#: filesystem visibility lag; a shard that died without a status is reported
#: as missing rather than waited on.
SHARD_STATUS_GRACE_SECONDS: Final[float] = 120.0


def _output_groups() -> tuple[str, ...]:
    """Return the ``outputs`` array's groups, indexed by task index."""
    from ._cli_output_manager import (
        FINALIZE_OUTPUT_GROUP_ANALYSIS,
        FINALIZE_OUTPUT_GROUP_TABLES,
    )

    return (FINALIZE_OUTPUT_GROUP_ANALYSIS, FINALIZE_OUTPUT_GROUP_TABLES)


def stage_task_count(stage: str, *, shards: int) -> int:
    """Return how many array tasks *stage* runs.

    Args:
        stage: One of the ``STAGE_*`` names.
        shards: K, the aggregation shard count. ``0`` means the chain has no
            shard stage (recompile brings its own shards).

    Returns:
        The task count.

    Raises:
        ValueError: *stage* is unknown.
    """
    if stage == STAGE_SHARDS:
        return shards
    if stage == STAGE_OUTPUTS:
        return len(_output_groups())
    if stage in {STAGE_PREPARE, STAGE_MASTER, STAGE_QC, STAGE_PUBLISH}:
        return 1
    raise ValueError(f"Unknown finalizer chain stage: {stage!r}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def chain_spec_path(output_dir: Path, generation: str) -> Path:
    """Return the chain's ``chain.json``."""
    from phenotypic.sdk_ import finalize_chain_dir

    return finalize_chain_dir(output_dir, generation) / "chain.json"


def handoff_path(output_dir: Path, generation: str) -> Path:
    """Return the ``handoff.json`` the ``master`` job leaves behind."""
    from phenotypic.sdk_ import finalize_chain_dir

    return finalize_chain_dir(output_dir, generation) / "handoff.json"


def stage_status_path(
    output_dir: Path, generation: str, stage: str, task_index: int
) -> Path:
    """Return one chain task's status file."""
    from phenotypic.sdk_ import finalize_chain_dir

    return (
        finalize_chain_dir(output_dir, generation)
        / "status"
        / f"{stage}_{task_index}.json"
    )


def stage_log_pattern(output_dir: Path, stage: str) -> Path:
    """Return a stage's SLURM log path pattern (``%A_%a`` unexpanded)."""
    from phenotypic.sdk_ import logs_dir

    return logs_dir(output_dir) / "slurm" / f"finalize_{stage}_%A_%a.log"


# ---------------------------------------------------------------------------
# The spec, written at submission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainSpec:
    """One finalizer chain, as ``chain.json`` records it.

    Attributes:
        mode: One of :data:`MODES`.
        output_dir: Run output root.
        generation: The SLURM lifecycle generation every job is fenced by.
        shards: K; ``0`` when the chain has no shard stage.
        scripts: Stage name to its SBATCH script.
        recompile_manifest: The recompile attempt's task manifest, recompile
            mode only.
        recompile_finalizer_index: The manifest index of its finalizer task,
            whose status file the CLI's ``--wait`` watches.
    """

    mode: str
    output_dir: Path
    generation: str
    shards: int
    scripts: Mapping[str, Path]
    recompile_manifest: Path | None = None
    recompile_finalizer_index: int | None = None

    @property
    def epoch(self) -> str | None:
        """The staged orchestration epoch, or ``None`` outside staged mode."""
        return self.generation if self.mode == MODE_STAGED else None

    def stages(self) -> tuple[str, ...]:
        """Return the stages ``prepare`` submits, in dependency order."""
        return tuple(
            stage
            for stage in SUBMITTED_STAGES
            if stage != STAGE_SHARDS or self.shards > 0
        )

    def to_json(self) -> dict[str, Any]:
        """Return the JSON payload."""
        return {
            "version": CHAIN_SPEC_VERSION,
            "mode": self.mode,
            "output_dir": str(self.output_dir),
            "generation": self.generation,
            "shards": self.shards,
            "scripts": {name: str(path) for name, path in self.scripts.items()},
            "recompile_manifest": (
                None
                if self.recompile_manifest is None
                else str(self.recompile_manifest)
            ),
            "recompile_finalizer_index": self.recompile_finalizer_index,
        }


def load_chain_spec(output_dir: Path, generation: str) -> ChainSpec:
    """Read and validate ``chain.json``.

    Raises:
        RuntimeError: The file is missing, of another version or mode, or
            belongs to another generation.
    """
    path = chain_spec_path(output_dir, generation)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read finalizer chain spec {path}") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("version") != CHAIN_SPEC_VERSION
        or raw.get("mode") not in MODES
        or raw.get("generation") != generation
    ):
        raise RuntimeError(f"Finalizer chain spec {path} is not valid")
    manifest = raw.get("recompile_manifest")
    finalizer_index = raw.get("recompile_finalizer_index")
    return ChainSpec(
        mode=str(raw["mode"]),
        output_dir=Path(str(raw["output_dir"])),
        generation=generation,
        shards=int(raw.get("shards", 0)),
        scripts={
            str(name): Path(str(script))
            for name, script in (raw.get("scripts") or {}).items()
        },
        recompile_manifest=None if manifest is None else Path(str(manifest)),
        recompile_finalizer_index=(
            None if finalizer_index is None else int(finalizer_index)
        ),
    )


def write_finalize_chain(
    output_dir: Path,
    *,
    mode: str,
    generation: str,
    slurm_args: Mapping[str, Any],
    shards: int,
    environment: Mapping[str, str] | None = None,
    recompile_manifest: Path | None = None,
    recompile_finalizer_index: int | None = None,
) -> Path:
    """Write one chain's scripts and ``chain.json``; return the ``prepare`` script.

    **Runs once, in the submitting process, before any chain job exists**, so
    the clear below has exactly one writer -- the same argument
    :func:`~phenotypic._cli._cli_finalize_fanout.begin_aggregation_fanout`
    makes for the shard directory.

    Args:
        output_dir: Run output root.
        mode: One of :data:`MODES`.
        generation: The lifecycle generation (staged: the epoch; recompile:
            the attempt id).
        slurm_args: The run's ``--slurm`` profile, used for every job.
        shards: K. ``0`` omits the shard stage.
        environment: Variables each script exports before running Python.
        recompile_manifest: Recompile mode: the attempt's task manifest.
        recompile_finalizer_index: Recompile mode: its finalizer task index.

    Returns:
        The ``prepare`` script, which the caller submits where it submitted
        the single finalizer before.

    Raises:
        ValueError: An unknown mode, a negative K, or recompile arguments
            missing in recompile mode.
    """
    import shutil

    from phenotypic.sdk_ import atomic_write_json, finalize_chain_dir, slurm_scripts_dir

    if mode not in MODES:
        raise ValueError(f"Unknown finalizer chain mode: {mode!r}")
    if shards < 0:
        raise ValueError(f"shards={shards} must not be negative")
    if mode == MODE_RECOMPILE and (
        recompile_manifest is None or recompile_finalizer_index is None
    ):
        raise ValueError(
            "A recompile chain needs its task manifest and finalizer index"
        )

    output_dir = Path(output_dir).absolute()
    chain_dir = finalize_chain_dir(output_dir, generation)
    if chain_dir.is_dir():
        shutil.rmtree(chain_dir)
    (chain_dir / "status").mkdir(parents=True, exist_ok=True)
    script_dir = slurm_scripts_dir(output_dir) / "finalize_chain" / generation
    script_dir.mkdir(parents=True, exist_ok=True)
    (stage_log_pattern(output_dir, STAGE_PREPARE).parent).mkdir(
        parents=True, exist_ok=True
    )

    stages = (STAGE_PREPARE, *SUBMITTED_STAGES)
    scripts: dict[str, Path] = {}
    for stage in stages:
        if stage == STAGE_SHARDS and shards == 0:
            continue
        scripts[stage] = _write_stage_script(
            script_dir / f"finalize_{stage}.sh",
            output_dir=output_dir,
            generation=generation,
            stage=stage,
            tasks=stage_task_count(stage, shards=shards),
            slurm_args=slurm_args,
            environment=environment or {},
        )
    spec = ChainSpec(
        mode=mode,
        output_dir=output_dir,
        generation=generation,
        shards=shards,
        scripts=scripts,
        recompile_manifest=(
            None
            if recompile_manifest is None
            else Path(recompile_manifest).absolute()
        ),
        recompile_finalizer_index=recompile_finalizer_index,
    )
    atomic_write_json(chain_spec_path(output_dir, generation), spec.to_json())
    return scripts[STAGE_PREPARE]


def _write_stage_script(
    path: Path,
    *,
    output_dir: Path,
    generation: str,
    stage: str,
    tasks: int,
    slurm_args: Mapping[str, Any],
    environment: Mapping[str, str],
) -> Path:
    """Render one stage's SBATCH array script."""
    from phenotypic.sdk_.slurm import (
        SlurmArrayScriptSpec,
        write_slurm_array_script,
    )

    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(shlex.quote(part) for part in python_cmd)
    q_output = shlex.quote(str(output_dir))
    q_generation = shlex.quote(generation)
    if stage == STAGE_SHARDS:
        # The P5 shard worker, unchanged. Its status files are what `master`
        # checks the shard set against.
        body = (
            f"{python_str} -m phenotypic._cli._cli_finalize_fanout "
            f"--output-dir {q_output} "
            '--task-index "$CURRENT_TASK_INDEX" '
            f"--epoch {q_generation}"
        )
    else:
        body = (
            f"{python_str} -m phenotypic._cli._cli_finalize_chain "
            f"--output-dir {q_output} "
            f"--generation {q_generation} "
            f"--stage {stage} "
            '--task-index "$CURRENT_TASK_INDEX"'
        )
    prelude = SLURM_THREAD_PIN_BASH
    for name, value in environment.items():
        prelude += f"\nexport {name}={shlex.quote(str(value))}"
    return write_slurm_array_script(
        path,
        SlurmArrayScriptSpec(
            job_name=f"pht-finalize-{stage}",
            slurm_args=slurm_args,
            log_path=stage_log_pattern(output_dir, stage),
            task_indices=list(range(tasks)),
            body=f'echo "Finalizer chain stage: {stage}"\n\n{body}',
            prelude=prelude,
            comments=[
                "# PhenoTypic SLURM finalizer chain",
                f"# Stage: {stage} ({tasks} task(s))",
                f"# Generation: {generation}",
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Status and handoff files
# ---------------------------------------------------------------------------


def write_stage_status(
    spec: ChainSpec,
    stage: str,
    task_index: int,
    status: str,
    **fields: Any,
) -> Path:
    """Record one chain task's outcome."""
    import os

    from phenotypic.sdk_ import atomic_write_json

    path = stage_status_path(spec.output_dir, spec.generation, stage, task_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "stage": stage,
            "task_index": task_index,
            "status": status,
            "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID")
            or os.environ.get("SLURM_JOB_ID"),
            **fields,
        },
    )
    return path


def read_stage_status(
    spec: ChainSpec, stage: str, task_index: int
) -> dict[str, Any] | None:
    """Return one task's status payload, or ``None`` when it wrote none."""
    path = stage_status_path(spec.output_dir, spec.generation, stage, task_index)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def chain_failures(spec: ChainSpec) -> list[str]:
    """Return one line per ``master``/``outputs``/``qc`` task that did not complete.

    The shard stage is not listed separately: ``master`` refuses an
    incomplete shard set, so a failed shard surfaces as a failed ``master``
    whose message names it.

    A task with no status file was killed (walltime, OOM, node loss) before
    it could write one; it is reported with its log pattern, because the log
    is the only place its cause was recorded.
    """
    failures: list[str] = []
    for stage in (STAGE_MASTER, STAGE_OUTPUTS, STAGE_QC):
        for task_index in range(stage_task_count(stage, shards=spec.shards)):
            status = read_stage_status(spec, stage, task_index)
            log = stage_log_pattern(spec.output_dir, stage)
            if status is None:
                failures.append(
                    f"{stage}[{task_index}]: no status (killed before it "
                    f"finished -- walltime, memory or node loss); see {log}"
                )
            elif status.get("status") == STATUS_FAILED:
                failures.append(
                    f"{stage}[{task_index}]: {status.get('error', 'failed')}; "
                    f"see {log}"
                )
    return failures


def _file_sha256(path: Path) -> str:
    from ._cli_completion import _sha256

    return _sha256(path)


def read_handoff(spec: ChainSpec) -> dict[str, Any]:
    """Return ``handoff.json``.

    Raises:
        RuntimeError: ``master`` left no handoff.
    """
    path = handoff_path(spec.output_dir, spec.generation)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"The master job left no handoff at {path}; it did not finish"
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"Handoff {path} is not a JSON object")
    return raw


def verify_handoff_bytes(spec: ChainSpec, handoff: Mapping[str, Any]) -> None:
    """Refuse to go on when the master or mirror changed since ``master`` wrote them.

    Raises:
        RuntimeError: Either file is missing or its SHA-256 differs.
    """
    from phenotypic.sdk_ import (
        master_measurements_parquet_path,
        measurements_parquet_path,
    )

    for key, path in (
        ("master_sha256", master_measurements_parquet_path(spec.output_dir)),
        ("mirror_sha256", measurements_parquet_path(spec.output_dir)),
    ):
        if not path.is_file() or _file_sha256(path) != handoff.get(key):
            raise RuntimeError(
                f"{path} changed after the master job published it. The jobs "
                "after it will not certify bytes it did not write. RECOVERY: "
                "re-run the same command."
            )


# ---------------------------------------------------------------------------
# Per-mode context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunContext:
    """What a stage needs to know about the run, read from its own records."""

    dataset_names: list[str]
    include_dataset_column: bool
    metadata_csv: Path | None
    no_qc: bool
    job_metadata: dict[str, Any] | None
    recompile_task: dict[str, Any] | None


def _run_context(spec: ChainSpec) -> _RunContext:
    """Return the run context for *spec*'s mode."""
    from phenotypic.sdk_ import JobMetadataKey, progress_dir

    from ._cli_utils import load_job_metadata

    if spec.mode == MODE_RECOMPILE:
        assert spec.recompile_manifest is not None
        assert spec.recompile_finalizer_index is not None
        manifest = json.loads(spec.recompile_manifest.read_text(encoding="utf-8"))
        task = manifest["tasks"][spec.recompile_finalizer_index]
        metadata_csv = task.get(JobMetadataKey.METADATA_CSV)
        return _RunContext(
            dataset_names=[str(name) for name in task.get("dataset_names", [])],
            include_dataset_column=bool(
                task.get("include_dataset_column", True)
            ),
            metadata_csv=Path(str(metadata_csv)) if metadata_csv else None,
            no_qc=bool(task.get(JobMetadataKey.NO_QC, False)),
            job_metadata=load_job_metadata(progress_dir(spec.output_dir)),
            recompile_task=task,
        )

    job_metadata = load_job_metadata(progress_dir(spec.output_dir))
    if job_metadata is None:
        raise RuntimeError("No job_metadata.json; cannot finalize")
    from ._cli_checkpoint_handler import _dataset_totals

    metadata_csv = job_metadata.get(JobMetadataKey.METADATA_CSV)
    return _RunContext(
        dataset_names=list(_dataset_totals(job_metadata)),
        include_dataset_column=bool(
            job_metadata.get(JobMetadataKey.INCLUDE_DATASET_COLUMN, True)
        ),
        metadata_csv=Path(str(metadata_csv)) if metadata_csv else None,
        no_qc=bool(job_metadata.get(JobMetadataKey.NO_QC, False)),
        job_metadata=job_metadata,
        recompile_task=None,
    )


def _assert_chain_active(spec: ChainSpec) -> None:
    """Raise unless the chain's generation (and staged epoch) is still active."""
    from ._cli_slurm_lifecycle import assert_generation_active

    if spec.epoch is not None:
        from ._cli_staged_orchestration import assert_active_epoch

        assert_active_epoch(spec.output_dir, spec.epoch)
    assert_generation_active(spec.output_dir, spec.generation)


def _commit_guard(spec: ChainSpec):
    """Return the per-write publication guard for *spec*, or ``None``.

    **Recompile only, and only for the serial jobs** (``master``,
    ``publish``). ``generation_publication_guard`` holds the lifecycle lock
    for the whole of each guarded write, and several writes here are long --
    the mirror CSV, the aggregate proof's hashing, an analysis fit. Handed to
    the two ``outputs`` tasks, which run at once, one task's write would hold
    the lock past the other's 300 s timeout, and past cancellation's 60 s.

    That matches what each finalizer had before the chain, where it had one:
    the single-task recompile finalizer wrapped all of ``finalize_run`` in the
    guard (longer than any one write here), and the ordinary and staged
    finalizers ran unguarded, fenced only by an epoch check between phases,
    which every chain job still makes at its start (:func:`_assert_chain_active`).
    """
    if spec.mode != MODE_RECOMPILE:
        return None
    from ._cli_slurm_lifecycle import generation_publication_guard

    return lambda: generation_publication_guard(
        spec.output_dir, spec.generation
    )


def _aggregate_publication_lock(output_dir: Path):
    """Return the lock that serializes every aggregate publisher."""
    from phenotypic.sdk_ import phenotypic_cache_dir
    from phenotypic.sdk_._file_locking import exclusive_path_lock

    return exclusive_path_lock(
        phenotypic_cache_dir(output_dir) / ".aggregate_publication.lock",
        timeout=60.0,
    )


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def run_prepare_stage(spec: ChainSpec) -> dict[str, str]:
    """Prepare the run for aggregation, then submit the rest of the chain.

    Returns:
        Stage name to submitted job id.
    """
    from phenotypic.sdk_ import progress_dir

    _assert_chain_active(spec)
    if spec.mode != MODE_RECOMPILE:
        from ._cli_checkpoint_handler import _prepare_finalization

        context = _run_context(spec)
        assert context.job_metadata is not None
        _prepare_finalization(
            spec.output_dir,
            progress_dir(spec.output_dir),
            context.job_metadata,
            epoch=spec.epoch,
        )
    _assert_chain_active(spec)
    with logged_step(logger, "finalize chain: submit stages"):
        job_ids = submit_chain_stages(spec)
    if spec.mode == MODE_STAGED:
        _hand_staged_controller_to(spec, job_ids[STAGE_PUBLISH])
    return job_ids


def submit_chain_stages(spec: ChainSpec) -> dict[str, str]:
    """Submit every stage after ``prepare``, each ``afterany`` on the one before.

    The first stage has no dependency: ``prepare`` runs only once the image
    arrays are terminal, so there is nothing left for it to wait on.

    Returns:
        Stage name to job id, in submission order.
    """
    from ._cli_slurm_lifecycle import submit_with_lifecycle

    job_ids: dict[str, str] = {}
    previous: str | None = None
    for stage in spec.stages():
        job_id = submit_with_lifecycle(
            spec.output_dir,
            generation=spec.generation,
            token=f"finalize-{stage}",
            role="finalizer",
            script_path=spec.scripts[stage],
            dependencies=() if previous is None else (previous,),
            dependency_kind="afterany",
        )
        job_ids[stage] = job_id
        previous = job_id
        logger.info("Submitted finalizer chain stage %s as job %s", stage, job_id)
    return job_ids


def _hand_staged_controller_to(spec: ChainSpec, publish_job_id: str) -> None:
    """Point the staged controller at the chain's last job.

    The controller submitted ``prepare`` as its ``"finalizer"`` work job and
    its recovery controller waits on it. Left alone, that controller would run
    as soon as ``prepare`` exits and read the phase as finished while the
    chain is still queued. So ``active_job_id`` becomes the ``publish`` job,
    which makes a controller that runs early re-arm and wait, and the pending
    controller's dependency is widened to include it.
    """
    from phenotypic.sdk_._file_locking import exclusive_path_lock

    from ._cli_staged_orchestration import (
        load_orchestration_state,
        orchestration_lock_path,
        save_orchestration_state,
        update_job_dependency,
    )

    with exclusive_path_lock(
        orchestration_lock_path(spec.output_dir), timeout=60.0
    ):
        state = load_orchestration_state(spec.output_dir)
        if state is None or state.get("epoch") != spec.generation:
            raise RuntimeError("Staged orchestration state was replaced")
        state["active_job_id"] = publish_job_id
        successor = state.get("expected_controller_id")
        # `publish` alone: it depends, transitively, on every chain job, and
        # naming `prepare` too would ask the scheduler to resolve a job that
        # may already have been purged from its table.
        if successor and not update_job_dependency(
            str(successor), [publish_job_id]
        ):
            state["dependency_update_failed"] = True
        save_orchestration_state(spec.output_dir, state)


# ---------------------------------------------------------------------------
# master
# ---------------------------------------------------------------------------


def run_master_stage(spec: ChainSpec) -> None:
    """Build and publish the master and the mirror, then write the handoff."""
    from phenotypic.sdk_ import (
        atomic_write_json,
        master_measurements_parquet_path,
        measurements_parquet_path,
    )

    from ._cli_finalize_run import publish_master_and_mirror

    _assert_chain_active(spec)
    context = _run_context(spec)
    guard = _commit_guard(spec)

    if spec.mode == MODE_RECOMPILE:
        kwargs = _recompile_master_inputs(spec, context)
    else:
        shard_paths: list[Path] | None = None
        planned: list[str] | None = None
        if spec.shards > 0:
            from ._cli_finalize_fanout import resolve_finalizer_shard_inputs

            with logged_step(
                logger, "finalize chain: check aggregation shards"
            ):
                fanout_inputs = resolve_finalizer_shard_inputs(
                    spec.output_dir,
                    spec.generation,
                    timeout=SHARD_STATUS_GRACE_SECONDS,
                )
            if fanout_inputs is None:
                raise RuntimeError(
                    "The aggregation fan-out left no task manifest; the "
                    "chain was submitted without one"
                )
            shard_paths, planned = fanout_inputs
        # K = 0 (a staged run without Stage-3 markers): no shards, so the
        # master reads the embedded tables itself, as the single-job
        # finalizer did.
        kwargs = {
            "dataset_names": context.dataset_names,
            "include_dataset_column": context.include_dataset_column,
            "metadata_csv": context.metadata_csv,
            "shard_paths": shard_paths,
            "planned_work_ids": planned,
        }
    kwargs.pop("no_qc", None)

    with _aggregate_publication_lock(spec.output_dir):
        if spec.mode == MODE_RECOMPILE:
            from ._cli_recompile_worker import _restore_overlay_marker_authority

            assert spec.recompile_manifest is not None
            # Re-fingerprint co-located overlay repairs once every recompile
            # task is terminal, so each marker describes the final bytes.
            _restore_overlay_marker_authority(
                spec.output_dir,
                spec.recompile_manifest,
                slurm_generation=spec.generation,
            )
        publication = publish_master_and_mirror(
            spec.output_dir, commit_guard=guard, **kwargs
        )
        if publication is None:
            handoff: dict[str, Any] = {"status": HANDOFF_EMPTY}
        else:
            mirror = measurements_parquet_path(spec.output_dir)
            if not mirror.is_file():
                raise RuntimeError(
                    f"The mirror {mirror} was not written; the jobs after "
                    "this one read it, so the chain cannot continue"
                )
            handoff = {
                "status": STATUS_COMPLETED,
                "master_sha256": _file_sha256(
                    master_measurements_parquet_path(spec.output_dir)
                ),
                "mirror_sha256": _file_sha256(mirror),
                "authorized": publication.authorized,
                "source_work_ids": publication.source_work_ids,
                "dataset_names": context.dataset_names,
                "no_qc": context.no_qc,
            }
        _assert_chain_active(spec)
        from phenotypic.sdk_ import publication_commit

        with publication_commit(guard):
            atomic_write_json(
                handoff_path(spec.output_dir, spec.generation), handoff
            )


def _recompile_master_inputs(
    spec: ChainSpec, context: _RunContext
) -> dict[str, Any]:
    """Check the recompile tasks and return the master's inputs."""
    from ._cli_recompile_worker import (
        _refuse_blocking_recompile_failures,
        _wait_for_non_finalizer_statuses,
        recompile_finalize_kwargs,
    )

    assert spec.recompile_manifest is not None
    assert context.recompile_task is not None
    attempt_dir = spec.recompile_manifest.parent
    statuses = _wait_for_non_finalizer_statuses(
        attempt_dir / "status",
        int(context.recompile_task.get("expected_non_finalizer_tasks", 0)),
        timeout=int(SHARD_STATUS_GRACE_SECONDS),
    )
    _refuse_blocking_recompile_failures(statuses)
    return recompile_finalize_kwargs(
        context.recompile_task, attempt_dir=attempt_dir, statuses=statuses
    )


# ---------------------------------------------------------------------------
# outputs and qc
# ---------------------------------------------------------------------------


def run_output_stage(spec: ChainSpec, stage: str, task_index: int) -> str:
    """Publish one group of derived outputs from the on-disk mirror.

    Returns:
        :data:`STATUS_COMPLETED`, or :data:`STATUS_SKIPPED` when ``master``
        did not publish a master to derive from.
    """
    import polars as pl

    from phenotypic.sdk_ import (
        master_measurements_parquet_path,
        measurements_parquet_path,
    )

    from ._cli_output_manager import (
        FINALIZE_OUTPUT_GROUP_QC,
        _load_pipeline_from_output_dir,
        publish_finalization_outputs,
    )

    if stage == STAGE_OUTPUTS:
        groups = _output_groups()
        if not 0 <= task_index < len(groups):
            raise ValueError(
                f"outputs task {task_index} is outside [0, {len(groups)})"
            )
        group = groups[task_index]
    elif stage == STAGE_QC:
        group = FINALIZE_OUTPUT_GROUP_QC
    else:  # pragma: no cover - dispatch guards the stage name
        raise ValueError(f"{stage!r} is not an output stage")

    _assert_chain_active(spec)
    try:
        handoff = read_handoff(spec)
    except RuntimeError:
        logger.warning("No master was published; skipping %s", group)
        return STATUS_SKIPPED
    if handoff.get("status") != STATUS_COMPLETED:
        logger.warning("No master was published; skipping %s", group)
        return STATUS_SKIPPED
    verify_handoff_bytes(spec, handoff)

    with logged_step(logger, f"finalize chain: read master and mirror ({group})"):
        master_df = pl.read_parquet(
            master_measurements_parquet_path(spec.output_dir)
        )
        post_df = pl.read_parquet(measurements_parquet_path(spec.output_dir))
    pipeline = _load_pipeline_from_output_dir(spec.output_dir)
    publish_finalization_outputs(
        spec.output_dir,
        master_df=master_df,
        post_df=post_df,
        pipeline=pipeline,
        no_qc=bool(handoff.get("no_qc", False)),
        commit_guard=None,
        groups=(group,),
    )
    return STATUS_COMPLETED


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


def run_publish_stage(spec: ChainSpec) -> None:
    """Publish the aggregate proof and close the run, or close it incomplete.

    Raises:
        RuntimeError: An earlier task failed (after the lifecycle is closed as
            terminal-incomplete), or the proof could not be published.
    """
    from phenotypic.sdk_ import master_measurements_parquet_path

    from ._cli_finalize_run import publish_finalization_proof

    _assert_chain_active(spec)
    failures = chain_failures(spec)
    handoff: dict[str, Any] | None = None
    if not failures:
        try:
            handoff = read_handoff(spec)
        except RuntimeError as exc:
            failures.append(f"{STAGE_MASTER}[0]: {exc}")

    aggregate_path: Path | None = None
    if not failures and handoff is not None:
        if handoff.get("status") == STATUS_COMPLETED:
            with _aggregate_publication_lock(spec.output_dir):
                verify_handoff_bytes(spec, handoff)
                publish_finalization_proof(
                    spec.output_dir,
                    dataset_names=[
                        str(name) for name in handoff.get("dataset_names", [])
                    ],
                    authorized=bool(handoff.get("authorized")),
                    source_work_ids=handoff.get("source_work_ids"),
                    commit_guard=_commit_guard(spec),
                )
            aggregate_path = master_measurements_parquet_path(spec.output_dir)

    reason = (
        None
        if not failures
        else (
            "The SLURM finalizer chain did not complete, so this run is "
            "left incomplete and nothing certifies it. RECOVERY: re-run the "
            "same command.\n  " + "\n  ".join(failures)
        )
    )
    if spec.mode == MODE_RECOMPILE:
        _publish_recompile_tail(spec, aggregate_path, reason)
        return

    from phenotypic.sdk_ import progress_dir

    from ._cli_checkpoint_handler import _publish_finalization_tail

    context = _run_context(spec)
    assert context.job_metadata is not None
    _publish_finalization_tail(
        spec.output_dir,
        progress_dir(spec.output_dir),
        context.job_metadata,
        epoch=spec.epoch,
        aggregate_path=aggregate_path,
        incomplete_reason=reason,
    )


def _publish_recompile_tail(
    spec: ChainSpec, aggregate_path: Path | None, reason: str | None
) -> None:
    """Recompile's completion, dashboard and terminal status.

    The terminal status is the finalizer task's status file, which the CLI's
    ``--wait`` watches; the chain writes it where the single-task finalizer
    did, so the waiter is unchanged.
    """
    from phenotypic.sdk_ import progress_dir

    from ._cli_recompile_slurm_scripts import TASK_FINALIZE
    from ._cli_recompile_worker import (
        _deactivate_generation_value,
        _regenerate_recompile_dashboard,
        _write_status,
    )
    from ._cli_slurm_lifecycle import generation_publication_guard

    assert spec.recompile_manifest is not None
    assert spec.recompile_finalizer_index is not None
    context = _run_context(spec)
    assert context.recompile_task is not None
    try:
        if reason is None and aggregate_path is not None:
            from ._cli_completion import (
                _all_accepted_images_succeeded,
                publish_run_completion_evidence,
                state_requires_success_markers,
            )

            if state_requires_success_markers(spec.output_dir):
                with generation_publication_guard(
                    spec.output_dir, spec.generation
                ):
                    if _all_accepted_images_succeeded(spec.output_dir) is True:
                        publish_run_completion_evidence(
                            spec.output_dir,
                            execution_epoch=spec.generation,
                        )
        _regenerate_recompile_dashboard(
            spec.output_dir,
            progress_dir(spec.output_dir),
            context.recompile_task,
            slurm_generation=spec.generation,
        )
    except Exception as exc:
        reason = reason or f"{type(exc).__name__}: {exc}"
    _write_status(
        spec.recompile_manifest,
        spec.recompile_finalizer_index,
        TASK_FINALIZE,
        {"status": STATUS_COMPLETED}
        if reason is None
        else {"status": STATUS_FAILED, "error": reason},
        output_dir=spec.output_dir,
        slurm_generation=spec.generation,
    )
    _deactivate_generation_value(spec.output_dir, spec.generation)
    if reason is not None:
        from ._cli_checkpoint_handler import FinalizationIncomplete

        raise FinalizationIncomplete(reason)


def _close_after_chain_error(spec: ChainSpec, error: BaseException) -> None:
    """Close the lifecycle when ``prepare`` or ``publish`` itself raised.

    Those two jobs are the chain's only publishers of lifecycle state, so if
    either dies with an exception nothing else would close the generation,
    and ``--wait`` and the GUI would read it as still running.
    """
    from ._cli_checkpoint_handler import FinalizationIncomplete
    from ._cli_slurm_lifecycle import generation_is_active

    if isinstance(error, FinalizationIncomplete):
        # Already closed as terminal-incomplete, on purpose. Marking it
        # failed now would overwrite the verdict the chain just recorded.
        return
    if not generation_is_active(spec.output_dir, spec.generation):
        # Cancelled, restarted or already closed: this chain no longer owns
        # the lifecycle, and must not write a verdict over the owner's.
        return
    message = f"Finalizer chain failed: {type(error).__name__}: {error}"
    try:
        if spec.mode == MODE_STAGED:
            from ._cli_staged_orchestration import deactivate_orchestration

            deactivate_orchestration(spec.output_dir, "failed")
        elif spec.mode == MODE_RECOMPILE:
            from ._cli_recompile_slurm_scripts import TASK_FINALIZE
            from ._cli_recompile_worker import (
                _deactivate_generation_value,
                _write_status,
            )

            assert spec.recompile_manifest is not None
            assert spec.recompile_finalizer_index is not None
            try:
                _write_status(
                    spec.recompile_manifest,
                    spec.recompile_finalizer_index,
                    TASK_FINALIZE,
                    {"status": STATUS_FAILED, "error": message},
                    output_dir=spec.output_dir,
                    slurm_generation=spec.generation,
                )
            finally:
                _deactivate_generation_value(spec.output_dir, spec.generation)
        else:
            from ._cli_slurm_lifecycle import mark_generation_failed

            mark_generation_failed(spec.output_dir, spec.generation, message)
    except Exception:  # noqa: BLE001 - the original error is what is raised
        logger.warning("Could not close the lifecycle after %s", message, exc_info=True)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run_chain_task(
    output_dir: Path, generation: str, stage: str, task_index: int
) -> None:
    """Run one chain task and record its status.

    Args:
        output_dir: Run output root.
        generation: The chain's lifecycle generation.
        stage: One of the ``STAGE_*`` names other than :data:`STAGE_SHARDS`.
        task_index: The array task index.

    Raises:
        RuntimeError: The task failed. Its status file says so first.
        ValueError: *stage* is not dispatched here.
    """
    from ._cli_preload import preload_custom_operation_modules

    spec = load_chain_spec(Path(output_dir), generation)
    preload_custom_operation_modules()
    if stage == STAGE_PREPARE:
        try:
            run_prepare_stage(spec)
        except BaseException as exc:
            _close_after_chain_error(spec, exc)
            raise
        return
    if stage == STAGE_PUBLISH:
        try:
            run_publish_stage(spec)
        except BaseException as exc:
            _close_after_chain_error(spec, exc)
            raise
        return
    if stage not in {STAGE_MASTER, STAGE_OUTPUTS, STAGE_QC}:
        raise ValueError(f"Stage {stage!r} is not dispatched by this module")
    try:
        if stage == STAGE_MASTER:
            with logged_step(logger, "finalize chain: master and mirror"):
                run_master_stage(spec)
            outcome = STATUS_COMPLETED
        else:
            with logged_step(
                logger, f"finalize chain: {stage}[{task_index}]"
            ):
                outcome = run_output_stage(spec, stage, task_index)
    except BaseException as exc:
        # The status is the only thing `publish` reads. Without it a failed
        # task and a killed one would be indistinguishable, and the error
        # below would live only in this task's log.
        try:
            write_stage_status(
                spec,
                stage,
                task_index,
                STATUS_FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001 - keep the original error
            logger.warning("Could not record the failed status", exc_info=True)
        raise
    write_stage_status(spec, stage, task_index, outcome)


@click.command("finalize-chain")
@click.option(
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--generation", required=True)
@click.option(
    "--stage",
    type=click.Choice(
        [STAGE_PREPARE, STAGE_MASTER, STAGE_OUTPUTS, STAGE_QC, STAGE_PUBLISH]
    ),
    required=True,
)
@click.option("--task-index", type=int, required=True)
def finalize_chain_cli(
    output_dir: Path, generation: str, stage: str, task_index: int
) -> None:
    """Run one task of a SLURM finalizer chain (``$SLURM_ARRAY_TASK_ID``)."""
    from phenotypic._startup_perf import load_runtime_dependencies

    load_runtime_dependencies()
    run_chain_task(output_dir, generation, stage, task_index)


if __name__ == "__main__":  # pragma: no cover - module entry point
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    finalize_chain_cli()
