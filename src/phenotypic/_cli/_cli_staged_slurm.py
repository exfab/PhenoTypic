"""Crash-recoverable SLURM orchestration for the staged GPU engine."""

from __future__ import annotations

import logging
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypeVar

from phenotypic.sdk_ import (
    JobMetadataKey,
    atomic_write_json,
    event_log_path,
    job_metadata_path,
    logs_dir,
    progress_dir,
    slurm_scripts_dir,
)
from phenotypic.sdk_.slurm import (
    SlurmArrayScriptSpec,
    calculate_optimal_array_chunks,
    get_slurm_array_limit,
    get_slurm_max_submit_jobs,
    write_slurm_array_script,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_execution_strategies import ExecutionStrategy
from ._cli_staged_orchestration import (
    StagedManifestEntry,
    completed_inventory_images,
    deactivate_orchestration,
    initialize_orchestration,
    load_orchestration_state,
    new_orchestration_epoch,
    orchestration_lock_path,
    save_orchestration_state,
    snapshot_inventory_parquets,
    staged_completion_matches,
    submit_with_intent,
    write_staged_manifest,
)
from ._cli_types import (
    Dataset,
    DatasetResults,
    ExecutionConfig,
    ExecutionResults,
    ImageFailure,
)
from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

def partition_shards(items: List[_T], n_shards: int) -> List[List[_T]]:
    """Split *items* into up to *n_shards* near-even contiguous shards (no loss)."""
    n = max(1, n_shards)
    k, r = divmod(len(items), n)
    shards: List[List[_T]] = []
    start = 0
    for i in range(n):
        size = k + (1 if i < r else 0)
        shards.append(items[start : start + size])
        start += size
    return shards


def resolve_stage_slurm_args(
    gpu_slurm_args: Dict[str, Any],
    cpu_slurm_args: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """GPU-stage (Stage 2) SBATCH args: inherit/delta over the CPU profile.

    Effective args = ``{**cpu_slurm_args, **gpu_slurm_args}``. Then resolve the
    GPU count:

    - absent -> auto-add ``slurm_gpus_per_node=1`` (request one whole GPU);
    - explicit ``0`` -> **omit** the key entirely (a CPU-only run of the GPU
      stage, e.g. the live dispatch test) — ``format_sbatch_directives`` would
      otherwise emit ``--gpus-per-node=0``, which SLURM rejects (OQ4);
    - explicit ``>0`` -> keep as given.

    Shared keys (account, qos, time) set in ``--slurm`` carry over; a separate
    GPU partition/account in ``--gpu-slurm`` overrides.
    """
    args = {**(cpu_slurm_args or {}), **gpu_slurm_args}
    gpus = args.get("slurm_gpus_per_node")
    if gpus == 0:
        args.pop("slurm_gpus_per_node", None)
    elif gpus is None:
        args["slurm_gpus_per_node"] = 1
    return args


def _stage_worker_body(
    python_str: str,
    stage: int,
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest_path: Path,
    ext: str,
    epoch: str,
    resume: bool,
    markers_required: bool = True,
    n_shards: int | None = None,
    index_var: str = "$SLURM_ARRAY_TASK_ID",
) -> str:
    """The per-array-task command line that invokes the staged SLURM worker.

    ``index_var`` is the shell expression the worker uses as its manifest index.
    Image stages (1 & 3) pass ``$CURRENT_TASK_INDEX`` — the per-chunk
    ``TASK_INDICES`` window maps a 0-based array id to the ABSOLUTE manifest
    index, so a chunked stage never emits an array index above ``MaxArraySize``.
    Stage 2 keeps ``$SLURM_ARRAY_TASK_ID`` (the shard index; never chunked).
    """
    q = shlex.quote
    parts = [
        f"{python_str} -m phenotypic._cli._cli_staged_slurm_worker",
        f"--stage {stage}",
        f"--pipeline {q(str(Path(pipeline_path).absolute()))}",
        f"--output-dir {q(str(output_dir.absolute()))}",
        f"--image-type {image_type}",
        f"--manifest {q(str(manifest_path.absolute()))}",
        f"--index {index_var}",
        f"--ext {q(ext)}",
        f"--epoch {q(epoch)}",
    ]
    if resume:
        parts.append("--resume")
    if markers_required:
        parts.append("--stage3-markers-required")
    if stage == 2:
        parts.append(f"--n-shards {n_shards}")
    return " \\\n    ".join(parts)


def _write_stage_script(
    script_dir: Path,
    log_dir: Path,
    stage_name: str,
    job_name: str,
    slurm_args: Dict[str, Any],
    task_indices: Sequence[int],
    body: str,
) -> Path:
    """Render + write one stage SBATCH array script (returns its path).

    ``task_indices`` are the manifest indices this array covers. The rendered
    ``--array`` directive is always 0-based over ``len(task_indices)`` and the
    ``TASK_INDICES`` bash array maps the array id to the (possibly offset)
    manifest index — so a chunk covering images 2500-3217 is submitted as
    ``--array=0-717`` with ``TASK_INDICES=(2500 … 3217)``, never an index that
    exceeds the cluster ``MaxArraySize``.
    """
    out_log = log_dir / f"{stage_name}_%A_%a.log"
    err_log = log_dir / f"{stage_name}_%A_%a.err"
    path = script_dir / f"{stage_name}.sh"
    return write_slurm_array_script(
        path,
        SlurmArrayScriptSpec(
            job_name=job_name,
            slurm_args=slurm_args,
            log_path=out_log,
            error_log_path=err_log,
            task_indices=list(task_indices) or [0],
            body=body,
            prelude=SLURM_THREAD_PIN_BASH,
            comments=[f"# Stage: {stage_name}"],
        ),
    )


def _write_image_stage_chunks(
    *,
    script_dir: Path,
    log_dir: Path,
    stage: int,
    slurm_args: Dict[str, Any],
    body: str,
    chunks: List[tuple[int, int]],
) -> List[Path]:
    """Write one CPU array script per image chunk for stage 1 or 3.

    A single chunk keeps the plain ``stageN.sh`` name (byte-identical to the
    pre-chunking output); multiple chunks are ``stageN_chunk{i}.sh``. Each
    covers the absolute manifest window ``[start, end)`` via its ``TASK_INDICES``
    array, so no chunk's SLURM array index reaches the cluster ``MaxArraySize``.
    """
    single = len(chunks) == 1
    scripts: List[Path] = []
    for i, (start, end) in enumerate(chunks):
        stage_name = f"stage{stage}" if single else f"stage{stage}_chunk{i}"
        scripts.append(
            _write_stage_script(
                script_dir,
                log_dir,
                stage_name,
                f"phenotypic-stage{stage}",
                slurm_args,
                list(range(start, end)),
                body,
            )
        )
    return scripts


def _finalizer_body(python_str: str, output_dir: Path, epoch: str) -> str:
    """Build the canonical aggregate/finalize command for staged runs."""
    return (
        f"{python_str} -m phenotypic._cli._cli_checkpoint_handler "
        f"--output-dir {shlex.quote(str(output_dir.absolute()))} "
        "--checkpoint-type finalize "
        f"--epoch {shlex.quote(epoch)}"
    )


def _controller_body(python_str: str, config_path: Path) -> str:
    """Build the command for one controller transition."""
    return (
        f"{python_str} -m phenotypic._cli._cli_staged_controller "
        f"--config {shlex.quote(str(config_path.absolute()))}"
    )


def generate_staged_scripts(
    *,
    pipeline_path: Path,
    datasets_manifest: Sequence[StagedManifestEntry],
    output_dir: Path,
    image_type: ImageTypeName,
    cpu_slurm_args: Dict[str, Any],
    gpu_slurm_args: Dict[str, Any],
    n_shards: int,
    ext: str = ".tiff",
    array_limit: int | None = None,
    epoch: str,
    resume: bool = False,
    markers_required: bool = True,
) -> Dict[str, Any]:
    """Write the per-stage SBATCH array scripts (no submission).

    Stage 1 & 3 are arrays over images (CPU ``cpu_slurm_args``); Stage 2 is an
    array over shards (GPU args resolved from ``gpu_slurm_args`` over the CPU
    profile, auto-1-GPU). When the image count exceeds ``array_limit`` (the
    cluster ``MaxArraySize``, queried when ``None``), the two image stages are
    split into ``ceil(n_images / array_limit)`` chunk scripts; Stage 2 is never
    chunked. The manifest is written to disk for the workers to index.

    Returns ``{"stage1": [Path, ...], "stage2": Path, "stage3": [Path, ...],
    "finalizer": Path}`` — the image stages are always lists (length 1 when
    unchunked). The one-task finalizer uses the CPU profile and invokes the same
    aggregate/finalize entry point as ordinary SLURM runs.
    """
    script_dir = slurm_scripts_dir(output_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = logs_dir(output_dir) / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = script_dir / "staged_manifest.json"
    write_staged_manifest(manifest_path, datasets_manifest)

    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(python_cmd)
    n_images = len(datasets_manifest)
    if array_limit is None:
        array_limit = get_slurm_array_limit()
    gpu_args = resolve_stage_slurm_args(gpu_slurm_args, cpu_slurm_args)

    # Image chunks tile the manifest into <=array_limit-wide windows; each stage-1
    # and stage-3 chunk resolves its absolute index via $CURRENT_TASK_INDEX.
    chunks = calculate_optimal_array_chunks(n_images, array_limit) if n_images else []

    def _image_stage_body(stage: int) -> str:
        return _stage_worker_body(
            python_str,
            stage,
            pipeline_path,
            output_dir,
            image_type,
            manifest_path,
            ext,
            epoch,
            resume,
            markers_required,
            index_var="$CURRENT_TASK_INDEX",
        )

    stage1 = _write_image_stage_chunks(
        script_dir=script_dir,
        log_dir=log_dir,
        stage=1,
        slurm_args=cpu_slurm_args,
        body=_image_stage_body(1),
        chunks=chunks,
    )
    stage2 = _write_stage_script(
        script_dir,
        log_dir,
        "stage2",
        "phenotypic-stage2",
        gpu_args,
        list(range(max(1, n_shards))),
        _stage_worker_body(
            python_str,
            2,
            pipeline_path,
            output_dir,
            image_type,
            manifest_path,
            ext,
            epoch,
            resume,
            markers_required,
            n_shards=n_shards,
        ),
    )
    stage3 = _write_image_stage_chunks(
        script_dir=script_dir,
        log_dir=log_dir,
        stage=3,
        slurm_args=cpu_slurm_args,
        body=_image_stage_body(3),
        chunks=chunks,
    )
    finalizer = _write_stage_script(
        script_dir,
        log_dir,
        "finalizer",
        "phenotypic-finalize",
        cpu_slurm_args,
        [0],
        _finalizer_body(python_str, output_dir, epoch),
    )
    controller_config = script_dir / "staged_controller.json"
    controller = _write_stage_script(
        script_dir,
        log_dir,
        "controller",
        "phenotypic-controller",
        cpu_slurm_args,
        [0],
        _controller_body(python_str, controller_config),
    )
    atomic_write_json(
        controller_config,
        {
            "version": 1,
            "epoch": epoch,
            "output_dir": str(output_dir.absolute()),
            "resume": resume,
            "stage3_markers_required": markers_required,
            "manifest_path": str(manifest_path.absolute()),
            "stage1_scripts": [str(path.absolute()) for path in stage1],
            "stage2_script": str(stage2.absolute()),
            "stage3_scripts": [str(path.absolute()) for path in stage3],
            "finalizer_script": str(finalizer.absolute()),
            "controller_script": str(controller.absolute()),
        },
    )
    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "finalizer": finalizer,
        "controller": controller,
        "controller_config": controller_config,
        "manifest": manifest_path,
    }


def _ordered_staged_scripts(scripts: Dict[str, Any]) -> List[Path]:
    """Return all scripts in logical lifecycle order."""
    return [
        *scripts["stage1"],
        scripts["controller"],
        scripts["stage2"],
        *scripts["stage3"],
        scripts["finalizer"],
    ]


def _write_staged_job_metadata(
    *,
    datasets: Sequence[Dataset],
    output_dir: Path,
    config: ExecutionConfig,
    scripts: Dict[str, Any],
    job_ids: Sequence[str],
    start_time: datetime,
    epoch: str,
) -> Path:
    """Write the metadata consumed by the canonical SLURM finalizer.

    The drip-feed submission knows only the first chunk job id at launch; later
    chunk ids are allocated by dispatcher jobs. Stage workers remain observable
    through their stage-tagged event records, so the initial metadata records
    the known ids and leaves the optional image/task mapping empty.
    """
    prog_dir = progress_dir(output_dir)
    prog_dir.mkdir(parents=True, exist_ok=True)

    ordered_scripts = _ordered_staged_scripts(scripts)
    metadata = {
        JobMetadataKey.START_TIME: start_time.isoformat(
            timespec="milliseconds"
        ),
        JobMetadataKey.EXECUTION_MODE: "slurm",
        JobMetadataKey.DATASETS: {
            name: {"total": len(images), "images": images}
            for name, images in (
                getattr(config, "full_dataset_inventory", {})
                or {
                    ds.name: [image.name for image in ds.images]
                    for ds in datasets
                }
            ).items()
        },
        JobMetadataKey.CHUNK_SCRIPTS: [str(path) for path in ordered_scripts],
        JobMetadataKey.CHUNK_JOB_IDS: {
            str(index): str(job_id) for index, job_id in enumerate(job_ids)
        },
        JobMetadataKey.IMAGE_TASK_MAPPING: {},
        JobMetadataKey.INCLUDE_DATASET_COLUMN: config.include_dataset_column,
        JobMetadataKey.METADATA_CSV: (
            str(config.metadata_csv) if config.metadata_csv else None
        ),
        JobMetadataKey.NO_QC: config.no_qc,
        JobMetadataKey.INPUT_PATH: config.input_path.stem,
        JobMetadataKey.ORCHESTRATION_EPOCH: epoch,
        JobMetadataKey.PIPELINE_PATH: str(
            Path(getattr(config, "pipeline_json", "pipeline.json")).absolute()
        ),
        JobMetadataKey.IMAGE_TYPE: getattr(config, "image_type", "Image"),
        JobMetadataKey.NROWS: getattr(config, "nrows", None),
        JobMetadataKey.NCOLS: getattr(config, "ncols", None),
        JobMetadataKey.SLURM_JOB_IDS: {
            str(index): str(job_id) for index, job_id in enumerate(job_ids)
        },
    }
    metadata_path = job_metadata_path(output_dir)
    atomic_write_json(metadata_path, metadata)
    return metadata_path


class StagedSlurmStrategy(ExecutionStrategy):
    """Submit the 3 staged stages as a SLURM ``afterany`` dependency chain."""

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        start = datetime.now()
        cfg = self.config
        manifest = [
            StagedManifestEntry(
                dataset=ds.name,
                image_name=img.name,
                stem=img.stem,
                input_path=str(Path(img).absolute()),
            )
            for ds in datasets
            for img in ds.images
        ]

        # Chunk to the TIGHTER of MaxArraySize and the conservative
        # MaxSubmitJobs estimate. Reserve slots for the running controller and
        # its pre-armed recovery controller while an array is active.
        array_limit = get_slurm_array_limit()
        max_submit = get_slurm_max_submit_jobs()
        if max_submit is not None and max_submit < 3:
            raise ValueError(
                "SLURM MaxSubmitJobs must be at least 3 for staged GPU "
                "orchestration (controller, array, recovery controller)."
            )
        submit_capacity = max_submit - 2 if max_submit else array_limit
        chunk_limit = min(array_limit, submit_capacity)
        # Image stages chunk to fit the limit; the Stage-2 shard array cannot
        # (a shard worker streams its whole shard on one GPU), so guard it.
        if max(1, cfg.gpu_shards) > chunk_limit:
            raise ValueError(
                f"--gpu-shards ({cfg.gpu_shards}) exceeds the SLURM chunk limit "
                f"({chunk_limit}); reduce the shard count."
            )

        epoch = new_orchestration_epoch()
        scripts = generate_staged_scripts(
            pipeline_path=cfg.pipeline_json,
            datasets_manifest=manifest,
            output_dir=output_dir,
            image_type=cfg.image_type,
            cpu_slurm_args=cfg.slurm_args,
            gpu_slurm_args=cfg.gpu_slurm_args,
            n_shards=max(1, cfg.gpu_shards),
            ext=cfg.ext,
            array_limit=chunk_limit,
            epoch=epoch,
            resume=getattr(cfg, "resume", False),
            markers_required=getattr(cfg, "staged_stage3_markers", True),
        )
        _write_staged_job_metadata(
            datasets=datasets,
            output_dir=output_dir,
            config=cfg,
            scripts=scripts,
            job_ids=[],
            start_time=start,
            epoch=epoch,
        )
        state = initialize_orchestration(
            output_dir,
            epoch=epoch,
            mode=(
                "restart"
                if getattr(cfg, "restart", False)
                else "resume"
                if getattr(cfg, "resume", False)
                else "fresh"
            ),
            controller_config_path=scripts["controller_config"],
        )
        if getattr(cfg, "restart", False):
            state["restart_parquet_fingerprints"] = snapshot_inventory_parquets(
                output_dir, cfg.full_dataset_inventory
            )
        initial_phase = getattr(cfg, "staged_resume_phase", None) or "stage1"
        state.update(
            {
                "phase": initial_phase,
                "stage1_index": 0,
                "resume_initial_phase": initial_phase,
                "stage3_markers_required": getattr(
                    cfg, "staged_stage3_markers", True
                ),
            }
        )
        if getattr(cfg, "staged_finalizer_only", False):
            state["phase"] = "stage3"
            state["stage3_index"] = len(scripts["stage3"])
        save_orchestration_state(output_dir, state)
        try:
            controller_id = submit_with_intent(
                output_dir,
                epoch=epoch,
                token="controller-initial",
                role="controller",
                round_index=0,
                script_path=scripts["controller"],
            )
        except Exception:
            deactivate_orchestration(output_dir, "failed")
            raise
        self.submitted_job_ids = [controller_id]
        with exclusive_path_lock(
            orchestration_lock_path(output_dir), timeout=60.0
        ):
            current = load_orchestration_state(output_dir)
            if current is not None and current.get("epoch") == epoch:
                if current.get("expected_controller_id") is None:
                    current["expected_controller_id"] = controller_id
                    save_orchestration_state(output_dir, current)
        logger.info(
            "Submitted staged GPU controller lifecycle "
            "(%d stage-1 chunks -> stage2 rounds -> %d stage-3 chunks -> finalizer): "
            "initial controller=%s",
            len(scripts["stage1"]),
            len(scripts["stage3"]),
            controller_id,
        )

        if not cfg.wait:
            return self._submitted_results(start)
        return self._wait_for_finalizer(output_dir, start, epoch)

    def _submitted_results(self, start: datetime) -> ExecutionResults:
        """Return immediately after SLURM accepted the initial jobs."""
        inventory = self.config.full_dataset_inventory
        return ExecutionResults(
            datasets={
                name: DatasetResults(name, len(images), 0, 0, [])
                for name, images in inventory.items()
            },
            total_images=sum(len(images) for images in inventory.values()),
            total_completed=0,
            total_failed=0,
            execution_mode="slurm",
            start_time=start,
            end_time=datetime.now(),
            submitted=True,
            remote_managed=True,
        )

    def _wait_for_finalizer(
        self, output_dir: Path, start: datetime, epoch: str
    ) -> ExecutionResults:
        """Monitor the active epoch until its remote finalizer is terminal."""
        detached = False
        try:
            while True:
                state = load_orchestration_state(output_dir)
                if state is None or state.get("epoch") != epoch:
                    raise RuntimeError(
                        "Staged orchestration state was replaced"
                    )
                if state.get("phase") in {"complete", "failed", "cancelled"}:
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            detached = True
        return self._results_from_events(output_dir, start, epoch, detached)

    def _results_from_events(
        self, output_dir: Path, start: datetime, epoch: str, detached: bool
    ) -> ExecutionResults:
        """Build full-inventory results from terminal Stage-3 events."""
        from ._cli_update_state import aggregate_state_from_events

        states = aggregate_state_from_events(event_log_path(output_dir))
        inventory = self.config.full_dataset_inventory
        dataset_results: dict[str, DatasetResults] = {}
        for name, images in inventory.items():
            ds_state = states.get(name)
            completed = completed_inventory_images(output_dir, name, images)
            event_failed = set() if ds_state is None else ds_state.failed
            failed = (set(images) - completed) | event_failed
            failed -= completed
            errors = {} if ds_state is None else ds_state.errors
            failures = [
                ImageFailure(
                    dataset=name,
                    image_filename=image_name,
                    error_type="StageFailure",
                    error_message=errors.get(
                        image_name, "Staged processing failed"
                    ),
                    traceback="",
                    timestamp=datetime.now(),
                )
                for image_name in sorted(failed)
            ]
            dataset_results[name] = DatasetResults(
                name=name,
                total=len(images),
                completed=len(completed),
                failed=len(failed),
                failures=failures,
            )
        state = load_orchestration_state(output_dir) or {}
        succeeded = (
            state.get("epoch") == epoch
            and state.get("phase") == "complete"
            and staged_completion_matches(output_dir, epoch)
        )
        return ExecutionResults(
            datasets=dataset_results,
            total_images=sum(
                result.total for result in dataset_results.values()
            ),
            total_completed=sum(
                result.completed for result in dataset_results.values()
            ),
            total_failed=sum(
                result.failed for result in dataset_results.values()
            ),
            execution_mode="slurm",
            start_time=start,
            end_time=datetime.now(),
            detached=detached,
            remote_finalized=succeeded,
            remote_managed=True,
        )
