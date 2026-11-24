"""Crash-recoverable SLURM orchestration for the staged GPU engine."""

from __future__ import annotations

import logging
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence, TypeVar
from uuid import uuid4

from phenotypic.sdk_ import (
    JobMetadataKey,
    atomic_write_json,
    job_metadata_path,
    slurm_logs_dir,
    progress_dir,
    slurm_scripts_dir,
    source_image_stem,
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

from ._cli_execution_strategies import (
    ExecutionStrategy,
    echo_slurm_monitor_commands,
    finalizer_aggregation_options,
)
from ._cli_failure_tracker import WorkIdentity, work_identity_for_image
from ._cli_stage2_token import staged_detector_slot
from ._cli_staged_orchestration import (
    StagedManifestEntry,
    deactivate_orchestration,
    initialize_orchestration,
    load_orchestration_state,
    new_orchestration_epoch,
    orchestration_lock_path,
    save_orchestration_state,
    snapshot_inventory_parquets,
    stage4_worklist_path,
    submit_with_intent,
    write_staged_manifest,
)
from ._cli_types import (
    Dataset,
    DatasetResults,
    ExecutionConfig,
    ExecutionResults,
)
from ._cli_update_state import PROCESSING_GENERATION_ENV_VAR
from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: What the staged worker's Stage 3 runs: the staged merge-and-measure
#: (``"staged"``, a run with a ``GpuDetector``) or the single-pass core with
#: its embeddings deferred (``"single_pass"``, a Stage-4 run without one,
#: #282 P5 T2), or the ``--mode measure`` core over an existing store with its
#: embeddings carried and deferred (``"measure"``, #282 P6 T2, U10).
Stage3Role = Literal["staged", "single_pass", "measure"]

#: Every phase of a staged run with a ``GpuDetector`` that embeds on the GPU.
STAGED_PHASES_WITH_STAGE4 = ("stage1", "stage2", "stage3", "stage4")
#: A staged run with a ``GpuDetector`` that embeds inline (a v1 config's phases).
STAGED_PHASES_INLINE = ("stage1", "stage2", "stage3")
#: A Stage-4 run without a ``GpuDetector`` (spec §4.1).
NO_DETECTOR_PHASES = ("stage3", "stage4")

#: The Stage-4 script prelude: the thread pins, then the deterministic cuBLAS
#: workspace, exported before Python starts so the shard's first cuBLAS handle
#: sees it (#282 spec §4.3).
STAGE4_PRELUDE_BASH = SLURM_THREAD_PIN_BASH + "\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8"

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


def staged_slurm_limit_errors(
    max_submit: int | None, array_limit: int, gpu_shards: int
) -> list[str]:
    """Why a staged GPU SLURM run cannot be orchestrated, or nothing.

    The one statement of the two limits the staged strategy enforces; the run
    preflight (``PF-SLURM-LIMIT``) and :class:`StagedSlurmStrategy` both call
    it, so the early refusal and the late one cannot disagree (spec
    2026-09-24-cli-preflight §6, F18).

    Args:
        max_submit: Conservative ``MaxSubmitJobs`` (``None`` when unknown).
        array_limit: ``MaxArraySize``.
        gpu_shards: ``--gpu-shards``.

    Returns:
        Error messages; empty when the run fits.
    """
    if max_submit is not None and max_submit < 3:
        return [
            "SLURM MaxSubmitJobs must be at least 3 for staged GPU "
            "orchestration (controller, array, recovery controller)."
        ]
    submit_capacity = max_submit - 2 if max_submit else array_limit
    chunk_limit = min(array_limit, submit_capacity)
    if max(1, gpu_shards) > chunk_limit:
        return [
            f"--gpu-shards ({gpu_shards}) exceeds the SLURM chunk limit "
            f"({chunk_limit}); reduce the shard count."
        ]
    return []


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
    overlay_alpha: float = 0.3,
    processing_generation: str | None = None,
    durable_writes: bool | None = None,
    drop_originals: bool = False,
    pipeline_identity: Mapping[str, str] | None = None,
    stage3_role: Stage3Role = "staged",
    worklist_path: Path | None = None,
) -> str:
    """The per-array-task command line that invokes the staged SLURM worker.

    ``index_var`` is the shell expression the worker uses as its manifest index.
    Image stages (1 & 3) pass ``$CURRENT_TASK_INDEX`` — the per-chunk
    ``TASK_INDICES`` window maps a 0-based array id to the ABSOLUTE manifest
    index, so a chunked stage never emits an array index above ``MaxArraySize``.
    Stages 2 and 4 keep ``$SLURM_ARRAY_TASK_ID`` (the shard index; never
    chunked). Stage 4 reads its shard from ``worklist_path``, the per-round
    worklist the controller writes (#282 P5 T1). A Stage 3 whose role is not
    ``"staged"`` names it (``--stage3-role``); the single-pass role reads the
    input as Stage 1 does, so it carries Stage 1's provenance identity and
    ``--drop-originals`` too.
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
        # Internal worker contract, not a public lifecycle option. This keeps
        # automatic continuation distinct from --restart after removing the
        # public continuation flag.
        parts.append("--reuse-existing")
    if markers_required:
        parts.append("--stage3-markers-required")
    reads_input = stage == 1 or (stage == 3 and stage3_role == "single_pass")
    if reads_input and pipeline_identity is not None:
        parts.extend(
            [
                "--provenance-pipeline-source-path "
                f"{q(pipeline_identity['source_path'])}",
                f"--provenance-pipeline-sha256 "
                f"{q(pipeline_identity['sha256'])}",
            ]
        )
    if reads_input and drop_originals:
        parts.append("--drop-originals")
    if stage == 2:
        parts.append(f"--n-shards {n_shards}")
    if stage == 3 and stage3_role != "staged":
        parts.append(f"--stage3-role {stage3_role}")
    if stage in (3, 4):
        parts.append(f"--overlay-alpha {overlay_alpha}")
    if stage == 4:
        if worklist_path is None:
            raise ValueError("a Stage-4 worker reads its shard from a worklist")
        parts.append(f"--worklist {q(str(Path(worklist_path).absolute()))}")
    # Tri-state, so ONLY an explicit choice is emitted: an unset flag has to
    # reach the worker unset, where ngff_.durable_writes_enabled re-detects
    # SLURM on the worker's own node. --no-durable-writes, by contrast, exists
    # only in this submitting process; dropping it would leave every staged
    # worker fsyncing and make the flag look inert on the one path where it
    # costs the most (spec §3.7). Stage 2 writes no store, but the flag is
    # harmless there and omitting it per stage would be one more branch to get
    # wrong.
    if durable_writes is not None:
        parts.append(
            "--durable-writes" if durable_writes else "--no-durable-writes"
        )
    command = " \\\n    ".join(parts)
    if processing_generation:
        return (
            f"export {PROCESSING_GENERATION_ENV_VAR}="
            f"{shlex.quote(processing_generation)}\n{command}"
        )
    return command


def _write_stage_script(
    script_dir: Path,
    log_dir: Path,
    stage_name: str,
    job_name: str,
    slurm_args: Dict[str, Any],
    task_indices: Sequence[int],
    body: str,
    prelude: str = SLURM_THREAD_PIN_BASH,
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
            prelude=prelude,
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
    detector_slot: str | None,
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
    overlay_alpha: float = 0.3,
    processing_generation: str | None = None,
    durable_writes: bool | None = None,
    drop_originals: bool = False,
    pipeline_identity: Mapping[str, str] | None = None,
    phases: Sequence[str] = STAGED_PHASES_INLINE,
    stage3_role: Stage3Role = "staged",
    pipeline_sha256: str | None = None,
) -> Dict[str, Any]:
    """Write the per-stage SBATCH array scripts and the controller config (no submission).

    Stage 1 & 3 are arrays over images (CPU ``cpu_slurm_args``); Stages 2 and
    4 are arrays over shards (GPU args resolved from ``gpu_slurm_args`` over
    the CPU profile, auto-1-GPU). When the image count exceeds
    ``array_limit`` (the cluster ``MaxArraySize``, queried when ``None``),
    the two image stages are split into ``ceil(n_images / array_limit)``
    chunk scripts; Stages 2 and 4 are never chunked. The manifest is written
    to disk for the workers to index.

    *phases* says which stages the run has (controller config v2, #282 spec
    §4.3): ``["stage1", "stage2", "stage3"]`` for a ``GpuDetector`` run that
    embeds inline, plus ``"stage4"`` when it embeds on the GPU, or
    ``["stage3", "stage4"]`` for a Stage-4 run without a detector, whose
    Stage 3 runs in *stage3_role* ``"single_pass"`` and which has no
    *detector_slot*. Stage-1 and Stage-2 scripts are written only when
    ``"stage1"`` is in *phases*. ``stage4.sh`` is an array of *n_shards*
    tasks on the Stage-2 GPU profile, with the deterministic cuBLAS
    workspace exported in its prelude; no script pins a GPU type (U6), so
    shards may land on different GPU models and each entry's ``compute``
    block records which ran. *pipeline_sha256* is the measure pipeline's
    (P6); a forward run passes ``None``. The ``"measure"`` role (``--mode
    measure``, U10) writes the manifest flagged *republish* (version 5,
    :func:`~phenotypic._cli._cli_staged_orchestration.staged_manifest_republishes`),
    the one carrier of republish mode.

    Returns ``{"stage1": [Path, ...], "stage2": Path, "stage3": [Path, ...],
    "stage4": Path, "finalizer": Path, "controller": Path,
    "controller_config": Path, "manifest": Path}``; ``stage1``/``stage2``
    only with ``"stage1"`` in *phases* and ``stage4`` only with ``"stage4"``.
    The image stages are always lists (length 1 when unchunked). The one-task
    finalizer uses the CPU profile and invokes the same aggregate/finalize
    entry point as ordinary SLURM runs.

    Raises:
        ValueError: *phases* has ``"stage1"`` but no *detector_slot*, or
            *stage3_role* does not fit *phases*.
    """
    phases = list(phases)
    detected = "stage1" in phases
    if detected and detector_slot is None:
        raise ValueError("a staged run with Stage 1 needs its detector slot")
    if (stage3_role == "staged") is not detected:
        raise ValueError(
            f"stage3 role {stage3_role!r} does not fit phases {phases!r}"
        )
    script_dir = slurm_scripts_dir(output_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir = slurm_logs_dir(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = script_dir / "staged_manifest.json"
    write_staged_manifest(
        manifest_path, datasets_manifest, republish=stage3_role == "measure"
    )

    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(python_cmd)
    n_images = len(datasets_manifest)
    if array_limit is None:
        array_limit = get_slurm_array_limit()
    gpu_args = resolve_stage_slurm_args(gpu_slurm_args, cpu_slurm_args)

    # Image chunks tile the manifest into <=array_limit-wide windows; each stage-1
    # and stage-3 chunk resolves its absolute index via $CURRENT_TASK_INDEX.
    chunks = calculate_optimal_array_chunks(n_images, array_limit) if n_images else []

    def _body(stage: int, **kwargs: Any) -> str:
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
            overlay_alpha=overlay_alpha,
            processing_generation=processing_generation,
            durable_writes=durable_writes,
            **kwargs,
        )

    image_stage = {
        "index_var": "$CURRENT_TASK_INDEX",
        "drop_originals": drop_originals,
        "pipeline_identity": pipeline_identity,
    }
    shards = list(range(max(1, n_shards)))
    scripts: Dict[str, Any] = {}
    if detected:
        scripts["stage1"] = _write_image_stage_chunks(
            script_dir=script_dir,
            log_dir=log_dir,
            stage=1,
            slurm_args=cpu_slurm_args,
            body=_body(1, **image_stage),
            chunks=chunks,
        )
        scripts["stage2"] = _write_stage_script(
            script_dir,
            log_dir,
            "stage2",
            "phenotypic-stage2",
            gpu_args,
            shards,
            _body(2, n_shards=n_shards),
        )
    scripts["stage3"] = _write_image_stage_chunks(
        script_dir=script_dir,
        log_dir=log_dir,
        stage=3,
        slurm_args=cpu_slurm_args,
        body=_body(3, stage3_role=stage3_role, **image_stage),
        chunks=chunks,
    )
    worklist_path = stage4_worklist_path(output_dir)
    if "stage4" in phases:
        scripts["stage4"] = _write_stage_script(
            script_dir,
            log_dir,
            "stage4",
            "phenotypic-stage4",
            gpu_args,
            shards,
            _body(4, worklist_path=worklist_path),
            prelude=STAGE4_PRELUDE_BASH,
        )
    scripts["finalizer"] = _write_stage_script(
        script_dir,
        log_dir,
        "finalizer",
        "phenotypic-finalize",
        cpu_slurm_args,
        [0],
        _finalizer_body(python_str, output_dir, epoch),
    )
    controller_config = script_dir / "staged_controller.json"
    scripts["controller"] = _write_stage_script(
        script_dir,
        log_dir,
        "controller",
        "phenotypic-controller",
        cpu_slurm_args,
        [0],
        _controller_body(python_str, controller_config),
    )
    config: Dict[str, Any] = {
        "version": 2,
        "phases": phases,
        "epoch": epoch,
        "output_dir": str(output_dir.absolute()),
        "resume": resume,
        "stage3_markers_required": markers_required,
        "stage3_role": stage3_role,
        "pipeline_sha256": pipeline_sha256,
        "manifest_path": str(manifest_path.absolute()),
        "stage3_scripts": [str(path.absolute()) for path in scripts["stage3"]],
        "finalizer_script": str(scripts["finalizer"].absolute()),
        "controller_script": str(scripts["controller"].absolute()),
        "n_shards": max(1, n_shards),
    }
    if detected:
        # The recovery controller probes the Stage-2 signal but never
        # loads the pipeline, so the slot is recorded here. It arrives as
        # a parameter rather than being re-derived from `pipeline_path`:
        # this function WRITES scripts, and `pipeline_path` is a string it
        # embeds in them, not a file it reads. Parsing here also made
        # script generation fail on a pipeline that was never written --
        # see the strategy's preflight for where the parse lives now.
        config["detector_slot"] = detector_slot
        config["stage1_scripts"] = [str(path.absolute()) for path in scripts["stage1"]]
        config["stage2_script"] = str(scripts["stage2"].absolute())
    if "stage4" in phases:
        config["stage4_script"] = str(scripts["stage4"].absolute())
        config["stage4_worklist"] = str(worklist_path.absolute())
    atomic_write_json(controller_config, config)
    scripts["controller_config"] = controller_config
    scripts["manifest"] = manifest_path
    return scripts


def _ordered_staged_scripts(scripts: Dict[str, Any]) -> List[Path]:
    """Return all scripts in logical lifecycle order (the stages a run has)."""
    ordered: List[Path] = [*scripts.get("stage1", []), scripts["controller"]]
    if scripts.get("stage2") is not None:
        ordered.append(scripts["stage2"])
    ordered.extend(scripts["stage3"])
    if scripts.get("stage4") is not None:
        ordered.append(scripts["stage4"])
    ordered.append(scripts["finalizer"])
    return ordered


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
        **finalizer_aggregation_options(config),
        JobMetadataKey.INPUT_PATH: source_image_stem(config.input_path),
        JobMetadataKey.ORCHESTRATION_EPOCH: epoch,
        JobMetadataKey.GUI_RECORD_GENERATION: os.environ.get(
            "PHENOTYPIC_GUI_RECORD_GENERATION"
        ),
        JobMetadataKey.PROCESSING_GENERATION: getattr(
            config,
            "processing_generation",
            None,
        ),
        "slurm_metadata_version": 2,
        "slurm_generation": epoch,
        JobMetadataKey.PIPELINE_PATH: str(
            Path(getattr(config, "pipeline_json", "pipeline.json")).absolute()
        ),
        JobMetadataKey.IMAGE_TYPE: getattr(config, "image_type", "Image"),
        JobMetadataKey.NROWS: getattr(config, "nrows", None),
        JobMetadataKey.NCOLS: getattr(config, "ncols", None),
        JobMetadataKey.BIT_DEPTH: getattr(config, "bit_depth", None),
        JobMetadataKey.DETECT_MODE: getattr(config, "detect_mode", "gray"),
        JobMetadataKey.EMBED_STAGE: getattr(config, "embed_stage", "inline"),
        # U10: the finalizer serves a measure resubmission differently (no
        # forward run proof), so it is told which run this is.
        JobMetadataKey.RUN_MODE: (
            "measure" if getattr(config, "measure_only", False) else "full"
        ),
        # U5: every shard and round of a Stage-4 run loads these commits.
        JobMetadataKey.EMBED_REVISIONS: getattr(config, "embed_revisions", None),
        JobMetadataKey.SLURM_JOB_IDS: {
            str(index): {
                "job_id": str(job_id),
                "role": "unknown",
                "generation": epoch,
            }
            for index, job_id in enumerate(job_ids)
        },
    }
    if getattr(config, "measure_only", False):
        # Measure mode keeps no processing state, so its workers read the
        # invocation's initial call from here, as the ordinary SLURM measure
        # worker does (figures spec §1a, `metadata_run_initiation`).
        from ._cli_state_management import run_initiation_config

        metadata |= run_initiation_config(config.run_initiation)
    metadata_path = job_metadata_path(output_dir)
    atomic_write_json(metadata_path, metadata)
    return metadata_path


def staged_manifest_entry(
    config: ExecutionConfig, dataset_name: str, image: Path
) -> StagedManifestEntry:
    """The image's one identity for the whole staged SLURM run.

    Computed once, at submission and before any stage applies anything. Every
    stage publishes and records failures under ``work_id`` and pins its
    reference context to ``reference_digest``, the digest that work-id was
    computed from, so a manifest re-planned by a later invocation is refused
    rather than applied under one plan and certified under another (review
    F1).

    Args:
        config: The run's execution configuration.
        dataset_name: The image's dataset.
        image: The input image.

    Returns:
        The manifest entry the stage workers read.
    """
    return manifest_entry_for_identity(
        dataset_name, image, work_identity_for_image(config, dataset_name, image)
    )


def manifest_entry_for_identity(
    dataset_name: str, image: Path, identity: WorkIdentity
) -> StagedManifestEntry:
    """The manifest entry of *image* under an identity already computed for it.

    Hashes nothing: a caller that holds the image's one identity for the run
    (a local strategy, Stage 1) builds its entry from it, rather than
    recomputing a work id after Stages 1-3 (#282 review I4; ``_cli/CLAUDE.md``
    "Reference metadata").
    """
    return StagedManifestEntry(
        dataset=dataset_name,
        image_name=image.name,
        stem=source_image_stem(image),
        input_path=str(Path(image).absolute()),
        work_id=identity.work_id,
        relative_image_path=identity.relative_path,
        attempt_id=uuid4().hex,
        reference_digest=identity.reference_digest,
    )


def _measure_pipeline_sha256(cfg: ExecutionConfig) -> str | None:
    """The measure pipeline's sha256, the run folder its measure core writes (P6 T2).

    The CLI's one digest of the file (``pipeline_identity``, review M6),
    else the file's own for a hand-built config. The controller config
    records it; the measure step and every Stage-4 shard read it from there
    (``recorded_measure_pipeline_sha256``).
    """
    from ._cli_execution_strategies import recorded_pipeline_sha256
    from ._cli_stage4_worker import _measure_pipeline_sha256 as sha256_of

    return recorded_pipeline_sha256(cfg) or sha256_of(cfg.pipeline_json)


def _staged_manifest_entries(
    cfg: ExecutionConfig, tasks: Sequence[tuple[Dataset, Path]]
) -> dict[tuple[str, Path], StagedManifestEntry]:
    """Each image's :func:`staged_manifest_entry`, hashed on threads (P6 review M8).

    The same function, called once per image as before, only on a thread
    pool sized to the allocation: ``hashlib`` releases the GIL. It is not
    ``stage4_entries``, whose identity function is the failure tracker's
    rather than this module's, so the submitter keeps its one identity path.
    """
    from joblib import Parallel, delayed

    from ._cli_utils import resolve_local_worker_count

    n_jobs = resolve_local_worker_count(getattr(cfg, "n_jobs", 1), len(tasks))
    entries = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(staged_manifest_entry)(cfg, dataset.name, image) for dataset, image in tasks
    )
    return {
        (dataset.name, image): entry for (dataset, image), entry in zip(tasks, entries)
    }


def _staged_plan(cfg: ExecutionConfig) -> tuple[list[str], Stage3Role, bool]:
    """``(phases, stage3 role, has a GpuDetector)`` of a staged SLURM run (#282 spec §4.1).

    A ``GpuDetector`` run has Stages 1-3, plus Stage 4 when it embeds on the
    GPU; a run without one reaches this strategy only to embed on the GPU
    (``uses_staged_controller``), so it has ``["stage3", "stage4"]`` and its
    Stage 3 is the single-pass core. A ``--mode measure`` run (U10) has
    ``["stage3", "stage4"]`` whatever its pipeline holds, since measure mode
    applies no op: its Stage 3 is the measure core, and its Stage 4 runs in
    republish mode.
    """
    from ._cli_validation import pipeline_requires_gpu

    gpu_embed = getattr(cfg, "embed_stage", "inline") == "gpu"
    # getattr, as for every other config field here: callers hand duck-typed configs.
    if getattr(cfg, "measure_only", False):
        if not gpu_embed:
            raise ValueError(
                "the staged SLURM strategy runs --mode measure only when it "
                "embeds in a GPU Stage 4"
            )
        return list(NO_DETECTOR_PHASES), "measure", False
    has_detector = pipeline_requires_gpu(cfg.pipeline_json)
    if has_detector:
        phases = STAGED_PHASES_WITH_STAGE4 if gpu_embed else STAGED_PHASES_INLINE
        return list(phases), "staged", True
    if not gpu_embed:
        raise ValueError(
            "the staged SLURM strategy runs a pipeline without a GpuDetector "
            "only when it embeds in a GPU Stage 4"
        )
    return list(NO_DETECTOR_PHASES), "single_pass", False


class StagedSlurmStrategy(ExecutionStrategy):
    """Submit the staged stages through the recoverable SLURM controller.

    A ``GpuDetector`` run: Stages 1-3, plus the GPU Stage 4 when it embeds on
    the GPU. A Stage-4 run without one: the single-pass "stage3" role, then
    Stage 4 (#282 P5 T2).
    """

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        start = datetime.now()
        cfg = self.config
        # A continuation's split already built each image's entry under its
        # one identity for the invocation (#282 review I4): reuse it.
        known = getattr(cfg, "staged_manifest_entries", None) or {}
        # The rest are hashed on threads (P6 review M8): a measure manifest
        # hashes every whole store tree, which serially on the submitting
        # node is the slow half of a submission. `hashlib` releases the GIL;
        # each identity is still `staged_manifest_entry`'s, computed once.
        missing = [
            (dataset, image)
            for dataset in datasets
            for image in dataset.images
            if (dataset.name, image) not in known
        ]
        hashed = (
            _staged_manifest_entries(cfg, missing) if missing else {}
        )
        manifest: list[StagedManifestEntry] = [
            known.get((dataset.name, image)) or hashed[(dataset.name, image)]
            for dataset in datasets
            for image in dataset.images
        ]

        # Chunk to the TIGHTER of MaxArraySize and the conservative
        # MaxSubmitJobs estimate. Reserve slots for the running controller and
        # its pre-armed recovery controller while an array is active.
        array_limit = get_slurm_array_limit()
        max_submit = get_slurm_max_submit_jobs()
        # Image stages chunk to fit the limit; the Stage-2 shard array cannot
        # (a shard worker streams its whole shard on one GPU), so guard it.
        # The run preflight asks the same function first (PF-SLURM-LIMIT).
        limit_errors = staged_slurm_limit_errors(
            max_submit, array_limit, cfg.gpu_shards
        )
        if limit_errors:
            raise ValueError(limit_errors[0])
        submit_capacity = max_submit - 2 if max_submit else array_limit
        chunk_limit = min(array_limit, submit_capacity)

        epoch = new_orchestration_epoch()
        # SUBMISSION PREFLIGHT, and the slot's one producer on this path.
        # Parsing and splitting the pipeline here carries every structural
        # refusal `split_pipeline_at_gpu` has -- two GpuDetectors, one in a
        # slot the staged engine cannot drive, none at all -- so an unstageable
        # pipeline is refused with nothing on disk to clean up
        # (reviewer finding F3). `new_orchestration_epoch` above only mints a
        # uuid; `initialize_orchestration` below is the first writer, so this
        # still precedes all scheduler state.
        #
        # Placed HERE, immediately before script generation, rather than at the
        # top of `execute`: this is the exact point the parse happened when it
        # lived inside `generate_staged_scripts`, so the refusal keeps its
        # ordering against the submit-capacity guard above it. Moving it
        # earlier would pre-empt that guard's error message.
        phases, stage3_role, has_detector = _staged_plan(cfg)
        staged_slot = (
            staged_detector_slot(cfg.pipeline_json) if has_detector else None
        )
        scripts = generate_staged_scripts(
            pipeline_path=cfg.pipeline_json,
            detector_slot=staged_slot,
            phases=phases,
            stage3_role=stage3_role,
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
            overlay_alpha=cfg.overlay_alpha,
            durable_writes=cfg.durable_writes,
            drop_originals=cfg.drop_originals,
            pipeline_identity=getattr(cfg, "pipeline_identity", None),
            processing_generation=getattr(
                cfg,
                "processing_generation",
                None,
            ),
            # The measure pipeline's sha256, as the measure core computes it,
            # so the controller and every Stage-4 shard name the run folder the
            # measure step wrote (P6 T2); a forward run records none.
            pipeline_sha256=(
                _measure_pipeline_sha256(cfg)
                if stage3_role == "measure"
                else None
            ),
        )
        if stage3_role == "measure":
            # U18: an earlier pass's measure-done markers are never read
            # again (each reader names its own epoch); clear them before
            # this epoch exists, so they do not accumulate pass after pass.
            from ._cli_staged_resume import clear_measure_done_markers

            clear_measure_done_markers(output_dir)
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
            # P6 review M11: the forward run's staged-completion marker is not
            # a measure pass's to delete.
            keep_completion_marker=stage3_role == "measure",
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
        if getattr(cfg, "restart", False):
            state["restart_parquet_fingerprints"] = snapshot_inventory_parquets(
                output_dir, cfg.full_dataset_inventory
            )
        initial_phase = getattr(cfg, "staged_resume_phase", None) or (
            "stage1" if has_detector else "stage3"
        )
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
                role="controller-initial",
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
            "Submitted staged controller lifecycle (phases %s; %d stage-1 "
            "chunk(s), %d stage-3 chunk(s), then the finalizer): initial "
            "controller=%s",
            ",".join(phases),
            len(scripts.get("stage1", [])),
            len(scripts["stage3"]),
            controller_id,
        )

        echo_slurm_monitor_commands(output_dir, dashboard=False)
        return self._submitted_results(datasets, start)

    def _submitted_results(
        self, datasets: List[Dataset], start: datetime
    ) -> ExecutionResults:
        """Return immediately after SLURM accepted the initial jobs.

        .. versionchanged:: 0.21.0
           Always: ``--wait`` was removed, and with it the loop that followed
           the epoch to its finalizer. The finalizer publishes the run's
           outputs and completion marker; the CLI prints a submission summary.
        """
        return ExecutionResults(
            datasets={
                dataset.name: DatasetResults(
                    dataset.name, len(dataset.images), 0, 0, []
                )
                for dataset in datasets
            },
            total_images=sum(len(dataset.images) for dataset in datasets),
            total_completed=0,
            total_failed=0,
            execution_mode="slurm",
            start_time=start,
            end_time=datetime.now(),
            submitted=True,
        )
