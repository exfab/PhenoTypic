"""SLURM 3-stage chaining for the staged GPU engine (Spec 1 §7).

Stage 1 (CPU array over images) -> Stage 2 (GPU array over shards, resident
model) -> Stage 3 (CPU array over images), wired with ``afterany`` between
stages so a few per-image failures never block the next stage.
"""

from __future__ import annotations

import json
import logging
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypeVar

from phenotypic.sdk_ import (
    JOB_METADATA_JSON,
    JobMetadataKey,
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
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_execution_strategies import ExecutionStrategy
from ._cli_types import Dataset, DatasetResults, ExecutionConfig, ExecutionResults
from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Dependency type linking the three stages. ``afterany`` (not ``afterok``) so a
#: handful of per-image failures in one stage never block the next — the staged
#: workers are content-defined and skip already-done images.
STAGE_DEPENDENCY = "afterany"


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
    ]
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
    signal_grace: int | None = None,
    requeue: bool = False,
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
    # ``--requeue`` lets the Stage-2 worker requeue its own array task on the
    # pre-walltime SIGTERM (so Stage 3's afterany dependency waits for it).
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
            signal_grace=signal_grace,
            requeue=requeue,
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


def _finalizer_body(python_str: str, output_dir: Path) -> str:
    """Build the canonical aggregate/finalize command for staged runs."""
    return (
        f"{python_str} -m phenotypic._cli._cli_checkpoint_handler "
        f"--output-dir {shlex.quote(str(output_dir.absolute()))} "
        "--checkpoint-type finalize"
    )


def generate_staged_scripts(
    *,
    pipeline_path: Path,
    datasets_manifest: Sequence[Sequence[str]],
    output_dir: Path,
    image_type: ImageTypeName,
    cpu_slurm_args: Dict[str, Any],
    gpu_slurm_args: Dict[str, Any],
    n_shards: int,
    signal_grace: int = 120,
    ext: str = ".tiff",
    array_limit: int | None = None,
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
    manifest_path.write_text(
        json.dumps([list(e) for e in datasets_manifest]), encoding="utf-8"
    )

    python_cmd, _ = get_python_command(for_slurm=True)
    python_str = " ".join(python_cmd)
    n_images = len(datasets_manifest)
    if array_limit is None:
        array_limit = get_slurm_array_limit()
    gpu_args = resolve_stage_slurm_args(gpu_slurm_args, cpu_slurm_args)

    # Image chunks tile the manifest into <=array_limit-wide windows; each stage-1
    # and stage-3 chunk resolves its absolute index via $CURRENT_TASK_INDEX.
    chunks = calculate_optimal_array_chunks(n_images, array_limit) or [(0, 1)]

    def _image_stage_body(stage: int) -> str:
        return _stage_worker_body(
            python_str,
            stage,
            pipeline_path,
            output_dir,
            image_type,
            manifest_path,
            ext,
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
            n_shards=n_shards,
        ),
        signal_grace=signal_grace,
        requeue=True,
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
        _finalizer_body(python_str, output_dir),
    )
    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "finalizer": finalizer,
    }


def _ordered_staged_scripts(scripts: Dict[str, Any]) -> List[Path]:
    """Return staged scripts in their strict submission order."""
    return [
        *scripts["stage1"],
        scripts["stage2"],
        *scripts["stage3"],
        scripts["finalizer"],
    ]


def flatten_staged_scripts(scripts: Dict[str, Any]) -> List[Path]:
    """Ordered flat chunk list for the drip-feed dispatcher.

    Order: every Stage-1 chunk, Stage 2, every Stage-3 chunk, then the aggregate
    finalizer. The dispatcher submits them one at a time, so the linear order
    encodes both stage dependencies and final publication.
    """
    return _ordered_staged_scripts(scripts)


def submit_staged_chain(
    scripts: Dict[str, Any],
    *,
    output_dir: Path,
    slurm_args: Dict[str, Any],
    console: Any = None,
) -> List[str]:
    """Submit the staged scripts via the shared drip-feed dispatcher.

    Reuses :func:`submit_slurm_script_chain` (the same funnel the CPU
    autonomous path uses) so **only the first chunk + a dispatcher job are
    queued up front**; when chunk N finishes, its dispatcher submits chunk N+1.
    Peak queue occupancy stays at ~1 chunk + 1 dispatcher instead of every
    chunk at once — critical because a run's total array tasks
    (~2 x n_images) otherwise blows ``MaxSubmitJobs`` (Spec 1 §7/§9). The tiny
    dispatcher runs on the CPU ``slurm_args`` profile. Returns the initially
    submitted job ids (chunk 0 and, if multiple chunks, dispatcher 1).
    """
    # Local import avoids a circular import at module load
    # (_cli_slurm_submission -> sdk_ -> ... ).
    from rich.console import Console

    from ._cli_slurm_submission import submit_slurm_script_chain

    submission = submit_slurm_script_chain(
        flat_chunk_scripts=flatten_staged_scripts(scripts),
        output_dir=output_dir,
        slurm_args=slurm_args,
        console=console or Console(),
    )
    return submission.job_ids


def _write_staged_job_metadata(
    *,
    datasets: Sequence[Dataset],
    output_dir: Path,
    config: ExecutionConfig,
    scripts: Dict[str, Any],
    job_ids: Sequence[str],
    start_time: datetime,
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
        JobMetadataKey.START_TIME: start_time.isoformat(timespec="milliseconds"),
        JobMetadataKey.EXECUTION_MODE: "slurm",
        JobMetadataKey.DATASETS: {
            dataset.name: {
                "total": len(dataset.images),
                "images": [image.name for image in dataset.images],
            }
            for dataset in datasets
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
    }
    metadata_path = prog_dir / JOB_METADATA_JSON
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata_path


class StagedSlurmStrategy(ExecutionStrategy):
    """Submit the 3 staged stages as a SLURM ``afterany`` dependency chain."""

    def execute(
        self, datasets: List[Dataset], output_dir: Path
    ) -> ExecutionResults:
        start = datetime.now()
        cfg = self.config
        manifest = [
            (ds.name, img.stem, str(Path(img).absolute()))
            for ds in datasets
            for img in ds.images
        ]

        # Chunk to the TIGHTER of MaxArraySize and the conservative
        # MaxSubmitJobs estimate. Reserve one submission slot for the dependent
        # dispatcher that is queued alongside the active array chunk.
        array_limit = get_slurm_array_limit()
        max_submit = get_slurm_max_submit_jobs()
        submit_capacity = max(1, max_submit - 1) if max_submit else array_limit
        chunk_limit = min(array_limit, submit_capacity)
        # Image stages chunk to fit the limit; the Stage-2 shard array cannot
        # (a shard worker streams its whole shard on one GPU), so guard it.
        if max(1, cfg.gpu_shards) > chunk_limit:
            raise ValueError(
                f"--gpu-shards ({cfg.gpu_shards}) exceeds the SLURM chunk limit "
                f"({chunk_limit}); reduce the shard count."
            )

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
        )

        # Drip-feed dispatcher: only chunk 0 + a dispatcher are queued now; each
        # chunk's dispatcher submits the next after it ends. Keeps peak queue at
        # ~1 chunk (never the ~2 x n_images tasks that blow MaxSubmitJobs) while
        # afterany between chunks lets per-image failures pass (Spec 1 §7/§9).
        self.submitted_job_ids = submit_staged_chain(
            scripts, output_dir=output_dir, slurm_args=cfg.slurm_args
        )
        _write_staged_job_metadata(
            datasets=datasets,
            output_dir=output_dir,
            config=cfg,
            scripts=scripts,
            job_ids=self.submitted_job_ids[:1],
            start_time=start,
        )
        logger.info(
            "Submitted staged GPU chain via drip-feed dispatcher "
            "(%d stage-1 chunks -> stage2 -> %d stage-3 chunks -> finalizer): "
            "initial jobs=%s",
            len(scripts["stage1"]),
            len(scripts["stage3"]),
            self.submitted_job_ids,
        )

        return ExecutionResults(
            datasets={
                ds.name: DatasetResults(
                    name=ds.name,
                    total=len(ds.images),
                    completed=0,
                    failed=0,
                    failures=[],
                )
                for ds in datasets
            },
            total_images=len(manifest),
            total_completed=0,
            total_failed=0,
            execution_mode="slurm",
            start_time=start,
            end_time=datetime.now(),
        )
