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

from phenotypic.tools_ import logs_dir, slurm_scripts_dir
from phenotypic.tools_.slurm import _sbatch, get_slurm_array_limit
from phenotypic.tools_.slurm._sbatch import format_sbatch_directives
from phenotypic.tools_.typing_ import ImageTypeName

from ._cli_execution_strategies import ExecutionStrategy
from ._cli_types import Dataset, DatasetResults, ExecutionResults
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
        shards.append(items[start:start + size])
        start += size
    return shards


def resolve_stage_slurm_args(
    gpu_slurm_args: Dict[str, Any], cpu_slurm_args: Dict[str, Any] | None = None
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
) -> str:
    """The per-array-task command line that invokes the staged SLURM worker."""
    q = shlex.quote
    parts = [
        f"{python_str} -m phenotypic._cli._cli_staged_slurm_worker",
        f"--stage {stage}",
        f"--pipeline {q(str(Path(pipeline_path).absolute()))}",
        f"--output-dir {q(str(output_dir.absolute()))}",
        f"--image-type {image_type}",
        f"--manifest {q(str(manifest_path.absolute()))}",
        "--index $SLURM_ARRAY_TASK_ID",
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
    array_size: int,
    body: str,
    signal_grace: int | None = None,
) -> Path:
    """Render + write one stage SBATCH array script (returns its path)."""
    out_log = log_dir / f"{stage_name}_%A_%a.log"
    err_log = log_dir / f"{stage_name}_%A_%a.err"
    directives = format_sbatch_directives(job_name, slurm_args, out_log, err_log)
    array_directive = f"#SBATCH --array=0-{max(0, array_size - 1)}"
    signal_directive = (
        f"\n#SBATCH --signal=B:TERM@{signal_grace}" if signal_grace else ""
    )
    script_content = f"""#!/bin/bash
{directives}
{array_directive}{signal_directive}

set -e
set -u

{SLURM_THREAD_PIN_BASH}

echo "===== {stage_name} task ${{SLURM_ARRAY_TASK_ID:-?}} (job ${{SLURM_JOB_ID:-?}}) on ${{SLURMD_NODENAME:-$(hostname)}} ====="
echo "Start: $(date)"

set +e
{body}
EXIT_CODE=$?
set -e

echo "===== {stage_name} exit $EXIT_CODE at $(date) ====="
exit $EXIT_CODE
"""
    path = script_dir / f"{stage_name}.sh"
    path.write_text(script_content, encoding="utf-8")
    path.chmod(0o755)
    return path


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
) -> Dict[str, Path]:
    """Write the three per-stage SBATCH array scripts (no submission).

    Stage 1 & 3 are arrays over images (CPU ``cpu_slurm_args``); Stage 2 is an
    array over shards (GPU args resolved from ``gpu_slurm_args`` over the CPU
    profile, auto-1-GPU). The manifest is written to disk for the workers to
    index. Returns ``{"stage1": path, "stage2": path, "stage3": path}``.
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
    gpu_args = resolve_stage_slurm_args(gpu_slurm_args, cpu_slurm_args)

    return {
        "stage1": _write_stage_script(
            script_dir, log_dir, "stage1", "phenotypic-stage1", cpu_slurm_args,
            n_images,
            _stage_worker_body(
                python_str, 1, pipeline_path, output_dir, image_type,
                manifest_path, ext,
            ),
        ),
        "stage2": _write_stage_script(
            script_dir, log_dir, "stage2", "phenotypic-stage2", gpu_args,
            n_shards,
            _stage_worker_body(
                python_str, 2, pipeline_path, output_dir, image_type,
                manifest_path, ext, n_shards=n_shards,
            ),
            signal_grace=signal_grace,
        ),
        "stage3": _write_stage_script(
            script_dir, log_dir, "stage3", "phenotypic-stage3", cpu_slurm_args,
            n_images,
            _stage_worker_body(
                python_str, 3, pipeline_path, output_dir, image_type,
                manifest_path, ext,
            ),
        ),
    }


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

        array_limit = get_slurm_array_limit()
        if len(manifest) > array_limit or cfg.gpu_shards > array_limit:
            logger.warning(
                "staged SLURM work-list (%d images / %d shards) exceeds the "
                "array limit (%d); per-stage chunking is not implemented — "
                "submit fewer items or raise MaxArraySize.",
                len(manifest), cfg.gpu_shards, array_limit,
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
        )

        # 3-link afterany chain: stage1 -> stage2 -> stage3. submit_script uses
        # --dependency afterany:<prev>, so a few per-image failures in one stage
        # never block the next (Spec 1 §7/§9).
        job1 = _sbatch.submit_script(scripts["stage1"])
        job2 = _sbatch.submit_script(scripts["stage2"], dependency_job_id=job1)
        job3 = _sbatch.submit_script(scripts["stage3"], dependency_job_id=job2)
        self.submitted_job_ids = [job1, job2, job3]
        logger.info(
            "Submitted staged GPU chain: stage1=%s -> stage2=%s -> stage3=%s",
            job1, job2, job3,
        )

        return ExecutionResults(
            datasets={
                ds.name: DatasetResults(
                    name=ds.name, total=len(ds.images), completed=0, failed=0,
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
