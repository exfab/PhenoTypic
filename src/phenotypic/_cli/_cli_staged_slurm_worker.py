"""Per-stage SLURM array-task entrypoints for the staged GPU engine.

One module invoked per SLURM array task as ``python -m
phenotypic._cli._cli_staged_slurm_worker --stage {1|2|3} ...``:

- Stage 1 / Stage 3 are arrays over **images**: ``$SLURM_ARRAY_TASK_ID`` indexes
  the manifest and runs the matching CPU stage core.
- Stage 2 is an array over **shards**: ``$SLURM_ARRAY_TASK_ID`` is the shard
  index; the resident model streams that shard of HDFs to objmap sidecars.

All stages are content-defined: a requeued/duplicate task is idempotent because
each stage skips work whose durable artifact already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import signal as _signal
from pathlib import Path
from typing import List, Sequence, Tuple

from phenotypic import ImagePipeline
from phenotypic.tools_ import slurm_scripts_dir
from phenotypic.tools_.slurm import _sbatch
from phenotypic.tools_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_sidecar import sidecar_exists
from ._cli_staged_slurm import partition_shards
from ._cli_staged_workers import (
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
)

# A manifest entry is [dataset, image_stem, image_path]; Stage 1 needs the path,
# Stages 2/3 only the (dataset, stem).
Manifest = Sequence[Sequence[str]]

# --- Stage-2 walltime survival ------------------------------------------------
# SLURM does NOT auto-requeue a TIMEOUT job. The durable half is already in
# place (each sidecar write is atomic and the worker skips done images), so no
# work is lost on a kill. This adds the TRIGGER: the Stage-2 script carries
# ``--signal=B:TERM@<grace>`` so SLURM sends SIGTERM shortly before walltime;
# the worker catches it and resubmits its shard (afterany on itself) to finish
# the remainder. Content-defined skip makes the continuation re-run only the
# sidecar-less images, so it converges.
_STOP = False


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):  # noqa: ANN001 - signal handler signature
        global _STOP
        _STOP = True

    _signal.signal(_signal.SIGTERM, _handler)


def _should_stop() -> bool:
    return _STOP


def resubmit_stage2_continuation(
    *,
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    shard_index: int,
    n_shards: int,
) -> None:
    """Resubmit THIS shard's Stage-2 array task (afterany on the current job).

    Re-runs the original ``stage2.sh`` restricted to this shard's array index;
    the resident model reloads and content-defined skip means only the
    remaining (sidecar-less) images are processed.
    """
    stage2_script = slurm_scripts_dir(output_dir) / "stage2.sh"
    self_job = os.environ.get("SLURM_JOB_ID", "")
    _sbatch.submit_script(
        stage2_script,
        dependency_job_id=self_job or None,
        array_index=shard_index,
    )


def run_stage1_step(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    index: int,
    ext: str = ".tiff",
) -> None:
    """Stage-1 array task: preprocess one manifest image -> staged HDF."""
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    om = OutputManager.from_config(output_dir, ext, save_overlays=False)
    dataset, stem, image_path = manifest[index][0], manifest[index][1], manifest[index][2]
    stage1_preprocess_core(
        plan, Path(image_path), dataset, stem, output_dir, om, image_type
    )


def run_stage2_shard(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    shard_index: int,
    n_shards: int,
) -> None:
    """Stage-2 array task: model loaded once, stream this shard of HDFs -> sidecars.

    Catches the pre-walltime SIGTERM and resubmits the shard for the remainder.
    """
    _install_sigterm_handler()
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    plan.gpu_detector._ensure_model_loaded()  # ONCE per worker
    my_shard = partition_shards(list(manifest), n_shards)[shard_index]
    for entry in my_shard:
        if _should_stop():
            break
        dataset, stem = entry[0], entry[1]
        if sidecar_exists(output_dir, dataset, stem):
            continue  # content-defined resume
        stage2_detect_core(plan.gpu_detector, output_dir, dataset, stem, image_type)

    if _should_stop():
        # Pre-walltime SIGTERM: resume the remainder. Guard on remaining work
        # (NOT merely "stopped") so a deterministically-failing image never
        # triggers an infinite resubmit loop.
        remaining = [
            e for e in my_shard if not sidecar_exists(output_dir, e[0], e[1])
        ]
        if remaining:
            resubmit_stage2_continuation(
                pipeline_path=pipeline_path, output_dir=output_dir,
                image_type=image_type, manifest=manifest,
                shard_index=shard_index, n_shards=n_shards,
            )


def run_stage3_step(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    index: int,
    ext: str = ".tiff",
) -> None:
    """Stage-3 array task: merge one manifest image's sidecar -> measure -> parquet."""
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    om = OutputManager.from_config(output_dir, ext, save_overlays=False)
    dataset, stem = manifest[index][0], manifest[index][1]
    stage3_merge_measure_core(plan, output_dir, dataset, stem, om, image_type)


def _load_manifest(manifest_path: Path) -> List[Tuple[str, ...]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [tuple(entry) for entry in data]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phenotypic-staged-slurm-worker")
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-type", default="Image")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)  # array task id
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--ext", default=".tiff")
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest)
    image_type: ImageTypeName = (
        "GridImage" if args.image_type == "GridImage" else "Image"
    )
    if args.stage == 1:
        run_stage1_step(
            args.pipeline, args.output_dir, image_type, manifest, args.index, args.ext
        )
    elif args.stage == 2:
        run_stage2_shard(
            args.pipeline, args.output_dir, image_type, manifest, args.index,
            args.n_shards,
        )
    else:
        run_stage3_step(
            args.pipeline, args.output_dir, image_type, manifest, args.index, args.ext
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
