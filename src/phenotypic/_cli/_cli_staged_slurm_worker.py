"""Per-stage SLURM array-task entrypoints for the staged GPU engine.

One module invoked per SLURM array task as ``python -m
phenotypic._cli._cli_staged_slurm_worker --stage {1|2|3} ...``:

- Stage 1 / Stage 3 are arrays over **images**: ``$SLURM_ARRAY_TASK_ID`` indexes
  the manifest and runs the matching CPU stage core (one image per task).
- Stage 2 is an array over **shards**: ``$SLURM_ARRAY_TASK_ID`` is the shard
  index; the resident model streams that shard of HDFs to objmap sidecars.

All stages are content-defined and emit stage-tagged events: a requeued/duplicate
task is idempotent, and the per-stage dashboard view is populated for SLURM runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal as _signal
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

from phenotypic import ImagePipeline
from phenotypic.sdk_ import dataset_hdf_dir, event_log_path
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_sidecar import sidecar_exists
from ._cli_staged_slurm import partition_shards
from ._cli_staged_workers import (
    emit_missing_prereq,
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
    stage_event,
)
from ._stages import STAGE_GPU_DETECT, STAGE_MEASURE, STAGE_PREPROCESS

# A manifest entry is [dataset, image_stem, image_path]; Stage 1 needs the path,
# Stages 2/3 only the (dataset, stem).
Manifest = Sequence[Sequence[str]]

# --- Stage-2 walltime survival ------------------------------------------------
# SLURM does NOT auto-requeue a TIMEOUT job. The durable half is already in place
# (each sidecar write is atomic and the worker skips done images), so no work is
# lost on a kill. This adds the TRIGGER: the Stage-2 script carries
# ``--signal=B:TERM@<grace>`` so SLURM sends SIGTERM shortly before walltime; the
# worker catches it and REQUEUES its own array task (not a new job), so Stage 3's
# ``afterany`` dependency on the Stage-2 array automatically waits for the
# requeued run — no race where Stage 3 merges before the shard is complete.
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
    """Requeue THIS Stage-2 array task to finish its shard after a SIGTERM.

    Requeuing (rather than submitting a NEW continuation job) keeps the Stage-2
    array job alive, so Stage 3's ``afterany`` dependency on that array job
    automatically waits for the requeued run to finish before merging — there is
    no race where Stage 3 starts on a shard whose continuation is still detecting.
    Content-defined skip means the requeued run reprocesses only the remaining
    sidecar-less images. (Args beyond the job id are kept for the call contract /
    test seam.)
    """
    job = os.environ.get("SLURM_JOB_ID", "")
    if job:
        subprocess.run(["scontrol", "requeue", job], check=False)


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
    dataset, stem, image_path = (
        manifest[index][0], manifest[index][1], manifest[index][2]
    )
    log = event_log_path(output_dir)
    # stage_event re-raises on failure: the task fails (visible in sacct) and
    # afterany lets the chain proceed.
    with stage_event(log, dataset, stem, STAGE_PREPROCESS):
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

    Per-image isolation (S6): a missing staged HDF or an inference error skips
    that image (recorded) instead of aborting the shard. Catches the pre-walltime
    SIGTERM and requeues the shard for the remainder.
    """
    _install_sigterm_handler()
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    plan.gpu_detector._ensure_model_loaded()  # ONCE per worker
    my_shard = partition_shards(list(manifest), n_shards)[shard_index]
    log = event_log_path(output_dir)
    attempted: set[tuple[str, str]] = set()

    for entry in my_shard:
        if _should_stop():
            break
        dataset, stem = entry[0], entry[1]
        if sidecar_exists(output_dir, dataset, stem):
            continue  # content-defined resume
        attempted.add((dataset, stem))
        hdf = dataset_hdf_dir(output_dir, dataset) / f"{stem}.h5"
        if not hdf.is_file():
            emit_missing_prereq(
                log, dataset, stem, STAGE_GPU_DETECT, "staged HDF"
            )
            continue
        try:
            with stage_event(log, dataset, stem, STAGE_GPU_DETECT):
                stage2_detect_core(
                    plan.gpu_detector, output_dir, dataset, stem, image_type
                )
        except Exception:  # isolate one bad image from the rest of the shard
            pass

    if _should_stop():
        # Walltime SIGTERM: requeue only if images were NOT yet attempted (we ran
        # out of time before reaching them). Images we attempted-and-failed are
        # excluded, so a deterministic failure never requeues forever (S10).
        remaining = [
            e for e in my_shard
            if (e[0], e[1]) not in attempted
            and not sidecar_exists(output_dir, e[0], e[1])
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
    log = event_log_path(output_dir)
    if not sidecar_exists(output_dir, dataset, stem):
        # S6: Stage 2 failed/absent for this image — record + fail the task
        # (rather than letting load_sidecar raise an opaque FileNotFoundError).
        emit_missing_prereq(
            log, dataset, stem, STAGE_MEASURE, "objmap sidecar"
        )
        raise SystemExit(1)
    with stage_event(log, dataset, stem, STAGE_MEASURE):
        stage3_merge_measure_core(
            plan, output_dir, dataset, stem, om, image_type
        )


def _load_manifest(manifest_path: Path) -> List[Tuple[str, ...]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [tuple(entry) for entry in data]


def _preload_custom_op_modules() -> None:
    """Import modules named in ``PHENOTYPIC_PRELOAD_MODULES`` (comma-separated).

    A SLURM worker is a fresh process; ``ImagePipeline.from_json`` resolves op
    classes from the ``phenotypic`` namespace, so operations defined OUTSIDE the
    package (a user's custom detector module — or, in tests, the fake detector)
    must register themselves before deserialization. Listing such a module here
    imports it (and runs its registration side effect) on worker startup.
    """
    import importlib

    for mod in os.environ.get("PHENOTYPIC_PRELOAD_MODULES", "").split(","):
        mod = mod.strip()
        if mod:
            importlib.import_module(mod)


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

    _preload_custom_op_modules()
    manifest = _load_manifest(args.manifest)
    image_type: ImageTypeName = (
        "GridImage" if args.image_type == "GridImage" else "Image"
    )
    if args.stage == 1:
        run_stage1_step(
            args.pipeline, args.output_dir, image_type, manifest, args.index,
            args.ext,
        )
    elif args.stage == 2:
        run_stage2_shard(
            args.pipeline, args.output_dir, image_type, manifest, args.index,
            args.n_shards,
        )
    else:
        run_stage3_step(
            args.pipeline, args.output_dir, image_type, manifest, args.index,
            args.ext,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
