"""Per-stage SLURM workers for the staged GPU engine.

Workers are content-defined and epoch fenced. Stage 2 builds its pending shard
before loading the GPU model; walltime continuation is owned by the dependent
controller rather than by signal handling inside this process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from phenotypic import ImagePipeline
from phenotypic.sdk_ import (
    dataset_hdf_dir,
    dataset_measurements_dir,
    event_log_path,
)
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_preload import preload_custom_operation_modules
from ._cli_sidecar import sidecar_exists
from ._cli_staged_orchestration import (
    StagedManifestEntry,
    append_stage2_terminal_failure,
    assert_active_epoch,
    epoch_is_active,
    load_orchestration_state,
    load_staged_manifest,
    terminal_stage2_identities,
)
from ._cli_staged_slurm import partition_shards
from ._cli_staged_resume import (
    clear_downstream_artifacts_for_stage1,
    stage3_completion_exists,
    valid_staged_hdf,
)
from ._cli_staged_workers import (
    emit_missing_prereq,
    ensure_staged_overlay,
    stage1_preprocess_core,
    stage2_detect_core,
    stage3_merge_measure_core,
    stage_event,
)
from ._stages import STAGE_GPU_DETECT, STAGE_MEASURE, STAGE_PREPROCESS

Manifest = Sequence[StagedManifestEntry]


def _active_check(output_dir: Path, epoch: str | None):
    """Return a publication fence callback for a staged worker."""
    if epoch is None:
        return None

    def _check() -> None:
        assert_active_epoch(output_dir, epoch)

    return _check


def _entry(entry: StagedManifestEntry | Sequence[str]) -> StagedManifestEntry:
    """Coerce legacy direct-test tuples while disk manifests stay versioned."""
    if isinstance(entry, StagedManifestEntry):
        return entry
    dataset, stem = entry[:2]
    input_path = entry[2] if len(entry) > 2 else str(stem)
    return StagedManifestEntry(
        dataset=str(dataset),
        image_name=Path(str(input_path)).name or str(stem),
        stem=str(stem),
        input_path=str(input_path),
    )


def run_stage1_step(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    index: int,
    ext: str = ".tiff",
    *,
    epoch: str | None = None,
    resume: bool = False,
) -> None:
    """Preprocess one manifest image into its staged HDF."""
    item = _entry(manifest[index])
    hdf = dataset_hdf_dir(output_dir, item.dataset) / f"{item.stem}.h5"
    if resume and valid_staged_hdf(hdf):
        return
    check = _active_check(output_dir, epoch)
    if check is not None:
        check()
    if resume:
        clear_downstream_artifacts_for_stage1(
            output_dir, item.dataset, item.stem
        )
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    output_manager = OutputManager.from_config(
        output_dir, ext, save_overlays=False
    )
    log = event_log_path(output_dir)
    with stage_event(log, item.dataset, item.image_name, STAGE_PREPROCESS):
        stage1_preprocess_core(
            plan,
            Path(item.input_path),
            item.dataset,
            item.stem,
            output_dir,
            output_manager,
            image_type,
            active_check=check,
        )


def run_stage2_shard(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    shard_index: int,
    n_shards: int,
    *,
    epoch: str | None = None,
    resume: bool = False,
    markers_required: bool = True,
) -> None:
    """Stream one pending shard through one resident GPU model."""
    check = _active_check(output_dir, epoch)
    if check is not None:
        check()
    shard = [
        _entry(item)
        for item in partition_shards(list(manifest), n_shards)[shard_index]
    ]
    terminal = (
        terminal_stage2_identities(output_dir, epoch) if epoch is not None else set()
    )
    candidates = [
        item
        for item in shard
        if item.identity not in terminal
        if not sidecar_exists(output_dir, item.dataset, item.stem)
        and not (
            stage3_completion_exists(output_dir, item.dataset, item.stem)
            or (
                resume
                and not markers_required
                and (
                dataset_measurements_dir(output_dir, item.dataset)
                / f"{item.stem}.parquet"
                ).is_file()
            )
        )
    ]
    if not candidates:
        return

    log = event_log_path(output_dir)
    state = load_orchestration_state(output_dir)
    round_index = 0 if state is None else int(state.get("round", 0))
    pending: list[StagedManifestEntry] = []
    for item in candidates:
        hdf = dataset_hdf_dir(output_dir, item.dataset) / f"{item.stem}.h5"
        if valid_staged_hdf(hdf):
            pending.append(item)
            continue
        emit_missing_prereq(
            log,
            item.dataset,
            item.image_name,
            STAGE_GPU_DETECT,
            "staged HDF",
        )
        if epoch is not None and epoch_is_active(output_dir, epoch):
            append_stage2_terminal_failure(
                output_dir,
                epoch=epoch,
                round_index=round_index,
                entry=item,
                error_type="MissingPrerequisite",
                error_message="Stage 1 HDF is absent",
            )
    if not pending:
        return

    if check is not None:
        check()
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    plan.gpu_detector._ensure_model_loaded()
    for item in pending:
        if check is not None:
            check()
        try:
            with stage_event(
                log, item.dataset, item.image_name, STAGE_GPU_DETECT
            ):
                stage2_detect_core(
                    plan.gpu_detector,
                    output_dir,
                    item.dataset,
                    item.stem,
                    image_type,
                    active_check=check,
                )
        except Exception as exc:
            if epoch is not None and epoch_is_active(output_dir, epoch):
                append_stage2_terminal_failure(
                    output_dir,
                    epoch=epoch,
                    round_index=round_index,
                    entry=item,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )


def run_stage3_step(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    index: int,
    ext: str = ".tiff",
    *,
    epoch: str | None = None,
    resume: bool = False,
    markers_required: bool = True,
    overlay_alpha: float = 0.3,
) -> None:
    """Merge one sidecar, measure it, and publish final per-image outputs."""
    item = _entry(manifest[index])
    parquet = (
        dataset_measurements_dir(output_dir, item.dataset)
        / f"{item.stem}.parquet"
    )
    output_manager = OutputManager.from_config(
        output_dir,
        ext,
        overlay_alpha=overlay_alpha,
        save_overlays=True,
    )
    terminal = stage3_completion_exists(output_dir, item.dataset, item.stem) or (
        resume and not markers_required and parquet.is_file()
    )
    if terminal:
        ensure_staged_overlay(
            output_dir,
            item.dataset,
            item.stem,
            output_manager,
            image_type,
            active_check=_active_check(output_dir, epoch),
        )
        return
    check = _active_check(output_dir, epoch)
    if check is not None:
        check()
    log = event_log_path(output_dir)
    if not sidecar_exists(output_dir, item.dataset, item.stem):
        emit_missing_prereq(
            log,
            item.dataset,
            item.image_name,
            STAGE_MEASURE,
            "objmap sidecar",
        )
        raise SystemExit(1)
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    with stage_event(log, item.dataset, item.image_name, STAGE_MEASURE):
        stage3_merge_measure_core(
            plan,
            output_dir,
            item.dataset,
            item.stem,
            output_manager,
            image_type,
            active_check=check,
            image_name=item.image_name,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one SLURM array task to its staged worker."""
    parser = argparse.ArgumentParser(prog="phenotypic-staged-slurm-worker")
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-type", default="Image")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--ext", default=".tiff")
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage3-markers-required", action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=0.3)
    args = parser.parse_args(argv)

    preload_custom_operation_modules()
    manifest = load_staged_manifest(args.manifest)
    image_type: ImageTypeName = (
        "GridImage" if args.image_type == "GridImage" else "Image"
    )
    common = {"epoch": args.epoch, "resume": args.resume}
    if args.stage == 1:
        run_stage1_step(
            args.pipeline,
            args.output_dir,
            image_type,
            manifest,
            args.index,
            args.ext,
            **common,
        )
    elif args.stage == 2:
        run_stage2_shard(
            args.pipeline,
            args.output_dir,
            image_type,
            manifest,
            args.index,
            args.n_shards,
            markers_required=args.stage3_markers_required,
            **common,
        )
    else:
        run_stage3_step(
            args.pipeline,
            args.output_dir,
            image_type,
            manifest,
            args.index,
            args.ext,
            markers_required=args.stage3_markers_required,
            overlay_alpha=args.overlay_alpha,
            **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
