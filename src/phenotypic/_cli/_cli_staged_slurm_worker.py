"""Per-stage SLURM workers for the staged GPU engine.

Workers are content-defined and epoch fenced. Stage 2 builds its pending shard
before loading the GPU model; walltime continuation is owned by the dependent
controller rather than by signal handling inside this process.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from phenotypic import ImagePipeline
from phenotypic.sdk_ import (
    dataset_measurements_dir,
    event_log_path,
    zarr_store_path,
)
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_preload import preload_custom_operation_modules
from ._cli_stage2_token import (
    delete_stage2_raw,
    delete_stage2_token,
    stage2_result_replayable,
)
from ._cli_staged_orchestration import (
    StagedManifestEntry,
    assert_active_epoch,
    epoch_is_active,
    load_staged_manifest,
)
from ._cli_failure_tracker import (
    PerImageScientificError,
    append_terminal_failure,
    read_terminal_failures,
)
from ._cli_completion import (
    image_data_artifact,
    publish_image_success,
    valid_image_success,
)
from ._cli_staged_slurm import partition_shards
from ._cli_staged_resume import (
    clear_downstream_artifacts_for_stage1,
    stage3_completion_exists,
    staged_store_matches_work_id,
    valid_staged_store,
    write_stage3_completion_marker,
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


def _record_terminal_scientific_failure(
    output_dir: Path,
    item: StagedManifestEntry,
    exception: Exception,
    epoch: str | None,
) -> bool:
    """Commit one current-epoch staged scientific failure when identifiable."""
    if (
        not isinstance(exception, PerImageScientificError)
        or not item.work_id
        or epoch is None
        or not epoch_is_active(output_dir, epoch)
    ):
        return False
    return append_terminal_failure(
        output_dir,
        work_id=item.work_id,
        dataset=item.dataset,
        relative_image_path=item.relative_image_path or item.image_name,
        failed_stage=exception.stage,
        exception=exception.cause,
        attempt_id=item.attempt_id or uuid4().hex,
        lifecycle_epoch=epoch,
        traceback=traceback.format_exc(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )


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
    """Preprocess one manifest image into its staged OME-Zarr store."""
    item = _entry(manifest[index])
    store = zarr_store_path(output_dir, item.dataset, item.stem)
    if resume and (
        (
            item.work_id
            and staged_store_matches_work_id(store, item.work_id)
        )
        or (not item.work_id and valid_staged_store(store))
    ):
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
    try:
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
                work_id=item.work_id,
            )
    except Exception as exc:
        _record_terminal_scientific_failure(output_dir, item, exc, epoch)
        raise


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
    terminal = {
        record.work_id
        for record in read_terminal_failures(output_dir)
        if epoch is not None and record.lifecycle_epoch == epoch
    }
    candidates = [
        item
        for item in shard
        if item.work_id not in terminal
        if not stage2_result_replayable(output_dir, item.dataset, item.stem)
        and not (
            bool(
                item.work_id
                and valid_image_success(
                    output_dir,
                    dataset=item.dataset,
                    image_stem=item.stem,
                    work_id=item.work_id,
                )
            )
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
    pending: list[StagedManifestEntry] = []
    for item in candidates:
        store = zarr_store_path(output_dir, item.dataset, item.stem)
        if (
            item.work_id
            and staged_store_matches_work_id(store, item.work_id)
        ) or (not item.work_id and valid_staged_store(store)):
            pending.append(item)
            continue
        emit_missing_prereq(
            log,
            item.dataset,
            item.image_name,
            STAGE_GPU_DETECT,
            "staged store",
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
            _record_terminal_scientific_failure(output_dir, item, exc, epoch)


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
    """Replay one Stage-2 result, measure it, and publish per-image outputs."""
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
    terminal = bool(
        item.work_id
        and valid_image_success(
            output_dir,
            dataset=item.dataset,
            image_stem=item.stem,
            work_id=item.work_id,
        )
    ) or (
        resume
        and (not markers_required or not item.work_id)
        and stage3_completion_exists(output_dir, item.dataset, item.stem)
        and parquet.is_file()
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
    if (
        resume
        and item.work_id
        and stage3_completion_exists(output_dir, item.dataset, item.stem)
        and staged_store_matches_work_id(
            zarr_store_path(output_dir, item.dataset, item.stem),
            item.work_id,
        )
        and parquet.is_file()
    ):
        ensure_staged_overlay(
            output_dir,
            item.dataset,
            item.stem,
            output_manager,
            image_type,
            active_check=_active_check(output_dir, epoch),
        )
        data_key, data_path = image_data_artifact(
            output_dir, output_manager, item.dataset, item.stem
        )
        artifacts = {
            "measurements": parquet,
            data_key: data_path,
            "overlay": output_manager.get_output_path(
                item.dataset, "overlays", item.stem
            ),
        }
        publish_image_success(
            output_dir,
            work_id=item.work_id,
            dataset=item.dataset,
            relative_image_path=item.relative_image_path or item.image_name,
            image_stem=item.stem,
            mode="full",
            attempt_id=item.attempt_id or uuid4().hex,
            lifecycle_epoch=epoch or "slurm-unfenced",
            artifacts=artifacts,
        )
        return
    check = _active_check(output_dir, epoch)
    if check is not None:
        check()
    log = event_log_path(output_dir)
    # BOTH halves. The token is only a flag; Stage 3's input is the raw .npy,
    # and a token-present/raw-missing image would otherwise raise
    # FileNotFoundError inside stage_event and be recorded as a terminal
    # SCIENTIFIC failure rather than a missing prereq (ledger FLOW-17/M7).
    if not stage2_result_replayable(output_dir, item.dataset, item.stem):
        emit_missing_prereq(
            log,
            item.dataset,
            item.image_name,
            STAGE_MEASURE,
            "Stage 2 result",
        )
        raise SystemExit(1)
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    try:
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
                work_id=item.work_id,
            )
            if item.work_id:
                data_key, data_path = image_data_artifact(
                    output_dir, output_manager, item.dataset, item.stem
                )
                artifacts = {
                    "measurements": output_manager.get_output_path(
                        item.dataset, "measurements", item.stem
                    ),
                    data_key: data_path,
                }
                if output_manager.save_overlays:
                    artifacts["overlay"] = output_manager.get_output_path(
                        item.dataset, "overlays", item.stem
                    )
                publish_image_success(
                    output_dir,
                    work_id=item.work_id,
                    dataset=item.dataset,
                    relative_image_path=(
                        item.relative_image_path or item.image_name
                    ),
                    image_stem=item.stem,
                    mode="full",
                    attempt_id=item.attempt_id or uuid4().hex,
                    lifecycle_epoch=epoch or "slurm-unfenced",
                    artifacts=artifacts,
                )
            write_stage3_completion_marker(
                output_dir,
                item.dataset,
                item.image_name,
                item.stem,
            )
            # Token FIRST at every consumption site: the only reachable
            # intermediate state must be "no token, orphan raw".
            delete_stage2_token(output_dir, item.dataset, item.stem)
            delete_stage2_raw(output_dir, item.dataset, item.stem)
    except Exception as exc:
        _record_terminal_scientific_failure(output_dir, item, exc, epoch)
        raise


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
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--stage3-markers-required", action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=0.3)
    args = parser.parse_args(argv)

    preload_custom_operation_modules()
    manifest = load_staged_manifest(args.manifest)
    image_type: ImageTypeName = (
        "GridImage" if args.image_type == "GridImage" else "Image"
    )
    common = {"epoch": args.epoch, "resume": args.reuse_existing}
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
