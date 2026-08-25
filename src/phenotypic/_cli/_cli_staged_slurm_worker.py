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
from typing import Mapping, Sequence
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
    valid_stage1_store,
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
    durable_writes: bool | None = None,
    drop_originals: bool = False,
    pipeline_identity: Mapping[str, str] | None = None,
) -> None:
    """Preprocess one manifest image into its staged OME-Zarr store.

    ``durable_writes`` is the transported ``--durable-writes`` /
    ``--no-durable-writes`` tri-state. ``None`` lets this process auto-detect
    SLURM; an explicit value only exists here because the submitter emitted it
    (spec §3.7).
    """
    item = _entry(manifest[index])
    store = zarr_store_path(output_dir, item.dataset, item.stem)
    if resume and (
        (
            item.work_id
            and staged_store_matches_work_id(store, item.work_id)
        )
        or (not item.work_id and valid_stage1_store(store))
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
        output_dir, ext, save_overlays=False, durable_writes=durable_writes
    )
    log = event_log_path(output_dir)
    try:
        with stage_event(
            log,
            item.dataset,
            item.image_name,
            STAGE_PREPROCESS,
            active_check=check,
        ):
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
                pipeline_path=pipeline_path,
                pipeline_identity=pipeline_identity,
                drop_originals=drop_originals,
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
        ) or (not item.work_id and valid_stage1_store(store)):
            pending.append(item)
            continue
        if check is not None:
            check()
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
                log,
                item.dataset,
                item.image_name,
                STAGE_GPU_DETECT,
                active_check=check,
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
    durable_writes: bool | None = None,
) -> None:
    """Replay one Stage-2 result, measure it, and publish per-image outputs.

    ``durable_writes`` is the transported tri-state; see
    :func:`run_stage1_step`. Stage 3 re-promotes the store, so a lost value
    here costs exactly what a lost value in Stage 1 costs.
    """
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
        durable_writes=durable_writes,
    )
    check = _active_check(output_dir, epoch)
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
            active_check=check,
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
            active_check=check,
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
        if check is not None:
            check()
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
        with stage_event(
            log,
            item.dataset,
            item.image_name,
            STAGE_MEASURE,
            active_check=check,
        ):
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
                if check is not None:
                    check()
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
            if check is not None:
                check()
            write_stage3_completion_marker(
                output_dir,
                item.dataset,
                item.image_name,
                item.stem,
            )
            # Token FIRST at every consumption site: the only reachable
            # intermediate state must be "no token, orphan raw".
            if check is not None:
                check()
            delete_stage2_token(output_dir, item.dataset, item.stem)
            if check is not None:
                check()
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
    parser.add_argument("--drop-originals", action="store_true")
    parser.add_argument(
        "--provenance-pipeline-source-path",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--provenance-pipeline-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.3)
    # BooleanOptionalAction, not store_true: the flag is TRI-state and its
    # default must stay ``None``. ``store_true`` would make an unset flag
    # ``False`` and permanently disable fsync in every staged SLURM worker --
    # the opposite of the spec's default on a cluster (spec §3.7).
    parser.add_argument(
        "--durable-writes",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args(argv)
    provenance_identity_values = (
        args.provenance_pipeline_source_path,
        args.provenance_pipeline_sha256,
    )
    if any(value is not None for value in provenance_identity_values):
        if not all(value is not None for value in provenance_identity_values):
            parser.error("incomplete provenance pipeline identity")
        pipeline_identity = {
            "source_path": args.provenance_pipeline_source_path,
            "sha256": args.provenance_pipeline_sha256,
        }
    else:
        pipeline_identity = None

    preload_custom_operation_modules()
    manifest = load_staged_manifest(args.manifest)
    image_type: ImageTypeName = (
        "GridImage" if args.image_type == "GridImage" else "Image"
    )
    common = {
        "epoch": args.epoch,
        "resume": args.reuse_existing,
        "durable_writes": args.durable_writes,
    }
    if args.stage == 1:
        run_stage1_step(
            args.pipeline,
            args.output_dir,
            image_type,
            manifest,
            args.index,
            args.ext,
            drop_originals=args.drop_originals,
            pipeline_identity=pipeline_identity,
            **common,
        )
    elif args.stage == 2:
        # Stage 2 writes no store (it drops a token plus a retained raw array
        # under .phenotypic/progress/), so durability does not apply to it.
        stage2_common = {
            k: v for k, v in common.items() if k != "durable_writes"
        }
        run_stage2_shard(
            args.pipeline,
            args.output_dir,
            image_type,
            manifest,
            args.index,
            args.n_shards,
            markers_required=args.stage3_markers_required,
            **stage2_common,
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
