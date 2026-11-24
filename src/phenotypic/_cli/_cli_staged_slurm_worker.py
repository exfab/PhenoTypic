"""Per-stage SLURM workers for the staged GPU engine.

Workers are content-defined and epoch fenced. Stage 2 builds its pending shard
before loading the GPU model; walltime continuation is owned by the dependent
controller rather than by signal handling inside this process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from phenotypic import ImagePipeline
from phenotypic._startup_perf import load_runtime_dependencies
from phenotypic.sdk_ import (
    CommitGuard,
    JobMetadataKey,
    event_log_path,
    journal_measurement_table,
    job_metadata_path,
    progress_dir,
    zarr_store_path,
)
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_process_single import process_single_image_core
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_reference import ReferencePin
from ._cli_preload import preload_custom_operation_modules
from ._cli_stage2_token import (
    delete_stage2_raw,
    delete_stage2_token,
    detector_slot,
    stage2_result_replayable,
)
from ._cli_staged_orchestration import (
    StagedManifestEntry,
    assert_active_epoch,
    load_staged_manifest,
)
from ._cli_failure_tracker import (
    PerImageScientificError,
    append_terminal_failure,
    read_terminal_failures,
)
from ._cli_completion import (
    image_data_artifact,
    image_measurement_artifact,
    publish_image_success,
    valid_image_success,
)
from ._cli_staged_slurm import partition_shards
from ._cli_utils import input_read_settings, load_job_metadata
from ._cli_state_management import recorded_run_initiation
from ._cli_staged_resume import (
    clear_downstream_artifacts_for_stage1,
    stage3_completion_exists,
    staged_store_matches_work_id,
    stages_1_to_3_done,
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
from ._cli_slurm_lifecycle import (
    generation_publication_guard,
    slurm_generation_inactive_cause,
)
from ._stages import STAGE_GPU_DETECT, STAGE_MEASURE, STAGE_PREPROCESS

Manifest = Sequence[StagedManifestEntry]


def _record_terminal_scientific_failure(
    output_dir: Path,
    item: StagedManifestEntry,
    exception: Exception,
    epoch: str | None,
    commit_guard: CommitGuard | None,
) -> bool:
    """Commit one current-epoch staged scientific failure when identifiable."""
    if (
        not isinstance(exception, PerImageScientificError)
        or not item.work_id
        or epoch is None
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
        commit_guard=commit_guard,
    )


def _reference_pin(item: StagedManifestEntry) -> ReferencePin:
    """Pin a stage to the digest the entry's work-id was computed from.

    The submitter computed both once, before any stage ran
    (``staged_manifest_entry``), so all three stages apply the plan the
    image's records are published under.
    """
    return ReferencePin(item.stem, item.reference_digest)


def _input_root(item: StagedManifestEntry) -> Path | None:
    """The ``--input`` root the submitter resolved *item*'s relative path against.

    The manifest records the input's absolute path and its input-relative
    path (``staged_manifest_entry``), not the root, so the root is the input
    path with the relative path's parts removed. Stage 1 hands it to
    ``upstream_entries_for_input``, which re-derives exactly that relative
    path as the upstream ``source`` (#281 spec D8). ``None`` -- an entry with
    no relative path, or one the input path does not end with -- records the
    input's file name instead.
    """
    if not item.relative_image_path:
        return None
    relative = Path(item.relative_image_path).parts
    source = Path(item.input_path).parts
    if not relative or len(relative) >= len(source) or source[-len(relative):] != relative:
        return None
    return Path(*source[: -len(relative)])


def _active_check(output_dir: Path, epoch: str | None):
    """Return a publication fence callback for a staged worker."""
    if epoch is None:
        return None

    def _check() -> None:
        assert_active_epoch(output_dir, epoch)

    return _check


def _commit_guard(output_dir: Path, epoch: str | None) -> CommitGuard | None:
    """Return the staged lifecycle lock held only at canonical commit points."""
    if epoch is None:
        return None

    @contextmanager
    def _guard() -> Iterator[None]:
        with generation_publication_guard(output_dir, epoch):
            assert_active_epoch(output_dir, epoch)
            yield

    return _guard


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


def _recorded_image_read(
    output_dir: Path,
) -> tuple[dict[str, Any], int | None, int | None]:
    """How the submitter's flags say to read an input: settings, nrows, ncols.

    The staged submitter records ``--bit-depth``, ``--detect-mode``,
    ``--nrows`` and ``--ncols`` in ``job_metadata.json`` before Controller 0
    is submitted (``_write_staged_job_metadata``), so every Stage-1 task
    reads the same values. A key absent from the file (one written before it
    was recorded) reads as unset, which is what that run did.

    Raises:
        RuntimeError: The file is missing or unreadable. A Stage-1 task never
            starts before the submitter has written it, so this is a broken
            run, and reading the image with the defaults instead would
            silently apply the wrong grid, bit depth and detect mode.
    """
    metadata = load_job_metadata(progress_dir(output_dir))
    if not isinstance(metadata, dict):
        path = job_metadata_path(output_dir)
        state = "is unreadable" if path.exists() else "does not exist"
        raise RuntimeError(
            f"Staged Stage 1 reads --nrows, --ncols, --bit-depth and "
            f"--detect-mode from {path}, which {state}. The submitter writes "
            "it before any worker starts, so this run's machine state is "
            "incomplete; re-run the same command (or with --restart) to "
            "submit it again."
        )
    settings = input_read_settings(
        metadata.get(JobMetadataKey.BIT_DEPTH),
        metadata.get(JobMetadataKey.DETECT_MODE),
    )
    return (
        settings,
        metadata.get(JobMetadataKey.NROWS),
        metadata.get(JobMetadataKey.NCOLS),
    )


def _recorded_defers_embeddings(output_dir: Path) -> bool:
    """Whether the run embeds in Stage 4, as the submitter recorded it (#282).

    The worker never re-derives deferral from flags: it reads
    ``JobMetadataKey.EMBED_STAGE`` from ``job_metadata.json``, which the
    submitter writes before Controller 0 (``_write_staged_job_metadata``).
    A file without the key, or no readable file (a run written before #282,
    or a hand-built test tree), reads as ``"inline"``, which is what such a
    run did.
    """
    metadata = load_job_metadata(progress_dir(output_dir))
    if not isinstance(metadata, dict):
        return False
    return metadata.get(JobMetadataKey.EMBED_STAGE) == "gpu"


def _waiting_on_stage4(
    output_dir: Path, item: StagedManifestEntry, *, deferring: bool
) -> bool:
    """Whether *item* finished Stages 1-3 of a deferring run (``stages_1_to_3_done``)."""
    return bool(
        deferring
        and item.work_id
        and stages_1_to_3_done(output_dir, item.dataset, item.stem, item.work_id)
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
        (item.work_id and staged_store_matches_work_id(store, item.work_id))
        or (not item.work_id and valid_stage1_store(store))
    ):
        return
    check = _active_check(output_dir, epoch)
    commit_guard = _commit_guard(output_dir, epoch)
    if check is not None:
        check()
    # Before anything is cleared or written: a run with no recorded flags
    # must not touch the image at all.
    read_kwargs, cli_nrows, cli_ncols = _recorded_image_read(output_dir)
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    if resume:
        clear_downstream_artifacts_for_stage1(
            output_dir,
            item.dataset,
            item.stem,
            detector_slot(plan.gpu_path),
            commit_guard=commit_guard,
        )
    output_manager = OutputManager.from_config(
        output_dir,
        ext,
        save_overlays=False,
        durable_writes=durable_writes,
        run_initiation=recorded_run_initiation(output_dir),
    )
    log = event_log_path(output_dir)
    try:
        with stage_event(
            log,
            item.dataset,
            item.image_name,
            STAGE_PREPROCESS,
            active_check=check,
            commit_guard=commit_guard,
        ):
            stage1_preprocess_core(
                plan,
                Path(item.input_path),
                item.dataset,
                item.stem,
                output_dir,
                output_manager,
                image_type,
                read_kwargs,
                cli_nrows=cli_nrows,
                cli_ncols=cli_ncols,
                active_check=check,
                work_id=item.work_id,
                commit_guard=commit_guard,
                pipeline_path=pipeline_path,
                pipeline_identity=pipeline_identity,
                drop_originals=drop_originals,
                reference_pin=_reference_pin(item),
                input_root=_input_root(item),
            )
    except Exception as exc:
        _record_terminal_scientific_failure(
            output_dir, item, exc, epoch, commit_guard
        )
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
    commit_guard = _commit_guard(output_dir, epoch)
    if check is not None:
        check()
    shard = [
        _entry(item)
        for item in partition_shards(list(manifest), n_shards)[shard_index]
    ]
    # Hoisted above the candidate filter, which now needs the slot. This loads
    # and splits the pipeline (a JSON read) even when the shard turns out to be
    # fully done; the *model* load stays below the `pending` check, so the
    # full-dataset-sweep-with-resident-model invariant (spec §13.4) is
    # untouched -- nothing here rebuilds a model per image.
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    slot = detector_slot(plan.gpu_path)
    terminal = {
        record.work_id
        for record in read_terminal_failures(output_dir)
        if epoch is not None and record.lifecycle_epoch == epoch
    }
    # An image waiting on Stage 4 has consumed its Stage-2 signal; without
    # this it would be re-detected (#282 plan review C1).
    deferring = _recorded_defers_embeddings(output_dir)
    candidates = [
        item
        for item in shard
        if item.work_id not in terminal
        if not _waiting_on_stage4(output_dir, item, deferring=deferring)
        if not stage2_result_replayable(
            output_dir, item.dataset, item.stem, slot
        )
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
                and journal_measurement_table(
                    zarr_store_path(output_dir, item.dataset, item.stem)
                )
                is not None
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
            item.work_id and staged_store_matches_work_id(store, item.work_id)
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
            commit_guard=commit_guard,
        )
    if not pending:
        return

    if check is not None:
        check()
    # Load ONCE, after the shard is known to have work: one resident model per
    # full pass over the shard, never one per image.
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
                commit_guard=commit_guard,
            ):
                stage2_detect_core(
                    plan.gpu_detector,
                    output_dir,
                    item.dataset,
                    item.stem,
                    slot,
                    image_type,
                    active_check=check,
                    commit_guard=commit_guard,
                    stage2_prefix=plan.stage2_prefix,
                    reference_pin=_reference_pin(item),
                )
        except Exception as exc:
            inactive = slurm_generation_inactive_cause(exc)
            if inactive is not None:
                raise inactive
            _record_terminal_scientific_failure(
                output_dir, item, exc, epoch, commit_guard
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
    durable_writes: bool | None = None,
) -> None:
    """Replay one Stage-2 result, measure it, and publish per-image outputs.

    ``durable_writes`` is the transported tri-state; see
    :func:`run_stage1_step`. Stage 3 re-promotes the store, so a lost value
    here costs exactly what a lost value in Stage 1 costs.
    """
    item = _entry(manifest[index])
    store = zarr_store_path(output_dir, item.dataset, item.stem)
    output_manager = OutputManager.from_config(
        output_dir,
        ext,
        overlay_alpha=overlay_alpha,
        save_overlays=True,
        durable_writes=durable_writes,
        run_initiation=recorded_run_initiation(output_dir),
    )
    check = _active_check(output_dir, epoch)
    commit_guard = _commit_guard(output_dir, epoch)
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
        and journal_measurement_table(store) is not None
    )
    if terminal:
        ensure_staged_overlay(
            output_dir,
            item.dataset,
            item.stem,
            output_manager,
            image_type,
            active_check=check,
            commit_guard=commit_guard,
        )
        return
    deferring = _recorded_defers_embeddings(output_dir)
    if _waiting_on_stage4(output_dir, item, deferring=deferring):
        # Stages 1-3 are done and Stage 4 owns the image: publish nothing,
        # and return before the Stage-2 prereq check below, which would
        # otherwise SystemExit(1) on the signal Stage 3 already consumed
        # (#282 plan review C1).
        return
    if (
        not deferring
        and resume
        and item.work_id
        and stage3_completion_exists(output_dir, item.dataset, item.stem)
        and staged_store_matches_work_id(store, item.work_id)
        and journal_measurement_table(store) is not None
    ):
        ensure_staged_overlay(
            output_dir,
            item.dataset,
            item.stem,
            output_manager,
            image_type,
            active_check=check,
            commit_guard=commit_guard,
        )
        data_key, data_path = image_data_artifact(
            output_dir, output_manager, item.dataset, item.stem
        )
        artifacts = {
            "measurements": image_measurement_artifact(
                output_dir, item.dataset, item.stem
            ),
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
            commit_guard=commit_guard,
        )
        return
    if check is not None:
        check()
    log = event_log_path(output_dir)
    # Hoisted above the prereq probe, which now needs the slot.
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    slot = detector_slot(plan.gpu_path)
    # BOTH halves. The token is only a flag; Stage 3's input is the raw .npy,
    # and a token-present/raw-missing image would otherwise raise
    # FileNotFoundError inside stage_event and be recorded as a terminal
    # SCIENTIFIC failure rather than a missing prereq (ledger FLOW-17/M7).
    if not stage2_result_replayable(
        output_dir, item.dataset, item.stem, slot
    ):
        emit_missing_prereq(
            log,
            item.dataset,
            item.image_name,
            STAGE_MEASURE,
            "Stage 2 result",
            commit_guard=commit_guard,
        )
        raise SystemExit(1)
    try:
        with stage_event(
            log,
            item.dataset,
            item.image_name,
            STAGE_MEASURE,
            active_check=check,
            commit_guard=commit_guard,
        ):
            stage3_merge_measure_core(
                plan,
                output_dir,
                item.dataset,
                item.stem,
                output_manager,
                image_type,
                active_check=check,
                commit_guard=commit_guard,
                image_name=item.image_name,
                work_id=item.work_id,
                reference_pin=_reference_pin(item),
                defer_embeddings=deferring,
            )
            # Under deferral Stage 4 publishes the record (spec §4.2).
            if item.work_id and not deferring:
                data_key, data_path = image_data_artifact(
                    output_dir, output_manager, item.dataset, item.stem
                )
                artifacts = {
                    "measurements": image_measurement_artifact(
                        output_dir, item.dataset, item.stem
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
                    commit_guard=commit_guard,
                )
            if check is not None:
                check()
            write_stage3_completion_marker(
                output_dir,
                item.dataset,
                item.image_name,
                item.stem,
                commit_guard=commit_guard,
            )
            # Token FIRST at every consumption site: the only reachable
            # intermediate state must be "no token, orphan raw".
            if check is not None:
                check()
            delete_stage2_token(
                output_dir,
                item.dataset,
                item.stem,
                slot,
                commit_guard=commit_guard,
            )
            if check is not None:
                check()
            delete_stage2_raw(
                output_dir,
                item.dataset,
                item.stem,
                slot,
                commit_guard=commit_guard,
            )
    except Exception as exc:
        inactive = slurm_generation_inactive_cause(exc)
        if inactive is not None:
            raise inactive
        _record_terminal_scientific_failure(
            output_dir, item, exc, epoch, commit_guard
        )
        raise


def run_single_pass_step(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    index: int,
    ext: str = ".tiff",
    *,
    epoch: str | None = None,
    resume: bool = False,
    overlay_alpha: float = 0.3,
    durable_writes: bool | None = None,
    drop_originals: bool = False,
    pipeline_identity: Mapping[str, str] | None = None,
) -> None:
    """The "stage3" role of a Stage-4 run without a ``GpuDetector`` (#282 spec §4.1).

    Runs the single-pass core with its embeddings deferred
    (``defer_embeddings=True``) on one manifest image: the store with its
    measurements, then ``stages.stage3``, and **no** record; Stage 4
    publishes it once its embeddings commit (spec §4.2).

    It returns at once, touching nothing, when the image's record is valid
    or it already finished Stages 1-3 (``stages_1_to_3_done``): a
    continuation must not rewrite a root a Stage-4 signal was computed
    against (plan review C1). That holds with or without *resume*: a
    ``--restart`` clears the record and the Stage-3 marker first, so a
    restarted image is never skipped.

    A scientific failure is journalled under *epoch*, as Stage 3 journals
    one, and re-raised so the task fails.

    Args:
        pipeline_path: The run's pipeline file.
        output_dir: Run output root.
        image_type: ``"Image"`` or ``"GridImage"``.
        manifest: The staged manifest.
        index: The image's absolute manifest index.
        ext: The run's ``--ext``.
        epoch: The lifecycle epoch, the fence.
        resume: Accepted for the stage worker's common arguments; the skip
            above does not depend on it.
        overlay_alpha: The overlay alpha.
        durable_writes: The transported ``--durable-writes`` tri-state.
        drop_originals: ``--drop-originals``.
        pipeline_identity: The user pipeline's source identity.
    """
    del resume
    item = _entry(manifest[index])
    if item.work_id and (
        valid_image_success(
            output_dir,
            dataset=item.dataset,
            image_stem=item.stem,
            work_id=item.work_id,
        )
        or stages_1_to_3_done(output_dir, item.dataset, item.stem, item.work_id)
    ):
        return
    check = _active_check(output_dir, epoch)
    commit_guard = _commit_guard(output_dir, epoch)
    if check is not None:
        check()
    # Before anything is written: the input is read as Stage 1 reads it.
    read_kwargs, cli_nrows, cli_ncols = _recorded_image_read(output_dir)
    output_manager = OutputManager.from_config(
        output_dir,
        ext,
        overlay_alpha=overlay_alpha,
        save_overlays=True,
        durable_writes=durable_writes,
        run_initiation=recorded_run_initiation(output_dir),
    )
    log = event_log_path(output_dir)
    try:
        with stage_event(
            log,
            item.dataset,
            item.image_name,
            STAGE_MEASURE,
            active_check=check,
            commit_guard=commit_guard,
        ):
            process_single_image_core(
                pipeline_path=pipeline_path,
                image_path=Path(item.input_path),
                output_dir=output_dir,
                dataset_name=item.dataset,
                image_type=image_type,
                read_kwargs=read_kwargs,
                output_manager=output_manager,
                cli_nrows=cli_nrows,
                cli_ncols=cli_ncols,
                drop_originals=drop_originals,
                pipeline_identity=pipeline_identity,
                work_id=item.work_id,
                active_check=check,
                commit_guard=commit_guard,
                reference_pin=_reference_pin(item),
                input_root=_input_root(item),
                defer_embeddings=True,
            )
    except Exception as exc:
        inactive = slurm_generation_inactive_cause(exc)
        if inactive is not None:
            raise inactive
        _record_terminal_scientific_failure(
            output_dir, item, exc, epoch, commit_guard
        )
        raise


def run_measure_step(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: Manifest,
    index: int,
    *,
    epoch: str,
    durable_writes: bool | None = None,
) -> None:
    """The "stage3" role of a SLURM ``--mode measure`` on the staged controller (#282 U10).

    Re-measures one manifest store with
    ``process_single_store_measure_core(..., defer_embeddings=True)``: the
    measure pipeline's run folder is rewritten, the image's comparable
    embeddings are carried forward (U8), and neither ``embed`` nor the
    record's republish runs; the republish-mode Stage 4 does both.

    It returns at once when this pass's Stage 4 already republished the
    image (``stage4_status(republish=True) == "done"`` under *epoch*): a
    resubmitted task (U15) must not rewrite a root Stage 4 certified.

    The core runs inside ``measure_scientific_boundary()``, so a failure is
    journalled under the entry's measure work id and *epoch* with stage
    ``MEASURE_FAILURE_STAGE``, as the ordinary SLURM measure worker journals
    one, and re-raised so the task fails.

    Args:
        pipeline_path: The measure pipeline file.
        output_dir: Run output root.
        image_type: ``"Image"`` or ``"GridImage"``, the fallback for a store
            whose block names no class.
        manifest: The staged manifest (one entry per store; ``input_path``
            is the store).
        index: The image's absolute manifest index.
        epoch: The lifecycle epoch, the fence.
        durable_writes: The transported ``--durable-writes`` tri-state.
    """
    from ._cli_process_single import (
        measure_scientific_boundary,
        process_single_store_measure_core,
    )
    from ._cli_stage4_worker import _measure_pipeline_sha256
    from ._cli_staged_resume import stage4_status
    from ._cli_state_management import metadata_run_initiation

    item = _entry(manifest[index])
    if stage4_status(
        output_dir, item.dataset, item.stem, item.work_id, republish=True,
        pipeline_sha256=_measure_pipeline_sha256(pipeline_path), epoch=epoch,
    ) == "done":
        return
    check = _active_check(output_dir, epoch)
    commit_guard = _commit_guard(output_dir, epoch)
    if check is not None:
        check()
    metadata = load_job_metadata(progress_dir(output_dir))
    output_manager = OutputManager.from_config(
        output_dir,
        ".tiff",
        include_dataset_column=(
            metadata.get(JobMetadataKey.INCLUDE_DATASET_COLUMN, True)
            if isinstance(metadata, dict)
            else True
        ),
        save_overlays=False,
        durable_writes=durable_writes,
        run_initiation=metadata_run_initiation(output_dir),
    )
    try:
        with stage_event(
            event_log_path(output_dir),
            item.dataset,
            item.image_name,
            STAGE_MEASURE,
            active_check=check,
            commit_guard=commit_guard,
        ):
            with measure_scientific_boundary():
                process_single_store_measure_core(
                    pipeline_path=pipeline_path,
                    store_path=Path(item.input_path),
                    output_dir=output_dir,
                    dataset_name=item.dataset,
                    image_type=image_type,
                    output_manager=output_manager,
                    commit_guard=commit_guard,
                    defer_embeddings=True,
                )
    except Exception as exc:
        inactive = slurm_generation_inactive_cause(exc)
        if inactive is not None:
            raise inactive
        _record_terminal_scientific_failure(output_dir, item, exc, epoch, commit_guard)
        raise


def _republish_revisions(output_dir: Path) -> "dict[str, str | None] | None":
    """The revisions a SLURM ``--mode measure`` submission resolved (U5, P6 T2).

    Measure mode keeps no processing state, so the state's revisions are the
    forward run's; the measure submitter recorded its own in
    ``job_metadata.json`` (``EMBED_REVISIONS``), as a local measure uses its
    invocation's.
    """
    metadata = load_job_metadata(progress_dir(output_dir))
    recorded = metadata.get(JobMetadataKey.EMBED_REVISIONS) if isinstance(metadata, dict) else None
    if not isinstance(recorded, dict):
        return None
    return {str(k): (None if v is None else str(v)) for k, v in recorded.items()}


def run_stage4_shard(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    worklist_path: Path,
    shard_index: int,
    *,
    epoch: str,
    ext: str = ".tiff",
    overlay_alpha: float = 0.3,
    durable_writes: bool | None = None,
) -> int:
    """Embed, commit and publish one shard of the round's Stage-4 worklist on this GPU.

    The shard is ``load_stage4_worklist(worklist_path, shard_index)``, the
    controller's partition of the round's retryable images (#282 P5 T1). An
    empty shard (more shards than images) exits ``0`` before loading
    anything. Otherwise it calls ``run_stage4`` with:

    - ``epoch``, the run's real lifecycle epoch, stamped on every record and
      failure, and the epoch's fence: ``_active_check`` before each durable
      step and ``_commit_guard`` around each durable write;
    - the revisions the submitter recorded in ``job_metadata.json``
      (``EMBED_REVISIONS``, U5), so every shard of every round loads the same
      commit;
    - ``resolve_local_worker_count(-1, n) - 1`` loader workers: the
      allocation's CPUs, never a bare ``SLURM_CPUS_PER_TASK`` read (plan
      review I15);
    - the durability the submitter transported.

    The device, the determinism settings, the licence and the physical
    batch (from this GPU's total memory) are ``run_stage4``'s, at shard
    start. **Republish** (``--mode measure``, U4/U10) is read from the
    worklist's manifest (``staged_manifest_republishes``), the mode's one
    carrier (C14): then each record is republished stamped with *epoch*,
    and the revisions are the measure submission's.

    Args:
        pipeline_path: The run's pipeline file.
        output_dir: Run output root.
        image_type: ``"Image"`` or ``"GridImage"``.
        worklist_path: The round's worklist.
        shard_index: This task's shard (``$SLURM_ARRAY_TASK_ID``).
        epoch: The lifecycle epoch.
        ext: The run's ``--ext`` (artifact paths).
        overlay_alpha: The overlay alpha (the output manager's).
        durable_writes: The transported ``--durable-writes`` tri-state.

    Returns:
        ``0`` when the shard ran (per-image failures are journalled, and an
        image left pending is the next round's); ``1`` on node or model state
        (no usable CUDA, a GPU or host out of memory, a failed forward pass, a
        refused licence), journalling no image, so the next round retries.

    Raises:
        SlurmGenerationInactiveError: The epoch was fenced.
    """
    from phenotypic.abc_._measure_embedding import is_out_of_memory
    from phenotypic.detect.nn._helper._batch_encoder import Stage4OutOfMemory

    from ._cli_embed_stage import recorded_embed_revisions
    from ._cli_stage4_worker import Stage4DeviceError, Stage4ShardError, run_stage4
    from ._cli_staged_orchestration import (
        load_stage4_worklist,
        staged_manifest_republishes,
    )
    from ._cli_utils import resolve_local_worker_count

    check = _active_check(output_dir, epoch)
    commit_guard = _commit_guard(output_dir, epoch)
    if check is not None:
        check()
    shard = load_stage4_worklist(worklist_path, shard_index)
    entries = shard.entries
    if not entries:
        return 0
    republish = staged_manifest_republishes(shard.manifest_path)
    output_manager = OutputManager.from_config(
        output_dir,
        ext,
        overlay_alpha=overlay_alpha,
        save_overlays=True,
        durable_writes=durable_writes,
        run_initiation=recorded_run_initiation(output_dir),
    )
    try:
        outcome = run_stage4(
            pipeline_path,
            output_dir,
            entries,
            image_type=image_type,
            epoch=epoch,
            republish=republish,
            republish_epoch=epoch if republish else None,
            # The one reader of the recorded revisions (review M15): the
            # state first, then the job metadata, as a continuation reads them.
            # A measure submission's are its own (job metadata only).
            revisions=(
                _republish_revisions(output_dir)
                if republish
                else recorded_embed_revisions(output_dir)
            ),
            num_workers=resolve_local_worker_count(-1, len(entries)) - 1,
            output_manager=output_manager,
            active_check=check,
            commit_guard=commit_guard,
        )
    except Exception as exc:
        inactive = slurm_generation_inactive_cause(exc)
        if inactive is not None:
            raise inactive
        if not (
            isinstance(
                exc, (Stage4DeviceError, Stage4OutOfMemory, Stage4ShardError, MemoryError)
            )
            or is_out_of_memory(exc)
        ):
            raise
        print(
            f"Stage 4 shard {shard_index} stopped: {type(exc).__name__}: {exc}. "
            "No image was recorded as failed; the next round retries them.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Stage 4 shard {shard_index}: {len(outcome.published)} published, "
        f"{len(outcome.failed)} failed, {len(outcome.pending)} left for the next "
        f"round, {outcome.signals_written} signal(s) written"
    )
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one SLURM array task to its staged worker."""
    parser = argparse.ArgumentParser(prog="phenotypic-staged-slurm-worker")
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2, 3, 4))
    # Stage 3's role (#282 P5 T2): the staged merge-and-measure, or, in a
    # Stage-4 run without a GpuDetector, the single-pass core, or, in a
    # SLURM --mode measure on the controller (P6 T2), the measure core.
    parser.add_argument(
        "--stage3-role", choices=("staged", "single_pass", "measure"), default="staged"
    )
    # Stage 4: the round's worklist; --index is the shard.
    parser.add_argument("--worklist", type=Path, default=None)
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
    load_runtime_dependencies()
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
    image_type: ImageTypeName = (
        "GridImage" if args.image_type == "GridImage" else "Image"
    )
    if args.stage == 4:
        if args.worklist is None:
            parser.error("--stage 4 needs --worklist")
        return run_stage4_shard(
            args.pipeline,
            args.output_dir,
            image_type,
            args.worklist,
            args.index,
            epoch=args.epoch,
            ext=args.ext,
            overlay_alpha=args.overlay_alpha,
            durable_writes=args.durable_writes,
        )
    manifest = load_staged_manifest(args.manifest)
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
    elif args.stage3_role == "measure":
        run_measure_step(
            args.pipeline,
            args.output_dir,
            image_type,
            manifest,
            args.index,
            epoch=args.epoch,
            durable_writes=args.durable_writes,
        )
    elif args.stage3_role == "single_pass":
        run_single_pass_step(
            args.pipeline,
            args.output_dir,
            image_type,
            manifest,
            args.index,
            args.ext,
            overlay_alpha=args.overlay_alpha,
            drop_originals=args.drop_originals,
            pipeline_identity=pipeline_identity,
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
