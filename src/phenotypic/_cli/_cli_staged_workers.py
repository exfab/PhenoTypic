"""Per-image stage workers for the local staged GPU engine (Spec 1 §5-§6).

Three content-defined stages, each a pure per-image function:

- Stage 1 (``stage1_preprocess_core``): read raw image, apply the pre-detector
  ops, publish the staged ``results/<ds>/zarr/<stem>.ome.zarr`` store -- objmap
  included, as zeros, because ``valid_staged_store`` requires it.
- Stage 2 (``stage2_detect_core``): load the input layer (store read-only), run
  the resident detector, retain the **raw** result under
  ``.phenotypic/progress/stage2_raw/`` and drop the consumable token. It does
  **not** write into the store: only the final store needs third-party interop,
  and an in-store write here would be visible to the uncached crop route as raw
  pre-``drop_frame_background`` labels.
- Stage 3 (``stage3_merge_measure_core``): load the store, replay the retained
  **raw** array through the accessor, apply post-ops + measure, re-promote the
  store with the post-refined objmap, consume the token and the raw array
  (mandatory cleanup).
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from phenotypic import GridImage, Image
from phenotypic._core._provenance import (
    append_operation_provenance,
    initialize_cli_provenance,
    new_provenance_journal,
    provenance_success_sink,
    set_provenance_status,
    set_retry_base_length,
    truncate_provenance_to_retry_base,
    write_provenance_checkpoint,
)
from phenotypic.abc_ import GpuDetector
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_.ngff_ import valid_staged_store
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import StagePlan
from ._cli_stage2_token import (
    delete_stage2_raw,
    delete_stage2_token,
    load_stage2_raw,
    read_stage2_token,
    write_stage2_raw,
    write_stage2_token,
)
from ._cli_staged_resume import write_stage3_completion_marker
from ._stages import (
    STAGE_GPU_DETECT,
    STAGE_MEASURE,
    STAGE_PREPROCESS,
    StageTag,
)
from ._cli_update_state import append_completion_event, append_event
from ._cli_failure_tracker import PerImageScientificError
from ._cli_slurm_lifecycle import (
    SlurmGenerationInactiveError,
    slurm_generation_inactive_cause,
)

logger = logging.getLogger(__name__)

ActiveCheck = Callable[[], None]


def _check_active(active_check: ActiveCheck | None) -> None:
    if active_check is not None:
        active_check()


def _image_class(image_type: ImageTypeName):
    return GridImage if image_type == "GridImage" else Image


def _initialize_stage1_provenance(
    image: Image,
    pipeline_path: Path | None,
    pipeline_identity: Mapping[str, str] | None,
) -> None:
    if pipeline_path is not None:
        initialize_cli_provenance(
            image,
            pipeline_path,
            pipeline_identity=pipeline_identity,
        )
        return
    journal = new_provenance_journal()
    journal["status"] = "in_progress"
    image._metadata.provenance_journal = journal


def _write_provenance_checkpoint_fenced(
    store: Path,
    image: Image,
    active_check: ActiveCheck | None,
    *,
    journal_only: bool = False,
) -> None:
    """Publish one root journal only while this worker owns the lifecycle."""
    _check_active(active_check)
    write_provenance_checkpoint(store, image, journal_only=journal_only)


def _mark_failed_checkpoint(
    store: Path,
    image: Image,
    active_check: ActiveCheck | None,
) -> None:
    """Best-effort failure status without hiding the scientific exception."""
    try:
        _check_active(active_check)
        set_provenance_status(image, "failed")
        write_provenance_checkpoint(store, image)
    except SlurmGenerationInactiveError:
        # A superseded worker must not replace the active owner's prefix.
        return
    except Exception:
        logger.exception("Failed to mark staged provenance failed: %s", store)


def _checkpoint_successful_operation(
    store: Path,
    image: Image,
    active_check: ActiveCheck | None,
) -> None:
    """Publish an appended staged operation or roll it back on sink failure."""
    operations = image._metadata.provenance_journal["operations"]
    prior_length = len(operations) - 1
    try:
        _check_active(active_check)
        write_provenance_checkpoint(store, image)
    except BaseException:
        del operations[prior_length:]
        raise


@contextmanager
def stage_event(
    event_log: Path,
    dataset: str,
    image: str,
    stage: StageTag,
    *,
    active_check: ActiveCheck | None = None,
) -> Iterator[None]:
    """Emit ``started`` -> ``completed`` around a stage body; on exception emit a
    stage-tagged ``failed`` event (``"<ExcType>: <msg>"``) and re-raise.

    Centralizes the per-image event bookkeeping shared by the local strategy and
    the SLURM workers. The SLURM workers want the re-raise (fail the task);
    local callers that isolate a bad image wrap the ``with`` in ``try/except``.
    """
    _check_active(active_check)
    append_event(event_log, dataset, image, "started", stage=stage)
    try:
        yield
    except Exception as e:
        inactive = slurm_generation_inactive_cause(e)
        if inactive is not None:
            raise inactive
        _check_active(active_check)
        append_event(
            event_log,
            dataset,
            image,
            "failed",
            error_msg=f"{type(e).__name__}: {e}",
            stage=stage,
        )
        raise
    _check_active(active_check)
    append_completion_event(
        event_log, dataset, image, "completed", stage=stage
    )


def emit_missing_prereq(
    event_log: Path, dataset: str, image: str, stage: StageTag, what: str
) -> None:
    """Record an S6 skip: a stage's input artifact is absent.

    *what* is e.g. ``"staged store"`` (Stage 2) or ``"Stage 2 result"``
    (Stage 3).
    """
    append_event(
        event_log,
        dataset,
        image,
        "failed",
        error_msg=f"{stage} skipped: {what} missing",
        stage=stage,
    )


def stage1_preprocess_core(
    plan: StagePlan,
    image_path: Path,
    dataset_name: str,
    image_stem: str,
    output_dir: Path,
    output_manager: OutputManager,
    image_type: ImageTypeName,
    read_kwargs: Optional[Dict[str, Any]] = None,
    active_check: ActiveCheck | None = None,
    work_id: str | None = None,
    pipeline_path: Path | None = None,
    pipeline_identity: Mapping[str, str] | None = None,
    drop_originals: bool = False,
) -> None:
    """Read raw image, apply the pre-detector ops, publish the staged store."""
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image: Image | None = None
    checkpoint_ready = False
    try:
        read_kwargs = dict(read_kwargs or {})
        image_cls = _image_class(image_type)
        detect_mode = read_kwargs.pop("detect_mode", "gray")
        image = image_cls.imread(image_path, **read_kwargs)
        _initialize_stage1_provenance(
            image, pipeline_path, pipeline_identity
        )
        _check_active(active_check)
        if drop_originals:
            write_provenance_checkpoint(store, image, journal_only=True)
        else:
            image._retain_original()
            initial_store = output_manager.save_image_store(
                image, dataset_name, image_stem, work_id=work_id
            )
            if initial_store is None or not valid_staged_store(initial_store):
                raise RuntimeError(
                    "Initial Stage 1 checkpoint failed for "
                    f"{dataset_name}/{image_stem}"
                )
        checkpoint_ready = True
        if detect_mode != "gray":
            image.set_detect_mode(detect_mode)
        with provenance_success_sink(
            lambda updated: _write_provenance_checkpoint_fenced(
                store, updated, active_check
            )
        ):
            plan.pre_pipeline.apply(image, inplace=True)
        operation_count = len(
            image._metadata.provenance_journal["operations"]
        )
        _check_active(active_check)
        set_retry_base_length(image, operation_count)
        set_provenance_status(image, "staged")
        saved_store = output_manager.save_image_store(
            image, dataset_name, image_stem, work_id=work_id
        )
        if saved_store is None or not valid_staged_store(saved_store):
            raise RuntimeError(
                f"Stage 1 store publication failed for {dataset_name}/{image_stem}"
            )
    except SlurmGenerationInactiveError:
        raise
    except MemoryError:
        if image is not None and checkpoint_ready:
            _mark_failed_checkpoint(store, image, active_check)
        raise
    except Exception as exc:
        inactive = slurm_generation_inactive_cause(exc)
        if inactive is not None:
            raise inactive
        if image is not None and checkpoint_ready:
            _mark_failed_checkpoint(store, image, active_check)
        raise PerImageScientificError(STAGE_PREPROCESS, exc) from exc


def stage2_detect_core(
    detector: GpuDetector,
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    image_type: ImageTypeName = "Image",
    active_check: ActiveCheck | None = None,
) -> None:
    """Load the input layer (store read-only), infer, retain the raw + token.

    The detector's model must already be resident (caller invokes
    ``_ensure_model_loaded()`` once before streaming a shard).
    """
    image_cls = _image_class(image_type)
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image = image_cls.load_zarr(store)  # read-only use; never re-promoted here
    array = getattr(image, detector.input_layer)[:]
    try:
        compute_started = perf_counter()
        sample = detector._preprocess(array)
        batch = detector._collate([sample])
        result = detector._infer_batch(batch)[0]
        detector_duration = perf_counter() - compute_started
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError(STAGE_GPU_DETECT, exc) from exc
    _check_active(active_check)
    # Stage 2 does NOT write into the store. Only the final store needs
    # third-party interop, and an in-store write here would be visible to the
    # uncached crop route as raw pre-drop_frame_background labels. The raw
    # array precedes the token, so a crash before the token leaves no
    # "Stage 2 done" signal and Stage 2 simply recomputes.
    write_stage2_raw(output_dir, dataset_name, image_stem, result)
    _check_active(active_check)
    write_stage2_token(
        output_dir,
        dataset_name,
        image_stem,
        objmap_shape=(int(result.shape[0]), int(result.shape[1])),
        detector_duration_seconds=detector_duration,
    )


def ensure_staged_overlay(
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    output_manager: OutputManager,
    image_type: ImageTypeName,
    active_check: ActiveCheck | None = None,
) -> Path | None:
    """Publish a missing staged-run overlay from the completed image store."""
    if not output_manager.save_overlays:
        return None
    overlay_path = output_manager.get_output_path(
        dataset_name, "overlays", image_stem
    )
    if overlay_path.is_file():
        return overlay_path

    _check_active(active_check)
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image = _image_class(image_type).load_zarr(store)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    _check_active(active_check)
    return output_manager.save_overlay(image, dataset_name, image_stem)


def stage3_merge_measure_core(
    plan: StagePlan,
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    output_manager: OutputManager,
    image_type: ImageTypeName,
    active_check: ActiveCheck | None = None,
    image_name: str | None = None,
    work_id: str | None = None,
) -> None:
    """Replay the raw result, measure, re-promote the store, consume both."""
    image_cls = _image_class(image_type)
    store = zarr_store_path(output_dir, dataset_name, image_stem)
    image = image_cls.load_zarr(store)
    image.name = image_stem

    # Replay from Stage 2's RETAINED RAW output, never from the store's own
    # objmap. Stage 3 re-promotes over that objmap, so using it as input makes
    # a retried Stage 3 re-run _write_object_output on already-refined labels
    # -- and drop_frame_background then deletes a real colony. See D1.
    #
    # NOTE (ledger FLOW-21): this restores idempotency for the OBJMAP only. The
    # image loaded here is the already-post-processed store, so a post-op that
    # touches detect_mat or gray is applied twice on a retry. Pre-existing --
    # the HDF path re-saved the same way -- and out of scope here.
    try:
        _check_active(active_check)
        truncate_provenance_to_retry_base(image)
        set_provenance_status(image, "in_progress")
        write_provenance_checkpoint(store, image)

        result = load_stage2_raw(output_dir, dataset_name, image_stem)
        token = read_stage2_token(output_dir, dataset_name, image_stem)
        _check_active(active_check)
        merge_started = perf_counter()
        plan.gpu_detector._write_object_output(image, result)
        merge_duration = perf_counter() - merge_started
        append_operation_provenance(
            image,
            plan.gpu_detector,
            duration_seconds=(
                float(token.get("detector_duration_seconds", 0.0))
                + merge_duration
            ),
            pipeline_step_path=[plan.gpu_key],
        )
        _checkpoint_successful_operation(store, image, active_check)

        # post-detector ops (refiners incl. watershed) then measurement.
        with provenance_success_sink(
            lambda updated: _write_provenance_checkpoint_fenced(
                store, updated, active_check
            )
        ):
            plan.post_pipeline.apply(image, inplace=True)
        measurements = plan.post_pipeline.measure(image, apply_post=False)

        _check_active(active_check)
        output_manager.save_measurements(measurements, dataset_name, image_stem)
        if output_manager.save_overlays:
            _check_active(active_check)
            output_manager.save_overlay(image, dataset_name, image_stem)
        from phenotypic.plotting._pipeline import PlotCoordinator

        _check_active(active_check)
        PlotCoordinator(plan.post_pipeline, output_dir).emit_image(
            image,
            dataset=dataset_name,
            image_stem=image_stem,
            strict=True,
        )

        _check_active(active_check)
        set_provenance_status(image, "complete")
        saved_store = output_manager.save_image_store(
            image, dataset_name, image_stem, work_id=work_id
        )
        if saved_store is None or not valid_staged_store(saved_store):
            raise RuntimeError(
                f"Stage 3 store publication failed for {dataset_name}/{image_stem}"
            )
        if work_id is None:
            _check_active(active_check)
            write_stage3_completion_marker(
                output_dir,
                dataset_name,
                image_name or image_stem,
                image_stem,
            )
            _check_active(active_check)
            # Token first: the only partial cleanup is an inert orphan raw.
            delete_stage2_token(output_dir, dataset_name, image_stem)
            _check_active(active_check)
            delete_stage2_raw(output_dir, dataset_name, image_stem)
    except SlurmGenerationInactiveError:
        raise
    except MemoryError:
        _mark_failed_checkpoint(store, image, active_check)
        raise
    except Exception as exc:
        inactive = slurm_generation_inactive_cause(exc)
        if inactive is not None:
            raise inactive
        _mark_failed_checkpoint(store, image, active_check)
        if isinstance(exc, RuntimeError) and str(exc).startswith(
            "Stage 3 store publication failed"
        ):
            raise
        raise PerImageScientificError(STAGE_MEASURE, exc) from exc
