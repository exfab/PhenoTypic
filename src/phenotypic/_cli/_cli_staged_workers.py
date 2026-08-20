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
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from phenotypic import GridImage, Image
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

ActiveCheck = Callable[[], None]


def _check_active(active_check: ActiveCheck | None) -> None:
    if active_check is not None:
        active_check()


def _image_class(image_type: ImageTypeName):
    return GridImage if image_type == "GridImage" else Image


@contextmanager
def stage_event(
    event_log: Path, dataset: str, image: str, stage: StageTag
) -> Iterator[None]:
    """Emit ``started`` -> ``completed`` around a stage body; on exception emit a
    stage-tagged ``failed`` event (``"<ExcType>: <msg>"``) and re-raise.

    Centralizes the per-image event bookkeeping shared by the local strategy and
    the SLURM workers. The SLURM workers want the re-raise (fail the task);
    local callers that isolate a bad image wrap the ``with`` in ``try/except``.
    """
    append_event(event_log, dataset, image, "started", stage=stage)
    try:
        yield
    except Exception as e:
        append_event(
            event_log,
            dataset,
            image,
            "failed",
            error_msg=f"{type(e).__name__}: {e}",
            stage=stage,
        )
        raise
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
) -> None:
    """Read raw image, apply the pre-detector ops, publish the staged store."""
    try:
        read_kwargs = dict(read_kwargs or {})
        image_cls = _image_class(image_type)
        detect_mode = read_kwargs.pop("detect_mode", "gray")
        image = image_cls.imread(image_path, **read_kwargs)
        if detect_mode != "gray":
            image.set_detect_mode(detect_mode)
        plan.pre_pipeline.apply(image, inplace=True)
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError(STAGE_PREPROCESS, exc) from exc
    _check_active(active_check)
    saved_store = output_manager.save_image_store(
        image, dataset_name, image_stem, work_id=work_id
    )
    if saved_store is None or not valid_staged_store(saved_store):
        raise RuntimeError(
            f"Stage 1 store publication failed for {dataset_name}/{image_stem}"
        )


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
        sample = detector._preprocess(array)
        batch = detector._collate([sample])
        result = detector._infer_batch(batch)[0]
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
    write_stage2_token(
        output_dir,
        dataset_name,
        image_stem,
        objmap_shape=(int(result.shape[0]), int(result.shape[1])),
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
    result = load_stage2_raw(output_dir, dataset_name, image_stem)
    try:
        plan.gpu_detector._write_object_output(image, result)

        # post-detector ops (refiners incl. watershed) then measurement.
        plan.post_pipeline.apply(image, inplace=True)
        measurements = plan.post_pipeline.measure(image, apply_post=False)
    except MemoryError:
        raise
    except Exception as exc:
        raise PerImageScientificError(STAGE_MEASURE, exc) from exc

    _check_active(active_check)
    output_manager.save_measurements(measurements, dataset_name, image_stem)
    _check_active(active_check)
    # Re-promote: post-ops mutate the objmap, and this is what publishes the
    # POST-REFINED segmentation. Without it the stored label image disagrees
    # with the parquet and with a single-pass run.
    saved_store = output_manager.save_image_store(
        image, dataset_name, image_stem, work_id=work_id
    )
    if saved_store is None or not valid_staged_store(saved_store):
        raise RuntimeError(
            f"Stage 3 store publication failed for {dataset_name}/{image_stem}"
        )
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
    if work_id is None:
        _check_active(active_check)
        write_stage3_completion_marker(
            output_dir,
            dataset_name,
            image_name or image_stem,
            image_stem,
        )
        _check_active(active_check)
        # Consume both. The completion marker is already written above, so a
        # crash between these two deletes classifies "complete" either way and
        # the survivor is inert garbage -- but delete the token FIRST, so the
        # only reachable intermediate state is "no token, orphan raw" (Stage 2
        # would recompute and overwrite it) rather than "token present, raw
        # missing" (Stage 3 would replay into a FileNotFoundError).
        delete_stage2_token(output_dir, dataset_name, image_stem)
        delete_stage2_raw(output_dir, dataset_name, image_stem)
