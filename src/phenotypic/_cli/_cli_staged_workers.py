"""Per-image stage workers for the local staged GPU engine (Spec 1 §5-§6).

Three content-defined stages, each a pure per-image function:

- Stage 1 (``stage1_preprocess_core``): read raw image, apply the pre-detector
  ops, save the staged ``results/<ds>/hdf/<stem>.h5``.
- Stage 2 (``stage2_detect_core``): load the input layer (HDF read-only), run
  the resident detector, write the ``.npy`` objmap sidecar.
- Stage 3 (``stage3_merge_measure_core``): load HDF + sidecar, write the object
  output via the accessor, apply post-ops + measure, atomically re-save the HDF,
  delete the sidecar (mandatory cleanup).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from phenotypic import GridImage, Image
from phenotypic.abc_ import GpuDetector
from phenotypic.sdk_ import dataset_hdf_dir
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import StagePlan
from ._cli_sidecar import delete_sidecar, load_sidecar, write_sidecar
from ._cli_staged_resume import (
    valid_staged_hdf,
    write_stage3_completion_marker,
)
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

    *what* is e.g. ``"staged HDF"`` (Stage 2) or ``"objmap sidecar"`` (Stage 3).
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
    """Read raw image, apply the pre-detector ops, save the staged HDF."""
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
    saved_hdf = output_manager.save_image_hdf(
        image,
        dataset_name,
        image_stem,
        root_attributes=(
            {"phenotypic_work_id": work_id} if work_id is not None else None
        ),
    )
    if saved_hdf is None or not valid_staged_hdf(saved_hdf):
        raise RuntimeError(
            f"Stage 1 HDF publication failed for {dataset_name}/{image_stem}"
        )


def stage2_detect_core(
    detector: GpuDetector,
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    image_type: ImageTypeName = "Image",
    active_check: ActiveCheck | None = None,
) -> None:
    """Load the input layer (HDF read-only), run inference, write the sidecar.

    The detector's model must already be resident (caller invokes
    ``_ensure_model_loaded()`` once before streaming a shard).
    """
    image_cls = _image_class(image_type)
    hdf = dataset_hdf_dir(output_dir, dataset_name) / f"{image_stem}.h5"
    image = image_cls.load_hdf5(hdf)  # read-only use; never re-saved here
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
    write_sidecar(output_dir, dataset_name, image_stem, result)


def ensure_staged_overlay(
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    output_manager: OutputManager,
    image_type: ImageTypeName,
    active_check: ActiveCheck | None = None,
) -> Path | None:
    """Publish a missing staged-run overlay from the completed image HDF."""
    if not output_manager.save_overlays:
        return None
    overlay_path = output_manager.get_output_path(
        dataset_name, "overlays", image_stem
    )
    if overlay_path.is_file():
        return overlay_path

    _check_active(active_check)
    hdf = dataset_hdf_dir(output_dir, dataset_name) / f"{image_stem}.h5"
    image = _image_class(image_type).load_hdf5(hdf)
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
    """Merge the sidecar, apply post-ops + measure, re-save HDF, delete sidecar."""
    image_cls = _image_class(image_type)
    hdf = dataset_hdf_dir(output_dir, dataset_name) / f"{image_stem}.h5"
    image = image_cls.load_hdf5(hdf)
    image.name = image_stem

    result = load_sidecar(output_dir, dataset_name, image_stem)
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
    saved_hdf = output_manager.save_image_hdf(
        image,
        dataset_name,
        image_stem,
        root_attributes=(
            {"phenotypic_work_id": work_id} if work_id is not None else None
        ),
    )  # atomic re-save
    if saved_hdf is None or not valid_staged_hdf(saved_hdf):
        raise RuntimeError(
            f"Stage 3 HDF publication failed for {dataset_name}/{image_stem}"
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
        delete_sidecar(output_dir, dataset_name, image_stem)
