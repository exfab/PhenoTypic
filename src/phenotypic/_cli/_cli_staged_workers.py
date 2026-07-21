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
from typing import Any, Dict, Iterator, Optional

from phenotypic import GridImage, Image
from phenotypic.abc_ import GpuDetector
from phenotypic.sdk_ import dataset_hdf_dir
from phenotypic.sdk_.typing_ import ImageTypeName

from ._cli_output_manager import OutputManager
from ._cli_pipeline_split import StagePlan
from ._cli_sidecar import delete_sidecar, load_sidecar, write_sidecar
from ._stages import StageTag
from ._cli_update_state import append_completion_event, append_event


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
            event_log, dataset, image, "failed",
            error_msg=f"{type(e).__name__}: {e}", stage=stage,
        )
        raise
    append_completion_event(event_log, dataset, image, "completed", stage=stage)


def emit_missing_prereq(
    event_log: Path, dataset: str, image: str, stage: StageTag, what: str
) -> None:
    """Record an S6 skip: a stage's input artifact is absent.

    *what* is e.g. ``"staged HDF"`` (Stage 2) or ``"objmap sidecar"`` (Stage 3).
    """
    append_event(
        event_log, dataset, image, "failed",
        error_msg=f"{stage} skipped: {what} missing", stage=stage,
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
) -> None:
    """Read raw image, apply the pre-detector ops, save the staged HDF."""
    read_kwargs = dict(read_kwargs or {})
    image_cls = _image_class(image_type)
    detect_mode = read_kwargs.pop("detect_mode", "gray")
    image = image_cls.imread(image_path, **read_kwargs)
    if detect_mode != "gray":
        image.set_detect_mode(detect_mode)
    plan.pre_pipeline.apply(image, inplace=True)
    output_manager.save_image_hdf(image, dataset_name, image_stem)


def stage2_detect_core(
    detector: GpuDetector,
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    image_type: ImageTypeName = "Image",
) -> None:
    """Load the input layer (HDF read-only), run inference, write the sidecar.

    The detector's model must already be resident (caller invokes
    ``_ensure_model_loaded()`` once before streaming a shard).
    """
    image_cls = _image_class(image_type)
    hdf = dataset_hdf_dir(output_dir, dataset_name) / f"{image_stem}.h5"
    image = image_cls.load_hdf5(hdf)  # read-only use; never re-saved here
    array = getattr(image, detector.input_layer)[:]
    sample = detector._preprocess(array)
    batch = detector._collate([sample])
    result = detector._infer_batch(batch)[0]
    write_sidecar(output_dir, dataset_name, image_stem, result)


def stage3_merge_measure_core(
    plan: StagePlan,
    output_dir: Path,
    dataset_name: str,
    image_stem: str,
    output_manager: OutputManager,
    image_type: ImageTypeName,
) -> None:
    """Merge the sidecar, apply post-ops + measure, re-save HDF, delete sidecar."""
    image_cls = _image_class(image_type)
    hdf = dataset_hdf_dir(output_dir, dataset_name) / f"{image_stem}.h5"
    image = image_cls.load_hdf5(hdf)

    result = load_sidecar(output_dir, dataset_name, image_stem)
    plan.gpu_detector._write_object_output(image, result)

    # post-detector ops (refiners incl. watershed) then measurement. measure()
    # runs only the measurement queue (apply_post=False keeps per-image parquets
    # clean), so the refiners applied above do not run twice.
    plan.post_pipeline.apply(image, inplace=True)
    measurements = plan.post_pipeline.measure(image, apply_post=False)

    output_manager.save_measurements(measurements, dataset_name, image_stem)
    output_manager.save_image_hdf(image, dataset_name, image_stem)  # atomic re-save
    from phenotypic.plotting import PlotCoordinator

    PlotCoordinator(plan.post_pipeline, output_dir).emit_image(
        image,
        dataset=dataset_name,
        image_stem=image_stem,
    )
    delete_sidecar(output_dir, dataset_name, image_stem)  # mandatory cleanup
