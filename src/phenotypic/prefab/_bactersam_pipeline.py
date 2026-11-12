from __future__ import annotations

from phenotypic.abc_ import PrefabPipeline
from phenotypic.correction import CropImage
from phenotypic.detect.nn import Sam2
from phenotypic.detect.nn._helper._checkpoint_manager import Device, Sam2ModelSize
from phenotypic.enhance import BlurGauss, SubtractGaussian
from phenotypic.measure import (
    MeasureColor,
    MeasureIntensity,
    MeasureNeighborDist,
    MeasureShape,
    MeasureSize,
    MeasureTexture,
)
from phenotypic.refine import (
    KeepSectionLargest,
    RemoveBorderObjects,
    RemoveLowCircularity,
    RemoveSmallObjects,
)


class BacterSamPipeline(PrefabPipeline):
    """Detect and measure bacterial colonies on arrayed plates with SAM2.

    Crops the plate to its colony area, smooths and flattens the background
    of ``detect_mat``, and segments every colony with Meta's SAM2 foundation
    model. Shape, border and size filters then remove agar texture, edge
    fragments and debris, and only the largest object in each grid section is
    kept. The defaults reproduce the pipeline used on a 96-well (8 x 12)
    bacterial growth-curve screen.

    **Input assumptions.** Images are expected to be **already colour-corrected
    and denoised**: this pipeline performs neither. Run those steps first, for
    example with a ``ColorCorrector`` or ``CalibrateColorRpcc`` followed by
    ``ColorDenoise`` in a separate pipeline exported with
    ``python -m phenotypic --mode process --layer rgb``. Then pass the
    resulting OME-Zarr tree to this pipeline as ``--input``. Uncorrected
    input still runs, but colour measurements are then not comparable across
    plates, and sensor noise reaches SAM2 and the texture measurements.

    Steps:
        1. ``Crop`` -- CropImage: trim the plate border and holder.
        2. ``Blur`` -- BlurGauss: smooth ``detect_mat`` before segmentation.
        3. ``RemoveBG`` -- SubtractGaussian: remove slow illumination gradients.
        4. ``Detector`` -- Sam2: segment colonies from the prepared
           ``detect_mat`` (GPU; staged automatically by the CLI).
        5. ``CircularityFilter`` -- RemoveLowCircularity: drop irregular,
           non-colony masks.
        6. ``BorderFilter`` -- RemoveBorderObjects: drop objects touching the
           cropped image edge.
        7. ``SmallObjectFilter`` -- RemoveSmallObjects: drop debris and dust.
        8. ``GridLargest`` -- KeepSectionLargest: keep one colony per grid
           section.

    Measurements: MeasureNeighborDist, MeasureShape, MeasureSize,
    MeasureIntensity, MeasureColor, and one MeasureTexture per entry of
    ``texture_scales``.

    SAM2 needs a GPU for practical throughput. In a CLI run the pipeline is
    split at ``Detector`` into CPU preprocessing, GPU detection and CPU
    measurement automatically. SAM2 weights download on first use.

    Best For:
        - Bacterial colonies pinned in a grid on solid agar, imaged over a
          growth time course.
        - Colonies whose edges a global threshold merges with agar texture
          or uneven lighting.
        - Plates where small early-timepoint colonies must be found alongside
          large late ones.

    Consider Also:
        - :class:`HeavyOtsuPipeline` for a CPU-only pipeline on clean,
          evenly lit plates.
        - :class:`HeavyWatershedPipeline` for colonies that grow into one
          another.
        - :class:`RoundPeaksPipeline` for fast detection of round,
          well-separated yeast colonies.

    Examples:
        Build the pipeline for a 384-colony plate (construction loads no
        model):

        >>> from phenotypic.prefab import BacterSamPipeline
        >>> pipeline = BacterSamPipeline(nrows=16, ncols=24)
        >>> list(pipeline.get_ops())[3]
        'Detector'
    """

    def __init__(
        self,
        *,
        crop_left: int = 450,
        crop_right: int = 480,
        crop_top: int = 430,
        crop_bottom: int = 450,
        blur_sigma: float = 5.0,
        background_sigma: float = 800.0,
        sam_model_size: Sam2ModelSize = "tiny",
        sam_points_per_side: int = 32,
        sam_pred_iou_thresh: float = 0.7,
        sam_stability_score_thresh: float = 0.92,
        sam_min_mask_region_area: int = 100,
        sam_crop_n_layers: int = 1,
        sam_device: Device = "auto",
        circularity_cutoff: float = 0.5,
        border_size: int = 20,
        min_object_size: int = 600,
        texture_scales: tuple[int, ...] = (10, 20, 30, 40),
        nrows: int = 8,
        ncols: int = 12,
        benchmark: bool = False,
        verbose: bool = False,
    ) -> None:
        """Build the pipeline.

        Args:
            crop_left: Pixels cropped from the left edge. The default suits
                the imager the pipeline was built on; set it so the crop
                removes the plate holder and wall but keeps every colony.
            crop_right: Pixels cropped from the right edge.
            crop_top: Pixels cropped from the top edge.
            crop_bottom: Pixels cropped from the bottom edge.
            blur_sigma: Gaussian blur sigma, in pixels, applied to
                ``detect_mat``. Higher values suppress agar texture and
                condensation at the cost of softer colony edges.
            background_sigma: Sigma, in pixels, of the Gaussian background
                estimate subtracted from ``detect_mat``. Keep it much larger
                than a colony so colonies are not subtracted with the
                background.
            sam_model_size: SAM2 variant. ``"tiny"`` is fastest; larger
                variants trade VRAM and time for mask quality.
            sam_points_per_side: SAM2 prompt-grid density per axis
                (``points_per_side ** 2`` prompts). Raise it for many small
                colonies.
            sam_pred_iou_thresh: Minimum SAM2-predicted mask quality kept.
                Lower it to catch faint colonies.
            sam_stability_score_thresh: Minimum SAM2 mask stability kept.
                Higher values keep only crisp-edged masks.
            sam_min_mask_region_area: SAM2's own minimum mask area, in
                pixels, applied before the pipeline's filters.
            sam_crop_n_layers: Extra SAM2 crop-pyramid layers. ``1`` adds four
                overlapping crops, which helps small colonies on a large
                plate; ``0`` runs one full-image pass.
            sam_device: PyTorch device for SAM2. ``"auto"`` picks an available
                accelerator and falls back to CPU.
            circularity_cutoff: Minimum circularity (0--1) an object needs
                to be kept. Lower it for colonies with irregular edges.
            border_size: Objects within this many pixels of the cropped image
                edge are removed.
            min_object_size: Objects smaller than this many pixels are
                removed.
            texture_scales: Pixel offsets for texture measurement. Each scale
                gets its own MeasureTexture, keyed ``MeasureTexture``,
                ``MeasureTexture_1``, and so on.
            nrows: Grid rows on the plate.
            ncols: Grid columns on the plate.
            benchmark: Record per-operation timing.
            verbose: Log each operation as it runs.
        """
        ops = {
            "Crop": CropImage(
                left=crop_left, right=crop_right, top=crop_top, bottom=crop_bottom
            ),
            "Blur": BlurGauss(sigma=blur_sigma),
            "RemoveBG": SubtractGaussian(sigma=background_sigma),
            # Fixed to detect_mat: Blur and RemoveBG exist only to shape what
            # SAM2 sees, and on "rgb" they would not reach detection at all.
            "Detector": Sam2(
                input_layer="detect_mat",
                model_size=sam_model_size,
                points_per_side=sam_points_per_side,
                pred_iou_thresh=sam_pred_iou_thresh,
                stability_score_thresh=sam_stability_score_thresh,
                min_mask_region_area=sam_min_mask_region_area,
                crop_n_layers=sam_crop_n_layers,
                device=sam_device,
            ),
            "CircularityFilter": RemoveLowCircularity(cutoff=circularity_cutoff),
            "BorderFilter": RemoveBorderObjects(border_size=border_size),
            "SmallObjectFilter": RemoveSmallObjects(min_size=min_object_size),
            "GridLargest": KeepSectionLargest(),
        }
        meas = {
            "MeasureNeighborDist": MeasureNeighborDist(),
            "MeasureShape": MeasureShape(),
            "MeasureSize": MeasureSize(),
            "MeasureIntensity": MeasureIntensity(),
            "MeasureColor": MeasureColor(),
        }
        for index, scale in enumerate(texture_scales):
            key = "MeasureTexture" if index == 0 else f"MeasureTexture_{index}"
            meas[key] = MeasureTexture(scale=[scale])
        super().__init__(
            ops=ops,
            meas=meas,
            nrows=nrows,
            ncols=ncols,
            benchmark=benchmark,
            verbose=verbose,
        )


__all__ = ["BacterSamPipeline"]
