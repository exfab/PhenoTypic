from typing import Literal

import numpy as np

from phenotypic.abc_ import PrefabPipeline
from phenotypic.enhance import CLAHE, MedianFilter, BM3DDenoiser
from phenotypic.detect import RoundPeaksDetector
from phenotypic.correction import GridAligner
from phenotypic.refine import ReduceMultipleGridObjects, GridOversizedObjectRemover
from phenotypic.refine import BorderObjectRemover, SmallObjectRemover
from phenotypic.refine import MaskFill, MaskOpener
from phenotypic.measure import (
    MeasureIntensity,
    MeasureShape,
    MeasureTexture,
    MeasureColor,
)


class HeavyRoundPeaksPipeline(PrefabPipeline):
    """Detect and measure round colonies using peak detection with full refinement.

    An extended version of :class:`RoundPeaksPipeline` that adds BM3D denoising,
    CLAHE contrast enhancement, morphological refinement, grid alignment, and a
    second detection pass for improved accuracy on challenging plates.

    Steps:
        1. BM3DDenoiser — high-quality denoising
        2. CLAHE — boost local contrast
        3. MedianFilter — remove residual speckle
        4. RoundPeaksDetector — first detection pass
        5. MaskOpener, BorderObjectRemover, SmallObjectRemover, MaskFill — cleanup
        6. GridOversizedObjectRemover, ReduceMultipleGridObjects — grid refinement
        7. GridAligner — straighten the grid
        8. RoundPeaksDetector — second detection pass after alignment
        9. BorderObjectRemover, SmallObjectRemover, MaskFill — final cleanup
        10. ReduceMultipleGridObjects — final grid refinement

    Measurements: MeasureShape, MeasureColor, MeasureIntensity, MeasureTexture.

    Best For:
        - Round colonies on grid plates that need thorough refinement.
        - Noisy or low-contrast plates where :class:`RoundPeaksPipeline`
          produces too many artifacts.
        - Plates with slight rotation that benefits from grid alignment
          between detection passes.

    Consider Also:
        - :class:`RoundPeaksPipeline` for a faster, lighter version when
          plates are clean and well-lit.
        - :class:`HeavyOtsuPipeline` for general-purpose Otsu-based detection.

    See Also:
        :doc:`/tutorials/notebooks/08_using_prefab_pipelines` for a visual
        comparison of prefab pipelines.
    """

    def __init__(
            self,
            # Preprocessing / enhancement
            bm3d_sigma: float = 0.02,
            bm3d_block_size: int = 8,
            bm3d_stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages",
            bm3d_clip: bool = True,
            clahe_kernel_size: int | None = None,
            clahe_clip_limit: float = 0.01,
            median_mode: Literal["nearest", "reflect", "constant", "mirror", "wrap"] = "nearest",
            median_shape: Literal["disk", "square", "diamond"] = "diamond",
            median_radius: int = 5,
            median_cval: float = 0.0,
            # Detection settings
            detector_thresh_method: Literal[
                "otsu", "mean", "local", "triangle", "minimum", "isodata", "li"
            ] = "otsu",
            detector_subtract_background: bool = True,
            detector_remove_noise: bool = True,
            detector_footprint_width: int = 6,
            detector_noise_radius: int = 1,
            detector_smoothing_sigma: float = 2.0,
            detector_min_peak_distance: int | None = None,
            detector_peak_prominence: float | None = None,
            detector_edge_refinement: bool = True,
            detector_selection_mode: Literal["dominant", "centered", "regularized"] = "dominant",
            detector_split_merged: bool = True,
            # Grid alignment
            grid_aligner_axis: int = 0,
            grid_aligner_mode: str = "edge",
            # Morphology / refinement
            mask_opener_footprint: Literal["auto", "disk", "square", "diamond"] | np.ndarray | None = "auto",
            mask_opener_width: int = 5,
            mask_opener_n_iter: int = 1,
            border_remover_size: int | float = 1,
            small_object_min_size: int = 50,
            mask_fill_structure: np.ndarray | None = None,
            mask_fill_origin: int = 0,
            # Measurements
            texture_scale: int | list[int] = 5,
            texture_quant_lvl: Literal[8, 16, 32, 64] = 32,
            texture_enhance: bool = False,
            texture_warn: bool = False,
            color_white_chroma_max: float = 4.0,
            color_chroma_min: float = 8.0,
            color_include_XYZ: bool = False,
            # Pipeline bookkeeping
            benchmark: bool = False,
            verbose: bool = False,
    ) -> None:
        """
        Represents an image processing pipeline for analyzing microbe colonies on solid media agar.
        The pipeline includes preprocessing, detection, morphological refinement, and measurement
        steps.

        Attributes:
            bm3d_sigma: Controls the degree of noise reduction during BM3D denoising. Lower values
                retain more fine details, which might preserve subtle colony textures. Higher values
                remove more noise but may blur colony edges, affecting detection accuracy.
            bm3d_block_size: Block size for BM3D denoising. Larger blocks capture more spatial
                context for noise estimation but increase computation time.
            bm3d_stage_arg: Specifies the stage of BM3D denoising. "all_stages" applies more
                comprehensive denoising, potentially enhancing signal uniformity but may result
                in detail loss. "hard_thresholding" retains more high-frequency details but may
                leave more background noise intact.
            bm3d_clip: Whether to clip denoised values to the [0, 1] range. Disabling may
                preserve subtle intensity variations but can produce out-of-range values.
            clahe_kernel_size: Determines the size of the kernel used for local contrast enhancement
                via CLAHE. Larger sizes improve contrast over broader areas, but may over-amplify
                large background variations. Smaller sizes enhance localized details but may
                introduce noise.
            clahe_clip_limit: Contrast clipping limit for CLAHE. Lower values produce more subtle
                enhancement; higher values amplify local contrast more aggressively, which can
                over-enhance noise.
            median_mode: Boundary handling mode for median filtering. Controls how pixel values
                are extrapolated at image edges.
            median_shape: Defines the morphological shape ("disk", "square", "diamond") used for
                median filtering. The choice impacts how texture and artifacts are smoothed.
                For instance, "disk" may preserve radial features, whereas "square" provides
                edge-focused filtering.
            median_radius: Dictates the width for median filtering. Smaller values enhance fine
                textural differences, whereas larger radii smooth broader regions, potentially
                affecting the precise detection of small colonies.
            median_cval: Constant value used to pad image borders when median_mode is "constant".
            detector_thresh_method: Specifies the thresholding method for binary segmentation.
                "otsu" (default) applies global thresholding, "mean" uses mean-based threshold,
                "local" adapts to background variations, "li" uses Li's iterative minimum
                cross-entropy method, and "triangle", "minimum", "isodata" offer alternative
                thresholding strategies.
            detector_subtract_background: Toggles white tophat background subtraction before
                thresholding. Enabling this helps standardize varying lighting or agar density but
                may also obscure genuine gradients or subtle ring colonies.
            detector_remove_noise: Sets whether morphological opening is applied to remove small
                noise artifacts. True ensures a cleaner output but may falsely discard tiny colonies.
                False retains all details, which can increase false-positive noise levels.
            detector_footprint_width: Width in pixels for the morphological footprint used in
                background subtraction. Larger values remove larger-scale background variations
                but may erode colony edges.
            detector_noise_radius: Radius in pixels for the morphological opening used to remove
                small noise artifacts. Larger values remove larger noise but may erode fine
                colony features.
            detector_smoothing_sigma: Standard deviation for Gaussian smoothing of intensity
                profiles before peak detection. Higher values increase robustness to noise but may
                merge nearby peaks. Set to 0 to disable smoothing.
            detector_min_peak_distance: Minimum distance between detected peaks in pixels. If None,
                automatically estimated from grid dimensions. Prevents detection of spurious peaks
                too close together.
            detector_peak_prominence: Minimum prominence of peaks for detection. If None,
                automatically estimated from signal statistics. Higher values are more selective.
            detector_edge_refinement: Whether to refine grid edges using local intensity profiles.
                Improves accuracy but adds computational cost.
            detector_selection_mode: Strategy for selecting the primary grid when multiple
                candidate grids are found. "dominant" selects the grid with the most colonies,
                "centered" prefers grids near the image center, "regularized" balances colony
                count with spatial regularity.
            detector_split_merged: Whether to attempt splitting merged colonies that appear as
                single large objects. Improves accuracy for dense plates where colonies grow
                together but may over-segment irregular colony morphologies.
            grid_aligner_axis: Axis along which to align the grid (0 for rows, 1 for columns).
            grid_aligner_mode: Padding mode used when rotating the image for alignment.
            mask_opener_footprint: Describes the morphological shape for noise removal or
                mask refinement. "auto" lets the system adapt, named shapes ("disk", "square",
                "diamond") use a standard footprint, an ndarray specifies a custom structuring
                element, and None disables opening.
            mask_opener_width: Width of the structuring element when using a named shape for
                mask opening. Larger values remove more noise but may erode small colonies.
            mask_opener_n_iter: Number of iterations for the morphological opening. More
                iterations apply stronger smoothing to the mask.
            border_remover_size: Specifies the width of the border region to remove. Larger sizes
                eliminate edge artifacts and colonies cropped by image edges but may discard valid
                colonies near borders.
            small_object_min_size: Specifies the size threshold for considering objects as colonies.
                Increasing this parameter reduces false detection of small artifacts but risks
                ignoring small colonies.
            mask_fill_structure: Binary structuring element for hole filling in masks. Larger or
                more connected structures fill bigger holes. None uses the default cross-shaped
                element.
            mask_fill_origin: Origin offset for the structuring element used in hole filling.
            texture_scale: Defines the spatial scale(s) at which texture features are measured.
                Larger scales focus on macro-textures; smaller scales enhance granular detail
                assessment. Can be a list to compute features at multiple scales simultaneously.
            texture_quant_lvl: Number of gray levels for quantizing intensity values in texture
                analysis. Higher values capture finer intensity distinctions but increase
                computation time and may be sensitive to noise.
            texture_enhance: Whether to apply contrast enhancement to the texture input before
                computing GLCM features. May improve texture discrimination for low-contrast
                colonies.
            texture_warn: Boolean that enables warnings when texture measurements may not be
                reliable. Use this to flag potential inconsistencies in the captured texture data
                or image quality issues.
            color_white_chroma_max: Maximum chroma value for classifying a colony as white.
                Colonies with chroma below this threshold are labeled as white/achromatic.
            color_chroma_min: Minimum chroma value for assigning a chromatic hue. Colonies with
                chroma below this value receive a neutral color classification.
            color_include_XYZ: Whether to include CIE XYZ color space measurements in addition
                to the standard Lab and descriptive color features.
            benchmark: Enables time benchmarking for each pipeline step. Useful for performance
                debugging but adds overhead to the computation.
            verbose: Specifies whether to output detailed process information during execution.
                True provides step-by-step logs, which are useful for debugging, while False
                ensures silent execution suitable for batch processing.
        """

        # Construct the operations pipeline
        detector_kwargs = dict(
                thresh_method=detector_thresh_method,
                subtract_background=detector_subtract_background,
                remove_noise=detector_remove_noise,
                footprint_width=detector_footprint_width,
                noise_radius=detector_noise_radius,
                smoothing_sigma=detector_smoothing_sigma,
                min_peak_distance=detector_min_peak_distance,
                peak_prominence=detector_peak_prominence,
                edge_refinement=detector_edge_refinement,
                selection_mode=detector_selection_mode,
                split_merged=detector_split_merged,
        )

        ops = [
            BM3DDenoiser(sigma_psd=bm3d_sigma, block_size=bm3d_block_size, stage_arg=bm3d_stage_arg, clip=bm3d_clip),
            CLAHE(kernel_size=clahe_kernel_size, clip_limit=clahe_clip_limit),
            MedianFilter(mode=median_mode, shape=median_shape, width=median_radius, cval=median_cval),
            # First detection pass
            RoundPeaksDetector(**detector_kwargs),
            MaskOpener(shape=mask_opener_footprint, width=mask_opener_width, n_iter=mask_opener_n_iter),
            BorderObjectRemover(border_size=border_remover_size),
            SmallObjectRemover(min_size=small_object_min_size),
            MaskFill(structure=mask_fill_structure, origin=mask_fill_origin),
            GridOversizedObjectRemover(),
            ReduceMultipleGridObjects(),
            GridAligner(axis=grid_aligner_axis, mode=grid_aligner_mode),
            # Second detection pass
            RoundPeaksDetector(**detector_kwargs),
            MaskOpener(shape=None),
            BorderObjectRemover(border_size=border_remover_size),
            SmallObjectRemover(min_size=small_object_min_size),
            GridOversizedObjectRemover(),
            MaskFill(structure=mask_fill_structure, origin=mask_fill_origin),
            ReduceMultipleGridObjects(),
        ]

        meas = [
            MeasureShape(),
            MeasureColor(
                white_chroma_max=color_white_chroma_max,
                chroma_min=color_chroma_min,
                include_XYZ=color_include_XYZ,
            ),
            MeasureTexture(
                scale=texture_scale,
                quant_lvl=texture_quant_lvl,
                enhance=texture_enhance,
                warn=texture_warn,
            ),
            MeasureIntensity(),
        ]

        super().__init__(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose)


__all__ = ("HeavyRoundPeaksPipeline",)
