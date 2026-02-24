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
    """
    Configures and initializes a robust image processing pipeline tailored for analyzing circular colonies grown on
    solid media agar. It incorporates preprocessing, detection, morphological refinement, and feature extraction
    stages, with customizable parameters to handle diverse experimental setups and imaging conditions. Adjusting
    attributes fine-tunes pipeline behavior and impacts colony detection and measurement accuracy.

    Operations:
        1. `BM3DDenoiser`
        2. `CLAHE`
        3. `MedianFilter`
        4. `RoundPeaksDetector`
        5. `MaskOpener`
        6. `BorderObjectRemover`
        7. `SmallObjectRemover`
        8. `MaskFill`
        9. `GridOversizedObjectRemover`
        10. `MinResidualRemover`
        11. `GridAligner`
        12. `RoundPeaksDetector` (second pass since alignment might improve detection)
        13. `MaskOpener`
        14. `BorderObjectRemover`
        15. `SmallObjectRemover`
        16. `MaskFill`
        17. `MinResidualReducer`

    Measurements:
        - `MeasureShape`
        - `MeasureColor`
        - `MeasureIntensity`
        - `MeasureTexture`
    """

    def __init__(
            self,
            # Preprocessing / enhancement
            bm3d_sigma: float = 0.02,
            bm3d_stage_arg: Literal["all_stages", "hard_thresholding"] = "all_stages",
            clahe_kernel_size: int | None = None,
            median_shape: Literal["disk", "square", "diamond"] = "diamond",
            median_radius: int = 5,
            # detection settings
            detector_thresh_method: Literal[
                "otsu", "mean", "local", "triangle", "minimum", "isodata"
            ] = "otsu",
            detector_subtract_background: bool = True,
            detector_remove_noise: bool = True,
            detector_footprint_radius: int = 3,
            detector_smoothing_sigma: float = 2.0,
            detector_min_peak_distance: int | None = None,
            detector_peak_prominence: float | None = None,
            detector_edge_refinement: bool = True,
            # Morphology / refinement
            mask_opener_footprint: Literal["auto"] | int | np.ndarray | None = "auto",
            border_remover_size: int = 1,
            small_object_min_size: int = 50,
            # Measurements
            texture_scale: int = 5,
            texture_warn: bool = False,
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
            bm3d_stage_arg: Specifies the stage of BM3D denoising. "all_stages" applies more
                comprehensive denoising, potentially enhancing signal uniformity but may result
                in detail loss. "hard_thresholding" retains more high-frequency details but may
                leave more background noise intact.
            clahe_kernel_size: Determines the size of the kernel used for local contrast enhancement
                via CLAHE. Larger sizes improve contrast over broader areas, but may over-amplify
                large background variations. Smaller sizes enhance localized details but may
                introduce noise.
            median_shape: Defines the morphological shape ("disk", "square", "diamond") used for
                median filtering. The choice impacts how texture and artifacts are smoothed.
                For instance, "disk" may preserve radial features, whereas "square" provides
                edge-focused filtering.
            median_radius: Dictates the width for median filtering. Smaller values enhance fine
                textural differences, whereas larger radii smooth broader regions, potentially
                affecting the precise detection of small colonies.
            detector_thresh_method: Specifies the thresholding method for binary segmentation.
                "otsu" (default) applies global thresholding, "mean" uses mean-based threshold,
                "local" adapts to background variations, and "triangle", "minimum", "isodata"
                offer alternative thresholding strategies.
            detector_subtract_background: Toggles white tophat background subtraction before
                thresholding. Enabling this helps standardize varying lighting or agar density but
                may also obscure genuine gradients or subtle ring colonies.
            detector_remove_noise: Sets whether morphological opening is applied to remove small
                noise artifacts. True ensures a cleaner output but may falsely discard tiny colonies.
                False retains all details, which can increase false-positive noise levels.
            detector_footprint_radius: Radius in pixels for morphological operations (noise removal
                and background subtraction). Larger values remove larger noise but may erode colony
                edges.
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
            mask_opener_footprint: Describes the morphological shape for noise removal or
                mask refinement. "auto" lets the system adapt, while specifying values allows
                control over the scale of mask cleanup or preservation of detailed structures.
            border_remover_size: Specifies the width of the border region to remove. Larger sizes
                eliminate edge artifacts and colonies cropped by image edges but may discard valid
                colonies near borders.
            small_object_min_size: Specifies the size threshold for considering objects as colonies.
                Increasing this parameter reduces false detection of small artifacts but risks
                ignoring small colonies.
            texture_scale: Defines the spatial scale at which texture features are measured. Larger
                scales focus on macro-textures; smaller scales enhance granular detail assessment.
            texture_warn: Boolean that enables warnings when texture measurements may not be
                reliable. Use this to flag potential inconsistencies in the captured texture data
                or image quality issues.
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
                footprint_radius=detector_footprint_radius,
                smoothing_sigma=detector_smoothing_sigma,
                min_peak_distance=detector_min_peak_distance,
                peak_prominence=detector_peak_prominence,
                edge_refinement=detector_edge_refinement,
        )

        ops = [
            BM3DDenoiser(sigma_psd=bm3d_sigma, stage_arg=bm3d_stage_arg),
            CLAHE(kernel_size=clahe_kernel_size),
            MedianFilter(shape=median_shape, radius=median_radius),
            # First detection pass
            RoundPeaksDetector(**detector_kwargs),
            MaskOpener(shape=mask_opener_footprint),
            BorderObjectRemover(border_size=border_remover_size),
            SmallObjectRemover(min_size=small_object_min_size),
            MaskFill(),
            GridOversizedObjectRemover(),
            ReduceMultipleGridObjects(),
            GridAligner(),
            # Second detection pass
            RoundPeaksDetector(**detector_kwargs),
            MaskOpener(shape=None),
            BorderObjectRemover(border_size=border_remover_size),
            SmallObjectRemover(min_size=small_object_min_size),
            GridOversizedObjectRemover(),
            MaskFill(),
            ReduceMultipleGridObjects(),
        ]

        meas = [
            MeasureShape(),
            MeasureColor(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
            MeasureIntensity(),
        ]

        super().__init__(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose)


__all__ = ("HeavyRoundPeaksPipeline",)
