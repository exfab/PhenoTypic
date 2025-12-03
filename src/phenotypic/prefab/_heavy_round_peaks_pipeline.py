from typing import Literal

import numpy as np

from phenotypic.abc_ import PrefabPipeline
from phenotypic.enhance import CLAHE, MedianFilter, BM3DDenoiser
from phenotypic.detect import RoundPeaksDetector
from phenotypic.correction import GridAligner
from phenotypic.refine import MinResidualErrorReducer, GridOversizedObjectRemover
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
            "gitter", "otsu", "mean", "local", "triangle", "minimum", "isodata"
        ] = "gitter",
        detector_subtract_background: bool = True,
        detector_remove_noise: bool = False,
        detector_fast_resize: int | None = 1000,
        detector_fixed_square: float = 2.0,
        detector_expf: float = 1.5,
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
            median_radius: Dictates the radius for median filtering. Smaller values enhance fine
                textural differences, whereas larger radii smooth broader regions, potentially
                affecting the precise detection of small colonies.
            detector_thresh_method: Specifies the thresholding method for binary segmentation.
                "gitter" uses iterative thresholding from the original algorithm, robust to uneven
                illumination. "otsu" or "triangle" focus on global thresholding, suitable for
                uniform backgrounds. "local" adapts to background variations but may increase runtime.
            detector_subtract_background: Toggles background normalization during the detection stage.
                Enabling this helps standardize varying lighting or agar density but may also
                obscure genuine gradients or subtle ring colonies.
            detector_remove_noise: Sets whether small noisy objects are removed during detection.
                True ensures a cleaner output but may falsely discard tiny colonies. False retains
                all details, which can increase false-positive noise levels.
            detector_fast_resize: Downsample height used during background correction; larger
                values better preserve small colonies at the cost of speed. None disables downsampling.
            detector_fixed_square: Fallback box multiplier when the center pixel is 0; raise
                for hollow or frayed colonies so bounding boxes still capture area.
            detector_expf: Expansion factor for rectangles around detected peaks; increase
                if colonies sprawl or have halos, decrease to reduce spillover into
                neighbors on dense plates.
            mask_opener_footprint: Describes the morphological footprint for noise removal or
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
            fast_resize=detector_fast_resize,
            fixed_square=detector_fixed_square,
            expf=detector_expf,
        )

        ops = [
            BM3DDenoiser(sigma_psd=bm3d_sigma, stage_arg=bm3d_stage_arg),
            CLAHE(kernel_size=clahe_kernel_size),
            MedianFilter(shape=median_shape, radius=median_radius),
            # First detection pass
            RoundPeaksDetector(**detector_kwargs),
            MaskOpener(footprint=mask_opener_footprint),
            BorderObjectRemover(border_size=border_remover_size),
            SmallObjectRemover(min_size=small_object_min_size),
            MaskFill(),
            GridOversizedObjectRemover(),
            MinResidualErrorReducer(),
            GridAligner(),
            # Second detection pass
            RoundPeaksDetector(**detector_kwargs),
            MaskOpener(footprint=None),
            BorderObjectRemover(border_size=border_remover_size),
            SmallObjectRemover(min_size=small_object_min_size),
            GridOversizedObjectRemover(),
            MaskFill(),
            MinResidualErrorReducer(),
        ]

        meas = [
            MeasureShape(),
            MeasureColor(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
            MeasureIntensity(),
        ]

        super().__init__(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose)


__all__ = ("HeavyRoundPeaksPipeline",)
