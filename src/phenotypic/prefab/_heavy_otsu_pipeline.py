from typing import Literal

import numpy as np

from phenotypic.abc_ import PrefabPipeline
from phenotypic.enhance import (
    EnhanceLocalContrast,
    GaussianBlur,
    MedianFilter,
    SobelFilter,
)
from phenotypic.detect import OtsuDetector
from phenotypic.correction import GridAligner
from phenotypic.refine import ReduceSectionsByLine, GridOversizedObjectRemover
from phenotypic.refine import (
    RemoveBorderObjects,
    SmallObjectRemover,
)
from phenotypic.refine import MaskFill, MaskOpening
from phenotypic.measure import (
    MeasureIntensity,
    MeasureShape,
    MeasureTexture,
    MeasureColor,
)


class HeavyOtsuPipeline(PrefabPipeline):
    """Detect and measure colonies using multi-stage Otsu thresholding with refinement.

    A robust general-purpose pipeline that chains preprocessing, Otsu detection,
    morphological cleanup, grid alignment, and re-detection for reliable colony
    segmentation on standard grid plates.

    Steps:
        1. GaussianBlur — smooth noise
        2. EnhanceLocalContrast — boost local contrast
        3. MedianFilter — remove residual speckle
        4. SobelFilter — enhance colony edges
        5. OtsuDetector — threshold-based detection
        6. MaskOpening — smooth mask boundaries
        7. RemoveBorderObjects — remove partial edge colonies
        8. SmallObjectRemover — remove noise fragments
        9. MaskFill — fill interior holes
        10. GridOversizedObjectRemover — remove merged multi-well objects
        11. ReduceSectionsByLine — keep one colony per well
        12. GridAligner — straighten the grid

    Measurements: MeasureShape, MeasureColor, MeasureTexture, MeasureIntensity.

    Best For:
        - Standard 96-well or 384-well yeast plates with clean backgrounds.
        - General-purpose colony detection when you are unsure which detector
          to use.
        - Plates with uniform illumination and bimodal intensity histograms.

    Consider Also:
        - :class:`HeavyWatershedPipeline` when colonies touch or overlap.
        - :class:`RoundPeaksPipeline` for a faster, lighter approach on
          well-separated round colonies.
        - :class:`FilamentousFungiPipeline` for filamentous organisms.

    See Also:
        :doc:`/tutorials/notebooks/08_using_prefab_pipelines` for a visual
        comparison of prefab pipelines.
        :doc:`/explanation/prefab_pipelines_guide` for guidance on choosing
        a prefab pipeline.
    """

    def __init__(
            self,
            gaussian_sigma: int = 5,
            gaussian_mode: str = "reflect",
            gaussian_truncate: float = 4.0,
            otsu_ignore_zeros: bool = True,
            otsu_ignore_borders: bool = True,
            mask_opener_footprint: Literal["auto"] | int | np.ndarray | None = "auto",
            border_remover_size: int = 1,
            small_object_min_size: int = 50,
            texture_scale: int = 5,
            texture_warn: bool = False,
            benchmark: bool = False,
            verbose: bool = False,
    ):
        """
        Initializes the object with a sequence of operations and measurements for image
        processing. The sequence includes smoothing, enhance, segmentation, border
        object removal, and various measurement steps for analyzing images. Customizable
        parameters allow for adjusting the processing pipeline for specific use cases such
        as image segmentation and feature extraction.

        Args:
            gaussian_sigma (int): Standard deviation for Gaussian kernel in smoothing.
            gaussian_mode (str): Mode for handling image boundaries in Gaussian smoothing.
            gaussian_truncate (float): Truncate filter at this many standard deviations.
            otsu_ignore_zeros (bool): Whether to ignore zero pixels in Otsu thresholding.
            otsu_ignore_borders (bool): Whether to ignore border objects in Otsu detection.
            mask_opener_footprint: Structuring element for morphological opening.
            border_remover_size (int): Size of border to remove objects from.
            small_object_min_size (int): Minimum size of objects to retain.
            texture_scale (int): Scale parameter for Haralick texture features.
            texture_warn (bool): Whether to warn on texture computation errors.
            shape: Deprecated, use mask_opener_footprint.
            min_size: Deprecated, use small_object_min_size.
            border_size: Deprecated, use border_remover_size.
        """
        border_remover = RemoveBorderObjects(border_size=border_remover_size)
        min_residual_reducer = ReduceSectionsByLine()

        ops = [
            GaussianBlur(
                    sigma=gaussian_sigma, mode=gaussian_mode, truncate=gaussian_truncate
            ),
            EnhanceLocalContrast(),
            MedianFilter(),
            SobelFilter(),
            OtsuDetector(
                    ignore_zeros=otsu_ignore_zeros, ignore_borders=otsu_ignore_borders
            ),
            MaskOpening(shape=mask_opener_footprint),
            border_remover,
            SmallObjectRemover(min_size=small_object_min_size),
            MaskFill(),
            GridOversizedObjectRemover(),
            min_residual_reducer,
            GridAligner(),
            OtsuDetector(
                    ignore_zeros=otsu_ignore_zeros, ignore_borders=otsu_ignore_borders
            ),
            MaskOpening(shape=None),
            border_remover,
            SmallObjectRemover(min_size=small_object_min_size),
            GridOversizedObjectRemover(),
            min_residual_reducer,
            MaskFill(),
        ]

        meas = [
            MeasureShape(),
            MeasureColor(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
            MeasureIntensity(),
        ]
        super().__init__(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose)


__all__ = "HeavyOtsuPipeline"
