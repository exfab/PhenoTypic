from typing import Literal

import numpy as np

from phenotypic.ABC_ import PrefabPipeline
from phenotypic.enhance import CLAHE, GaussianBlur, MedianEnhancer, ContrastStretching, SobelFilter
from phenotypic.detection import OtsuDetector, WatershedDetector
from phenotypic.correction import GridAligner
from phenotypic.grid import (MinResidualErrorReducer, GridOversizedObjectRemover)
from phenotypic.objedit import BorderObjectRemover, SmallObjectRemover, LowCircularityRemover
from phenotypic.objedit import MaskFill, MaskOpener
from phenotypic.measure import MeasureIntensity, MeasureShape, MeasureTexture, MeasureColor


class AdvOtsuPipeline(PrefabPipeline):
    """
    The AdvWatershedPipeline class is a composite image processing pipeline that combines multiple layers of preprocessing, detection, and filtering steps
    that can will select the right colonies in most cases. This comes at the cost of being a more computationally expensive pipeline.

    Pipeline Steps:
        1. Gaussian Smoothing
        2. CLAHE
        3. Median Enhancement
        4. Watershed Segmentation
        5. Border Object Removal
        6. Grid Oversized Object Removal
        7. Minimum Residual Error Reduction
        8. Grid Alignment
        9. Repeat Watershed Segmentation
        10. Repeat Border Object Removal
        11. Repeat Minimum Residual Error Reduction
        12. Mask Fill

    Measurements:
        - Shape
        - Color
        - Texture
        - Intensity
    """

    def __init__(self,
                 gaussian_sigma: int = 5,
                 gaussian_mode: str = 'reflect',
                 gaussian_truncate: float = 4.0,
                 otsu_ignore_zeros: bool = True,
                 otsu_ignore_borders: bool = True,
                 mask_opener_footprint: Literal['auto'] | int | np.ndarray | None = 'auto',
                 border_remover_size: int = 1,
                 small_object_min_size: int = 50,
                 texture_scale: int = 5,
                 texture_warn: bool = False,
                 benchmark: bool = False, verbose: bool = False):
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
            footprint: Deprecated, use mask_opener_footprint.
            min_size: Deprecated, use small_object_min_size.
            border_size: Deprecated, use border_remover_size.
        """
        border_remover = BorderObjectRemover(border_size=border_remover_size)
        min_residual_reducer = MinResidualErrorReducer()

        self._ops = [
            GaussianBlur(sigma=gaussian_sigma, mode=gaussian_mode, truncate=gaussian_truncate),
            CLAHE(),
            MedianEnhancer(),
            SobelFilter(),
            OtsuDetector(ignore_zeros=otsu_ignore_zeros, ignore_borders=otsu_ignore_borders),
            MaskOpener(footprint=mask_opener_footprint),
            border_remover,
            SmallObjectRemover(min_size=small_object_min_size),
            MaskFill(),
            GridOversizedObjectRemover(),
            min_residual_reducer,
            GridAligner(),
            OtsuDetector(ignore_zeros=otsu_ignore_zeros, ignore_borders=otsu_ignore_borders),
            MaskOpener(footprint=None),
            border_remover,
            SmallObjectRemover(min_size=small_object_min_size),
            GridOversizedObjectRemover(),
            min_residual_reducer,
            MaskFill()
        ]

        self._meas = [
            MeasureShape(),
            MeasureColor(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
            MeasureIntensity()
        ]
        super().__init__(benchmark=benchmark, verbose=verbose)
