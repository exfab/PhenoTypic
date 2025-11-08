from typing import Literal

import numpy as np

from phenotypic.abc_ import PrefabPipeline
from phenotypic.correction import GridAligner
from phenotypic.detect import WatershedDetector
from phenotypic.enhance import CLAHE, GaussianBlur, MedianEnhancer
from phenotypic.grid import (GridOversizedObjectRemover, MinResidualErrorReducer)
from phenotypic.measure import MeasureColor, MeasureIntensity, MeasureShape, MeasureTexture
from phenotypic.objedit import BorderObjectRemover, MaskFill


class AdvWatershedPipeline(PrefabPipeline):
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
                 watershed_footprint: Literal['auto'] | np.ndarray | int | None = None,
                 watershed_min_size: int = 50,
                 watershed_compactness: float = 0.001,
                 watershed_connectivity: int = 1,
                 watershed_relabel: bool = True,
                 watershed_ignore_zeros: bool = True,
                 border_remover_size: int = 1,
                 texture_scale: int = 5,
                 texture_warn: bool = False,
                 benchmark: bool = False, **kwargs):
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
            watershed_footprint: Structuring element for watershed peak detection.
            watershed_min_size (int): Minimum size for watershed segmentation.
            watershed_compactness (float): Compactness parameter for watershed.
            watershed_connectivity (int): Connectivity for watershed.
            watershed_relabel (bool): Whether to relabel after watershed.
            watershed_ignore_zeros (bool): Whether to ignore zeros in watershed thresholding.
            border_remover_size (int): Size of border to remove objects from.
            texture_scale (int): Scale parameter for Haralick texture features.
            texture_warn (bool): Whether to warn on texture computation errors.
            footprint: Deprecated, use watershed_footprint.
            min_size: Deprecated, use watershed_min_size.
            compactness: Deprecated, use watershed_compactness.
            border_size: Deprecated, use border_remover_size.
        """

        watershed_detector = WatershedDetector(footprint=watershed_footprint, min_size=watershed_min_size,
                                               compactness=watershed_compactness, connectivity=watershed_connectivity,
                                               relabel=watershed_relabel, ignore_zeros=watershed_ignore_zeros)
        border_remover = BorderObjectRemover(border_size=border_remover_size)
        min_residual_reducer = MinResidualErrorReducer()

        ops = [
            GaussianBlur(sigma=gaussian_sigma, mode=gaussian_mode, truncate=gaussian_truncate),
            CLAHE(),
            MedianEnhancer(),
            watershed_detector,
            border_remover,
            GridOversizedObjectRemover(),
            min_residual_reducer,
            GridAligner(),
            watershed_detector,
            min_residual_reducer,
            border_remover,
            MaskFill()
        ]

        meas = [
            MeasureShape(),
            MeasureColor(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
            MeasureIntensity()
        ]
        super().__init__(ops=ops, meas=meas, benchmark=benchmark, **kwargs)
