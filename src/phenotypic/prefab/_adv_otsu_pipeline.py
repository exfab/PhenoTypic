from typing import Literal

import numpy as np

from phenotypic.core._image_pipeline import ImagePipeline

from phenotypic.enhancement import CLAHE, GaussianSmoother, MedianEnhancer, ContrastStretching
from phenotypic.detection import OtsuDetector, WatershedDetector
from phenotypic.correction import GridAligner
from phenotypic.grid import (MinResidualErrorReducer, GridOversizedObjectRemover)
from phenotypic.objedit import BorderObjectRemover, SmallObjectRemover, LowCircularityRemover
from phenotypic.morphology import MaskFill
from phenotypic.measure import MeasureIntensity, MeasureShape, MeasureTexture, MeasureColor


class AdvOtsuPipeline(ImagePipeline):
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

    def __init__(self, scale: int = 5, sigma: int = 5,
                 footprint: Literal['auto'] | int | np.ndarray | None = 'auto',
                 min_size: int = 50,
                 compactness: float = 0.001,
                 border_size: int = 1, benchmark: bool = False, ):
        """
        Initializes the object with a sequence of operations and measurements for image
        processing. The sequence includes smoothing, enhancement, segmentation, border
        object removal, and various measurement steps for analyzing images. Customizable
        parameters allow for adjusting the processing pipeline for specific use cases such
        as image segmentation and feature extraction.

        Args:
            sigma (int): The standard deviation of the Gaussian kernel used for smoothing
                images during preprocessing. Helps reduce noise and improve segmentation.
            footprint (Literal['auto'] | int | np.ndarray | None): The structuring element
                used in the watershed segmentation algorithm. Can be 'auto' for automatic
                detection, an integer for a square structuring element of given size, a
                numpy array defining a custom structuring element, or None for no footprint.
            min_size (int): Minimum allowable size (in pixels) of segmented objects. Objects
                below this size are removed during processing.
            compactness (float): Compactness parameter for the watershed algorithm. Controls
                the smoothness of segment boundaries. Larger values produce smoother, more
                rounded segments.
            border_size (int): Size of the border (in pixels) within which segmented objects
                are removed. This helps to eliminate artifacts and unwanted objects at the
                edges of the processed images.
        """
        border_remover = BorderObjectRemover(border_size=border_size)
        min_residual_reducer = MinResidualErrorReducer()
        super().__init__(
                ops=[
                    GaussianSmoother(sigma=5),
                    CLAHE(footprint),
                    MedianEnhancer(),
                    SobelFilter(),
                    OtsuDetector(ignore_borders=True),
                    MaskOpener(),
                    border_remover,
                    SmallObjectRemover(min_size),
                    MaskFill(),
                    GridOversizedObjectRemover(),
                    min_residual_reducer,
                    GridAligner(),
                    OtsuDetector(ignore_borders=True),
                    MaskOpener(),
                    border_remover,
                    SmallObjectRemover(min_size),
                    GridOversizedObjectRemover(),
                    min_residual_reducer,
                    MaskFill()
                ],
                meas=[
                    MeasureShape(),
                    MeasureColor(),
                    MeasureTexture(scale=scale),
                    MeasureIntensity()
                ], benchmark=benchmark
        )
