from typing import Literal

import numpy as np

from phenotypic.abc_ import PrefabPipeline
from phenotypic.correction import GridAligner
from phenotypic.detect import WatershedDetector
from phenotypic.enhance import CLAHE, GaussianBlur, MedianEnhancer
from phenotypic.grid import (GridOversizedObjectRemover, MinResidualErrorReducer)
from phenotypic.measure import MeasureColor, MeasureIntensity, MeasureShape, MeasureTexture
from phenotypic.refine import BorderObjectRemover, MaskFill, LowCircularityRemover


class HeavyWatershedPipeline(PrefabPipeline):
    """
    Provides an image processing pipeline with robust preprocessing/post-processing and watershed segmentation.

    This class defines a sequence of operations and measurements designed for
    image analysis. It includes smoothing, enhancement, segmentation, border
    object removal, and various analysis steps. The pipeline is highly customizable
    for tasks such as image segmentation and feature extraction, making it suitable
    for applications involving image quantification and preprocessing.

    Note:
        This pipeline uses computationally intensive operations aimed at cases where there is heavy background noise

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
                 border_remover_size: int = 25,
                 circularity_cutoff: float = 0.5,
                 texture_scale: int = 5,
                 texture_warn: bool = False,
                 benchmark: bool = False, **kwargs):
        """
        Initializes an image processing pipeline for various image analysis tasks such as object detection,
        segmentation, and measurement. This pipeline uses a combination of operations, including filtering,
        segmentation, and morphological processing, followed by shape, intensity, texture, and color
        measurements.

        Args:
            gaussian_sigma (int, optional): Standard deviation for Gaussian blur filter. Defaults to 5.
            gaussian_mode (str, optional): Mode parameter for Gaussian blur filter (e.g., 'reflect').
                Defaults to 'reflect'.
            gaussian_truncate (float, optional): Truncate value for Gaussian kernel to limit its size.
                Defaults to 4.0.
            watershed_footprint (Literal['auto'] | np.ndarray | int | None, optional): Footprint size or
                structure for the watershed algorithm. Defaults to None.
            watershed_min_size (int, optional): Minimum size of the objects to be retained after watershed
                segmentation. Defaults to 50.
            watershed_compactness (float, optional): Compactness parameter for the watershed algorithm to
                control how tightly regions are formed. Defaults to 0.001.
            watershed_connectivity (int, optional): Connectivity parameter for region connectivity in
                watershed segmentation. Defaults to 1.
            watershed_relabel (bool, optional): Whether to relabel the regions after watershed segmentation.
                Defaults to True.
            watershed_ignore_zeros (bool, optional): Whether to ignore zero-valued regions in the watershed
                algorithm. Defaults to True.
            border_remover_size (int, optional): Size of the border in pixels to be removed during border
                object removal. Defaults to 25.
            circularity_cutoff (float, optional): Threshold for object circularity below which objects will
                be removed. Defaults to 0.5.
            texture_scale (int, optional): Scale parameter for texture measurement. Defaults to 5.
            texture_warn (bool, optional): Whether to issue warnings for invalid texture measurements.
                Defaults to False.
            benchmark (bool, optional): Whether to enable benchmarking of pipeline performance.
                Defaults to False.
            **kwargs: Additional keyword arguments for parent class initialization.
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
            LowCircularityRemover(cutoff=circularity_cutoff),
            min_residual_reducer,
            GridAligner(),
            watershed_detector,
            min_residual_reducer,
            border_remover,
            LowCircularityRemover(cutoff=circularity_cutoff),
            MaskFill()
        ]

        meas = [
            MeasureShape(),
            MeasureIntensity(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
            MeasureColor(),
        ]
        super().__init__(ops=ops, meas=meas, benchmark=benchmark, **kwargs)
