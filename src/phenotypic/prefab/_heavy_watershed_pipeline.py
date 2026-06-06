from typing import Literal

import numpy as np

from phenotypic.abc_ import PrefabPipeline
from phenotypic.correction import GridAligner
from phenotypic.detect import WatershedDetector
from phenotypic.enhance import EnhanceLocalContrast, GaussianBlur, MedianFilter
from phenotypic.measure import (
    MeasureColor,
    MeasureIntensity,
    MeasureShape,
    MeasureTexture,
)
from phenotypic.refine import (
    RemoveBorderObjects,
    MaskFill,
    RemoveLowCircularity,
    GridOversizedObjectRemover,
    ReduceSectionsByLine,
)


class HeavyWatershedPipeline(PrefabPipeline):
    """Detect and measure colonies using watershed segmentation for touching colonies.

    A robust pipeline that uses watershed region-growing to separate touching
    or overlapping colonies that single-threshold methods merge into one object.
    Includes multi-stage preprocessing, morphological refinement, and grid
    alignment.

    Steps:
        1. GaussianBlur — smooth noise
        2. EnhanceLocalContrast — boost local contrast
        3. MedianFilter — remove residual speckle
        4. WatershedDetector — region-growing segmentation
        5. RemoveBorderObjects — remove partial edge colonies
        6. RemoveLowCircularity — remove low-circularity artifacts
        7. GridOversizedObjectRemover — remove merged multi-well objects
        8. ReduceSectionsByLine — keep one colony per well
        9. GridAligner — straighten the grid
        10. MaskFill — fill interior holes

    Measurements: MeasureShape, MeasureColor, MeasureTexture, MeasureIntensity.

    Best For:
        - Dense plates where colonies touch or overlap.
        - Late time-point plates with large, merging colonies.
        - Plates where Otsu detection merges adjacent colonies into single objects.

    Consider Also:
        - :class:`HeavyOtsuPipeline` when colonies are well-separated.
        - :class:`RoundPeaksPipeline` for a faster approach on round,
          non-touching colonies.

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
            watershed_footprint: Literal["auto"] | np.ndarray | int | None = None,
            watershed_min_size: int = 50,
            watershed_compactness: float = 0.001,
            watershed_connectivity: int = 1,
            watershed_relabel: bool = True,
            watershed_ignore_zeros: bool = True,
            border_remover_size: int = 25,
            circularity_cutoff: float = 0.5,
            texture_scale: int = 5,
            texture_warn: bool = False,
            benchmark: bool = False,
            **kwargs,
    ):
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

        watershed_detector = WatershedDetector(
                footprint=watershed_footprint,
                min_size=watershed_min_size,
                compactness=watershed_compactness,
                connectivity=watershed_connectivity,
                relabel=watershed_relabel,
                ignore_zeros=watershed_ignore_zeros,
        )
        border_remover = RemoveBorderObjects(border_size=border_remover_size)
        min_residual_reducer = ReduceSectionsByLine()

        ops = [
            GaussianBlur(
                    sigma=gaussian_sigma, mode=gaussian_mode, truncate=gaussian_truncate
            ),
            EnhanceLocalContrast(),
            MedianFilter(),
            watershed_detector,
            border_remover,
            GridOversizedObjectRemover(),
            RemoveLowCircularity(cutoff=circularity_cutoff),
            min_residual_reducer,
            GridAligner(),
            watershed_detector,
            GridOversizedObjectRemover(),
            min_residual_reducer,
            border_remover,
            RemoveLowCircularity(cutoff=circularity_cutoff),
            MaskFill(),
        ]

        meas = [
            MeasureShape(),
            MeasureIntensity(),
            MeasureTexture(scale=texture_scale, warn=texture_warn),
            MeasureColor(),
        ]
        super().__init__(ops=ops, meas=meas, benchmark=benchmark, **kwargs)
