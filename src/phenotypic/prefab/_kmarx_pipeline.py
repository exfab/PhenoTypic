from phenotypic import ImagePipeline

from phenotypic.enhancement import CLAHE, GaussianSmoother, MedianEnhancer, ContrastStretching
from phenotypic.detection import OtsuDetector, WatershedDetector
from phenotypic.correction import GridAligner
from phenotypic.grid import (MinResidualErrorReducer, GridOversizedObjectRemover)
from phenotypic.objects import BorderObjectRemover, SmallObjectRemover, LowCircularityRemover
from phenotypic.morphology import MaskFill
from phenotypic.measure import MeasureIntensity, MeasureShape, MeasureTexture, MeasureColor


class KmarxPipeline(ImagePipeline):
    def __init__(self, sigma=5, footprint='auto', min_size=50, compactness=0.001):
        super().__init__(
            ops=[
                GaussianSmoother(sigma=sigma),
                CLAHE(),
                MedianEnhancer(),
                WatershedDetector(
                    footprint=footprint,
                    min_size=min_size,
                    compactness=compactness,
                ),
                BorderObjectRemover(),
                GridOversizedObjectRemover(),
                MinResidualErrorReducer(),
                GridAligner(),
                WatershedDetector(footprint=footprint, min_size=min_size, compactness=compactness),
                MaskFill()
            ],
            measurements=[
                MeasureShape(),
                MeasureColor(),
                MeasureTexture(),
                MeasureIntensity()
            ],
        )
