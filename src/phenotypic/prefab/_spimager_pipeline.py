from phenotypic.abc_ import PrefabPipeline
from phenotypic.correction import CropImage
from phenotypic.correction import DenoiseBlockMatch
from phenotypic.enhance import SubtractGaussian
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize


class SpImagerPipeline(PrefabPipeline):
    """
    A prefabricated pipeline for light image processing task for images from
    the S&P Robotics Imager
    """

    def __init__(self):
        super().__init__(
            ops=[
                CropImage(left=650, right=650, top=600, bottom=600),
                DenoiseBlockMatch(),
                SubtractGaussian(sigma=500),
                OtsuDetector(),
            ],
            meas=[MeasureSize()],
        )
