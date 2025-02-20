import pandas as pd

from ._object_detector import ObjectDetector
from ._image_preprocessor import ImagePreprocessor
from ._feature_extractor import FeatureExtractor
from ._object_filter import ObjectFilter
from ._map_modifier import MapModifier

from .. import Image
from ..util.constants import INTERFACE_ERROR_MSG


class ObjectProfiler:
    def __init__(
            self,
            detector: ObjectDetector,
            preprocessor: ImagePreprocessor = None,
            modifier: MapModifier = None,
            measurer: FeatureExtractor = None,
            measurement_filter: ObjectFilter = None
    ):
        self._object_table = pd.DataFrame(
                data={
                    'Location_CenterRR': [],
                    'Location_CenterCC': [],
                    'Boundary_Radius'  : []
                }
        )

        self._detector: ObjectDetector = detector
        self._preprocessor: ImagePreprocessor = preprocessor
        self._object_modifier: MapModifier = modifier
        self._measurer: FeatureExtractor = measurer
        self._measurement_filter: ObjectFilter = measurement_filter

    def profile(self, image: Image) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
