import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

from ..interface import ObjectProfiler, ThresholdDetector, ImagePreprocessor, MorphologyMorpher, FeatureExtractor, ObjectFilter
from ..detection import OtsuDetector


class ThresholdProfiler(ObjectProfiler):
    def __init__(self,
                 detector: ThresholdDetector = None,
                 preprocessor: ImagePreprocessor = None,
                 morpher: MorphologyMorpher = None,
                 measurer: FeatureExtractor = None,
                 measurement_filter: ObjectFilter = None
                 ):
        if detector is None: detector = OtsuDetector()

        if type(detector) is not ThresholdDetector: raise ValueError('For a ThresholdProfiler, detector should be a ThresholdDetector')

        super().__init__(
                detector=detector,
                preprocessor=preprocessor,
                morpher=morpher,
                measurer=measurer,
                measurement_filter=measurement_filter
        )

        # Binary Object Mask
        self._object_mask: np.ndarray = None

        # Labelled Object Mask
        self._object_map: np.ndarray = None

    def profile(self, image: np.ndarray) -> pd.DataFrame:
        if self._preprocessor is not None: image = self._preprocessor.preprocess(image)

        self._object_table = self._measure_object(image)

        if self._measurement_filter is not None:
            self._object_table = self._measurement_filter.filter(self._object_table)
            self._object_mask = np.isin(element=self._object_mask,
                                        test_elements=self._object_table.index.to_numpy())

    def _measure_object(self, image: np.ndarray):
        if self._object_mask is None: self._object_mask = self._detector.detect(image)

        if self._morpher is not None: self._object_mask = self._morpher.morph(image)

        self._object_map = label(self._object_mask)
        table = pd.DataFrame(regionprops_table(
                label_image=self._object_map,
                intensity_image=image,
                properties=['label', 'centroid', 'equivalent_diameter_area']
        ))
        table.index.name = 'object'
        table.rename(columns={
            'label'                   : 'map_value',
            'centroid-0'              : 'Location_CenterRR',
            'centroid-1'              : 'Location_CenterCC',
            'equivalent_diameter_area': 'Boundary_Radius',
        }, inplace=True)
        table.loc[:, 'Boundary_Radius'] = table.loc[:, 'Boundary_Radius'] / 2

        if self._measurer is not None:
            self._object_map, self._object_table = self._measurer.extract(self._object_map, self._object_table)
        return table
