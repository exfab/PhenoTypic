import pandas as pd
from skimage.measure import regionprops_table

from .. import Image
from ..interface import FeatureExtractor


class BoundaryExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def _operate(self, image: Image) -> pd.DataFrame:
        results = pd.DataFrame(regionprops_table(
                label_image=image.object_map,
                intensity_image=image.array,
                properties=['label', 'centroid', 'bbox']
        )).set_index('label')

        results.rename(columns={
            'centroid-0': 'center_rr',
            'centroid-1': 'center_cc',
            'bbox-0'    : 'min_rr',
            'bbox-1'    : 'min_cc',
            'bbox-2'    : 'max_rr',
            'bbox-3'    : 'max_cc',
        }, inplace=True)

        return results
