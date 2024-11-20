import pandas as pd
from skimage.measure import regionprops_table

from .. import Image
from ..interface import FeatureExtractor


class BoundaryExtractor(FeatureExtractor):
    """
    Extracts the object boundary coordinate info within the image using the object map
    """

    LABEL_OBJ_MAP_ID = 'label'
    LABEL_CENTER_RR = 'Bbox_CenterRR'
    LABEL_MIN_RR = 'Bbox_MinRR'
    LABEL_MAX_RR = 'Bbox_MaxRR'

    LABEL_CENTER_CC = 'Bbox_CenterCC'
    LABEL_MIN_CC = 'Bbox_MinCC'
    LABEL_MAX_CC = 'Bbox_MaxCC'

    def _operate(self, image: Image) -> pd.DataFrame:
        results = pd.DataFrame(regionprops_table(
                label_image=image.object_map,
                properties=['label', 'centroid', 'bbox']
        )).set_index('label')

        results.rename(columns={
            'centroid-0': self.LABEL_CENTER_RR,
            'centroid-1': self.LABEL_CENTER_CC,
            'bbox-0'    : self.LABEL_MIN_RR,
            'bbox-1'    : self.LABEL_MIN_CC,
            'bbox-2'    : self.LABEL_MAX_RR,
            'bbox-3'    : self.LABEL_MAX_CC,
        }, inplace=True)

        return results
