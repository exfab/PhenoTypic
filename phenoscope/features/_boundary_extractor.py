import pandas as pd
from skimage.measure import regionprops_table

from .. import Image
from ..interface import FeatureExtractor

from ..util.constants import C_ObjectInfo

class BoundaryExtractor(FeatureExtractor):
    """
    Extracts the object boundary coordinate info within the image using the object map
    """

    def _operate(self, image: Image) -> pd.DataFrame:
        results = pd.DataFrame(
            data=regionprops_table(
                label_image=image.obj_map[:],
                properties=['label', 'centroid', 'bbox']
            )
        ).rename(columns={
            'label': C_ObjectInfo.OBJECT_LABELS,
            'centroid-0': C_ObjectInfo.CENTER_RR,
            'centroid-1': C_ObjectInfo.CENTER_CC,
            'bbox-0': C_ObjectInfo.MIN_RR,
            'bbox-1': C_ObjectInfo.MIN_CC,
            'bbox-2': C_ObjectInfo.MAX_RR,
            'bbox-3': C_ObjectInfo.MAX_CC,
        }).set_index(keys=C_ObjectInfo.OBJECT_LABELS)

        return results
