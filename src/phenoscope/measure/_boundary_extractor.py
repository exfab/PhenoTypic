from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

import pandas as pd
from skimage.measure import regionprops_table

from phenoscope.abstract import FeatureExtractor

from ..util.constants_ import OBJECT_INFO

class BoundaryExtractor(FeatureExtractor):
    """
    Extracts the object boundary coordinate info within the image using the object map
    """

    def _operate(self, image: Image) -> pd.DataFrame:
        results = pd.DataFrame(
            data=regionprops_table(
                label_image=image.omap[:],
                properties=['label', 'centroid', 'bbox']
            )
        ).rename(columns={
            'label': OBJECT_INFO.OBJECT_LABELS,
            'centroid-0': OBJECT_INFO.CENTER_RR,
            'centroid-1': OBJECT_INFO.CENTER_CC,
            'bbox-0': OBJECT_INFO.MIN_RR,
            'bbox-1': OBJECT_INFO.MIN_CC,
            'bbox-2': OBJECT_INFO.MAX_RR,
            'bbox-3': OBJECT_INFO.MAX_CC,
        }).set_index(keys=OBJECT_INFO.OBJECT_LABELS)

        return results
