from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from skimage.measure import regionprops_table

from phenotypic.abstract import MeasureFeatures

from ..util.constants_ import OBJECT, BBOX

class MeasureBounds(MeasureFeatures):
    """
    Extracts the object boundary coordinate info within the image using the object map
    """

    def _operate(self, image: Image) -> pd.DataFrame:
        results = pd.DataFrame(
            data=regionprops_table(
                label_image=image.objmap[:],
                properties=['label', 'centroid', 'bbox']
            )
        ).rename(columns={
            'label': OBJECT.LABEL,
            'centroid-0': str(BBOX.CENTER_RR),
            'centroid-1': str(BBOX.CENTER_CC),
            'bbox-0': str(BBOX.MIN_RR),
            'bbox-1': str(BBOX.MIN_CC),
            'bbox-2': str(BBOX.MAX_RR),
            'bbox-3': str(BBOX.MAX_CC),
        }).set_index(keys=OBJECT.LABEL)

        return results

MeasureBounds.__doc__ = BBOX.append_rst_to_doc(MeasureBounds)