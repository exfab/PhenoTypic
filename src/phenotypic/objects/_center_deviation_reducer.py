from scipy.spatial.distance import euclidean

from phenotypic.abstract import MapModifier
from phenotypic import Image
from phenotypic.util.constants_ import OBJECT, BBOX


class CenterDeviationReducer(MapModifier):
    """Removes objects based on how far away they are from the center of the image.

    Useful for isolated colony images

    """

    def _operate(self, image: Image):
        img_center_cc = image.shape[1] // 2
        img_center_rr = image.shape[0] // 2

        bound_info = image.objects.info()

        # Add a column to the bound info for center deviation
        bound_info.loc[:, 'Measurement_CenterDeviation'] = bound_info.apply(
            lambda row: euclidean(u=[row[str(BBOX.CENTER_CC)], row[str(BBOX.CENTER_RR)]],
                                  v=[img_center_cc, img_center_rr]
                                  ),
            axis=1
        )

        # Get the label of the obj w/ the least deviation
        obj_to_keep = bound_info.loc[:, 'Measurement_CenterDeviation'].idxmin()

        # Get a working copy of the object map
        objmap = image.objmap[:]

        # Set Image object map to new other_image
        image.objmap[objmap != obj_to_keep] = 0

        return image
