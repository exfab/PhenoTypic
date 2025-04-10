from scipy.spatial.distance import euclidean

from ..abstract import MapModifier
from .. import Image
from ..measure import BoundaryMeasure


class CenterDeviationReducer(MapModifier):
    """Removes objects based on how far away they are from the center of the image.

    Useful for isolated colony images

    """

    def _operate(self, image: Image):
        img_center_cc = round(image.shape[1] / 2)
        img_center_rr = round(image.shape[0] / 2)

        bound_extractor = BoundaryMeasure()
        bound_info = bound_extractor.measure(image)

        # bound_info.loc[:, 'Measurement_CenterDeviation'] = bound_info.apply(
        #         lambda row: print(row),
        #         axis=0)

        bound_info.loc[:, 'Measurement_CenterDeviation'] = bound_info.apply(
                lambda row: euclidean(u=[row[bound_extractor.LABEL_CENTER_CC], row[bound_extractor.LABEL_CENTER_RR]],
                                      v=[img_center_cc, img_center_rr]),
                axis=1)

        # Get the label of the obj w/ the least deviation
        obj_to_keep = bound_info.loc[:,'Measurement_CenterDeviation'].idxmin()

        # Get a working copy of the object map
        obj_map = image.objmap

        # Set other objects to background
        obj_map[obj_map!=obj_to_keep] = 0

        # Set Image object map to new value
        image.objmap = obj_map

        return image


