import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
import math

from .. import Image
from ..interface import ObjectModifier


class CircularityRemovalModifier(ObjectModifier):
    def __init__(self, cutoff: float = 0.785):
        if cutoff < 0 or cutoff > 1: raise ValueError('threshold should be a number between 0 and 1.')
        self.__cutoff = cutoff

    def _operate(self, image: Image) -> Image:
        # Create intial measurement table
        table = pd.DataFrame(regionprops_table(label_image=image.object_map, intensity_image=image.array,
                                               properties=['label', 'area', 'perimeter'])).set_index('label')

        # Calculate circularity based on Polsby-Popper Score
        table['circularity'] = (4 * math.pi * table['area']) / (table['perimeter']**2)

        passing_objects = table[table['circularity'] > self.__cutoff]
        failed_object_boolean_indices = ~(np.isin(element=image.object_map, test_elements=passing_objects.index.to_numpy()))
        obj_map = image.object_map
        obj_map[failed_object_boolean_indices] = 0
        image.object_map = obj_map
        return image

