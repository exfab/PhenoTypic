import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
import math

from .. import Image
from ..abstract import MapModifier
from ..util.constants_ import OBJECT_INFO


class LowCircularityRemover(MapModifier):
    """
    LowCircularityRemover is a map modifier that removes objects in an image
    based on their circularity measurement.

    This class evaluates objects in an image using the Polsby-Popper score to
    calculate circularity. Objects with a circularity score below the specified
    cutoff value are removed from the image. The user can specify a `cutoff` value
    to determine the minimum acceptable circularity for objects to remain in the
    image. Objects meeting the cutoff criteria are preserved, while others are
    filtered out.

    The Polsby-Popper circularity score is calculated using the following formula:

    .. math::

        C = \\frac{4\\pi A}{P^2}

    where:
        - C is the circularity score (ranging from 0 to 1)
        - A is the area of the object
        - P is the perimeter of the object
        - π (pi) is the mathematical constant

    A perfect circle has a score of 1, while more complex or elongated shapes have
    lower scores approaching 0.

    Attributes:
        cutoff (float): The minimum threshold for the circularity score of
            objects. Must be a value between 0 and 1.
    """
    def __init__(self, cutoff: float = 0.785):
        if cutoff < 0 or cutoff > 1: raise ValueError('threshold should be a number between 0 and 1.')
        self.cutoff = cutoff

    def _operate(self, image: Image) -> Image:
        # Create intial measurement table
        table = (pd.DataFrame(regionprops_table(label_image=image.objmap[:], intensity_image=image.matrix[:],
                                                properties=['label', 'area', 'perimeter']
                                                )
                              )
                 .rename(columns={'label': OBJECT_INFO.OBJECT_LABELS})
                 .set_index(OBJECT_INFO.OBJECT_LABELS))

        # Calculate circularity based on Polsby-Popper Score
        table['circularity'] = (4 * math.pi * table['area']) / (table['perimeter'] ** 2)

        passing_objects = table[table['circularity'] > self.cutoff]
        failed_object_boolean_indices = ~(np.isin(element=image.objmap[:], test_elements=passing_objects.index.to_numpy()))
        image.objmap[failed_object_boolean_indices] = 0
        return image
