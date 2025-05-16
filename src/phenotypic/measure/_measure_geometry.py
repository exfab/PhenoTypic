from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt
import numpy as np

from phenotypic.abstract import FeatureMeasure

from phenotypic.util.constants_ import SHAPE


class MeasureShape(FeatureMeasure):
    r"""Calculates various geometric measures of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the geometric measures of the objects in the image.

    Notes:
        Area: The sum of the individual pixel's in the object's footprint
        Perimeter: The length of the object's boundary
        Circularity: Calculated as :math:`\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}`
        ConvexArea: The area of the convex hull of the object

    References:
        1. D. R. Stirling, M. J. Swain-Bowden, A. M. Lucas, A. E. Carpenter, B. A. Cimini, and A. Goodman,
            “CellProfiler 4: improvements in speed, utility and usability,” BMC Bioinformatics, vol. 22, no. 1, p. 433, Sep. 2021, doi: 10.1186/s12859-021-04344-9.
        2. “Shape factor (image analysis and microscopy),” Wikipedia. Oct. 09, 2021. Accessed: Apr. 08, 2025. [Online].
            Available: https://en.wikipedia.org/w/index.php?title=Shape_factor_(image_analysis_and_microscopy)&oldid=1048998776

    """

    @staticmethod
    def _operate(image: Image) -> pd.DataFrame:
        measurements = {
            str(SHAPE.AREA): [],
            str(SHAPE.PERIMETER): [],
            str(SHAPE.CIRCULARITY): [],
            str(SHAPE.CONVEX_AREA): [],
            str(SHAPE.MEAN_RADIUS): [],
            str(SHAPE.MEDIAN_RADIUS): []
        }
        obj_props = image.objects.props
        for idx, obj_image in enumerate(image.objects):
            measurements[str(SHAPE.AREA)].append(obj_props[idx].area)
            measurements[str(SHAPE.PERIMETER)].append(obj_props[idx].perimeter)

            circularity = (4 * np.pi * obj_props[idx].area) / (obj_props[idx].perimeter ** 2)
            measurements[str(SHAPE.CIRCULARITY)].append(circularity)

            convex_hull = ConvexHull(obj_props[idx].coords)
            measurements[str(SHAPE.CONVEX_AREA)].append(convex_hull.area)

            # TODO: Alter so that calculations are made simultaneously instead of iterating through each object
            dist_matrix = distance_transform_edt(obj_image.objmap[:])
            measurements[str(SHAPE.MEAN_RADIUS)].append(np.mean(dist_matrix))
            measurements[str(SHAPE.MEDIAN_RADIUS)].append(np.median(dist_matrix))

        return pd.DataFrame(measurements, index=image.objects.get_labels_series())
