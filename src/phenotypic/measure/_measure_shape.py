from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from scipy.spatial import ConvexHull
from scipy.ndimage import distance_transform_edt
import numpy as np

from phenotypic.abstract import MeasureFeatures


class SHAPE(Enum):
    """The labels and descriptions of the shape measurements."""
    CATEGORY = ('Shape', 'The category of the measurements')

    AREA = ('Area', "The sum of the object's pixels")
    PERIMETER = ('Perimeter', "The perimeter of the object's pixels")
    CIRCULARITY = ('Circularity', r'Calculated as :math:`\frac{4\pi*Area}{Perimeter^2}`. A perfect circle has a other_image of 1.')
    CONVEX_AREA = ('ConvexArea', 'The area of the convex hull of the object')
    ORIENTATION = ('Orientation', 'The orientation of the object in degrees')
    MEDIAN_RADIUS = ('MedianRadius', 'The median radius of the object')
    MEAN_RADIUS = ('MeanRadius', 'The mean radius of the object')
    ECCENTRICITY = ('Eccentricity', 'The eccentricity of the object')
    SOLIDITY = ('Solidity', 'The object Area/ConvexArea')
    EXTENT = ('Extent', 'The proportion of object pixels to the bounding box. ObjectArea/BboxArea')
    BBOX_AREA = ('BboxArea', 'The area of the bounding box of the object')
    MAJOR_AXIS_LENGTH = (
        'MajorAxisLength',
        'The length of the major axis of the ellipse that has the same normalized central moments as the object'
    )
    MINOR_AXIS_LENGTH = (
        'MinorAxisLength',
        'The length of the minor axis of the ellipse that has the same normalized central moments as the object'
    )

    def __init__(self, label, desc=None):
        self.label, self.desc = label, desc

    def __str__(self):
        return f'{self.CATEGORY.label}_{self.label}'

    def get_labels(self):
        return [key.label for key in SHAPE if key.label != self.CATEGORY.label]


class MeasureShape(MeasureFeatures):
    r"""Calculates various geometric measures of the objects in the _parent_image.

    Returns:
        pd.DataFrame: A dataframe containing the geometric measures of the objects in the _parent_image.

    Notes:
        Area: The sum of the individual pixel's in the object's footprint
        Perimeter: The length of the object's boundary
        Circularity: Calculated as :math:`\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}`
        ConvexArea: The area of the convex hull of the object

    References:
        1. D. R. Stirling, M. J. Swain-Bowden, A. M. Lucas, A. E. Carpenter, B. A. Cimini, and A. Goodman,
            “CellProfiler 4: improvements in speed, utility and usability,” BMC Bioinformatics, vol. 22, no. 1, p. 433, Sep. 2021, doi: 10.1186/s12859-021-04344-9.
        2. “Shape factor (_parent_image analysis and microscopy),” Wikipedia. Oct. 09, 2021. Accessed: Apr. 08, 2025. [Online].
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
            str(SHAPE.MEDIAN_RADIUS): [],
            str(SHAPE.ECCENTRICITY): [],
            str(SHAPE.SOLIDITY): [],
            str(SHAPE.EXTENT): [],
            str(SHAPE.BBOX_AREA): [],
            str(SHAPE.MAJOR_AXIS_LENGTH): [],
            str(SHAPE.MINOR_AXIS_LENGTH): []
        }
        obj_props = image.objects.props
        for idx, obj_image in enumerate(image.objects):
            current_props = obj_props[idx]
            measurements[str(SHAPE.AREA)].append(current_props.area)
            measurements[str(SHAPE.PERIMETER)].append(current_props.perimeter)
            measurements[str(SHAPE.ECCENTRICITY)].append(current_props.eccentricity)
            measurements[str(SHAPE.EXTENT)].append(current_props.extent)
            measurements[str(SHAPE.BBOX_AREA)].append(current_props.area_bbox)
            measurements[str(SHAPE.MAJOR_AXIS_LENGTH)].append(current_props.major_axis_length)
            measurements[str(SHAPE.MINOR_AXIS_LENGTH)].append(current_props.minor_axis_length)

            circularity = (4 * np.pi * obj_props[idx].area) / (current_props.perimeter ** 2)
            measurements[str(SHAPE.CIRCULARITY)].append(circularity)

            try:
                if current_props.area >= 3:
                    convex_hull = ConvexHull(current_props.coords)
                else:
                    convex_hull = None

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                warnings.warn(f'Error in computing convex hull for object {current_props.label}: {e}')
                convex_hull = None

            measurements[str(SHAPE.CONVEX_AREA)].append(convex_hull.area if convex_hull else np.nan)
            measurements[str(SHAPE.SOLIDITY)].append((current_props.area / convex_hull.area) if convex_hull else np.nan)

            # TODO: Alter so that calculations are made simultaneously instead of iterating through each object
            dist_matrix = distance_transform_edt(obj_image.objmap[:])
            measurements[str(SHAPE.MEAN_RADIUS)].append(np.mean(dist_matrix))
            measurements[str(SHAPE.MEDIAN_RADIUS)].append(np.median(dist_matrix))

        return pd.DataFrame(measurements, index=image.objects.get_labels_series())
