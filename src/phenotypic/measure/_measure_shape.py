from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import warnings
import pandas as pd
from scipy.spatial import ConvexHull, qhull
from scipy.ndimage import distance_transform_edt
import numpy as np

from phenotypic.abstract import MeasurementInfo, MeasureFeatures


class SHAPE(MeasurementInfo):
    """The labels and descriptions of the shape measurements."""
    @property
    def CATEGORY(self):
        return 'Shape'

    AREA = ('Area', "The sum of the object's pixels")
    PERIMETER = ('Perimeter', "The perimeter of the object's pixels")
    CIRCULARITY = (
        'Circularity', r'Calculated as :math:`\frac{4\pi*\text{Area}}{\text{Perimeter}^2}`. A perfect circle has a other_image of 1.'
    )
    CONVEX_AREA = ('ConvexArea', 'The area of the convex hull of the object')
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
    COMPACTNESS = (
        'Compactness',
        r'Calculated as :math:`Calculated as \frac{\text{Perimeter}^2}{4\pi*\text{Area}}`. A filled circle will have a value of 1, while irregular or objects with holes have a value greater than 1'
    )
    ORIENTATION = ('Orientation', 'The angle between the major axis and the horizontal axis in radians')


class MeasureShape(MeasureFeatures):
    r"""Calculates various geometric measures of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the geometric measures of the objects in the image.

    References:
        1. D. R. Stirling, M. J. Swain-Bowden, A. M. Lucas, A. E. Carpenter, B. A. Cimini, and A. Goodman,
            “CellProfiler 4: improvements in speed, utility and usability,” BMC Bioinformatics, vol. 22, no. 1, p. 433, Sep. 2021, doi: 10.1186/s12859-021-04344-9.
        2. “Shape factor (image analysis and microscopy),” Wikipedia. Oct. 09, 2021. Accessed: Apr. 08, 2025. [Online].
            Available: https://en.wikipedia.org/w/index.php?title=Shape_factor_(image_analysis_and_microscopy)&oldid=1048998776

    """

    def _operate(self, image: Image) -> pd.DataFrame:
        # Create empty numpy arrays to store measurements
        measurements = {str(feature): np.zeros(shape=image.num_objects) for feature in SHAPE if feature != SHAPE.CATEGORY}

        dist_matrix = distance_transform_edt(image.objmap[:])
        measurements[str(SHAPE.MEAN_RADIUS)] = self._calculate_mean(array=dist_matrix, labels=image.objmap[:])
        measurements[str(SHAPE.MEDIAN_RADIUS)] = self._calculate_median(array=dist_matrix, labels=image.objmap[:])

        obj_props = image.objects.props
        for idx, obj_image in enumerate(image.objects):
            current_props = obj_props[idx]
            measurements[str(SHAPE.AREA)][idx] = current_props.area
            measurements[str(SHAPE.PERIMETER)][idx] = current_props.perimeter
            measurements[str(SHAPE.ECCENTRICITY)][idx] = current_props.eccentricity
            measurements[str(SHAPE.EXTENT)][idx] = current_props.extent
            measurements[str(SHAPE.BBOX_AREA)][idx] = current_props.area_bbox
            measurements[str(SHAPE.MAJOR_AXIS_LENGTH)][idx] = current_props.major_axis_length
            measurements[str(SHAPE.MINOR_AXIS_LENGTH)][idx] = current_props.minor_axis_length
            measurements[str(SHAPE.ORIENTATION)][idx] = current_props.orientation

            numer = 4 * np.pi * current_props.area
            denom = current_props.perimeter ** 2

            measurements[str(SHAPE.CIRCULARITY)][idx] = numer / denom if denom != 0 else np.nan
            measurements[str(SHAPE.COMPACTNESS)][idx] = denom / numer if numer != 0 else np.nan

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message='Qhull')
                    convex_hull = ConvexHull(current_props.coords)


            except qhull.QhullError:
                convex_hull = None

            measurements[str(SHAPE.CONVEX_AREA)][idx] = (convex_hull.area if convex_hull else np.nan)
            measurements[str(SHAPE.SOLIDITY)][idx] = ((current_props.area / convex_hull.area) if convex_hull else np.nan)

        return pd.DataFrame(measurements, index=image.objects.labels2series())

MeasureShape.__doc__ = SHAPE.append_rst_to_doc(MeasureShape)