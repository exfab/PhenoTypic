from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic.tools.constants_ import OBJECT

if TYPE_CHECKING: from phenotypic import Image

import warnings
import pandas as pd
from scipy.spatial import ConvexHull, qhull
from scipy.ndimage import distance_transform_edt
import numpy as np

from phenotypic.abc_ import MeasurementInfo, MeasureFeatures


class SHAPE(MeasurementInfo):
    """The labels and descriptions of the shape measurements."""

    @classmethod
    def category(cls):
        return 'Shape'

    AREA = ('Area',
            "Total number of pixels occupied by the microbial colony. Represents colony biomass and growth extent on agar plates. Larger areas typically indicate more robust growth or longer incubation times.")
    PERIMETER = ('Perimeter',
                 "Total length of the colony's outer boundary in pixels. Measures colony edge complexity and surface irregularity. Smooth, circular colonies have shorter perimeters relative to their area compared to irregular or filamentous colonies.")
    CIRCULARITY = (
        'Circularity',
        r'Calculated as :math:`\frac{4\pi*\text{Area}}{\text{Perimeter}^2}`. Measures how closely a colony approximates a perfect circle (value = 1). Values < 1 indicate irregular colony morphology, which may result from genetic mutations, environmental stress, or mixed microbial populations on agar plates.'
    )
    CONVEX_AREA = ('ConvexArea',
                   'Area of the smallest convex polygon that completely contains the colony. Represents the colony\'s "filled-in" appearance if all indentations and holes were removed. Useful for detecting colony spreading patterns or invasive growth characteristics.')
    MEDIAN_RADIUS = ('MedianRadius',
                     'Median distance from colony center to edge across all directions. Provides a robust measure of typical colony size that is less sensitive to outliers than mean radius. Particularly useful for colonies with uneven growth or sectoring.')
    MEAN_RADIUS = ('MeanRadius',
                   'Average distance from colony center to edge across all directions. Represents overall colony expansion rate. In arrayed growth assays, this correlates with microbial fitness and growth kinetics under controlled conditions.')
    MAX_RADIUS = ('MaxRadius',
                  'Maximum distance from colony center to edge across all directions. Represents the furthest extent of colony growth from its center. In arrayed microbial assays, this measurement helps identify asymmetric growth patterns or colonies extending toward neighboring positions.')
    MIN_FERET_DIAMETER = ('MinFeretDiameter',
                          'Minimum caliper diameter - the shortest distance between two parallel tangent lines touching opposite sides of the colony. Represents the narrowest dimension of the colony regardless of orientation. Useful for detecting elongated or irregular colony morphologies and measuring colony width.')
    MAX_FERET_DIAMETER = ('MaxFeretDiameter',
                          'Maximum caliper diameter - the longest distance between two parallel tangent lines touching opposite sides of the colony. Represents the maximum dimension of the colony regardless of orientation. Often exceeds major axis length for irregular shapes and helps quantify maximum colony extent.')
    ECCENTRICITY = ('Eccentricity',
                    'Measure of colony elongation, ranging from 0 (perfect circle) to 1 (highly elongated). Values near 0 indicate compact, radially symmetric growth typical of healthy bacterial colonies, while higher values may suggest directional growth, motility, or environmental gradients on the agar surface.')
    SOLIDITY = ('Solidity',
                'Ratio of actual colony area to its convex hull area (Area/ConvexArea). Values near 1 indicate compact, solid colonies with minimal indentations. Lower values (< 0.9) may indicate invasive growth, colony spreading, or the presence of clearing zones around colonies.')
    EXTENT = ('Extent',
              'Ratio of colony area to its bounding box area (ObjectArea/BboxArea). Measures how efficiently the colony fills its allocated space. Compact colonies have higher extent values, while spread-out or irregular colonies have lower values.')
    BBOX_AREA = ('BboxArea',
                 'Area of the smallest rectangle that completely contains the colony. Represents the total spatial footprint of the colony including any empty space. In high-throughput assays, this helps assess colony positioning and potential interference with neighboring colonies.')
    MAJOR_AXIS_LENGTH = (
        'MajorAxisLength',
        'Length of the longest axis of the ellipse that best fits the colony shape. Represents the maximum colony dimension. In arrayed microbial growth, this measurement helps identify colonies that have grown beyond their intended grid positions.'
    )
    MINOR_AXIS_LENGTH = (
        'MinorAxisLength',
        'Length of the shortest axis of the ellipse that best fits the colony shape. Represents the minimum colony dimension. Together with major axis length, this helps characterize colony aspect ratio and growth anisotropy.'
    )
    COMPACTNESS = (
        'Compactness',
        r'Calculated as :math:`\frac{\text{Perimeter}^2}{4\pi*\text{Area}}`. Inverse of circularity (ranges from 1 for perfect circles to higher values for irregular shapes). Measures colony shape complexity - compact, circular colonies have values near 1, while irregular or filamentous colonies have much higher values.'
    )
    ORIENTATION = ('Orientation',
                   'Angle (in radians) between the colony\'s major axis and the horizontal axis. Measures colony alignment and growth directionality. Random orientations are typical for most bacterial colonies, while consistent orientations may indicate environmental gradients or mechanical stresses during plating.')


class MeasureShape(MeasureFeatures):
    r"""Calculates various geometric measures of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the geometric measures of the objects in the image.

    References:
        1. D. R. Stirling, M. J. Swain-Bowden, A. M. Lucas, A. E. Carpenter, B. A. Cimini, and A. Goodman,
            "CellProfiler 4: improvements in speed, utility and usability," BMC Bioinformatics, vol. 22, no. 1, p. 433, Sep. 2021, doi: 10.1186/s12859-021-04344-9.
        2. "Shape factor (image analysis and microscopy)," Wikipedia. Oct. 09, 2021. Accessed: Apr. 08, 2025. [Online].
            Available: https://en.wikipedia.org/w/index.php?title=Shape_factor_(image_analysis_and_microscopy)&oldid=1048998776

    """

    @staticmethod
    def _calculate_feret_diameters(hull_points: np.ndarray) -> tuple[float, float]:
        """Calculate minimum and maximum Feret diameters from convex hull points.
        
        The Feret diameter is the distance between two parallel lines tangent to the object.
        Maximum Feret diameter: longest distance between any two points on the convex hull.
        Minimum Feret diameter: computed using rotating calipers algorithm to find the 
        minimum width of the object across all orientations.
        
        Args:
            hull_points: Nx2 array of coordinates representing convex hull vertices
            
        Returns:
            tuple: (max_feret, min_feret) diameters
        """
        if len(hull_points) < 2:
            return (np.nan, np.nan)

        # Maximum Feret: compute pairwise distances and find maximum
        # This is the straightforward maximum distance between any two hull vertices
        distances = np.sqrt(((hull_points[:, None, :] - hull_points[None, :, :]) ** 2).sum(axis=2))
        max_feret = np.max(distances)

        # Minimum Feret: use rotating calipers algorithm
        # For each edge of the convex hull, calculate perpendicular distance to all other points
        n = len(hull_points)
        min_feret = np.inf

        for i in range(n):
            # Define edge vector from point i to point i+1
            p1 = hull_points[i]
            p2 = hull_points[(i + 1)%n]
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)

            if edge_length == 0:
                continue

            # Normalized perpendicular direction to the edge
            edge_unit = edge/edge_length
            perpendicular = np.array([-edge_unit[1], edge_unit[0]])

            # Project all hull points onto the perpendicular direction
            projections = np.dot(hull_points - p1, perpendicular)

            # The width in this direction is the range of projections
            width = np.max(projections) - np.min(projections)
            min_feret = min(min_feret, width)

        return (max_feret, min_feret)

    def _operate(self, image: Image) -> pd.DataFrame:
        # Create empty numpy arrays to store measurements
        measurements = {str(feature): np.zeros(shape=image.num_objects) for feature in SHAPE if
                        feature != SHAPE.CATEGORY}

        # Calculate radius-based measurements using distance transform
        # Distance transform gives the distance from each object pixel to the nearest background pixel
        dist_matrix = distance_transform_edt(image.objmap[:])
        measurements[str(SHAPE.MEAN_RADIUS)] = self._calculate_mean(array=dist_matrix, objmap=image.objmap[:])
        measurements[str(SHAPE.MEDIAN_RADIUS)] = self._calculate_median(array=dist_matrix, objmap=image.objmap[:])
        measurements[str(SHAPE.MAX_RADIUS)] = self._calculate_max(array=dist_matrix, objmap=image.objmap[:])

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

            numer = 4*np.pi*current_props.area
            denom = current_props.perimeter ** 2

            measurements[str(SHAPE.CIRCULARITY)][idx] = numer/denom if denom != 0 else np.nan
            measurements[str(SHAPE.COMPACTNESS)][idx] = denom/numer if numer != 0 else np.nan

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message='Qhull')
                    convex_hull = ConvexHull(current_props.coords)

            except qhull.QhullError:
                convex_hull = None

            measurements[str(SHAPE.CONVEX_AREA)][idx] = (convex_hull.area if convex_hull else np.nan)
            measurements[str(SHAPE.SOLIDITY)][idx] = ((current_props.area/convex_hull.area) if convex_hull else np.nan)

            # Calculate Feret diameters using convex hull vertices if available
            # Feret diameter is the distance between two parallel tangent lines
            if convex_hull is not None:
                # Get convex hull vertices (actual coordinate points)
                hull_points = current_props.coords[convex_hull.vertices]

                # Maximum Feret: longest distance between any two points on the convex hull
                max_feret, min_feret = self._calculate_feret_diameters(hull_points)
                measurements[str(SHAPE.MAX_FERET_DIAMETER)][idx] = max_feret
                measurements[str(SHAPE.MIN_FERET_DIAMETER)][idx] = min_feret
            else:
                measurements[str(SHAPE.MAX_FERET_DIAMETER)][idx] = np.nan
                measurements[str(SHAPE.MIN_FERET_DIAMETER)][idx] = np.nan

        measurements = pd.DataFrame(measurements)
        measurements.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return measurements


MeasureShape.__doc__ = SHAPE.append_rst_to_doc(MeasureShape)
