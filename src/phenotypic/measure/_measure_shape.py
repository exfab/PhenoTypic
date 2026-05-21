from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from phenotypic.tools_.constants_ import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import warnings
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import distance_transform_edt
import numpy as np

from phenotypic.abc_ import MeasureFeatures
from ..tools_.measurement_info import SHAPE


class MeasureShape(MeasureFeatures):
    r"""Measure comprehensive morphological characteristics of detected colonies.

    Extract geometric metrics from each colony shape: area, perimeter,
    circularity, convex hull properties, width-based measures, Feret
    diameters, eccentricity, and best-fit ellipse parameters. The output
    DataFrame provides a full morphological profile for phenotypic
    classification and growth-pattern analysis.

    Returns:
        pd.DataFrame: Object-level morphological measurements with
        columns:

            - Label, Area, Perimeter, Circularity, Compactness,
              ConvexArea, Solidity, Extent, BboxArea.
            - MeanRadius, MedianRadius, MaxRadius (distance-transform
              based).
            - MinFeretDiameter, MaxFeretDiameter (caliper diameters).
            - MajorAxisLength, MinorAxisLength, Eccentricity,
              Orientation.

    Best For:
        - Distinguishing colony morphotypes (smooth circular wild-type
          vs wrinkled, branching, or invasive mutants).
        - Assessing growth symmetry and directionality via eccentricity
          and orientation.
        - Detecting invasive or spreading growth through low solidity
          values.
        - Morphological clustering for automated strain identification.

    Consider Also:
        - :class:`MeasureSize` for a lightweight area-only measurement
          when full morphology is not needed.
        - :class:`MeasureTexture` for surface roughness and pattern
          features that complement shape metrics.
        - :class:`MeasureBounds` for bounding box and centroid data
          without shape statistics.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting shape metrics in a biological context.
    """

    _measurement_infoclass: ClassVar[type] = SHAPE

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
        distances = np.sqrt(
                ((hull_points[:, None, :] - hull_points[None, :, :]) ** 2).sum(axis=2)
        )
        max_feret = np.max(distances)

        # Minimum Feret: use rotating calipers algorithm
        # For each edge of the convex hull, calculate perpendicular distance to all other points
        n = len(hull_points)
        min_feret = np.inf

        for i in range(n):
            # Define edge vector from point i to point i+1
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n]
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)

            if edge_length == 0:
                continue

            # Normalized perpendicular direction to the edge
            edge_unit = edge / edge_length
            perpendicular = np.array([-edge_unit[1], edge_unit[0]])

            # Project all hull points onto the perpendicular direction
            projections = np.dot(hull_points - p1, perpendicular)

            # The width in this direction is the range of projections
            width = np.max(projections) - np.min(projections)
            min_feret = min(min_feret, width)

        return (max_feret, min_feret)

    def _operate(self, image: Image) -> pd.DataFrame:
        # Create empty numpy arrays to store measurements
        measurements = {
            str(feature): np.zeros(shape=image.num_objects)
            for feature in SHAPE
            if feature != SHAPE.CATEGORY
        }

        # Calculate width-based measurements using distance transform
        # Distance transform gives the distance from each object pixel to the nearest background pixel
        dist_matrix = distance_transform_edt(image.objmap[:])
        measurements[str(SHAPE.MEAN_RADIUS)] = self._calculate_mean(
                array=dist_matrix, objmap=image.objmap[:]
        )
        measurements[str(SHAPE.MEDIAN_RADIUS)] = self._calculate_median(
                array=dist_matrix, objmap=image.objmap[:]
        )
        measurements[str(SHAPE.MAX_RADIUS)] = self._calculate_maximum(
                array=dist_matrix, objmap=image.objmap[:]
        )

        obj_props = image.objects.props
        for idx, obj_image in enumerate(image.objects):
            current_props = obj_props[idx]
            measurements[str(SHAPE.AREA)][idx] = current_props.area
            measurements[str(SHAPE.PERIMETER)][idx] = current_props.perimeter
            measurements[str(SHAPE.ECCENTRICITY)][idx] = current_props.eccentricity
            measurements[str(SHAPE.EXTENT)][idx] = current_props.extent
            measurements[str(SHAPE.BBOX_AREA)][idx] = current_props.area_bbox
            measurements[str(SHAPE.MAJOR_AXIS_LENGTH)][idx] = (
                current_props.axis_major_length
            )
            measurements[str(SHAPE.MINOR_AXIS_LENGTH)][idx] = (
                current_props.axis_minor_length
            )
            measurements[str(SHAPE.ORIENTATION)][idx] = current_props.orientation

            numer = 4 * np.pi * current_props.area
            denom = current_props.perimeter ** 2

            measurements[str(SHAPE.CIRCULARITY)][idx] = (
                numer / denom if denom != 0 else np.nan
            )
            measurements[str(SHAPE.COMPACTNESS)][idx] = (
                denom / numer if numer != 0 else np.nan
            )

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Qhull")
                    convex_hull = ConvexHull(current_props.coords)

            except QhullError:
                convex_hull = None

            measurements[str(SHAPE.CONVEX_AREA)][idx] = (
                convex_hull.area if convex_hull else np.nan
            )
            measurements[str(SHAPE.SOLIDITY)][idx] = (
                (current_props.area / convex_hull.area) if convex_hull else np.nan
            )

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
        measurements.insert(
                loc=0, column=OBJECT.LABEL, value=image.objects.labels2series()
        )
        return measurements


MeasureShape.__doc__ = SHAPE.append_rst_to_doc(MeasureShape)
