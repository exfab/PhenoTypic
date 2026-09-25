from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from phenotypic.schema import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd

from phenotypic.abc_ import MeasureFeatures
from phenotypic.measure._object_geometry import convex_hull_area, object_edt
from phenotypic.schema import SHAPE


class MeasureShape(MeasureFeatures):
    r"""Measure the form of each detected colony.

    Extract form descriptors from each colony: circularity and compactness
    (boundary regularity), solidity and extent (how completely the colony
    fills its convex hull and bounding box), eccentricity and orientation
    (elongation and its direction), the Feret caliper diameters, and
    interior thickness (mean and median distance to the nearest edge).
    Colony size -- area, perimeter, radii, axis lengths -- is measured by
    :class:`MeasureSize`; add both to a pipeline for a full profile.

    Returns:
        pd.DataFrame: One row per colony with ``Object_Label`` and every
        ``Shape_*`` column.

    Best For:
        - Distinguishing colony morphotypes (smooth circular wild-type
          vs wrinkled, branching, or invasive mutants).
        - Assessing growth symmetry and directionality via eccentricity
          and orientation.
        - Detecting invasive or spreading growth through low solidity
          values.
        - Morphological clustering for automated strain identification.

    Consider Also:
        - :class:`MeasureSize` for colony area, perimeter, radii and axis
          lengths.
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
        n_objects = image.num_objects
        measurements = {
            str(feature): np.full(shape=n_objects, fill_value=np.nan)
            for feature in SHAPE
        }

        for idx, props in enumerate(image.objects.props):
            edt = object_edt(props.image)
            interior = edt[props.image]
            measurements[str(SHAPE.MEAN_BOUNDARY_DIST)][idx] = float(interior.mean())
            measurements[str(SHAPE.MEDIAN_BOUNDARY_DIST)][idx] = float(np.median(interior))

            measurements[str(SHAPE.ECCENTRICITY)][idx] = props.eccentricity
            measurements[str(SHAPE.EXTENT)][idx] = props.extent
            measurements[str(SHAPE.ORIENTATION)][idx] = props.orientation

            numer = 4 * np.pi * props.area
            denom = props.perimeter ** 2
            measurements[str(SHAPE.CIRCULARITY)][idx] = (
                numer / denom if denom != 0 else np.nan
            )
            measurements[str(SHAPE.COMPACTNESS)][idx] = (
                denom / numer if numer != 0 else np.nan
            )

            hull, hull_area = convex_hull_area(props.coords)
            if hull is not None:
                measurements[str(SHAPE.SOLIDITY)][idx] = props.area / hull_area
                max_feret, min_feret = self._calculate_feret_diameters(
                        props.coords[hull.vertices]
                )
                measurements[str(SHAPE.MAX_FERET_DIAMETER)][idx] = max_feret
                measurements[str(SHAPE.MIN_FERET_DIAMETER)][idx] = min_feret

        frame = pd.DataFrame(measurements)
        frame.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return frame


MeasureShape.__doc__ = SHAPE.append_rst_to_doc(MeasureShape)
