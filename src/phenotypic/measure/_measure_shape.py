from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic.tools_.constants_ import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import warnings
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import distance_transform_edt
import numpy as np

from phenotypic.abc_ import MeasureFeatures
from ..tools_.measurement_info_ import SHAPE


class MeasureShape(MeasureFeatures):
    r"""Measure morphological characteristics of detected microbial colonies.

    This class extracts comprehensive geometric metrics from colony shapes, including area, perimeter,
    circularity, convex hull properties, width-based measures, Feret diameters, elongation (eccentricity),
    and ellipse fitting parameters. These measurements quantify colony morphology, growth patterns, and
    spatial organization on agar plates.

    **Intuition:** Colony shape encodes biological and environmental information. Regular circular colonies
    indicate healthy, isotropic growth under uniform conditions. Irregular, elongated, or filamentous
    morphologies suggest mutations, directional growth (chemotaxis), nutrient stress, or environmental
    gradients on the agar surface. Shape measures are used to classify colony types, assess fitness,
    and detect phenotypic variants in high-throughput screening.

    **Use cases (agar plates):**
    - Distinguish colony morphotypes: smooth circular (wild-type) vs wrinkled, branching, or invasive
      (mutant phenotypes).
    - Assess growth symmetry via eccentricity and orientation; colonies with high eccentricity may
      indicate motility, chemotaxis, or unidirectional stress.
    - Detect invasive or spreading growth via low solidity (indented periphery) or high convex area
      relative to actual area.
    - Enable morphological clustering and classification for automated strain identification or
      phenotypic screening.
    - Measure colony compactness to predict growth kinetics: compact colonies often have higher growth
      rates than sprawling ones under nutrient limitation.

    **Caveats:**
    - Shape measurements depend entirely on segmentation quality; poor thresholding or edge detection
      yield misleading morphology metrics.
    - Perimeter is sensitive to pixel-level noise; small variations in boundary can inflate perimeter
      and reduce circularity. Consider smoothing or filtering for robust estimates.
    - Feret diameters and convex hull computation are sensitive to boundary artifacts; outlier or
      misdetected pixels at the edge disproportionately affect these metrics.
    - Radius-based measures (mean, median, max width) depend on centroid accuracy; off-center centroids
      from irregular shapes can yield biased width values.
    - Eccentricity ranges 0–1 (circle to line); values near 0 and 1 are rare for biological objects.
      Interpret eccentricity alongside aspect ratio and orientation for robust shape classification.

    Returns:
        pd.DataFrame: Object-level morphological measurements with columns:
            - Label: Unique object identifier.
            - Area: Number of pixels in the colony.
            - Perimeter: Boundary length in pixels.
            - Circularity: 4π·Area/Perimeter² (1.0 = perfect circle; <1 = irregular).
            - Compactness: Perimeter²/(4π·Area) (inverse of circularity; >1 for irregular shapes).
            - ConvexArea: Area of convex hull (smallest convex polygon containing the colony).
            - Solidity: Area/ConvexArea (1.0 = convex; <1 = indented/spreading).
            - Extent: Area/BboxArea (1.0 = fills bounding box; <1 = spread out).
            - BboxArea: Area of axis-aligned bounding rectangle.
            - MeanRadius, MedianRadius, MaxRadius: Distance from centroid to edge (robust size measures).
            - MinFeretDiameter, MaxFeretDiameter: Minimum/maximum caliper diameters (orientation-independent).
            - MajorAxisLength, MinorAxisLength: Axes of best-fit ellipse.
            - Eccentricity: Ellipse elongation (0 = circle; 1 = line).
            - Orientation: Angle of major axis (radians, –π/2 to π/2).

    Examples:
        Measure colony morphology for phenotypic classification:

        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.measure import MeasureShape
        >>> # Load plate with multiple morphotype colonies
        >>> image = Image.imread("morphotype_plate.jpg")  # doctest: +SKIP
        >>> detector = OtsuDetector()
        >>> image = detector.operate(image)  # doctest: +SKIP
        >>> # Measure morphology
        >>> shaper = MeasureShape()
        >>> shapes = shaper.operate(image)  # doctest: +SKIP
        >>> # Classify morphotypes by circularity and solidity
        >>> smooth_round = shapes[
        ...     (shapes['Shape_Circularity'] > 0.8) &
        ...     (shapes['Shape_Solidity'] > 0.95)
        ... ]  # doctest: +SKIP
        >>> invasive = shapes[shapes['Shape_Solidity'] < 0.85]  # doctest: +SKIP
        >>> print(f"Smooth/round colonies: {len(smooth_round)}")  # doctest: +SKIP
        >>> print(f"Invasive/spreading colonies: {len(invasive)}")  # doctest: +SKIP

        Detect elongated or directional growth:

        >>> # Use eccentricity and max width to find elongated colonies
        >>> shapes = shaper.operate(image)  # doctest: +SKIP
        >>> elongated = shapes[shapes['Shape_Eccentricity'] > 0.7]  # doctest: +SKIP
        >>> print(f"Highly elongated colonies: {len(elongated)}")  # doctest: +SKIP
        >>> # Visualize growth directionality
        >>> import numpy as np
        >>> for idx, row in elongated.iterrows():  # doctest: +SKIP
        ...     angle = np.degrees(row['Shape_Orientation'])
        ...     aspect = row['Shape_MajorAxisLength'] / row['Shape_MinorAxisLength']
        ...     print(f"Colony {row['OBJECT_Label']}: angle={angle:.1f}deg, aspect={aspect:.2f}")
    """

    _measurement_info_class = SHAPE

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
                current_props.major_axis_length
            )
            measurements[str(SHAPE.MINOR_AXIS_LENGTH)][idx] = (
                current_props.minor_axis_length
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
