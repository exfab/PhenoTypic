from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING, cast

from phenotypic.schema import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import warnings
import pandas as pd
from pydantic import Field
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import distance_transform_edt, label as ndi_label
from scipy.stats import trim_mean
from skimage.measure import find_contours
import numpy as np

from phenotypic.abc_ import MeasureFeatures
from phenotypic.schema import SHAPE


class MeasureShape(MeasureFeatures):
    r"""Measure comprehensive morphological characteristics of detected colonies.

    Extract geometric metrics from each colony shape: area, perimeter,
    circularity, convex hull properties, width-based measures, Feret
    diameters, eccentricity, and best-fit ellipse parameters. The output
    DataFrame provides a full morphological profile for phenotypic
    classification and growth-pattern analysis.

    Args:
        angular_bins: Number of equally spaced directions at which the
            radial signature is sampled. Higher values reduce the
            outermost-crossing bias but leave more empty bins to
            interpolate across on small colonies. Typical range: 180--720.
            Default: 360.
        trim_proportion: Fraction of the radial signature trimmed from each
            tail before averaging, which sets the breakdown point of
            RobustMeanRadius. At 0.2 the estimate tolerates up to 20% of
            directions being contaminated by protrusions. Setting it to 0.0
            gives the plain arithmetic mean. Typical range: 0.0--0.3.
            Default: 0.2.
        plateau_tolerance: Relative tolerance defining the near-maximal
            plateau of the distance transform. Its centroid is the colony
            center. Values near 0 make the center an argmax, whose location
            is arbitrary among exact ties; the default averages over the
            plateau instead. Default: 0.01.

    Returns:
        pd.DataFrame: Object-level morphological measurements with
        columns:

            - Label, Area, Perimeter, Circularity, Compactness,
              ConvexArea, Solidity, Extent, BboxArea.
            - MeanBoundaryDist, MedianBoundaryDist (mean/median depth from
              the boundary; not radii).
            - InscribedRadius, RobustMeanRadius, ReachRadius (smallest,
              typical, and largest radius from the colony center).
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

    angular_bins: int = Field(360, ge=8, le=3600)
    trim_proportion: float = Field(0.2, ge=0.0, lt=0.5)
    plateau_tolerance: float = Field(0.01, gt=0.0, lt=1.0)

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

    def _trace_radial_signature(
            self, obj_mask: np.ndarray, edt: np.ndarray
    ) -> np.ndarray | None:
        """Sample the colony boundary's distance from its center, uniformly by angle.

        The center is the centroid of the distance-transform's near-maximal
        plateau, not its argmax: the transform's values are square roots of
        exact integers, so exact ties are common and an argmax would resolve
        them by raster order. The boundary is the subpixel marching-squares
        contour at the 0.5 iso-level. Sampling is by angle rather than along
        the contour so that a narrow protrusion contributes only its angular
        width, which is what keeps the trimmed mean inside its breakdown point.

        Args:
            obj_mask (np.ndarray): Boolean mask of a single object within its
                bounding box (``regionprops.image``).
            edt (np.ndarray): Euclidean distance transform of *obj_mask*,
                computed with one pixel of background padding on every side.

        Returns:
            np.ndarray | None: Radii at ``self.angular_bins`` equally spaced
            angles, or None when the object has no interior or no contour.
        """
        peak = float(edt.max())
        if peak <= 0.0:
            return None

        plateau = edt >= (1.0 - self.plateau_tolerance) * peak
        # ndi_label returns int | tuple[ndarray, int]; with no output arg the
        # runtime value is always the tuple, so narrow the stub's union.
        components = cast("tuple[np.ndarray, int]", ndi_label(plateau))[0]
        dominant = components[np.unravel_index(np.argmax(edt), edt.shape)]
        center = np.argwhere(components == dominant).mean(axis=0)

        contours = find_contours(np.pad(obj_mask, 1).astype(float), 0.5)
        if not contours:
            return None
        outline = max(contours, key=len) - 1.0

        offsets = outline - center
        radii = np.hypot(offsets[:, 0], offsets[:, 1])
        angles = np.arctan2(offsets[:, 0], offsets[:, 1])

        n_bins = self.angular_bins
        bins = ((angles + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins
        signature = np.full(n_bins, -np.inf)
        np.maximum.at(signature, bins, radii)

        empty = np.isinf(signature)
        if empty.all():
            return None
        signature[empty] = np.nan
        if empty.any():
            # Circular interpolation: bins with no contour vertex are not
            # missing at random. They cluster where the contour is angularly
            # sparse, so dropping them biases the mean upward.
            index = np.arange(n_bins)
            filled = ~empty
            signature[empty] = np.interp(
                    index[empty],
                    np.concatenate([index[filled] - n_bins, index[filled], index[filled] + n_bins]),
                    np.tile(signature[filled], 3),
            )
        return signature

    def _measure_radial_profile(self, obj_mask: np.ndarray) -> dict[str, float]:
        """Compute distance-transform measures for one cropped object.

        The Euclidean distance transform binarizes its input, so it must be
        given a single object's mask rather than the whole labelled objmap;
        otherwise two touching colonies see no background between them and
        both report inflated distances. The mask is padded by one pixel so
        that the transform sees background on every side of the bounding box.

        Args:
            obj_mask (np.ndarray): Boolean mask of a single object within its
                bounding box, as produced by ``regionprops.image``. Already
                isolated from neighbouring labels.

        Returns:
            dict[str, float]: Mapping of ``SHAPE`` column header to value,
            with keys MeanBoundaryDist, MedianBoundaryDist, InscribedRadius,
            RobustMeanRadius, and ReachRadius.
        """
        # asarray narrows the scipy stub's tuple-or-ndarray return union; with
        # return_indices=False the call yields an ndarray and copies nothing.
        edt = np.asarray(distance_transform_edt(np.pad(obj_mask, 1)))[1:-1, 1:-1]
        interior = edt[obj_mask]
        values = {
            str(SHAPE.MEAN_BOUNDARY_DIST): float(interior.mean()),
            str(SHAPE.MEDIAN_BOUNDARY_DIST): float(np.median(interior)),
            str(SHAPE.INSCRIBED_RADIUS): float(edt.max()),
            str(SHAPE.ROBUST_MEAN_RADIUS): np.nan,
            str(SHAPE.REACH_RADIUS): np.nan,
        }

        signature = self._trace_radial_signature(obj_mask, edt)
        if signature is not None:
            values[str(SHAPE.ROBUST_MEAN_RADIUS)] = float(
                    trim_mean(signature, self.trim_proportion)
            )
            values[str(SHAPE.REACH_RADIUS)] = float(signature.max())
        return values

    def _operate(self, image: Image) -> pd.DataFrame:
        # Create empty numpy arrays to store measurements
        measurements = {
            str(feature): np.zeros(shape=image.num_objects)
            for feature in SHAPE
            if feature != SHAPE.CATEGORY
        }

        obj_props = image.objects.props
        for idx, obj_image in enumerate(image.objects):
            current_props = obj_props[idx]
            for header, value in self._measure_radial_profile(
                    current_props.image
            ).items():
                measurements[header][idx] = value
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

            # ConvexArea and Solidity come from regionprops, which counts the
            # pixels of the convex image. scipy's ConvexHull is retained only
            # for the Feret calipers below: in 2D its `.area` is the hull
            # perimeter and its `.volume` is the hull area, and even `.volume`
            # undercounts because the hull passes through pixel centres.
            measurements[str(SHAPE.CONVEX_AREA)][idx] = current_props.area_convex
            measurements[str(SHAPE.SOLIDITY)][idx] = current_props.solidity

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Qhull")
                    convex_hull = ConvexHull(current_props.coords)

            except QhullError:
                convex_hull = None

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
