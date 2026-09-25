from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING, cast

from phenotypic.schema import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.ndimage import binary_fill_holes, label as ndi_label
from scipy.stats import trim_mean
from skimage.measure import find_contours

from phenotypic.abc_ import MeasureFeatures
from phenotypic.measure._object_geometry import convex_hull_area, object_edt
from phenotypic.schema import SIZE


class MeasureSize(MeasureFeatures):
    """Measure the key size magnitudes of each detected colony.

    The single source of colony size: area, integrated intensity, perimeter,
    convex-hull and bounding-box areas, best-fit-ellipse axis lengths, and a
    family of five radii that are all measured from one center inside the
    colony (inscribed, median, mean, robust mean and maximum). These are the
    starting measurements for growth and fitness comparisons; see the
    :class:`~phenotypic.schema.SIZE` table below for what each column means.

    The radii come from the colony's *radial signature*: the distance from
    the center to the boundary, sampled in ``angular_bins`` equal directions.
    Sampling by angle rather than along the boundary gives a runner or spur
    only its true angular width, so the trimmed ``RobustMeanRadius`` stays on
    the compact body while ``MeanRadius`` and ``MaxRadius`` show the reach.

    Args:
        angular_bins: Number of equal angular directions in which the radial
            signature samples the boundary. More bins resolve narrower
            protrusions; 360 (one per degree) resolves any runner wider than
            about 1/57 of the colony radius.
        trim_proportion: Fraction trimmed from each end of the radial
            signature for ``RobustMeanRadius``. It tolerates a runner or spur
            covering up to this fraction of all directions; 0 makes
            ``RobustMeanRadius`` equal ``MeanRadius``.
        plateau_tolerance: Relative tolerance defining the distance-transform
            peak plateau whose centroid is the colony center. The default
            (1%) merges the exact ties that integer pixel geometry produces.

    Returns:
        pd.DataFrame: One row per colony with ``Object_Label`` and every
        ``Size_*`` column.

    Best For:
        - Growth and fitness comparisons across strains and conditions
          (area and the radius family).
        - Separating compact growth from runners or spreading
          (``MaxRadius`` against ``RobustMeanRadius``).
        - Filtering debris or aborted growth by minimum size before
          downstream measurement.

    Consider Also:
        - :class:`MeasureShape` for form descriptors (circularity, solidity,
          eccentricity, Feret diameters, interior thickness).
        - :class:`MeasureIntensity` for full intensity statistics.
        - :class:`MeasureGridSpread` for detecting multi-object wells in
          arrayed assays.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting size metrics in a biological context.
    """

    _measurement_infoclass: ClassVar[type] = SIZE

    angular_bins: int = Field(360, ge=8, le=3600)
    trim_proportion: float = Field(0.2, ge=0.0, lt=0.5)
    plateau_tolerance: float = Field(0.01, gt=0.0, lt=1.0)

    def _trace_radial_signature(
            self, obj_mask: np.ndarray, edt: np.ndarray
    ) -> np.ndarray | None:
        """Sample the colony boundary's distance from its center, uniformly by angle.

        The center is the centroid of the distance-transform's near-maximal
        plateau, not its argmax: the transform's values are square roots of
        exact integers, so exact ties are common and an argmax would resolve
        them by raster order. The plateau component holding the argmax is
        taken 8-connected, as objects are labelled, so a diagonal run of tied
        pixels is one plateau.

        The boundary is every subpixel marching-squares contour of the
        hole-filled label at the 0.5 iso-level, pooled: a label made of
        several pieces (a runner joined only at a corner, or fragments merged
        under one label) is sampled whole, so each bin keeps the outermost
        crossing of any piece. Holes are filled first because a small outline
        has fewer vertices than bins, and a hole's vertices would otherwise
        fill bins that the outer outline left empty. The contours are traced
        8-connected to match how objects are labelled; pooled, the result
        does not depend on it, since connectivity only regroups the same
        crossings into contours. Sampling is by angle rather than along the
        contour so that a narrow protrusion contributes only its angular
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
        components = cast(
                "tuple[np.ndarray, int]",
                ndi_label(plateau, structure=np.ones((3, 3), dtype=bool)),
        )[0]
        dominant = components[np.unravel_index(np.argmax(edt), edt.shape)]
        center = np.argwhere(components == dominant).mean(axis=0)

        contours = find_contours(
                np.pad(binary_fill_holes(obj_mask), 1).astype(float),
                0.5,
                fully_connected="high",
        )
        if not contours:
            return None
        outline = np.concatenate(contours) - 1.0

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
        """Compute the five radii for one cropped object.

        Args:
            obj_mask (np.ndarray): Boolean mask of a single object within its
                bounding box (``regionprops.image``), already isolated from
                neighbouring labels.

        Returns:
            dict[str, float]: ``SIZE`` header to value for InscribedRadius,
            MedianRadius, MeanRadius, RobustMeanRadius and MaxRadius. The
            four signature radii are NaN when the object has no contour.
        """
        edt = object_edt(obj_mask)
        values = {
            str(SIZE.INSCRIBED_RADIUS): float(edt.max()),
            str(SIZE.MEDIAN_RADIUS): np.nan,
            str(SIZE.MEAN_RADIUS): np.nan,
            str(SIZE.ROBUST_MEAN_RADIUS): np.nan,
            str(SIZE.MAX_RADIUS): np.nan,
        }
        signature = self._trace_radial_signature(obj_mask, edt)
        if signature is not None:
            values[str(SIZE.MEDIAN_RADIUS)] = float(np.median(signature))
            values[str(SIZE.MEAN_RADIUS)] = float(signature.mean())
            values[str(SIZE.ROBUST_MEAN_RADIUS)] = float(
                    trim_mean(signature, self.trim_proportion)
            )
            values[str(SIZE.MAX_RADIUS)] = float(signature.max())
        return values

    def _operate(self, image: Image) -> pd.DataFrame:
        n_objects = image.num_objects
        measurements = {
            str(feature): np.full(shape=n_objects, fill_value=np.nan)
            for feature in SIZE
        }

        objmap = image.objmap[:].copy()
        measurements[str(SIZE.AREA)] = self._calculate_sum(
                array=image.objmask[:], objmap=objmap
        )
        measurements[str(SIZE.INTEGRATED_INTENSITY)] = self._calculate_sum(
                array=image.gray[:].copy(), objmap=objmap
        )

        for idx, props in enumerate(image.objects.props):
            measurements[str(SIZE.PERIMETER)][idx] = props.perimeter
            measurements[str(SIZE.BBOX_AREA)][idx] = props.area_bbox
            measurements[str(SIZE.MAJOR_AXIS_LENGTH)][idx] = props.axis_major_length
            measurements[str(SIZE.MINOR_AXIS_LENGTH)][idx] = props.axis_minor_length
            measurements[str(SIZE.CONVEX_AREA)][idx] = convex_hull_area(props.coords)[1]
            for header, value in self._measure_radial_profile(props.image).items():
                measurements[header][idx] = value

        frame = pd.DataFrame(measurements)
        frame.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return frame


MeasureSize.__doc__ = SIZE.append_rst_to_doc(MeasureSize)
