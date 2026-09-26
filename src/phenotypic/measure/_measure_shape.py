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
    (elongation and its direction), and interior thickness (mean and
    median distance to the nearest edge). Colony size -- area,
    perimeter, radii, axis lengths, Feret diameters -- is measured by
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
        - :class:`MeasureSize` for colony area, perimeter, radii, axis
          lengths and Feret diameters.
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

        frame = pd.DataFrame(measurements)
        frame.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return frame


MeasureShape.__doc__ = SHAPE.append_rst_to_doc(MeasureShape)
