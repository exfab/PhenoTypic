from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from phenotypic.tools_.constants_ import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import MeasureFeatures
from ..tools_.measurement_info import SIZE


class MeasureSize(MeasureFeatures):
    """Measure colony area and integrated intensity as lightweight size proxies.

    Extract two fundamental size metrics per detected colony: pixel area
    (biomass extent) and integrated grayscale intensity (total brightness,
    a proxy for optical density). This is a convenience class for rapid
    size assessment without the overhead of full shape or intensity
    statistical analysis.

    Returns:
        pd.DataFrame: Object-level size measurements with columns:

            - Label: unique object identifier.
            - Area: number of pixels occupied by the colony.
            - IntegratedIntensity: sum of grayscale pixel values
              (proxy for biomass / optical density).

    Best For:
        - Quick quality-control screening of colony size distributions.
        - Time-course growth tracking via area at successive time points.
        - Filtering colonies by minimum size to exclude debris or aborted
          growth before downstream measurement.

    Consider Also:
        - :class:`MeasureShape` for comprehensive morphological metrics
          (circularity, convex hull, Feret diameters).
        - :class:`MeasureIntensity` for full intensity statistics
          (percentiles, variance, coefficient of variation).
        - :class:`MeasureGridSpread` for detecting multi-object wells in
          arrayed assays.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting size metrics in a biological context.
    """

    _measurement_infoclass: ClassVar[type] = SIZE

    def _operate(self, image: Image) -> pd.DataFrame:
        # Create empty numpy arrays to store measurements
        measurements = {
            str(feature): np.zeros(shape=image.num_objects)
            for feature in SIZE
            if feature != SIZE.CATEGORY
        }

        # Calculate integrated intensity using the sum calculation method from base class
        intensity_matrix = image.gray[:].copy()
        objmap = image.objmap[:].copy()

        measurements[SIZE.AREA] = self._calculate_sum(
                array=image.objmask[:], objmap=objmap
        )
        measurements[SIZE.INTEGRATED_INTENSITY] = self._calculate_sum(
                array=intensity_matrix, objmap=objmap
        )

        measurements = pd.DataFrame(measurements)
        measurements.insert(
                loc=0, column=OBJECT.LABEL, value=image.objects.labels2series()
        )
        return measurements


MeasureSize.__doc__ = SIZE.append_rst_to_doc(MeasureSize)
