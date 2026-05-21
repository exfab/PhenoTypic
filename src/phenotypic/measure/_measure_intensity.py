from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from phenotypic.tools_.constants_ import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd

from phenotypic.abc_ import MeasureFeatures
from ..tools_.measurement_info import INTENSITY


class MeasureIntensity(MeasureFeatures):
    """Measure grayscale intensity statistics of detected colonies.

    Compute per-colony intensity metrics from the grayscale channel:
    integrated intensity, percentiles (min, Q1, median, Q3, max),
    standard deviation, coefficient of variation, and area-normalized
    density. These statistics reflect colony optical density, biomass
    accumulation, and internal heterogeneity.

    Returns:
        pd.DataFrame: Object-level intensity statistics with columns:

            - Label, IntegratedIntensity, MinimumIntensity,
              MaximumIntensity, MeanIntensity, MedianIntensity.
            - LowerQuartileIntensity, UpperQuartileIntensity,
              InterquartileRangeIntensity.
            - StandardDeviationIntensity, CoefficientVarianceIntensity.
            - Density (integrated intensity / area), ConvexDensity
              (integrated intensity / convex area).

    Best For:
        - Tracking colony growth over time via integrated intensity as
          an optical-density proxy.
        - Detecting metabolically stressed or slow-growing colonies
          through low mean intensity.
        - Identifying sectored or chimeric colonies by high
          within-colony intensity variance.
        - Automated colony picking based on biomass thresholds.

    Consider Also:
        - :class:`MeasureSize` for lightweight area and integrated
          intensity without full statistics.
        - :class:`MeasureColor` for multi-channel color statistics when
          pigmentation is relevant.
        - :class:`MeasureTexture` for surface-roughness features that
          complement intensity metrics.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting intensity metrics in a biological context.
    """

    _measurement_infoclass: ClassVar[type] = INTENSITY

    def _operate(self, image: Image) -> pd.DataFrame:
        from phenotypic.measure._measure_shape import MeasureShape
        from ..tools_.measurement_info import SHAPE

        intensity_matrix, objmap = image.gray[:].copy(), image.objmap[:].copy()
        measurements = {
            str(INTENSITY.INTEGRATED_INTENSITY)        : self._calculate_sum(
                    array=intensity_matrix, objmap=objmap
            ),
            str(INTENSITY.MINIMUM_INTENSITY)           : self._calculate_minimum(
                    array=intensity_matrix, objmap=objmap
            ),
            str(INTENSITY.MAXIMUM_INTENSITY)           : self._calculate_maximum(
                    array=intensity_matrix, objmap=objmap
            ),
            str(INTENSITY.MEAN_INTENSITY)              : self._calculate_mean(
                    array=intensity_matrix, objmap=objmap
            ),
            str(INTENSITY.MEDIAN_INTENSITY)            : self._calculate_median(
                    array=intensity_matrix, objmap=objmap
            ),
            str(INTENSITY.STANDARD_DEVIATION_INTENSITY): self._calculate_stddev(
                    array=intensity_matrix, objmap=objmap
            ),
            str(
                    INTENSITY.COEFFICIENT_VARIANCE_INTENSITY
            )                                          : self._calculate_coeff_variation(
                    array=intensity_matrix,
                    objmap=objmap,
            ),
            str(INTENSITY.Q1_INTENSITY)                : self._calculate_q1(
                    array=intensity_matrix, objmap=objmap
            ),
            str(INTENSITY.Q3_INTENSITY)                : self._calculate_q3(
                    array=intensity_matrix, objmap=objmap
            ),
            str(INTENSITY.IQR_INTENSITY)               : self._calculate_iqr(
                    array=intensity_matrix, objmap=objmap
            ),
        }

        measurements = pd.DataFrame(measurements)
        measurements.insert(
                loc=0, column=OBJECT.LABEL, value=image.objects.labels2series()
        )
        shape_measurements = MeasureShape().measure(image)

        measurements[INTENSITY.DENSITY] = (
                measurements[INTENSITY.INTEGRATED_INTENSITY]
                / shape_measurements[SHAPE.AREA]
        )

        measurements[INTENSITY.CONVEX_DENSITY] = (
                measurements[INTENSITY.INTEGRATED_INTENSITY]
                / shape_measurements[SHAPE.CONVEX_AREA]
        )
        return measurements


MeasureIntensity.__doc__ = INTENSITY.append_rst_to_doc(MeasureIntensity)
