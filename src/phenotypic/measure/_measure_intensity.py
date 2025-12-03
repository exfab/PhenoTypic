from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from phenotypic.tools.constants_ import OBJECT

if TYPE_CHECKING:
    from phenotypic import Image
from enum import Enum
from functools import partial

import pandas as pd

from phenotypic.abc_ import MeasureFeatures, MeasurementInfo


class INTENSITY(MeasurementInfo):
    @classmethod
    def category(cls):
        return "Intensity"

    INTEGRATED_INTENSITY = ("IntegratedIntensity", "The sum of the object's pixels")
    MINIMUM_INTENSITY = ("MinimumIntensity", "The minimum intensity of the object")
    MAXIMUM_INTENSITY = ("MaximumIntensity", "The maximum intensity of the object")
    MEAN_INTENSITY = ("MeanIntensity", "The mean intensity of the object")
    MEDIAN_INTENSITY = ("MedianIntensity", "The median intensity of the object")
    STANDARD_DEVIATION_INTENSITY = (
        "StandardDeviationIntensity",
        "The standard deviation of the object",
    )
    COEFFICIENT_VARIANCE_INTENSITY = (
        "CoefficientVarianceIntensity",
        "The coefficient of variation of the object",
    )
    Q1_INTENSITY = (
        "LowerQuartileIntensity",
        "The lower quartile intensity of the object",
    )
    Q3_INTENSITY = (
        "UpperQuartileIntensity",
        "The upper quartile intensity of the object",
    )
    IQR_INTENSITY = (
        "InterquartileRangeIntensity",
        "The interquartile range of the object",
    )


class MeasureIntensity(MeasureFeatures):
    """Measure grayscale intensity statistics of detected microbial colonies.

    This class computes quantitative intensity metrics from the grayscale representation of each detected
    colony, including integrated intensity (total brightness), percentiles (min, Q1, median, Q3, max),
    and variability measures (standard deviation, coefficient of variation). These statistics reflect
    colony optical density, biomass, and internal heterogeneity on agar plates.

    **Intuition:** Colony brightness in grayscale images correlates with biomass accumulation, cell
    density, and optical density. A low average intensity indicates sparse or translucent growth, while
    high intensity suggests dense mycelial mat or concentrated cells. Intensity variance within a colony
    reveals sectoring, differential growth rates, or uneven mycelial coverage. Integrated intensity
    (sum of all pixels) scales with both biomass and area, making it useful for growth kinetics tracking.

    **Use cases (agar plates):**
    - Track colony growth over time via integrated intensity measurements (approximates optical density
      without requiring liquid suspension).
    - Detect metabolically stressed or slow-growing colonies via low mean intensity.
    - Identify sectored, mutant, or chimeric colonies by high intensity variance within a single object.
    - Assess pigmentation or mycelial density differences between wild-type and mutant strains.
    - Enable automated colony picking strategies based on intensity thresholds (e.g., select only colonies
      above a biomass threshold).

    **Caveats:**
    - Intensity depends critically on imaging conditions (lighting, exposure, camera gain); standardize
      these settings across plates and experiments for reliable comparisons.
    - Grayscale conversion to luminance (Y channel) may not capture all visual information from colored
      agar or pigmented colonies; use enhanced grayscale (enh_gray) for better contrast, or measure color
      separately.
    - Integrated intensity mixes area and brightness; normalize by area or use intensity density (mean)
      for size-independent comparisons.
    - Shadows or uneven lighting on the plate cause local intensity artifacts; preprocessing with background
      subtraction or illumination correction may improve quality.
    - Outliers (very bright or very dark pixels from debris, bubble, or focus issues) can inflate standard
      deviation; use robust statistics (IQR) for noise-resistant variability assessment.

    Returns:
        pd.DataFrame: Object-level intensity statistics with columns:
            - Label: Unique object identifier.
            - IntegratedIntensity: Sum of all grayscale pixel values in the colony.
            - MinimumIntensity, MaximumIntensity: Range of intensity values.
            - MeanIntensity, MedianIntensity: Central tendency (mean more sensitive to outliers).
            - LowerQuartileIntensity, UpperQuartileIntensity: 25th and 75th percentiles (robust measures).
            - InterquartileRangeIntensity: Q3 - Q1 (robust measure of spread).
            - StandardDeviationIntensity: Sample standard deviation.
            - CoefficientVarianceIntensity: Normalized variability (std dev / mean, unitless).

    Examples:
        .. dropdown:: Measure colony biomass via intensity

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import OtsuDetector
                from phenotypic.measure import MeasureIntensity

                # Load and process plate image
                image = Image.from_image_path("colony_plate_t24h.jpg")
                detector = OtsuDetector()
                image = detector.operate(image)

                # Measure intensity to estimate biomass
                measurer = MeasureIntensity()
                intensity = measurer.operate(image)

                # Track colonies by integrated intensity (proxy for biomass)
                high_biomass = intensity[intensity['Intensity_IntegratedIntensity'] > 100000]
                print(f"Colonies with high biomass (>100k): {len(high_biomass)}")

        .. dropdown:: Identify heterogeneous or sectored colonies

            .. code-block:: python

                # Measure intensity variance to detect sectoring
                intensity = measurer.operate(image)

                # Colonies with high variance may be sectored or chimeric
                sectored = intensity[
                    intensity['Intensity_CoefficientVarianceIntensity'] >
                    intensity['Intensity_CoefficientVarianceIntensity'].quantile(0.75)
                ]
                print(f"Potentially sectored colonies: {list(sectored.index)}")
    """

    def _operate(self, image: Image) -> pd.DataFrame:
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
        return measurements


MeasureIntensity.__doc__ = INTENSITY.append_rst_to_doc(MeasureIntensity)
