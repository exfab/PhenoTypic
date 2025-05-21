from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd

from phenotypic.abstract import MeasureFeatures


# TODO: Add more measurements
class INTENSITY(Enum):
    CATEGORY = ('Intensity', 'The category of the measurements')

    INTEGRATED_INTENSITY = ('IntegratedIntensity', 'The sum of the object\'s pixels')
    MINIMUM_INTENSITY = ('MinimumIntensity', 'The minimum intensity of the object')
    MAXIMUM_INTENSITY = ('MaximumIntensity', 'The maximum intensity of the object')
    MEAN_INTENSITY = ('MeanIntensity', 'The mean intensity of the object')
    MEDIAN_INTENSITY = ('MedianIntensity', 'The median intensity of the object')
    STANDARD_DEVIATION_INTENSITY = ('StandardDeviationIntensity', 'The standard deviation of the object')
    COEFFICIENT_VARIANCE_INTENSITY = ('CoefficientVarianceIntensity', 'The coefficient of variation of the object')

    def __init__(self, label, desc=None):
        self.label, self.desc = label, desc

    def __str__(self):
        return f"{INTENSITY.CATEGORY.label}_{self.label}"


class MeasureIntensity(MeasureFeatures):
    """Calculates various intensity measures of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the intensity measures of the objects in the image.

    Notes:
        Integrated Intensity: Sum of all pixel values in the object's grayscale footprint

    """

    @staticmethod
    def _operate(image: Image) -> pd.DataFrame:
        intensity_matrix, objmap = image.matrix[:], image.objmap[:]
        measurements = {
            str(INTENSITY.INTEGRATED_INTENSITY): MeasureIntensity.calculate_sum(intensity_matrix, objmap),
            str(INTENSITY.MINIMUM_INTENSITY): MeasureIntensity.calculate_minimum(intensity_matrix, objmap),
            str(INTENSITY.MAXIMUM_INTENSITY): MeasureIntensity.calculate_max(intensity_matrix, objmap),
            str(INTENSITY.MEAN_INTENSITY): MeasureIntensity.calculate_mean(intensity_matrix, objmap),
            str(INTENSITY.MEDIAN_INTENSITY): MeasureIntensity.calculate_median(intensity_matrix, objmap),
            str(INTENSITY.STANDARD_DEVIATION_INTENSITY): MeasureIntensity.calculate_stddev(intensity_matrix, objmap),
            str(INTENSITY.COEFFICIENT_VARIANCE_INTENSITY): MeasureIntensity.calculate_coeff_variation(
                intensity_matrix, objmap
            )
        }

        return pd.DataFrame(measurements, index=image.objects.get_labels_series())
