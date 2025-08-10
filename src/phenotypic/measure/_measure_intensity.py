from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING: from phenotypic import Image
from enum import Enum
from functools import partial

import pandas as pd

from phenotypic.abstract import MeasureFeatures, MeasurementInfo


# TODO: Add more measurements
class INTENSITY(MeasurementInfo):
    @property
    def CATEGORY(self):
        return 'Intensity'

    INTEGRATED_INTENSITY = ('IntegratedIntensity', 'The sum of the object\'s pixels')
    MINIMUM_INTENSITY = ('MinimumIntensity', 'The minimum intensity of the object')
    MAXIMUM_INTENSITY = ('MaximumIntensity', 'The maximum intensity of the object')
    MEAN_INTENSITY = ('MeanIntensity', 'The mean intensity of the object')
    MEDIAN_INTENSITY = ('MedianIntensity', 'The median intensity of the object')
    STANDARD_DEVIATION_INTENSITY = ('StandardDeviationIntensity', 'The standard deviation of the object')
    COEFFICIENT_VARIANCE_INTENSITY = ('CoefficientVarianceIntensity', 'The coefficient of variation of the object')
    Q1_INTENSITY = ('LowerQuartileIntensity', 'The lower quartile intensity of the object')
    Q3_INTENSITY = ('UpperQuartileIntensity', 'The upper quartile intensity of the object')
    IQR_INTENSITY = ('InterquartileRangeIntensity', 'The interquartile range of the object')


class MeasureIntensity(MeasureFeatures):
    """Calculates various intensity measures of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the intensity measures of the objects in the image.

    Notes:
        Integrated Intensity: Sum of all pixel values in the object's grayscale footprint

    """

    def _operate(self, image: Image) -> pd.DataFrame:
        intensity_matrix, objmap = image.matrix[:].copy(), image.objmap[:].copy()
        measurements = {
            str(INTENSITY.INTEGRATED_INTENSITY): self._calculate_sum(array=intensity_matrix, labels=objmap),
            str(INTENSITY.MINIMUM_INTENSITY): self._calculate_minimum(array=intensity_matrix, labels=objmap),
            str(INTENSITY.MAXIMUM_INTENSITY): self._calculate_max(array=intensity_matrix, labels=objmap),
            str(INTENSITY.MEAN_INTENSITY): self._calculate_mean(array=intensity_matrix, labels=objmap),
            str(INTENSITY.MEDIAN_INTENSITY): self._calculate_median(array=intensity_matrix, labels=objmap),
            str(INTENSITY.STANDARD_DEVIATION_INTENSITY): self._calculate_stddev(array=intensity_matrix, labels=objmap),
            str(INTENSITY.COEFFICIENT_VARIANCE_INTENSITY): self._calculate_coeff_variation(
                array=intensity_matrix, labels=objmap,
            ),
            str(INTENSITY.Q1_INTENSITY): self._calculate_q1(array=intensity_matrix, labels=objmap),
            str(INTENSITY.Q3_INTENSITY): self._calculate_q3(array=intensity_matrix, labels=objmap),
            str(INTENSITY.IQR_INTENSITY): self._calculate_iqr(array=intensity_matrix, labels=objmap),
        }

        return pd.DataFrame(measurements, index=image.objects.labels2series())


MeasureIntensity.__doc__ = INTENSITY.append_rst_to_doc(MeasureIntensity)
