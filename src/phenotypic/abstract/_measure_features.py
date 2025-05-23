from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy

from ._base_operation import BaseOperation
from phenotypic.util.exceptions_ import OperationFailedError
from phenotypic.util.funcs_ import validate_measure_integrity

# <<Interface>>
class MeasureFeatures(BaseOperation):
    """
    A FeatureExtractor is an abstract object intended to calculate measurements on the values within detected objects of
    the image array. The __init__ constructor & _operate method is meant to be the only parts overloaded in inherited classes. This is so
    that the main measure method call can contain all the necessary type validation and output validation checks to streamline development.
    """

    @validate_measure_integrity()
    def measure(self, image: Image) -> pd.DataFrame:
        try:
            matched_args = self._get_matched_operation_args()

            # Apply the operation to a copy so that the original image is not modified.
            return self._operate(image, **matched_args)

        except Exception as e:
            raise OperationFailedError(operation=self.__class__.__name__,
                                       image_name=image.name,
                                       err_type=type(e),
                                       message=str(e),
                                       )

    @staticmethod
    def _operate(image: Image) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def __repair_scipy_results(scipy_output) -> np.array:
        """Tests and ensures scipy result is a numpy array.
        This is helpful for bulk measurements using scipy.ndimage measurement functions"""
        if getattr(scipy_output, "__getitem__", False):
            return np.array(scipy_output)
        else:
            return np.array([scipy_output])

    @staticmethod
    def calculate_center_of_mass(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.center_of_mass(array, labels, index))

    @staticmethod
    def calculate_max(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.maximum(array, labels, index))

    @staticmethod
    def calculate_mean(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.mean(array, labels, index))

    @staticmethod
    def calculate_median(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.median(array, labels, index))

    @staticmethod
    def calculate_minimum(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.minimum(array, labels, index))

    @staticmethod
    def calculate_stddev(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.standard_deviation(array, labels, index))

    @staticmethod
    def calculate_sum(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.sum_labels(array, labels, index))

    @staticmethod
    def calculate_variance(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures.__repair_scipy_results(scipy.ndimage.variance(array, labels, index))

    @staticmethod
    def calculate_coeff_variation(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        """Calculates unbiased coefficient of variation (CV) for each object in the image, assuming normal distribution.

        References:
            - https://en.wikipedia.org/wiki/Coefficient_of_variation
        """
        unique_labels, unique_counts = np.unique(labels, return_counts=True)
        unique_counts = unique_counts[unique_labels != 0]
        biased_cv = MeasureFeatures.calculate_stddev(array, labels, index) / MeasureFeatures.calculate_mean(array, labels, index)
        return (1 + (1 / unique_counts)) * biased_cv

    @staticmethod
    def _calculate_extrema(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        min_extrema, max_extrema, min_pos, max_pos = MeasureFeatures.__repair_scipy_results(scipy.ndimage.extrema(array, labels, index))
        return (
            MeasureFeatures.__repair_scipy_results(min_extrema),
            MeasureFeatures.__repair_scipy_results(max_extrema),
            MeasureFeatures.__repair_scipy_results(min_pos),
            MeasureFeatures.__repair_scipy_results(max_pos)
        )

    @staticmethod
    def calculate_min_extrema(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        min_extrema, _, min_pos, _ = MeasureFeatures._calculate_extrema(array, labels, index)
        return min_extrema, min_pos

    @staticmethod
    def calculate_max_extrema(array:np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        _, max_extreme, _, max_pos = MeasureFeatures._calculate_extrema(array, labels, index)
        return max_extreme, max_pos