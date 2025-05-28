from __future__ import annotations
from typing import TYPE_CHECKING

from typing_extensions import Callable

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy
from functools import partial

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
    def _repair_scipy_results(scipy_output) -> np.array:
        """Tests and ensures scipy result is a numpy array.
        This is helpful for bulk measurements using scipy.ndimage measurement functions"""
        if getattr(scipy_output, "__getitem__", False):
            return np.array(scipy_output)
        else:
            return np.array([scipy_output])

    @staticmethod
    def calculate_center_of_mass(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.center_of_mass(array, labels, index))

    @staticmethod
    def calculate_max(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.maximum(array, labels, index))

    @staticmethod
    def calculate_mean(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.mean(array, labels, index))

    @staticmethod
    def calculate_median(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.median(array, labels, index))

    @staticmethod
    def calculate_minimum(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.minimum(array, labels, index))

    @staticmethod
    def calculate_stddev(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.standard_deviation(array, labels, index))

    @staticmethod
    def calculate_sum(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.sum_labels(array, labels, index))

    @staticmethod
    def calculate_variance(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.variance(array, labels, index))

    @staticmethod
    def calculate_coeff_variation(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        """Calculates unbiased coefficient of variation (CV) for each object in the image, assuming normal distribution.

        References:
            - https://en.wikipedia.org/wiki/Coefficient_of_variation
        """
        unique_labels, unique_counts = np.unique(labels, return_counts=True)
        unique_counts = unique_counts[unique_labels != 0]
        biased_cv = MeasureFeatures.calculate_stddev(array, labels, index) / MeasureFeatures.calculate_mean(array, labels, index)
        return (1 + (1 / unique_counts)) * biased_cv

    @staticmethod
    def _calculate_extrema(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        min_extrema, max_extrema, min_pos, max_pos = MeasureFeatures._repair_scipy_results(scipy.ndimage.extrema(array, labels, index))
        return (
            MeasureFeatures._repair_scipy_results(min_extrema),
            MeasureFeatures._repair_scipy_results(max_extrema),
            MeasureFeatures._repair_scipy_results(min_pos),
            MeasureFeatures._repair_scipy_results(max_pos)
        )

    @staticmethod
    def calculate_min_extrema(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        min_extrema, _, min_pos, _ = MeasureFeatures._calculate_extrema(array, labels, index)
        return min_extrema, min_pos

    @staticmethod
    def calculate_max_extrema(array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None):
        _, max_extreme, _, max_pos = MeasureFeatures._calculate_extrema(array, labels, index)
        return max_extreme, max_pos

    @staticmethod
    def funcmap2objects(func: Callable, out_dtype: np.dtype,
                        array: np.ndarray, labels: ArrayLike = None, index: ArrayLike = None,
                        default: int | float | np.nan = np.nan,
                        pass_positions: bool = False):
        """Apply a custom function to labeled regions in an array.
        
        This method applies the provided function to each labeled region in the input array
        and returns the results as a numpy array. It uses scipy.ndimage.labeled_comprehension
        internally and ensures a consistent output format.
        
        Args:
            func: Function to apply to each labeled region. It should accept as input the 
                elements of the object subarray, and optionally the positions if 
                pass_positions is True.
            out_dtype: Data type of the output array.
            array: Input array to process.
            labels: Array of labels of the same shape as an input array. If None, all non-zero
                elements of the input are treated as a single object.
            index: Labels to include in the calculation. If None, all labels are used.
            default: The value to use for labels that are not in the index. Defaults to np.nan.
            pass_positions: If True, the positions where the input array is non-zero are 
                passed to func. Defaults to False.
        
        Returns:
            np.ndarray: Result of applying func to each labeled region, returned as a numpy array.
        
        Notes:
            This is a wrapper around scipy.ndimage.labeled_comprehension that ensures the
            output is always a proper numpy array.
        """
        return MeasureFeatures._repair_scipy_results(
            scipy.ndimage.labeled_comprehension(input=array, labels=labels, index=index,
                                                func=func, out_dtype=out_dtype,
                                                pass_positions=pass_positions,
                                                default=default),
        )

    @staticmethod
    def calculate_q1(array, labels=None, index=None, method: str = 'closest_observation'):
        find_q1 = partial(np.quantile, q=0.25, method=method)
        q1 = MeasureFeatures.funcmap2objects(func=find_q1, out_dtype=array.dtype, array=array, labels=labels, index=index, default=np.nan,
                                             pass_positions=False)
        return MeasureFeatures._repair_scipy_results(q1)

    @staticmethod
    def calculate_q3(array, labels=None, index=None, method: str = 'closest_observation'):
        find_q3 = partial(np.quantile, q=0.75, method=method)
        q3 = MeasureFeatures.funcmap2objects(func=find_q3, out_dtype=array.dtype, array=array, labels=labels, index=index, default=np.nan,
                                             pass_positions=False)
        return MeasureFeatures._repair_scipy_results(q3)

    @staticmethod
    def calculate_iqr(array, labels=None, index=None, method: str = 'nearest', nan_policy: str = 'omit'):
        find_iqr = partial(scipy.stats.iqr, axis=None, nan_policy=nan_policy, interpolation=method)
        return MeasureFeatures._repair_scipy_results(
            MeasureFeatures.funcmap2objects(
                func=find_iqr, out_dtype=array.dtype,
                array=array, labels=labels, index=index,
                default=np.nan, pass_positions=False,
            ),
        )
