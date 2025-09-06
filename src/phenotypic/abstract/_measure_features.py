from __future__ import annotations
from typing import TYPE_CHECKING

from typing_extensions import Callable

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy
import warnings
import functools
from functools import partial, wraps

from ._base_operation import BaseOperation
from phenotypic.util.exceptions_ import OperationFailedError
from phenotypic.util.funcs_ import validate_measure_integrity


def catch_warnings_decorator(func):
    """
    A decorator that catches warnings, prepends the method name to the warning message,
    and reraises the warning.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            # Call the original function
            warnings.simplefilter("ignore")
            result = func(*args, **kwargs)

            # If any warnings were raised, prepend the method name and reraise
        for warning in recorded_warnings:
            message = f"{func.__name__}: {warning.message}"
            warnings.warn(message, warning.category, stacklevel=2)

        return result
    return wrapper


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

        This is helpful for bulk measurements using scipy.ndimage measurement functions.

        Args:
            scipy_output: The output from a scipy function that needs to be converted to a numpy array.

        Returns:
            np.array: A numpy array containing the scipy output.
        """
        if getattr(scipy_output, "__getitem__", False):
            return np.array(scipy_output)
        else:
            return np.array([scipy_output])

    @staticmethod
    @catch_warnings_decorator
    def _calculate_center_of_mass(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the center of mass for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Coordinates of the center of mass for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes=None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.center_of_mass(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_max(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the maximum value for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Maximum value for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.maximum(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_mean(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the mean value for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Mean value for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.mean(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_median(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the median value for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Median value for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.median(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_minimum(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the minimum value for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Minimum value for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.minimum(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_stddev(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the standard deviation for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Standard deviation for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.standard_deviation(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_sum(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the sum of values for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Sum of values for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.sum_labels(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_variance(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the variance for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Variance for each labeled object.
        """
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._repair_scipy_results(scipy.ndimage.variance(array, labels, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_coeff_variation(array: np.ndarray, labels: ArrayLike = None):
        """Calculates unbiased coefficient of variation (CV) for each object in the image, assuming normal distribution.

        References:
            - https://en.wikipedia.org/wiki/Coefficient_of_variation
        """
        if labels is not None:
            unique_labels, unique_counts = np.unique(labels, return_counts=True)
            unique_counts = unique_counts[unique_labels != 0]
            biased_cv = MeasureFeatures._calculate_stddev(array, labels) / MeasureFeatures._calculate_mean(array, labels)
            result = (1 + (1 / unique_counts)) * biased_cv
        else:
            # For the case when labels is None, we can't calculate the coefficient of variation
            # because we need the counts of each label
            result = np.nan
        return MeasureFeatures._repair_scipy_results(result)

    @staticmethod
    def _calculate_extrema(array: np.ndarray, labels: ArrayLike = None):
        if labels is not None:
            indexes = np.unique(labels)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        min_extrema, max_extrema, min_pos, max_pos = MeasureFeatures._repair_scipy_results(scipy.ndimage.extrema(array, labels, index=indexes))
        return (
            MeasureFeatures._repair_scipy_results(min_extrema),
            MeasureFeatures._repair_scipy_results(max_extrema),
            MeasureFeatures._repair_scipy_results(min_pos),
            MeasureFeatures._repair_scipy_results(max_pos)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_min_extrema(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the minimum extrema and their positions for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Minimum extrema values for each labeled object.
                - np.ndarray: Positions of minimum extrema for each labeled object.
        """
        min_extrema, _, min_pos, _ = MeasureFeatures._calculate_extrema(array, labels)
        return min_extrema, min_pos

    @staticmethod
    @catch_warnings_decorator
    def _calculate_max_extrema(array: np.ndarray, labels: ArrayLike = None):
        """Calculates the maximum extrema and their positions for each labeled object in the array.

        Args:
            array: Input array to process.
            labels: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Maximum extrema values for each labeled object.
                - np.ndarray: Positions of maximum extrema for each labeled object.
        """
        _, max_extreme, _, max_pos = MeasureFeatures._calculate_extrema(array, labels)
        return max_extreme, max_pos

    @staticmethod
    def _funcmap2objects(func: Callable, out_dtype: np.dtype,
                         array: np.ndarray, labels: ArrayLike = None,
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
        if labels is not None:
            index = np.unique(labels)
            index = index[index != 0]
        else:
            index=None

        return MeasureFeatures._repair_scipy_results(
            scipy.ndimage.labeled_comprehension(input=array, labels=labels, index=index,
                                                func=func, out_dtype=out_dtype,
                                                pass_positions=pass_positions,
                                                default=default),
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_q1(array, labels=None, method: str = 'linear'):
        find_q1 = partial(np.quantile, q=0.25, method=method)
        q1 = MeasureFeatures._funcmap2objects(func=find_q1, out_dtype=array.dtype, array=array, labels=labels, default=np.nan,
                                              pass_positions=False)
        return MeasureFeatures._repair_scipy_results(q1)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_q3(array, labels=None, method: str = 'linear'):
        find_q3 = partial(np.quantile, q=0.75, method=method)
        q3 = MeasureFeatures._funcmap2objects(func=find_q3, out_dtype=array.dtype, array=array, labels=labels, default=np.nan,
                                              pass_positions=False)
        return MeasureFeatures._repair_scipy_results(q3)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_iqr(array, labels=None, method: str = 'linear', nan_policy: str = 'omit'):
        find_iqr = partial(scipy.stats.iqr, axis=None, nan_policy=nan_policy, interpolation=method)
        return MeasureFeatures._repair_scipy_results(
            MeasureFeatures._funcmap2objects(
                func=find_iqr, out_dtype=array.dtype,
                array=array, labels=labels,
                default=np.nan, pass_positions=False,
            ),
        )
