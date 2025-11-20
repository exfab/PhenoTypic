from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Callable

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.api.types import is_scalar
import scipy
import warnings
from functools import partial, wraps

from ._base_operation import BaseOperation
from phenotypic.tools.exceptions_ import OperationFailedError
from phenotypic.tools.funcs_ import validate_measure_integrity
from phenotypic.tools.constants_ import OBJECT
from abc import ABC


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
class MeasureFeatures(BaseOperation, ABC):
    """
    A FeatureExtractor is an abc_ object intended to calculate measurements on the values within detected objects of
    the image array. The __init__ constructor & _operate method is meant to be the only parts overloaded in inherited classes. This is so
    that the main measure method call can contain all the necessary type validation and output validation checks to streamline development.
    """

    @validate_measure_integrity()
    def measure(self, image: Image, include_meta: bool = False) -> pd.DataFrame:
        """
        Executes a measurement operation on the provided image and optionally includes
        metadata for the output.

        This method applies a series of operations to the input image using arguments
        matched from pre-defined configurations. It returns the results in a
        pandas DataFrame. Optionally, additional metadata about the operation can
        be included. If any exceptions occur during operation, they are captured and
        re-raised as a custom `OperationFailedError`.

        Args:
            image (Image): The image on which the measurement operation is to be
                performed.
            include_meta (bool, optional): Determines whether to include metadata
                about the operation in the result. Defaults to False.

        Raises:
            OperationFailedError: If the operation fails due to any exception, this
                error is raised, encapsulating details about the failure.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the measurement
                operation. If `include_meta` is True, additional columns with the
                metadata are added.
        """

        try:
            matched_args = self._get_matched_operation_args()

            meas = self._operate(image, **matched_args)
            if include_meta:
                meta = (image.grid.info(include_metadata=True)
                        if hasattr(image, 'grid')
                        else image.objects.info(include_metadata=True)
                        )
                meas = meta.merge(meas, on=OBJECT.LABEL)

            return meas



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
    def _ensure_array(value) -> np.ndarray:
        """
        Ensures that the input value is converted into a numpy array. This utility method
        is particularly useful for scenarios where scalar inputs need to be handled
        uniformly as an array for consistency in computations.

        Args:
            value: The input value which can be a scalar or already an array-like entity.

        Returns:
            np.ndarray: A numpy array representation of the input value.
        """
        if is_scalar(value):
            return np.asarray(value)
        else:
            return np.asarray(value)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_center_of_mass(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the center of mass for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Coordinates of the center of mass for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.center_of_mass(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_max(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the maximum value for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Maximum value for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.maximum(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_mean(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the mean value for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Mean value for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.mean(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_median(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the median value for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Median value for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.median(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_minimum(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the minimum value for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Minimum value for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.minimum(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_stddev(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the standard deviation for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Standard deviation for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.standard_deviation(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_sum(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the sum of values for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Sum of values for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.sum_labels(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_variance(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the variance for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Variance for each labeled object.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(scipy.ndimage.variance(array, objmap, index=indexes))

    @staticmethod
    @catch_warnings_decorator
    def _calculate_coeff_variation(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates unbiased coefficient of variation (CV) for each object in the image, assuming normal distribution.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            np.ndarray: Coefficient of variation for each labeled object.

        References:
            - https://en.wikipedia.org/wiki/Coefficient_of_variation
        """
        if objmap is not None:
            unique_labels, unique_counts = np.unique(objmap, return_counts=True)
            unique_counts = unique_counts[unique_labels != 0]
            biased_cv = MeasureFeatures._calculate_stddev(array, objmap)/MeasureFeatures._calculate_mean(array, objmap)
            result = (1 + (1/unique_counts))*biased_cv
        else:
            # For the case when objmap is None, we can't calculate the coefficient of variation
            # because we need the counts of each label
            result = np.nan
        return MeasureFeatures._ensure_array(result)

    @staticmethod
    def _calculate_extrema(array: np.ndarray, objmap: ArrayLike = None):
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        min_extrema, max_extrema, min_pos, max_pos = MeasureFeatures._ensure_array(
                scipy.ndimage.extrema(array, objmap, index=indexes))
        return (
            MeasureFeatures._ensure_array(min_extrema),
            MeasureFeatures._ensure_array(max_extrema),
            MeasureFeatures._ensure_array(min_pos),
            MeasureFeatures._ensure_array(max_pos)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_min_extrema(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the minimum extrema and their positions for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Minimum extrema values for each labeled object.
                - np.ndarray: Positions of minimum extrema for each labeled object.
        """
        min_extrema, _, min_pos, _ = MeasureFeatures._calculate_extrema(array, objmap)
        return min_extrema, min_pos

    @staticmethod
    @catch_warnings_decorator
    def _calculate_max_extrema(array: np.ndarray, objmap: ArrayLike = None):
        """Calculates the maximum extrema and their positions for each labeled object in the array.

        Args:
            array: Input array to process.
            objmap: Array of labels of the same shape as the input array. If None, all non-zero
                elements of the input are treated as a single object.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Maximum extrema values for each labeled object.
                - np.ndarray: Positions of maximum extrema for each labeled object.
        """
        _, max_extreme, _, max_pos = MeasureFeatures._calculate_extrema(array, objmap)
        return max_extreme, max_pos

    @staticmethod
    def _funcmap2objects(func: Callable, out_dtype: np.dtype,
                         array: np.ndarray, objmap: ArrayLike = None,
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
            objmap: Array of labels of the same shape as an input array. If None, all non-zero
                elements of the input are treated as a single object.
            default: The value to use for labels that are not in the index. Defaults to np.nan.
            pass_positions: If True, the positions where the input array is non-zero are 
                passed to func. Defaults to False.

        Returns:
            np.ndarray: Result of applying func to each labeled region, returned as a numpy array.

        Notes:
            This is a wrapper around scipy.ndimage.labeled_comprehension that ensures the
            output is always a proper numpy array.
        """
        if objmap is not None:
            index = np.unique(objmap)
            index = index[index != 0]
        else:
            index = None

        return MeasureFeatures._ensure_array(
                scipy.ndimage.labeled_comprehension(input=array, labels=objmap, index=index,
                                                    func=func, out_dtype=out_dtype,
                                                    pass_positions=pass_positions,
                                                    default=default),
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_q1(array, objmap=None, method: str = 'linear'):
        find_q1 = partial(np.quantile, q=0.25, method=method)
        q1 = MeasureFeatures._funcmap2objects(func=find_q1, out_dtype=array.dtype, array=array, objmap=objmap,
                                              default=np.nan,
                                              pass_positions=False)
        return MeasureFeatures._ensure_array(q1)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_q3(array, objmap=None, method: str = 'linear'):
        find_q3 = partial(np.quantile, q=0.75, method=method)
        q3 = MeasureFeatures._funcmap2objects(func=find_q3, out_dtype=array.dtype, array=array, objmap=objmap,
                                              default=np.nan,
                                              pass_positions=False)
        return MeasureFeatures._ensure_array(q3)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_iqr(array, objmap=None, method: str = 'linear', nan_policy: str = 'omit'):
        find_iqr = partial(scipy.stats.iqr, axis=None, nan_policy=nan_policy, interpolation=method)
        return MeasureFeatures._ensure_array(
                MeasureFeatures._funcmap2objects(
                        func=find_iqr, out_dtype=array.dtype,
                        array=array, objmap=objmap,
                        default=np.nan, pass_positions=False,
                ),
        )
