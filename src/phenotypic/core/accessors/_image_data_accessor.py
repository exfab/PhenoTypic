import numpy as np

import warnings

from phenotypic.core.accessors import ImageAccessor
from phenotypic.util.exceptions_ import InterfaceError


class ImageDataAccessor(ImageAccessor):
    """
    Handles interaction with image data by providing access to image attributes and data.

    This class serves as a bridge for interacting with image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessor`.

    Attributes:
        _parent_image (Any): Parent image object that this accessor is linked to.
        _target_arr (Any): Main array storing the image-related data.
        _dtype (Any): Data type of the image data stored in the target array.
    """

    def __init__(self, parent_image, target_array, dtype):
        self._parent_image = parent_image
        self._target_arr = target_array
        self._dtype = dtype

    def shape(self) -> tuple[int, ...]:
        return self._target_arr.shape

    def isempty(self):
        return True if self.shape[0] == 0 else False

    def _norm2dtype(self, normalized_matrix: np.ndarray):
        """
        Converts a normalized matrix with values between 0 and 1 to a specified data type with the
        appropriate scaling. The method ensures that all values are clipped to the range [0, 1]
        before scaling them to the data type's maximum value.

        Args:
            normalized_matrix: A 2D NumPy array where all values are assumed to be in the range
                [0, 1]. These values will be converted using the specified data type scale.

        Returns:
            numpy.ndarray: A 2D NumPy array of the same shape as `normalized_matrix`, converted
            to the target data type with scaled values.
        """
        max_val = np.iinfo(self._dtype)
        return (np.clip(normalized_matrix, a_min=0, a_max=1) * max_val).astype(self._dtype)

    def _dtype2norm(self, matrix: np.ndarray):
        """
        Normalizes the given matrix to have values between 0.0 and 1.0 based on its data type.

        The method checks the data type of the input matrix against the expected data
        type. If the data type does not match, a warning is issued. The matrix is
        then normalized by dividing its values by the maximum possible value for its
        data type, ensuring all elements remain within the range of [0.0, 1.0].

        Args:
            matrix (np.ndarray): The input matrix to be normalized.

        Returns:
            np.ndarray: A normalized matrix where all values are within [0.0, 1.0].
        """
        if matrix.dtype != self._dtype:
            warnings.warn(f'Possible Integrity Error: Image dtype {matrix.dtype} does not match the expected dtype {self._dtype}.')

        max_val = np.iinfo(matrix.dtype).max
        return np.clip(matrix.astype(np.float64) / max_val, a_min=0.0, a_max=1.0)
