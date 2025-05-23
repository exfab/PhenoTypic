import numpy as np

import warnings

import skimage.util
from phenotypic.core.accessors import ImageAccessor
from phenotypic.util.exceptions_ import InterfaceError


class ImageDataAccessor(ImageAccessor):
    """
    Handles interaction with _parent_image data by providing access to _parent_image attributes and data.

    This class serves as a bridge for interacting with _parent_image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    _parent_image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessor`.

    Attributes:
        _parent_image (Any): Parent _parent_image object that this accessor is linked to.
        _target_arr (Any): Main array storing the _parent_image-related data.
        _dtype (Any): Data type of the _parent_image data stored in the target array.
    """

    def __init__(self, parent_image, target_array, dtype):
        self._parent_image = parent_image
        self._target_arr = target_array
        self._dtype = dtype

    def shape(self) -> tuple[int, ...]:
        return self._target_arr.shape

    def isempty(self):
        return True if self.shape[0] == 0 else False

    def _norm2dtype(self, normalized_value: np.ndarray) -> np.ndarray:
        """
        Converts a normalized matrix with values between 0 and 1 to a specified data type with the
        appropriate scaling. The method ensures that all values are clipped to the range [0, 1]
        before scaling them to the data type's maximum other_image.

        Args:
            normalized_value: A 2D NumPy array where all values are assumed to be in the range
                [0, 1]. These values will be converted using the specified data type scale.

        Returns:
            numpy.ndarray: A 2D NumPy array of the same shape as `normalized_matrix`, converted
            to the target data type with scaled values.
        """
        match self._dtype:
            case np.uint8:
                return skimage.util.img_as_ubyte(normalized_value)
            case np.uint16:
                return skimage.util.img_as_uint(normalized_value)
            case _:
                raise AttributeError(f'Unsupported dtype {self._dtype} for matrix storage conversion')

    def _dtype2norm(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes the given matrix to have values between 0.0 and 1.0 based on its data type.

        The method checks the data type of the input matrix against the expected data
        type. If the data type does not match, a warning is issued. The matrix is
        then normalized by dividing its values by the maximum possible other_image for its
        data type, ensuring all elements remain within the range of [0.0, 1.0].

        Args:
            matrix (np.ndarray): The input matrix to be normalized.

        Returns:
            np.ndarray: A normalized matrix where all values are within [0.0, 1.0].
        """
        return skimage.util.img_as_float(matrix)
