from __future__ import annotations

import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import ImageMatrixDataAccessor
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError, EmptyImageError


class ImageMatrix(ImageMatrixDataAccessor):
    """An accessor for managing and visualizing image matrix data. This is the greyscale representation converted using weighted luminance

    This class provides a set of tools to access image data, analyze it through
    histograms, and visualize results. The class utilizes a parent
    Image object to interact with the underlying matrix data while
    maintaining immutability for direct external modifications.
    Additionally, it supports overlaying annotations and labels on the image
    for data analysis purposes.
    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Provides functionality to retrieve a copy of a specified portion of the parent image's
        matrix. This class method is used to access the image matrix data, or slices of the parent image
        matrix based on the provided key.

        Args:
            key (any): A key used to index or slice the parent image's matrix.

        Returns:
            np.ndarray: A copy of the accessed subset of the parent image's matrix with normalized values.
        """
        if self.isempty():
            raise EmptyImageError
        else:
            return self._root_image._data.matrix[key].copy()

    def __setitem__(self, key, value):
        """
        Sets the other_image for a given key in the parent image's matrix. Changes are not reflected in the color matrix,
        and any objects detected are reset.

        Args:
            key: The key in the matrix to update.
            value: The new other_image to assign to the key. Must be an array of a compatible
                shape or a primitive type like int, float, or bool.

        Raises:
            ArrayKeyValueShapeMismatchError: If the shape of the other_image does not match
                the shape of the existing key in the parent image's matrix.
        """
        if isinstance(value, np.ndarray):
            if self._root_image._data.matrix[key].shape != value.shape: raise ArrayKeyValueShapeMismatchError
            assert (0 <= np.min(value) <= 1) and (0 <= np.max(value) <= 1), 'matrix values must be between 0 and 1'
        elif isinstance(value, (int, float)):
            assert 0 <= value <= 1, 'matrix values must be between 0 and 1'
        else:
            raise TypeError(f'Unsupported type for setting the matrix. Value should be scalar or a numpy array: {type(value)}')

        self._root_image._data.matrix[key] = value
        self._root_image.enh_matrix.reset()
        self._root_image.objmap.reset()

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the parent image matrix.

        This property retrieves the dimensions of the associated matrix from the
        parent image that this object references.

        Returns:
            tuple: A tuple representing the shape of the parent image's matrix.
        """
        return self._root_image._data.matrix.shape
