from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: pass

import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import ImageMatrixDataAccessor
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError, EmptyImageError


class ImageEnhancedMatrix(ImageMatrixDataAccessor):
    """An accessor class to an image's enhanced matrix which is a copy of the original image matrix that is preprocessed for enhanced detection.

    Provides functionalities to manipulate and visualize the image enhanced matrix. This includes
    retrieving and setting data, resetting the matrix, visualizing histograms, viewing the matrix
    with overlays, and accessing matrix properties. The class relies on a handler for matrix operations
    and object mapping.
    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Provides a method to retrieve a copy of a specific portion of a parent image's detection
        matrix based on the given key.

        Args:
            key: The index or slice used to access a specific part of the parent image's detection
                matrix.

        Returns:
            numpy.ndarray: A copy of the corresponding portion of the parent image's detection
                matrix.
        """
        if self.isempty():
            raise EmptyImageError
        else:
            return self._root_image._data.enh_matrix[key].copy()

    def __setitem__(self, key, value):
        """
        Sets a other_image in the detection matrix of the parent image for the provided key.

        The method updates or sets a other_image in the detection matrix of the parent image
        (`image._det_matrix`) at the specified key. It ensures that if the other_image
        is not of type `int`, `float`, or `bool`, its shape matches the shape of the
        existing other_image at the specified key. If the shape does not match,
        `ArrayKeyValueShapeMismatchError` is raised. When the other_image is successfully set,
        the object map (`objmap`) of the parent image is reset.

        Notes:
            Objects are reset after setting a other_image in the detection matrix

        Args:
            key: The key in the detection matrix where the other_image will be set.
            value: The other_image to be assigned to the detection matrix. Must be of type
                int, float, or bool, or must have a shape matching the existing array
                in the detection matrix for the provided key.

        Raises:
            ArrayKeyValueShapeMismatchError: If the other_image is an array and its shape
                does not match the shape of the existing other_image in `image._det_matrix`
                for the specified key.
        """
        if isinstance(value, np.ndarray):
            if self._root_image._data.enh_matrix[key].shape != value.shape: raise ArrayKeyValueShapeMismatchError
        elif isinstance(value, (int, float)):
            pass
        else:
            raise TypeError(f'Unsupported type for setting the matrix. Value should be scalar or a numpy array: {type(value)}')

        self._root_image._data.enh_matrix[key] = value
        self._root_image.objmap.reset()

    @property
    def shape(self):
        """
        Represents the shape property of the parent image's enhanced matrix.

        This property fetches and returns the dimensions (shape) of the enhanced
        matrix that belongs to the parent image linked with the current class.

        Returns:
            tuple: The shape of the determinant matrix.
        """
        return self._root_image._data.enh_matrix.shape

    def reset(self):
        """Resets the image's enhanced matrix to the original matrix representation."""
        self._root_image._data.enh_matrix = self._root_image._data.matrix.copy()
