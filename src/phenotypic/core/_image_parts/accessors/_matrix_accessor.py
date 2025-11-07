from __future__ import annotations

import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import ArrayKeyValueShapeMismatchError, EmptyImageError


class ImageMatrix(SingleChannelAccessor):
    """An accessor for managing and visualizing image gray data. This is the greyscale representation converted using weighted luminance

    This class provides a set of tools to access image data, analyze it through
    histograms, and visualize results. The class utilizes a parent
    Image object to interact with the underlying gray data while
    maintaining immutability for direct external modifications.
    Additionally, it supports overlaying annotations and labels on the image
    for data analysis purposes.
    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Provides functionality to retrieve a copy of a specified portion of the parent image's
        gray. This class method is used to access the image gray data, or slices of the parent image
        gray based on the provided key.

        Args:
            key (any): A key used to index or slice the parent image's gray.

        Returns:
            np.ndarray: A copy of the accessed subset of the parent image's gray with normalized values.
        """
        if self.isempty():
            raise EmptyImageError
        else:
            view = self._root_image._data.gray[key]
            view.flags.writeable = False
            return view

    def __setitem__(self, key, value):
        """
        Sets the other_image for a given key in the parent image's gray. Changes are not reflected in the color gray,
        and any objects detected are reset.

        Args:
            key: The key in the gray to update.
            value: The new other_image to assign to the key. Must be an array of a compatible
                shape or a primitive type like int, float, or bool.

        Raises:
            ArrayKeyValueShapeMismatchError: If the shape of the other_image does not match
                the shape of the existing key in the parent image's gray.
        """
        if isinstance(value, np.ndarray):
            if self._root_image._data.gray[key].shape != value.shape: raise ArrayKeyValueShapeMismatchError
            assert (0 <= np.min(value) <= 1) and (0 <= np.max(value) <= 1), 'gray values must be between 0 and 1'
        elif isinstance(value, (int, float)):
            assert 0 <= value <= 1, 'gray values must be between 0 and 1'
        else:
            raise TypeError(
                    f'Unsupported type for setting the gray. Value should be scalar or a numpy array: {type(value)}')

        self._root_image._data.gray[key] = value
        self._root_image.enh_gray.reset()
        self._root_image.objmap.reset()

    @property
    def _subject_arr(self) -> np.ndarray:
        return self._root_image._data.gray
