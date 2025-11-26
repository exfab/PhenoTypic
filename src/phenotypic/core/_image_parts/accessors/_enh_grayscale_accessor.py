from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: pass

import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import ArrayKeyValueShapeMismatchError, EmptyImageError


class EnhancedGrayscale(SingleChannelAccessor):
    """The enhanced grayscale is a copy of the grayscale matrix used for enhancement that don't maintain data integrity
    and detection.

    Provides functionalities to manipulate and visualize the enhanced grayscale image. This includes
    retrieving and setting data, resetting the gray, visualizing histograms, viewing the gray
    with overlays, and accessing gray properties. The class relies on a handler for gray operations
    and object mapping.
    """

    _accessor_property_name: str = "enh_gray"

    def __getitem__(self, key) -> np.ndarray:
        """Return a non-writeable view of the enhanced gray for the given index."""
        if self.isempty():
            raise EmptyImageError
        else:
            view = self._root_image._data.enh_gray[key]
            view.flags.writeable = False
            return view

    def __setitem__(self, key, value):
        """
        Sets an item in the underlying dataset with the specified key and value. The method validates
        input types and ensures compliance with specified constraints, raising appropriate errors
        when conditions are not met. Specifically, scalar values or numpy arrays are allowed. If
        a numpy array is used, its shape must match the predefined dimensions associated with the
        given key.

        Args:
            key: The key representing the identifier in the dataset. It is used to locate the entry to be
                updated. Must be compatible with self._root_image._data.enh_gray data structure.
            value: The value to be assigned to the specified key. Can be a scalar (int or float) or a
                numpy array. If scalar, it directly overwrites the entry. If it's a numpy array, its
                shape must match the predefined dimensions associated with the given key.

        Raises:
            ArrayKeyValueShapeMismatchError: If the provided value is a numpy array and its shape
                does not match the shape associated with the specified key.
            TypeError: If the provided value is neither a scalar (int or float) nor a numpy array.
        """
        if isinstance(value, np.ndarray):
            if self._root_image._data.enh_gray[key].shape != value.shape: raise ArrayKeyValueShapeMismatchError
        elif isinstance(value, (int, float)):
            pass
        else:
            raise TypeError(
                    f'Unsupported type for setting the gray. Value should be scalar or a numpy array: {type(value)}')

        self._root_image._data.enh_gray[key] = value
        self._root_image.objmap.reset()

    @property
    def _subject_arr(self) -> np.ndarray:
        return self._root_image._data.enh_gray

    def reset(self):
        """Resets the image's enhanced gray to the original gray representation."""
        self._root_image._data.enh_gray = self._root_image._data.gray.copy()
