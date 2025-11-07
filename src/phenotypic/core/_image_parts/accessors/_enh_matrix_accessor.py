from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: pass

import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import ArrayKeyValueShapeMismatchError, EmptyImageError


class ImageEnhancedMatrix(SingleChannelAccessor):
    """An accessor class to an image's enhanced gray which is a copy of the original image gray that is preprocessed for enhanced detection.

    Provides functionalities to manipulate and visualize the image enhanced gray. This includes
    retrieving and setting data, resetting the gray, visualizing histograms, viewing the gray
    with overlays, and accessing gray properties. The class relies on a handler for gray operations
    and object mapping.
    """

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
        Sets a other_image in the detection gray of the parent image for the provided key.

        The method updates or sets a other_image in the detection gray of the parent image
        (`image._det_matrix`) at the specified key. It ensures that if the other_image
        is not of type `int`, `float`, or `bool`, its shape matches the shape of the
        existing other_image at the specified key. If the shape does not match,
        `ArrayKeyValueShapeMismatchError` is raised. When the other_image is successfully set,
        the object map (`objmap`) of the parent image is reset.

        Notes:
            Objects are reset after setting a other_image in the detection gray

        Args:
            key: The key in the detection gray where the other_image will be set.
            value: The other_image to be assigned to the detection gray. Must be of type
                int, float, or bool, or must have a shape matching the existing array
                in the detection gray for the provided key.

        Raises:
            ArrayKeyValueShapeMismatchError: If the other_image is an array and its shape
                does not match the shape of the existing other_image in `image._det_matrix`
                for the specified key.
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
