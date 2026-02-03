from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np

from phenotypic._core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools_.exceptions_ import (
    ArrayKeyValueShapeMismatchError,
    EmptyImageError
)
from phenotypic.tools_.funcs_ import normalize_rgb_bitdepth


class DetectMatAccessor(SingleChannelAccessor):
    """
    Provides the detection matrix channel accessor for image data.

    The DetectMatAccessor class represents an accessor for managing and manipulating
    the detection matrix for an entire image or specific regions. This class
    extends SingleChannelAccessor to include special behaviors and restrictions when
    working with a normalized representation of microbial colony images on
    solid media agar. Detection matrix values are typically normalized to range between
    0.0 and 1.0.

    The detection matrix source channel is controlled by the image's ``detect_mode``
    setting (``'gray'``, ``'red'``, ``'green'``, or ``'blue'``).

    This accessor facilitates retrieving and modifying detection matrix values for both
    visualization and computational purposes while enforcing integrity checks to avoid
    data inconsistency.

    """

    @property
    def _accessor_property_name(self) -> str:
        return "detect_mat"

    def __getitem__(self, key) -> np.ndarray:
        """Return a non-writeable view of the detection matrix data for the given index.

        Retrieves a portion of the detection matrix array using standard NumPy indexing.
        The returned view is read-only to prevent unintended modifications outside of the
        proper __setitem__ interface.

        Args:
            key: Array indexing expression (integer, slice, tuple of indices, or boolean mask).
                Follows standard NumPy indexing conventions.

        Returns:
            np.ndarray: A non-writeable view of the detection matrix data at the specified
                index. The returned array shares memory with the underlying data but cannot
                be modified.

        Raises:
            EmptyImageError: If the image has no data loaded (empty shape).

        Examples:
            Retrieve a single pixel value:

            >>> pixel_value = detect_mat[100, 200]

            Retrieve a rectangular region:

            >>> region = detect_mat[100:200, 50:150]
        """
        if self.isempty():
            raise EmptyImageError
        else:
            view = self._root_image._data.detect_mat[key]
            view.flags.writeable = False

            return view

    def __setitem__(self, key, value):
        """Set detection matrix data at the specified index with validation.

        Sets data in the detection matrix array at the specified location. The method
        validates that the input is either a scalar numeric value (int or float) or a
        NumPy array with a shape matching the indexed region. After successful assignment,
        the parent image's object map is reset to maintain consistency with the modified
        detection matrix data.

        Args:
            key: Array indexing expression (integer, slice, tuple of indices, or boolean mask).
                Follows standard NumPy indexing conventions and must be compatible with the
                detection matrix array structure.
            value (int | float | np.ndarray): The value(s) to assign. Can be:
                - A scalar (int or float) that will be broadcast to all indexed elements
                - A NumPy array whose shape must exactly match the indexed region's shape

        Raises:
            ArrayKeyValueShapeMismatchError: If value is a NumPy array and its shape does
                not match the shape of the indexed region.
            TypeError: If value is neither a scalar (int or float) nor a NumPy array.

        Examples:
            Set a single pixel to a scalar value:

            >>> detect_mat[100, 200] = 128

            Set a rectangular region with an array:

            >>> region_data = np.ones((100, 100), dtype=np.uint8) * 150
            >>> detect_mat[100:200, 50:150] = region_data

            Broadcast a scalar to a region:

            >>> detect_mat[0:50, 0:50] = 255  # Set all pixels in region to 255
        """
        if isinstance(value, np.ndarray):
            if self._root_image._data.detect_mat[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError
        elif isinstance(value, (int, float)):
            pass
        else:
            raise TypeError(
                    f"Unsupported type for setting the detection matrix. "
                    f"Value should be scalar or a numpy array: {type(value)}"
            )

        self._root_image._data.detect_mat[key] = value
        self._root_image.objmap.reset()

    def vmax(self) -> float:
        """Returns the maximum value in the detection matrix. Since it is
        normalized to [0.0, 1.0], it returns 1.0."""
        return 1.0

    def vmin(self) -> float:
        """Returns the minimum value in the detection matrix. Since it is
        normalized to [0.0, 1.0], it returns 0.0."""
        return 0.0

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return the underlying detection matrix array.

        This property provides access to the detection matrix data array used by inherited
        visualization and analysis methods from ImageAccessorBase and SingleChannelAccessor.

        Returns:
            np.ndarray: The detection matrix image data with shape (rows, columns).
        """
        return self._root_image._data.detect_mat

    def _get_color_channel(self, mode: str) -> np.ndarray:
        """Extract a single color channel from the RGB data as float32.

        Args:
            mode: One of ``'red'``, ``'green'``, ``'blue'``.

        Returns:
            np.ndarray: The extracted channel as a float32 array in [0, 1].
        """
        rgb_normed = normalize_rgb_bitdepth(self._root_image._data.rgb)
        channel_map = {"red": 0, "green": 1, "blue": 2}
        return rgb_normed[:, :, channel_map[mode]].astype(np.float32).clip(0, 1)

    def reset(self):
        """Reset the detection matrix to a fresh copy of the current mode's source channel.

        Discards all modifications made to the detection matrix and restores it from
        the source determined by the image's ``detect_mode`` setting:

        - ``'gray'``: copies from ``image.gray``
        - ``'red'`` / ``'green'`` / ``'blue'``: extracts the corresponding RGB channel

        Examples:
            Reset after applying unsuccessful enhancement:

            >>> detect_mat.reset()  # Revert to source channel
        """
        mode = self._root_image._data.detect_mode
        if mode == "gray":
            self._root_image._data.detect_mat = self._root_image._data.gray.copy()
        else:
            self._root_image._data.detect_mat = self._get_color_channel(mode)
