from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import (
    ArrayKeyValueShapeMismatchError,
    EmptyImageError,
)


class EnhancedGrayscale(SingleChannelAccessor):
    """Accessor for manipulating and visualizing enhanced grayscale image data.

    EnhancedGrayscale provides access to a mutable copy of the original grayscale matrix
    that can be modified for enhancement operations without compromising the integrity of
    the original grayscale data or object detection results. This class enables retrieval
    and modification of enhanced grayscale data, resetting to original values, and accessing
    visualization and analysis methods inherited from SingleChannelAccessor.

    The enhanced grayscale representation is useful for applying non-destructive filters,
    adjustments, or transformations that should not affect the parent image's core data
    or segmentation results.

    Attributes:
        _accessor_property_name (str): The property name on the Image object that returns
            this accessor. Set to "enh_gray".
    """

    _accessor_property_name: str = "enh_gray"

    def __getitem__(self, key) -> np.ndarray:
        """Return a non-writeable view of the enhanced grayscale data for the given index.

        Retrieves a portion of the enhanced grayscale array using standard NumPy indexing.
        The returned view is read-only to prevent unintended modifications outside of the
        proper __setitem__ interface.

        Args:
            key: Array indexing expression (integer, slice, tuple of indices, or boolean mask).
                Follows standard NumPy indexing conventions.

        Returns:
            np.ndarray: A non-writeable view of the enhanced grayscale data at the specified
                index. The returned array shares memory with the underlying data but cannot
                be modified.

        Raises:
            EmptyImageError: If the image has no data loaded (empty shape).

        Examples:
            .. dropdown:: Retrieve a single pixel value

                .. code-block:: python

                    pixel_value = enh_gray[100, 200]

            .. dropdown:: Retrieve a rectangular region

                .. code-block:: python

                    region = enh_gray[100:200, 50:150]
        """
        if self.isempty():
            raise EmptyImageError
        else:
            view = self._root_image._data.enh_gray[key]
            view.flags.writeable = False
            return view

    def __setitem__(self, key, value):
        """Set enhanced grayscale data at the specified index with validation.

        Sets data in the enhanced grayscale array at the specified location. The method
        validates that the input is either a scalar numeric value (int or float) or a
        NumPy array with a shape matching the indexed region. After successful assignment,
        the parent image's object map is reset to maintain consistency with the modified
        grayscale data.

        Args:
            key: Array indexing expression (integer, slice, tuple of indices, or boolean mask).
                Follows standard NumPy indexing conventions and must be compatible with the
                enhanced grayscale array structure.
            value (int | float | np.ndarray): The value(s) to assign. Can be:
                - A scalar (int or float) that will be broadcast to all indexed elements
                - A NumPy array whose shape must exactly match the indexed region's shape

        Raises:
            ArrayKeyValueShapeMismatchError: If value is a NumPy array and its shape does
                not match the shape of the indexed region.
            TypeError: If value is neither a scalar (int or float) nor a NumPy array.

        Examples:
            .. dropdown:: Set a single pixel to a scalar value

                .. code-block:: python

                    enh_gray[100, 200] = 128

            .. dropdown:: Set a rectangular region with an array

                .. code-block:: python

                    region_data = np.ones((100, 100), dtype=np.uint8) * 150
                    enh_gray[100:200, 50:150] = region_data

            .. dropdown:: Broadcast a scalar to a region

                .. code-block:: python

                    enh_gray[0:50, 0:50] = 255  # Set all pixels in region to 255
        """
        if isinstance(value, np.ndarray):
            if self._root_image._data.enh_gray[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError
        elif isinstance(value, (int, float)):
            pass
        else:
            raise TypeError(
                f"Unsupported type for setting the gray. Value should be scalar or a numpy array: {type(value)}"
            )

        self._root_image._data.enh_gray[key] = value
        self._root_image.objmap.reset()

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return the underlying enhanced grayscale array.

        This property provides access to the enhanced grayscale data array used by inherited
        visualization and analysis methods from ImageAccessorBase and SingleChannelAccessor.

        Returns:
            np.ndarray: The enhanced grayscale image data with shape (rows, columns).
        """
        return self._root_image._data.enh_gray

    def reset(self):
        """Reset the enhanced grayscale to a copy of the original grayscale data.

        Discards all modifications made to the enhanced grayscale and restores it to match
        the original grayscale representation. This is useful when reverting failed or
        unwanted enhancement operations. The reset operation creates a fresh copy of the
        original grayscale data to ensure subsequent modifications do not affect the
        original.

        Examples:
            .. dropdown:: Reset after applying unsuccessful enhancement

                .. code-block:: python

                    enh_gray.reset()  # Revert to original grayscale
        """
        self._root_image._data.enh_gray = self._root_image._data.gray.copy()
