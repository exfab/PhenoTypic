from __future__ import annotations

import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import (
    ArrayKeyValueShapeMismatchError,
    EmptyImageError,
)


class Grayscale(SingleChannelAccessor):
    """Accessor for managing and visualizing grayscale image data.

    This class provides an interface for interacting with the luminance-based grayscale
    representation of an image. It supports data access, modification, visualization through
    histograms, and overlay annotations. The accessor maintains immutability of data at the
    external interface level while allowing controlled modifications through dedicated methods.

    The grayscale data is stored as a 2D floating-point array with normalized values in the
    range [0.0, 1.0].

    Attributes:
        _accessor_property_name (str): Property name used to access this accessor from the
            Image object (set to "gray").

    Examples:
        .. dropdown:: Access and display grayscale data

            .. code-block:: python

                img = Image("path/to/image.png")
                gray_array = img.gray[:]
                fig, ax = img.gray.show()

        .. dropdown:: Modify grayscale data with validation

            .. code-block:: python

                img.gray[10:20, 10:20] = 0.5  # Set region to mid-gray
                img.gray[100, 100] = 0.8      # Set single pixel to light gray

        .. dropdown:: Visualize histogram and overlay

            .. code-block:: python

                fig, axes = img.gray.histogram()
                fig, ax = img.gray.show_overlay(show_labels=True)
    """

    _accessor_property_name: str = "gray"

    def __getitem__(self, key) -> np.ndarray:
        """Retrieve a read-only view or slice of the grayscale image data.

        Allows array-style indexing and slicing to access subsets of the grayscale data.
        The returned array is marked read-only to prevent accidental modifications. Use
        __setitem__ for intentional modifications.

        Args:
            key: Index or slice specification. Can be an integer for single-row access,
                tuple of slices for multi-dimensional slicing (e.g., [10:20, 5:15]),
                or boolean arrays for advanced indexing.

        Returns:
            np.ndarray: A read-only view of the requested grayscale data. Values are
                normalized floating-point numbers in the range [0.0, 1.0].

        Raises:
            EmptyImageError: If the underlying image data is empty (shape[0] == 0).

        Examples:
            .. dropdown:: Access grayscale data with various indexing techniques

                .. code-block:: python

                    # Access entire grayscale array
                    full_gray = img.gray[:]

                    # Slice a region
                    region = img.gray[10:20, 5:15]

                    # Access single row
                    row = img.gray[10]

                    # Advanced indexing not recommended but supported
                    mask = img.gray > 0.5
        """
        if self.isempty():
            raise EmptyImageError
        else:
            view = self._root_image._data.gray[key]
            view.flags.writeable = False
            return view

    def __setitem__(self, key, value) -> None:
        """Modify grayscale image data at specified indices with validation.

        Allows assignment of new values to grayscale data at specified locations. All values
        must be normalized to the range [0.0, 1.0]. Modifications trigger automatic reset of
        dependent data structures (enhanced grayscale and object map) to maintain consistency.

        Args:
            key: Index or slice specification. Same indexing schemes as __getitem__
                (e.g., [10:20, 5:15], [100, 100], [:]).
            value (np.ndarray | int | float): New value(s) to assign. Can be:
                - A NumPy array with shape matching the indexed region.
                - A scalar (int or float) to broadcast to the indexed region.
                Must contain values in the range [0.0, 1.0].

        Raises:
            ArrayKeyValueShapeMismatchError: If value is an ndarray and its shape does not
                match the shape of the indexed region.
            TypeError: If value is not an ndarray, int, or float.
            AssertionError: If values are outside the valid range [0.0, 1.0].

        Notes:
            - All grayscale values must be normalized to [0.0, 1.0].
            - Modifications automatically reset `enh_gray` and `objmap` to prevent stale data.
            - For bulk operations, consider using direct array indexing on a copy.

        Examples:
            .. dropdown:: Modify grayscale data with different assignment patterns

                .. code-block:: python

                    # Set a region to a specific value
                    img.gray[10:20, 5:15] = 0.5

                    # Set a single pixel
                    img.gray[100, 100] = 0.8

                    # Assign from another array
                    region = np.random.rand(10, 10)
                    img.gray[10:20, 5:15] = region
        """
        if isinstance(value, np.ndarray):
            if self._root_image._data.gray[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError
            assert (0 <= np.min(value) <= 1) and (0 <= np.max(value) <= 1), (
                "gray values must be between 0 and 1"
            )
        elif isinstance(value, (int, float)):
            assert 0 <= value <= 1, "gray values must be between 0 and 1"
        else:
            raise TypeError(
                f"Unsupported type for setting the gray. Value should be scalar or a numpy array: {type(value)}"
            )

        self._root_image._data.gray[key] = value
        self._root_image.enh_gray.reset()
        self._root_image.objmap.reset()

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return the underlying grayscale image array.

        This property provides direct access to the 2D grayscale array stored in the
        parent Image object. It is used internally by parent class methods for histogram
        computation, visualization, and other image analysis operations.

        Returns:
            np.ndarray: The 2D grayscale image array with shape (height, width). Values
                are normalized floating-point numbers in the range [0.0, 1.0].

        Note:
            This is an abstract property implementation required by the ImageAccessorBase
            base class. Direct modifications via this property are discouraged; use
            __setitem__ instead to maintain data consistency and trigger dependent resets.
        """
        return self._root_image._data.gray
