import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import pandas as pd
import skimage

from phenotypic.core._image_parts.accessor_abstracts import MultiChannelAccessor
from phenotypic.tools.constants_ import IMAGE_MODE
from phenotypic.tools.exceptions_ import (
    ArrayKeyValueShapeMismatchError,
    NoArrayError,
    EmptyImageError,
)


class ImageRGB(MultiChannelAccessor):
    """Accessor for interacting with RGB multichannel image data.

    ImageRGB provides a comprehensive interface for accessing, modifying, visualizing,
    and analyzing RGB image arrays. It acts as a bridge to the underlying RGB data stored
    in the parent Image object, exposing the data through intuitive indexing operations
    and utility methods.

    This accessor supports advanced visualization capabilities including display of
    individual channels or composite images, overlays with segmentation maps, and
    channel-specific histograms. Users can seamlessly manipulate image arrays, explore
    geometric and structural attributes, and analyze segmented objects.

    The class inherits from MultiChannelAccessor and extends functionality for
    3-channel (RGB) images specifically. All visualization and analysis methods
    from the base class are available and can handle 3-channel color data.

    Attributes:
        _accessor_property_name (str): Property name on Image that returns this accessor.
            Set to "rgb" to identify this as the RGB accessor.

    Raises:
        EmptyImageError: If attempting to access data when no image is loaded.
        NoArrayError: If attempting to access RGB data when the image is 2D
            (no RGB array form exists).

    Examples:
        .. dropdown:: Access RGB data from an Image object

            .. code-block:: python

                rgb_accessor = image.rgb
                # Get a copy of the full RGB array
                rgb_data = rgb_accessor[:]
                # Modify a region
                rgb_accessor[10:20, 10:20] = [255, 0, 0]
                # Display the image
                fig, ax = rgb_accessor.show()
    """

    _accessor_property_name: str = "rgb"

    def __getitem__(self, key) -> np.ndarray:
        """Return a read-only view of the RGB subregion specified by the given key.

        This method supports NumPy-style indexing and slicing to extract subregions
        from the RGB image array. The returned array is read-only to prevent accidental
        modifications outside of the __setitem__ interface.

        Args:
            key: Index or slice specifying the subregion to extract. Supports standard
                NumPy indexing (e.g., integer indices, slices, boolean masks, advanced
                indexing). For 3D RGB data, can use notation like [:, :, channel] to
                select specific channels.

        Returns:
            np.ndarray: A read-only NumPy array containing the extracted subregion
                with shape matching the selected region.

        Raises:
            EmptyImageError: If the image contains no RGB data and no grayscale
                fallback is available.
            NoArrayError: If the image is 2D (grayscale only) and no RGB array
                form exists.

        Examples:
            .. dropdown:: Extract the full RGB array

                .. code-block:: python

                    full_rgb = image.rgb[:]

            .. dropdown:: Extract a rectangular region

                .. code-block:: python

                    region = image.rgb[10:50, 20:60, :]

            .. dropdown:: Extract a specific channel

                .. code-block:: python

                    red_channel = image.rgb[:, :, 0]
        """
        if self.isempty():
            if self._root_image.gray.isempty():
                raise EmptyImageError
            else:
                raise NoArrayError
        else:
            view = self._root_image._data.rgb[key]
            view.flags.writeable = False
            return view

    def __setitem__(self, key, value):
        """Modify a subregion of the RGB image array.

        This method allows in-place modification of the underlying RGB image data using
        NumPy-style indexing. The provided value must either be a numeric scalar or a
        NumPy array with shape matching the indexed subregion.

        Args:
            key: Index or slice specifying the subregion to modify. Supports standard
                NumPy indexing (e.g., integer indices, slices, boolean masks, advanced
                indexing).
            value (int | float | np.number | np.ndarray): The new value(s) to assign.
                Can be a numeric scalar (int, float, or np.number) which will be broadcast
                to all elements in the indexed region, or a NumPy array with dtype
                compatible with the image array. If an array, its shape must exactly
                match the shape of the indexed subregion.

        Raises:
            TypeError: If value is a scalar that is not numeric (int, float, or np.number).
            TypeError: If value is an array with non-numeric dtype.
            TypeError: If value is neither a scalar nor a numpy array.
            ArrayKeyValueShapeMismatchError: If value is an array and its shape does
                not match the shape of the indexed subregion.

        Note:
            To replace the entire RGB image data, use Image.set_image() instead of
            this method. This method only modifies specific regions of the existing
            image array.

            After modification, the parent image is updated to reflect changes made
            through this accessor.

        Examples:
            .. dropdown:: Set a region to a solid color

                .. code-block:: python

                    image.rgb[10:20, 10:20] = [255, 128, 64]

            .. dropdown:: Set a region from another array

                .. code-block:: python

                    patch = np.ones((10, 10, 3), dtype=np.uint8) * 128
                    image.rgb[10:20, 10:20] = patch

            .. dropdown:: Modify a single channel

                .. code-block:: python

                    image.rgb[:, :, 0] = 255  # Set red channel to maximum
        """
        if pd.api.types.is_scalar(value):  # handle scalar values
            if not isinstance(value, (int, float, np.number)):  # assert numeric value
                raise TypeError("Array values must be a numeric scalar or np.array")

        elif isinstance(value, np.ndarray):  # handle numpy arrays
            if not np.issubdtype(value.dtype, np.number):  # assert numeric numpy value
                raise TypeError("Array values must be a numeric scalar or np.array")
            if (
                value.shape != self._root_image._data.rgb[key].shape
            ):  # assert window shape equals value shape
                raise ArrayKeyValueShapeMismatchError

        else:
            raise ValueError(
                f"Unsupported type for setting the array. Value should be scalar or a numpy array: {type(value)}"
            )

        self._root_image._data.rgb[key] = value
        self._root_image._set_from_array(self._root_image._data.rgb)

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return a copy of the underlying RGB image array.

        This property implements the abstract _subject_arr interface from ImageAccessorBase,
        providing access to the RGB array data managed by the parent Image object.

        The returned array is a copy, allowing safe inspection and manipulation without
        affecting the underlying image data directly. Use __setitem__ to modify the
        image and ensure proper synchronization with the parent Image object.

        Returns:
            np.ndarray: A copy of the RGB image array with shape (height, width, 3)
                for RGB images. The array dtype matches the image's bit depth
                (typically uint8 or uint16).

        Note:
            This property always returns a copy, not a view. Modifications to the
            returned array do not affect the parent image. Use the indexing interface
            (image.rgb[key] = value) to modify the image.
        """
        return self._root_image._data.rgb.copy()
