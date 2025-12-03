from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from skimage.measure import label
import matplotlib.pyplot as plt
import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import (
    InvalidMaskValueError,
    InvalidMaskScalarValueError,
    ArrayKeyValueShapeMismatchError,
)


class ObjectMask(SingleChannelAccessor):
    """Manages a binary object mask linked to a parent image.

    ObjectMask provides array-like access and manipulation of a binary mask indicating
    object locations in the parent image. It supports slicing, item assignment, copying,
    and visualization. The mask is backed by a sparse representation stored in the parent
    image's `sparse_object_map`, with automatic relabeling via scikit-image's `label()`
    function to maintain consistent object IDs after modifications.

    The object mask distinguishes between foreground (object) pixels (value 1) and
    background pixels (value 0). Any modification to the mask via `__setitem__` triggers
    automatic relabeling to ensure object label consistency across the parent image's
    object map.

    Attributes:
        _root_image (Image): The parent Image object containing this mask.

    Note:
        Changes to the object mask will trigger automatic relabeling of the object map
        to maintain consistent object IDs. This ensures data integrity when mask regions
        are directly modified.

    Examples:
        .. dropdown:: Basic access and slicing

            .. code-block:: python

                # Access the mask as a dense array
                mask_array = np.array(objmask)

                # Slice operations
                region = objmask[10:50, 20:60]

                # Modify mask regions
                objmask[10:50, 20:60] = np.zeros((40, 40))

        .. dropdown:: Creating foreground images

            .. code-block:: python

                # Get foreground of a grayscale channel
                foreground = image.gray.foreground()
    """

    @property
    def _backend(self):
        """Return the sparse matrix backend of the object mask.

        Provides direct access to the underlying sparse representation of the binary
        mask stored in the parent image's data structure. This avoids redundant
        dense conversions for internal operations.

        Returns:
            scipy.sparse matrix: The sparse matrix representation of the object mask,
                matching the shape and values of the parent image.
        """
        return self._root_image._data.sparse_object_map

    def __array__(self, dtype=None, copy=None):
        """Implement the NumPy array interface for seamless integration with NumPy.

        Converts the sparse binary mask to a dense NumPy array, enabling direct use
        of NumPy functions and operations on the ObjectMask. The mask is always returned
        as binary values (0 for background, 1 for foreground).

        Args:
            dtype (type, optional): Target NumPy dtype for the returned array. If None,
                defaults to int. Defaults to None.
            copy (bool, optional): If True, ensures the returned array is a copy.
                Ignored if dtype is specified. For NumPy 2.0+ compatibility.
                Defaults to None.

        Returns:
            np.ndarray: A dense binary array of shape matching the parent image with
                int dtype (0 and 1 values), or the specified dtype.

        Examples:
            .. dropdown:: Using with NumPy functions and type conversion

                .. code-block:: python

                    # Use with NumPy functions
                    num_foreground_pixels = np.count_nonzero(objmask)
                    total_pixels = np.sum(objmask)

                    # Explicit type conversion
                    mask_float = np.array(objmask, dtype=np.float32)
        """
        arr = (self._backend.toarray() > 0).astype(int)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        elif copy:
            arr = arr.copy()
        return arr

    def __getitem__(self, key):
        """Return a slice of the binary object mask with NumPy-compatible indexing.

        Supports all standard NumPy slicing operations (integer indexing, slice objects,
        boolean masks, fancy indexing). The sparse mask is converted to dense form and
        binary values (0 and 1) are returned.

        Args:
            key (int, slice, tuple, np.ndarray): Index or slice to retrieve. Follows
                NumPy indexing conventions for 2D arrays.

        Returns:
            np.ndarray: Binary array (0 and 1 values) representing the sliced region of
                the mask, with the same shape as the indexed region.

        Examples:
            .. dropdown:: NumPy slicing and indexing operations

                .. code-block:: python

                    # Single row
                    row = objmask[10]

                    # Rectangular region
                    region = objmask[10:50, 20:60]

                    # Column
                    col = objmask[:, 5]

                    # Boolean indexing
                    foreground_indices = objmask > 0
        """
        return (self._backend.toarray()[key] > 0).astype(int)

    def __setitem__(self, key, value: np.ndarray):
        """Update mask values at specified locations with automatic relabeling.

        Sets mask values at the specified indices or slices, then automatically
        relabels the entire mask using scikit-image's `label()` function to ensure
        object IDs remain consistent across the parent image. This maintains data
        integrity when mask regions are directly modified.

        The operation accepts scalar values (0, 1, bool) or arrays. Non-zero values
        are converted to 1 (foreground), zero values to 0 (background). The updated
        mask is then relabeled atomically to avoid inconsistent states.

        Args:
            key (int, slice, tuple): Index or slice indicating which mask elements
                to update. Follows NumPy indexing conventions.
            value (int, bool, np.ndarray): Value(s) to set. Scalars are normalized
                to binary (0 or 1). Arrays are converted to binary and must match
                the shape of mask[key].

        Raises:
            InvalidMaskScalarValueError: If a scalar value cannot be converted to
                a valid binary value (not int, bool, or convertible to these types).
            InvalidMaskValueError: If value is not an int, bool, or ndarray.
            ArrayKeyValueShapeMismatchError: If an ndarray value's shape does not
                match the shape of the indexed mask region.

        Examples:
            .. dropdown:: Setting mask regions with scalars and arrays

                .. code-block:: python

                    # Set a rectangular region to background (0)
                    objmask[10:50, 20:60] = 0

                    # Set with a matching array
                    region_mask = np.ones((40, 40))
                    objmask[10:50, 20:60] = region_mask

                    # Set single pixel
                    objmask[5, 10] = True

        Note:
            The entire mask is relabeled after any modification. This ensures
            that object IDs in the parent image remain consistent, but may
            change existing object labels if the relabeling alters connectivity.
        """
        # Get current mask as dense array (convert once)
        mask = self._backend.toarray() > 0

        # Apply the value based on type
        if isinstance(value, (int, bool)):
            try:
                value = 1 if value != 0 else 0
                mask[key] = value
            except TypeError:
                raise InvalidMaskScalarValueError
        elif isinstance(value, np.ndarray):
            # Check arr and section have matching shape
            if mask[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError

            # Sets the section of the binary mask to the value array
            mask[key] = value > 0
        else:
            raise InvalidMaskValueError(type(value))

        # Relabel the mask and update the backend atomically
        # This is where the relabeling occurs to maintain consistent object IDs
        relabeled = label(mask)
        new_sparse = self._root_image.objmap._dense_to_sparse(relabeled)
        new_sparse.eliminate_zeros()
        self._root_image._data.sparse_object_map = new_sparse

    @property
    def shape(self):
        """Return the shape of the object mask.

        The shape is always identical to the parent image's shape, since the mask
        covers the entire image extent.

        Returns:
            tuple[int, ...]: Shape of the mask as (height, width), matching the
                parent image dimensions.

        Examples:
            .. dropdown:: Accessing mask shape

                .. code-block:: python

                    height, width = objmask.shape
                    assert objmask.shape == image.gray.shape
        """
        return self._root_image.objmap.shape

    def copy(self) -> np.ndarray:
        """Return an independent copy of the binary object mask.

        Creates a new array containing the same binary values (0 and 1) as the
        current mask. Modifications to the returned array do not affect the original
        mask or the parent image.

        Returns:
            np.ndarray: A dense copy of the binary mask with int dtype, independent
                of the original sparse representation.

        Examples:
            .. dropdown:: Creating an independent mask copy

                .. code-block:: python

                    # Create a modifiable copy for processing
                    mask_copy = objmask.copy()
                    mask_copy[10:50, 20:60] = 0  # Doesn't affect objmask
        """
        return (self._backend.toarray() > 0).astype(int).copy()

    def reset(self):
        """Reset the object mask and linked object map to a cleared state.

        Delegates to the parent image's object map's reset method, which clears
        the mask and resets all object labels and properties. This is useful when
        re-segmenting the image or clearing previous detection results.

        Examples:
            .. dropdown:: Resetting the mask to cleared state

                .. code-block:: python

                    # Clear the mask and object map
                    objmask.reset()
                    # Now objmask contains only background (0s)

        Note:
            This operation affects both the object mask and the parent image's object
            map and properties. Use with caution as it discards all segmentation data.
        """
        self._root_image.objmap.reset()

    def show(
        self,
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] | None = None,
        cmap: str = "gray",
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Display the binary object mask as a Matplotlib image.

        Renders the object mask using Matplotlib's imshow with customizable appearance.
        The mask is shown as a grayscale image where white represents foreground
        (objects) and black represents background.

        Args:
            ax (plt.Axes, optional): An existing Matplotlib Axes object to plot on.
                If None, a new figure and axes are created. Defaults to None.
            figsize (tuple[int, int], optional): Size of the figure in inches as
                (width, height). Only used if ax is None. Defaults to None
                (uses default size).
            cmap (str, optional): Colormap to apply. Defaults to 'gray', which shows
                foreground pixels in white and background in black.
            title (str, optional): Title for the plot. If None, no title is displayed.
                Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: Matplotlib Figure and Axes objects containing
                the rendered mask.

        Examples:
            .. dropdown:: Displaying the mask with various options

                .. code-block:: python

                    # Display with default settings
                    fig, ax = objmask.show()

                    # Display with custom size and title
                    fig, ax = objmask.show(figsize=(8, 8), title='Object Mask')

                    # Display on existing axes
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    objmask.show(ax=ax1, title='Mask')
                    image.gray.show(ax=ax2, title='Original')
        """
        return self._plot(
            arr=self._subject_arr, figsize=figsize, ax=ax, title=title, cmap=cmap
        )

    def _create_foreground(self, array: np.ndarray, bg_label: int = 0) -> np.ndarray:
        """Extract foreground from an array using the object mask.

        Applies the binary object mask to an input array, setting all background
        pixels (where mask is 0) to a specified value. For multi-channel arrays,
        the mask is replicated across channels. This is equivalent to masked array
        fill operation.

        Args:
            array (np.ndarray): Input array to mask. Can be 2D (single channel)
                or 3D (multi-channel). Must have shape matching the parent image.
            bg_label (int, optional): Value to assign to background pixels (where
                mask is 0). Defaults to 0.

        Returns:
            np.ndarray: A copy of array with background pixels set to bg_label.
                Foreground pixels retain their original values.

        Examples:
            .. dropdown:: Extracting foreground from various array types

                .. code-block:: python

                    # Extract foreground from a grayscale image
                    foreground = objmask._create_foreground(image.gray[:])

                    # Extract foreground from RGB image
                    foreground_rgb = objmask._create_foreground(image.rgb[:])

                    # Set background to white (255 for uint8)
                    foreground = objmask._create_foreground(image.gray[:], bg_label=255)

        Note:
            This is an internal method used by the parent image's accessor classes
            to implement the `foreground()` functionality. For 3D arrays, the mask
            is automatically broadcasted across all channels.
        """
        mask = self._backend.toarray() > 0
        if array.ndim == 3:
            mask = np.dstack([mask for _ in range(array.shape[-1])])

        array[~mask] = bg_label
        return array

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return the dense binary representation of the object mask.

        Converts the sparse backend representation to a dense NumPy array with
        binary values (0 and 1). This property is used internally by parent class
        methods (e.g., `show()`, `histogram()`) for visualization and analysis.

        Returns:
            np.ndarray: Dense binary array with ``uint16`` dtype, same shape as the
                parent image. Using ``uint16`` ensures that, when possible, the
                mask is saved as a 16-bit image (e.g. PNG/TIFF) while JPEG output
                is still handled as 8-bit by the shared ``imsave`` logic.
        """
        # NOTE:
        # ``ImageAccessorBase.imsave`` preserves ``uint16`` arrays for formats
        # that support 16-bit data (e.g. PNG/TIFF) and only downcasts to
        # ``uint8`` for JPEGs (with a warning). By exposing the object mask as a
        # ``uint16`` array here, objmask images will be written as 16-bit where
        # possible, and automatically converted to 8-bit for JPEGs. This matches
        # the behaviour of ``ObjectMap`` whose backend is already ``uint16``.
        return (self._backend.toarray() > 0).astype(np.uint16)
