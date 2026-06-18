from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import numpy as np

from phenotypic._core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.sdk_.exceptions_ import (
    ArrayKeyValueShapeMismatchError,
    EmptyImageError
)

# Separate napari viewer for detection mode previews
_detect_modes_viewer: napari.Viewer | None = None


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

    def reset(self):
        """Reset the detection matrix to a fresh copy of the current mode's source channel.

        Discards all modifications made to the detection matrix and restores it
        from the source determined by the image's ``detect_mode`` setting.

        Examples:
            Reset after applying unsuccessful enhancement:

            >>> detect_mat.reset()  # Revert to source channel
        """
        from phenotypic._core._image_parts.detection_modes import get_detection_mode

        mode = get_detection_mode(self._root_image._data.detect_mode)
        self._root_image._data.detect_mat = mode.compute(self._root_image)

    def preview_modes(self, reset: bool = False) -> napari.Viewer:
        """Open a napari viewer with all registered detection mode matrices.

        Computes every registered detection mode and adds each as a separate
        image layer in a dedicated napari viewer. The current (possibly
        enhanced) detection matrix is also included. Toggle layer visibility
        in napari's layer list to compare modes.

        This viewer is independent of the main ``image.gray.napari()`` viewer.

        Args:
            reset: If True, closes the existing preview viewer and creates
                a fresh one. Defaults to False.

        Returns:
            napari.Viewer: A viewer with one layer per detection mode plus
                the current detection matrix.

        Examples:
            Compare all detection modes interactively:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> viewer = image.detect_mat.preview_modes()

            Reset and re-preview after applying enhancers:

            >>> viewer = image.detect_mat.preview_modes(reset=True)
        """
        import napari as _napari

        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
            _viewer_is_alive,
        )
        from phenotypic._core._image_parts.detection_modes import (
            available_modes,
            get_detection_mode,
        )

        global _detect_modes_viewer

        # Reset viewer if requested
        if reset and _viewer_is_alive(_detect_modes_viewer):
            _detect_modes_viewer.close()
            _detect_modes_viewer = None

        # Create new viewer if needed
        if not _viewer_is_alive(_detect_modes_viewer):
            _detect_modes_viewer = _napari.Viewer(
                title="Detection Mode Preview",
            )

        viewer = _detect_modes_viewer
        has_rgb = not self._root_image.rgb.isempty()

        # Add RGB and grayscale reference layers (hidden).
        # Names use "[ref]" prefix to avoid collision with mode names
        # (e.g. the "gray" detection mode vs. the grayscale reference).
        if has_rgb:
            rgb_data = self._root_image.rgb[:]
            try:
                viewer.layers["rgb"].data = rgb_data
            except KeyError:
                viewer.add_image(
                    rgb_data, name="rgb", visible=False, rgb=True,
                    contrast_limits=(0, int(np.iinfo(rgb_data.dtype).max)), gamma=1.0,
                )

        gray_data = self._root_image.gray[:]
        try:
            viewer.layers["[ref] gray"].data = gray_data
        except KeyError:
            viewer.add_image(
                gray_data, name="[ref] gray", visible=False,
                contrast_limits=(0.0, 1.0), gamma=1.0,
            )

        # Add each registered mode as a layer
        for mode_name in available_modes():
            mode = get_detection_mode(mode_name)
            if mode.requires_rgb and not has_rgb:
                continue
            matrix = mode.compute(self._root_image)

            try:
                viewer.layers[mode_name].data = matrix
            except KeyError:
                viewer.add_image(
                    matrix, name=mode_name, visible=False,
                    contrast_limits=(0.0, 1.0), gamma=1.0,
                )

        # Add the current (possibly enhanced) detect_mat
        current_mode = self._root_image._data.detect_mode
        current_label = f"current ({current_mode})"
        current_mat = self._root_image._data.detect_mat

        try:
            viewer.layers[current_label].data = current_mat
        except KeyError:
            viewer.add_image(
                current_mat, name=current_label, visible=True,
                contrast_limits=(float(current_mat.min()), float(current_mat.max())),
                gamma=1.0,
            )

        return viewer
