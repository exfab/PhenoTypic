import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import skimage

from phenotypic.core._image_parts.accessor_abstracts import ImageArrDataAccessor
from phenotypic.util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError, NoArrayError, EmptyImageError


class ImageArray(ImageArrDataAccessor):
    """An accessor for handling image arrays with helper methods for accessing, modifying, visualizing, and analyzing the multichannel image data.

    It relies on the parent image handler object that serves as the bridge to the underlying image
    array and associated metadata or attributes.

    The class allows users to interact with image arrays intuitively while providing
    features such as advanced visualization (both for the raw images and their derived
    representations, like histograms or overlays). Through its properties and methods,
    users can explore, manipulate, and analyze the structural or geometrical attributes
    of the image and its segmented objects.

    Key use cases for this class include displaying selected channels or the entire
    image (including overlays and highlighted objects), generating channel-specific
    histograms, and accessing image data attributes, such as shape.

    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Returns a copy of the elements at the subregion specified by the given key.

        This class provides a mechanism for extracting a specific subregion from
        the multichannel image array. The extracted subregion is represented in the form of a
        NumPy array, and its indexable nature allows users to freely interact with the
        underlying array data.

        Returns:
            np.ndarray: A copy of the extracted subregion represented as a NumPy array.
        """
        if self.isempty():
            if self._root_image.matrix.isempty():
                raise EmptyImageError
            else:
                raise NoArrayError
        else:
            return self._root_image._data.array[key].copy()

    def __setitem__(self, key, value):
        """
        Sets a other_image for a given key in the parent image array. The other_image must either be of
        type int, float, or bool, or it must match the shape of the corresponding key's other_image
        in the parent image array. If the other_image's shape does not align with the required shape,
        an exception is raised.

        Note:
            If you want to change the entire image array data, use Image.set_image() instead.

        Args:
            key: Index key specifying the location in the parent image array to modify.
            value: The new other_image to assign to the specified key in the array. Can be of types
                int, float, or bool. If not, it must match the shape of the target array segment.

        Raises:
            ArrayKeyValueShapeMismatchError: If the other_image is an array and its shape does not match
        """
        if isinstance(value, (int, float, np.ndarray)):
            if isinstance(value, np.ndarray):
                if value.shape != self._root_image._data.array[key].shape:
                    raise ArrayKeyValueShapeMismatchError

            self._root_image._data.array[key] = value
            self._root_image._set_from_array(self._root_image._data.array, imformat=self._root_image.imformat)
        else:
            raise ValueError(f'Unsupported type for setting the array. Value should be scalar or a numpy array: {type(value)}')

    @property
    def _subject_arr(self):
        return self._root_image._data.array

    def show(self,
             channel: int | None = None,
             figsize: tuple[int, int] | None = None,
             title: str | None = None,
             ax: plt.Axes | None = None,
             mpl_params: dict | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays the image array, either the full array or a specific channel, using matplotlib.

        Args:
            channel (int | None): Specifies the channel to display from the image array. If None,
                the entire array is displayed. If an integer is provided, only the specified
                channel is displayed.
            figsize (None | tuple[int, int]): Optional tuple specifying the width and height of
                the figure in inches. If None, defaults to matplotlib's standard figure size.
            title (str | None): Title text for the plot. If None, no title will be displayed.
            ax (plt.Axes): Optional matplotlib Axes instance. If provided, the plot will be
                drawn on this axes object. If None, a new figure and axes will be created.
            mpl_params (dict | None): Optional dictionary of keyword arguments for customizing
                matplotlib parameters (e.g., color maps, fonts). If None, default settings will
                be used.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects
                associated with the plot. If `ax` is provided in the arguments, the returned
                tuple will include that Axes instance; otherwise, a new Figure and Axes pair
                will be returned.

        """

        if channel is None:
            return self._plot(arr=self._root_image.array[:], ax=ax, figsize=figsize, title=title, mpl_kwargs=mpl_params)

        else:
            title = f"{self._root_image.name} - Channel {channel}" if title is None else f'{title} - Channel {channel}'
            return self._plot(arr=self._root_image.array[:, :, channel], ax=ax, figsize=figsize, title=title, mpl_kwargs=mpl_params)
