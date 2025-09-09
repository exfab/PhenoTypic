from abc import ABC
from pathlib import Path

import numpy
import numpy as np
import skimage

import skimage as ski
import skimage.util
import matplotlib.pyplot as plt

from phenotypic.core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic.util.constants_ import METADATA_LABELS


class MultiChannelAccessor(ImageAccessorBase):
    """
    Handles interaction with Image data by providing access to Image attributes and data.

    This class serves as a bridge for interacting with Image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    Image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessorBase`.

    Attributes:
        image (Any): Root Image object that this accessor is linked to.
        _main_arr (Any): Main array storing the Image-related data.
        _dtype (Any): Data type of the Image data stored in the target array.
    """

    def imsave(self, fname: str | Path):
        fname = Path(fname)
        arr = self._subject_arr.copy()
        if (arr.dtype != np.uint8) or (arr.dtype != np.uint16):
            match self._root_image.metadata[METADATA_LABELS.BIT_DEPTH]:
                case 8:
                    arr = ski.util.img_as_ubyte(arr)
                case 16:
                    arr = ski.util.img_as_uint(arr)
                case _:
                    raise AttributeError(f"Unsupported bit depth: {self._root_image.metadata[METADATA_LABELS.BIT_DEPTH]}")
        ski.io.imsave(fname=fname, arr=arr, check_contrast=False)

    def show(self,
             figsize: tuple[int, int] | None = None,
             title: str | None = None,
             ax: plt.Axes | None = None,
             channel: int | None = None,
             foreground_only: bool = False,
             *,
             mpl_settings: dict | None = None) -> tuple[plt.Figure, plt.Axes]:
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
            mpl_settings (dict | None): Optional dictionary of keyword arguments for customizing
                matplotlib parameters (e.g., color maps, fonts). If None, default settings will
                be used.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects
                associated with the plot. If `ax` is provided in the arguments, the returned
                tuple will include that Axes instance; otherwise, a new Figure and Axes pair
                will be returned.

        """
        arr = self[:] if not foreground_only else self.foreground()
        if channel is None:
            return self._plot(arr=arr, ax=ax, figsize=figsize, title=title, mpl_settings=mpl_settings)

        else:
            title = f"{self._root_image.name} - Channel {channel}" if title is None else f'{title} - Channel {channel}'
            return self._plot(arr=arr[:, :, channel], ax=ax, figsize=figsize, title=title, mpl_settings=mpl_settings)
