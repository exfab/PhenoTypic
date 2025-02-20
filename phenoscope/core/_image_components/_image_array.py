import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from skimage.color import rgb2hsv, label2rgb

from ...util.constants import C_ImageArraySubhandler, C_ImageFormats


class ImageArraySubhandler:
    """An immutable accessor image's multichannel information. Access image elements similar to a numpy array."""

    def __init__(self, handler):
        self._handler = handler

    def __getitem__(self, key) -> np.ndarray:
        return self._handler._array[key].copy()

    def __setitem__(self, key, value):
        raise C_ImageArraySubhandler.IllegalElementAssignmentError

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Returns the shape of the image"""
        return self._handler._array.shape

    def copy(self)->np.ndarray:
        """Returns a copy of the image array"""
        return self._handler._array.copy()

    def get_hsv(self) -> np.ndarray:
        if self._handler.schema != C_ImageFormats.HSV:
            raise C_ImageArraySubhandler.InvalidSchemaHsv(self._handler.schema)
        else:
            return rgb2hsv(self._handler._array)

    def show(self, ax: plt.Axes = None, figsize: Tuple[int, int] = None, title: str = None) -> (plt.Figure, plt.Axes):
        """Display the image array with matplotlib.

        Args:
            ax: (plt.Axes) Axes object to use for plotting.
            figsize: (Tuple[int, int]): Figure size in inches.
            title: (str) a title for the plot

        Returns:
            tuple(plt.Figure, plt.Axes): matplotlib figure and axes object
        """
        # Defaults
        if figsize is None: figsize = (6, 4)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot array
        ax.imshow(self._handler.array[:])

        # Adjust ax settings
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    def show_overlay(self, object_label: Optional[int] = None, ax: plt.Axes = None,
                     figsize: Tuple[int, int] = None)->(plt.Figure, plt.Axes):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.grid(False)

        map_copy = self._handler.obj_map[:]
        if object_label is not None:
            map_copy[map_copy==object_label]=0

        ax.imshow(label2rgb(label=map_copy, image=self._handler.array[:]))

        return fig, ax


