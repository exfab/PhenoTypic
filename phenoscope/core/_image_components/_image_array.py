import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from ...util.constants import C_ImageArray


class ImageArray:
    """An immutable container for the image's multi-channel information. Access image elements similar to a numpy array.

    """

    def __init__(self, handler, image_array: np.ndarray):
        self._handler = handler
        self.__array = image_array

    def __getitem__(self, key)->np.ndarray:
        return self.__array[key]

    def __setitem__(self, key, value):
        raise C_ImageArray.IllegalElementAssignmentError

    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the image"""
        return self.__array.shape

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
        ax.imshow(self.__array)

        # Adjust ax settings
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax
