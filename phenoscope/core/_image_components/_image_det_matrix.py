from typing import Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from skimage.color import label2rgb

from ...util.constants import C_ImageDetectionMatrixSubhandler


class ImageDetectionMatrixSubhandler:
    """Provides access to the image detection matrix. The detection matrix is a copy of the original image matrix that can be preprocessed for detection.
    """

    def __init__(self, handler):
        """Initializes the ImageDetectionMatrix.

        Args:
            handler: The ImageHandler instance that contains the image detection matrix.
        """
        self._handler = handler

    def __getitem__(self, key) -> np.ndarray:
        return self._handler._det_matrix[key].copy()

    def __setitem__(self, key, value):
        if type(value) not in [int, float, bool]:
            if self._handler._det_matrix[key].shape != value.shape:
                raise C_ImageDetectionMatrixSubhandler.ArrayKeyValueShapeMismatchError
        else:
            self._handler._det_matrix[key] = value
            self._handler.obj_map.reset()

    @property
    def shape(self):
        return self._handler._det_matrix.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the Detection Matrix."""
        return self._handler._det_matrix.copy()

    def reset(self):
        """Resets the image detection matrix to the original matrix representation."""
        self._handler._det_matrix = self._handler.matrix[:].copy()

    def show(self, ax: plt.Axes = None, figsize: str = None, cmap: str = 'gray', title: str = None) -> (plt.Figure, plt.Axes):
        """Display the image detection matrix with matplotlib.

        Args:
            ax: (plt.Axes) Axes object to use for plotting.
            figsize: (Tuple[int, int]): Figure size in inches.
            cmap: (str) Colormap name.
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
        ax.imshow(self._handler.det_matrix[:], cmap=cmap)

        # Adjust ax settings
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    def show_overlay(self, object_label: Optional[int] = None, ax: plt.Axes = None,
                     figsize: Tuple[int, int] = None) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.grid(False)

        map_copy = self._handler.obj_map[:]
        if object_label is not None:
            map_copy[map_copy == object_label] = 0

        ax.imshow(label2rgb(label=map_copy, image=self._handler.det_matrix[:]))

        return fig, ax
