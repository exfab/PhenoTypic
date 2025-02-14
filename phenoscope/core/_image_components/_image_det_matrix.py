import numpy as np

import matplotlib.pyplot as plt

from ...util.constants import C_ImageDetectionMatrix

class ImageDetectionMatrix:
    """Holds the image detection matrix. This is a mutable copy of the image matrix which is detection is applied on.

    The image detection matrix is an enhanceable copy of the image matrix that can be preprocessed to improve detection performance.

    """

    def __init__(self, handler, preset_detection_matrix:np.ndarray=None):
        """Initializes the ImageDetectionMatrix.

        Args:
            handler: The ImageHandler instance that contains the image detection matrix.
            preset_detection_matrix: (np.ndarray) A preset image detection matrix that can be applied
        """
        self._handler = handler

        if preset_detection_matrix is None:
            self.__det_matrix = self._handler.matrix[:]
        else:
            if preset_detection_matrix.shape != self._handler.matrix.shape: raise C_ImageDetectionMatrix.InputShapeMismatchError(preset_detection_matrix)
            else:
                self.__det_matrix = preset_detection_matrix

    def __getitem__(self, key)->np.ndarray:
        return self.__det_matrix[key]

    def __setitem__(self, key, value):
        if self.__det_matrix[key].shape != value.shape: raise C_ImageDetectionMatrix.ArrayKeyValueShapeMismatchError
        self.__det_matrix[key] = value

    @property
    def shape(self):
        return self.__det_matrix.shape

    def reset(self):
        """Resets the image detection matrix to the original matrix representation."""
        self.__det_matrix = self._handler.matrix[:]

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
        ax.imshow(self.__det_matrix, cmap=cmap)

        # Adjust ax settings
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax
