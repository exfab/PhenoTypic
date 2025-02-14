import numpy as np

import matplotlib.pyplot as plt

from ...util.constants import C_ImageMatrix


class ImageMatrix:
    """An immutable container for the image matrix representation. Accessible like a numpy array. This is used for the actual measurements after detection.

    Note:
        The ImageMatrix is left unchanged so that any detection improvements don't bias the true value of the images
    """

    def __init__(self, handler, image_matrix: np.ndarray):
        if len(image_matrix.shape) != 2: raise ValueError("Image matrix must be 2D")

        self._handler = handler

        self.__matrix = image_matrix

    def __getitem__(self, key):
        return self.__matrix[key]

    def __setitem__(self, key, value):
        raise C_ImageMatrix.IllegalElementAssignmentError('Image.matrix')

    @property
    def shape(self) -> tuple:
        """Returns the shape of the image matrix (num_rows:int, num_columns:int)"""
        return self.__matrix.shape

    def show(self, ax: plt.Axes = None, figsize: str = None, cmap: str = 'gray', title: str = None) -> (plt.Figure, plt.Axes):
        """Display the image matrix with matplotlib.

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
        ax.imshow(self.__matrix, cmap=cmap)

        # Adjust ax settings
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax
