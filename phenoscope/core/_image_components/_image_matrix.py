from typing import Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import histogram

from ...util.constants import C_ImageMatrixSubhandler


class ImageMatrixSubhandler:
    """An immutable accessor for the image matrix data. Accessible like a numpy array. This is used for the actual measurements after detection.

    Note:
        The ImageMatrix is left unchanged so that any detection improvements don't bias the true value of the images
    """

    def __init__(self, handler):
        """Initiallizes the ImageMatrixSubhandler object.

              Args:
                  handler: (ImageHandler) The parent ImageHandler that the ImageMatrixSubhandler belongs to.
              """
        self._handler = handler

    def __getitem__(self, key):
        return self._handler._matrix[key].copy()

    def __setitem__(self, key, value):
        raise C_ImageMatrixSubhandler.IllegalElementAssignmentError('Image.matrix')

    @property
    def shape(self) -> tuple:
        """Returns the shape of the image matrix (num_rows:int, num_columns:int)"""
        return self._handler._matrix.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the image matrix"""
        return self._handler._matrix.copy()

    def histogram(self, figsize: Tuple[int, int] = (10, 5)):
        """Returns a histogram of the image matrix. Useful for troubleshooting detection results.
        Args:
            figsize:

        Returns:

        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes[0].imshow(self._handler.matrix[:])
        axes[0].set_title(self._handler.name)

        hist_one, histc_one = histogram(self._handler.matrix[:])
        axes[1].plot(hist_one, histc_one, lw=2)
        axes[1].set_title('Grayscale Histogram')
        return fig, axes

    def show(self, ax: plt.Axes = None, figsize: Tuple[int, int] = None, cmap: str = 'gray', title: str = None) -> (plt.Figure, plt.Axes):
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
        ax.imshow(self._handler.matrix[:], cmap=cmap)

        # Adjust ax settings
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    def show_overlay(self, object_label: Optional[int] = None, ax: plt.Axes = None,
                     figsize: Tuple[int, int] = None,
                     annotate: bool = False,
                     annotation_size: int = 12,
                     annotation_color: str = 'white',
                     annotation_facecolor: str = 'red',
                     ) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.grid(False)

        map_copy = self._handler.obj_map[:]
        if object_label is not None:
            map_copy[map_copy == object_label] = 0

        ax.imshow(label2rgb(label=map_copy, image=self._handler.matrix[:]))

        if annotate:
            for i, label in enumerate(self._handler.objects.labels):
                if object_label is None:
                    text_rr, text_cc = self._handler.objects.props[i].centroid
                    ax.text(
                        x=text_cc, y=text_rr,
                        s=f'{label}',
                        color=annotation_color,
                        fontsize=annotation_size,
                        bbox=dict(facecolor=annotation_facecolor, edgecolor='none', alpha=0.6, boxstyle='round')
                    )
                elif object_label == label:
                    text_rr, text_cc = self._handler.objects.props[i].centroid
                    ax.text(
                        x=text_cc, y=text_rr,
                        s=f'{label}',
                        color=annotation_color,
                        fontsize=annotation_size,
                        bbox=dict(facecolor=annotation_facecolor, edgecolor='none', alpha=0.6, boxstyle='round')
                    )


        return fig, ax
