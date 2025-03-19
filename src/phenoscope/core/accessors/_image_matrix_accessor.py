from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

from typing import Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import histogram

from phenoscope.util.exceptions_ import IllegalElementAssignmentError

class ImageMatrix:
    """An accessor for managing and visualizing image matrix data. This is the greyscale representation converted using weighted luminance

    This class provides a set of tools to access image data, analyze it through
    histograms, and visualize results. The class utilizes a parent
    Image object to interact with the underlying matrix data while
    maintaining immutability for direct external modifications.
    Additionally, it supports overlaying annotations and labels on the image
    for data analysis purposes.

    Attributes:
        _parent_image (ImageHandler): The parent image handler object that manages
            the image matrix and associated data.
    """

    def __init__(self, parent_image: Image):
        """Initiallizes the ImageMatrixSubhandler object.

              Args:
                  parent_image: (Image) The parent ImageHandler that the ImageMatrixSubhandler belongs to.
              """
        self._parent_image = parent_image

    def __getitem__(self, key):
        """
        Provides the ability to access and return a copy of an internal matrix element from
        the parent image using key-based indexing.

        Args:
            key: The index or slice used to access the corresponding element in the
                parent image's matrix.

        Returns:
            A copy of the matrix element or slice of the parent image's internal
            matrix corresponding to the provided key.

        """
        return self._parent_image._matrix[key].copy()

    def __setitem__(self, key, value):
        """
        Handles operations for managing and assigning sub-elements in an image matrix. The image matrix should be changed through the Image.set_images() method.

        Raises:
            IllegalElementAssignmentError: If an attempt is made to assign an invalid element to the image matrix.

        """
        raise IllegalElementAssignmentError('Image.matrix')

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the parent image matrix.

        This property retrieves the dimensions of the associated matrix from the
        parent image that this object references.

        Returns:
            tuple: A tuple representing the shape of the parent image's matrix.
        """
        return self._parent_image._matrix.shape

    def copy(self) -> np.ndarray:
        """
        Returns a copy of the matrix from the parent image.

        This method retrieves a copy of the parent image matrix, ensuring
        that modifications to the returned matrix do not affect the original
        data in the parent image's matrix.

        Returns:
            np.ndarray: A deep copy of the parent image matrix.
        """
        return self._parent_image._matrix.copy()

    def histogram(self, figsize: Tuple[int, int] = (10, 5))-> Tuple[plt.Figure, np.ndarray]:
        """
        Generates a 2x2 subplot figure that includes the parent image and its grayscale histogram.

        This method creates a subplot layout with 2 rows and 2 columns. The first subplot
        displays the parent image. The second subplot displays the grayscale histogram
        associated with the same image.

        Args:
            figsize (Tuple[int, int]): A tuple specifying the width and height of the created
                figure in inches. Default value is (10, 5).

        Returns:
            Tuple[plt.Figure, np.ndarray]: Returns a matplotlib Figure object containing
                the subplots and a NumPy array of axes for further customization.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes[0].imshow(self._parent_image.matrix[:])
        axes[0].set_title(self._parent_image.name)

        hist_one, histc_one = histogram(self._parent_image.matrix[:])
        axes[1].plot(hist_one, histc_one, lw=2)
        axes[1].set_title('Grayscale Histogram')
        return fig, axes

    def show(self, ax: plt.Axes = None, figsize: Tuple[int, int] = None, cmap: str = 'gray', title: str = None) -> (plt.Figure, plt.Axes):
        """Displays the matrix form of the image using matplotlib with various customizable options.

        This function visualizes an image associated with the instance, leveraging matplotlib.
        It provides flexibility in terms of figure size, colormap, axis title, and a predefined
        matplotlib axis. It is designed to simplify image visualization while allowing users
        to control specific display parameters.

        Args:
            ax (plt.Axes, optional): The matplotlib axis on which to plot. If no axis is
                provided, a new figure and axis are created. Defaults to None.
            figsize (Tuple[int, int], optional): Size of the figure in inches for the new
                axis. Ignored if `ax` is provided. Defaults to (6, 4) if not specified.
            cmap (str): Colormap to be applied for rendering the image. Defaults to
                the 'gray' colormap.
            title (str, optional): Title for the image. If provided, displays the specified
                title above the axis. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib figure (if created)
                and axis. The returned objects can be used for further customization outside
                this function.

        """
        # Defaults
        if figsize is None: figsize = (6, 4)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot array
        ax.imshow(self._parent_image.matrix[:], cmap=cmap)

        # Adjust ax settings
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    def show_overlay(
            self,
            object_label: Optional[int] = None,
            figsize: Tuple[int, int] = None,
            annotate: bool = False,
            annotation_size: int = 12,
            annotation_color: str = 'white',
            annotation_facecolor: str = 'red',
            ax: plt.Axes = None,
    ) -> (plt.Figure, plt.Axes):
        """isplays an overlay of labeled objects on the parent image, optionally with annotations.

        Args:
            object_label (Optional[int]): If specified, the overlay will exclude the provided object label.
            figsize (Tuple[int, int], optional): Size of the figure to create if no axis is provided.
            annotate (bool): Whether to annotate the image objects. Defaults to False.
            annotation_size (int): Font size for annotations. Defaults to 12.
            annotation_color (str): Font color for annotations. Defaults to 'white'.
            annotation_facecolor (str): Background color behind the annotation text. Defaults to 'red'.
            ax (plt.Axes, optional): Axis to draw the overlay on. If not provided, a new matplotlib axis is created.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes objects containing the overlay plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.grid(False)

        map_copy = self._parent_image.omap[:]
        if object_label is not None:
            map_copy[map_copy == object_label] = 0

        ax.imshow(label2rgb(label=map_copy, image=self._parent_image.matrix[:]))

        if annotate:
            for i, label in enumerate(self._parent_image.objects.labels):
                if object_label is None:
                    text_rr, text_cc = self._parent_image.objects.props[i].centroid
                    ax.text(
                        x=text_cc, y=text_rr,
                        s=f'{label}',
                        color=annotation_color,
                        fontsize=annotation_size,
                        bbox=dict(facecolor=annotation_facecolor, edgecolor='none', alpha=0.6, boxstyle='round')
                    )
                elif object_label == label:
                    text_rr, text_cc = self._parent_image.objects.props[i].centroid
                    ax.text(
                        x=text_cc, y=text_rr,
                        s=f'{label}',
                        color=annotation_color,
                        fontsize=annotation_size,
                        bbox=dict(facecolor=annotation_facecolor, edgecolor='none', alpha=0.6, boxstyle='round')
                    )

        return fig, ax
