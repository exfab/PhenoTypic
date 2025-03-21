from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

from typing import Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from skimage.color import label2rgb, gray2rgb
from skimage.exposure import histogram

from phenoscope.core.accessors import ImageAccessor
from phenoscope.util.exceptions_ import ArrayKeyValueShapeMismatchError
from phenoscope.util.constants_ import IMAGE_FORMATS


class ImageMatrix(ImageAccessor):
    """An accessor for managing and visualizing image matrix data. This is the greyscale representation converted using weighted luminance

    This class provides a set of tools to access image data, analyze it through
    histograms, and visualize results. The class utilizes a parent
    Image object to interact with the underlying matrix data while
    maintaining immutability for direct external modifications.
    Additionally, it supports overlaying annotations and labels on the image
    for data analysis purposes.
    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Provides functionality to retrieve a copy of a specified portion of the parent image's
        matrix. This class method is used to access the image matrix data, or slices of the parent image
        matrix based on the provided key.

        Args:
            key (any): A key used to index or slice the parent image's matrix.

        Returns:
            np.ndarray: A copy of the accessed subset of the parent image's matrix.
        """
        return self._parent_image._matrix[key].copy()

    def __setitem__(self, key, value):
        """
        Sets the value for a given key in the parent image's matrix. Updates the parent
        image data and schema accordingly to ensure consistency with the provided value.

        Args:
            key: The key in the matrix to update.
            value: The new value to assign to the key. Must be an array of a compatible
                shape or a primitive type like int, float, or bool.

        Raises:
            ArrayKeyValueShapeMismatchError: If the shape of the value does not match
                the shape of the existing key in the parent image's matrix.
        """
        if type(value) not in [int, float, bool]:
            if self._parent_image._matrix[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError

        self._parent_image._matrix[key] = value
        if self._parent_image.schema not in IMAGE_FORMATS.MATRIX_FORMATS:
            self._parent_image.set_image(input_image=gray2rgb(self._parent_image._matrix), input_schema=IMAGE_FORMATS.RGB)
        else:
            self._parent_image.set_image(input_image=self._parent_image._matrix, input_schema=IMAGE_FORMATS.GRAYSCALE)

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

    def histogram(self, figsize: Tuple[int, int] = (10, 5)) -> Tuple[plt.Figure, np.ndarray]:
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
