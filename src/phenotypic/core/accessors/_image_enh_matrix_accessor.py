from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from typing import Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import histogram

from phenotypic.core.accessors import ImageDataAccessor
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError, EmptyImageError


class ImageEnhancedMatrix(ImageDataAccessor):
    """An accessor class to an image's enhanced matrix which is a copy of the original image matrix that is preprocessed for enhanced detection.

    Provides functionalities to manipulate and visualize the image enhanced matrix. This includes
    retrieving and setting data, resetting the matrix, visualizing histograms, viewing the matrix
    with overlays, and accessing matrix properties. The class relies on a handler for matrix operations
    and object mapping.
    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Provides a method to retrieve a copy of a specific portion of a parent image's detection
        matrix based on the given key.

        Args:
            key: The index or slice used to access a specific part of the parent image's detection
                matrix.

        Returns:
            numpy.ndarray: A copy of the corresponding portion of the parent image's detection
                matrix.
        """
        if self.isempty():
            raise EmptyImageError
        else:
            return self._parent_image._data.enh_matrix[key].copy()

    def __setitem__(self, key, value):
        """
        Sets a value in the detection matrix of the parent image for the provided key.

        The method updates or sets a value in the detection matrix of the parent image
        (`_parent_image._det_matrix`) at the specified key. It ensures that if the value
        is not of type `int`, `float`, or `bool`, its shape matches the shape of the
        existing value at the specified key. If the shape does not match,
        `ArrayKeyValueShapeMismatchError` is raised. When the value is successfully set,
        the object map (`objmap`) of the parent image is reset.

        Notes:
            Objects are reset after setting a value in the detection matrix

        Args:
            key: The key in the detection matrix where the value will be set.
            value: The value to be assigned to the detection matrix. Must be of type
                int, float, or bool, or must have a shape matching the existing array
                in the detection matrix for the provided key.

        Raises:
            ArrayKeyValueShapeMismatchError: If the value is an array and its shape
                does not match the shape of the existing value in `_parent_image._det_matrix`
                for the specified key.
        """
        if isinstance(value, np.ndarray):
            if self._parent_image._data.enh_matrix[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError

        self._parent_image._data.enh_matrix[key] = value
        self._parent_image.objmap.reset()

    @property
    def shape(self):
        """
        Represents the shape property of the parent image's enhanced matrix.

        This property fetches and returns the dimensions (shape) of the enhanced
        matrix that belongs to the parent image linked with the current class.

        Returns:
            tuple: The shape of the determinant matrix.
        """
        return self._parent_image._data.enh_matrix.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the Detection Matrix."""
        return self._parent_image._data.enh_matrix.copy()

    def reset(self):
        """Resets the image's enhanced matrix to the original matrix representation."""
        self._parent_image._data.enh_matrix = self._parent_image.matrix.copy()

    def histogram(self, figsize: Tuple[int, int] = (10, 5)):
        """Returns a histogram of the image matrix. Useful for troubleshooting detection results.
        Args:
            figsize:

        Returns:

        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes[0].imshow(self._parent_image.enh_matrix[:])
        axes[0].set_title(self._parent_image.name)

        hist_one, histc_one = histogram(self._parent_image.enh_matrix[:])
        axes[1].plot(hist_one, histc_one, lw=2)
        axes[1].set_title('Grayscale Histogram (Detection Matrix)')
        return fig, axes

    def show(self,
             figsize: str = None,
             cmap: str = 'gray',
             title: str = None,
             ax: plt.Axes = None,
             mpl_params: None | dict = None) -> (plt.Figure, plt.Axes):
        """Displays the enhanced matrix of the parent image.

        This method is used to visualize the enhanced matrix of the parent image using
        matplotlib. Different parameters can be configured for customization such as
        figure size, colormap, title of the plot, and specific matplotlib axes. Additional
        matplotlib parameters can also be passed for further customization.

        Args:
            figsize (tuple[int, int] | None): Tuple defining the width and height of
                the figure in inches. If None, defaults to matplotlib's standard size.
            cmap (str): Colormap used for plotting. Defaults to 'gray'.
            title (str | None): Title of the plot. If None, no title is set.
            ax (plt.Axes | None): Matplotlib axes on which the plot will be rendered.
                If None, a new figure and axes will be created.
            mpl_params (dict | None): Additional matplotlib parameters in dictionary
                format for adjusting plot properties. If None, no additional parameters
                are applied.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and
            Axes objects corresponding to the rendered plot.

        Raises:
            None
        """

        return self._plot(
            arr=self._parent_image.enh_matrix[:],
            figsize=figsize,
            ax=ax,
            title=title,
            cmap=cmap,
            mpl_params=mpl_params,
        )

    def show_overlay(
            self,
            object_label: Optional[int] = None,
            figsize: Tuple[int, int] = None,
            title: None | str = None,
            annotate: bool = False,
            annotation_params: None | dict = None,
            ax: plt.Axes = None,
            overlay_params: None | dict = None,
            imshow_params: None | dict = None,
    ) -> (plt.Figure, plt.Axes):
        """
        Displays an overlay visualization of a labeled image matrix and its annotations.

        This method generates an overlay of a labeled image using the 'label2rgb'
        function from skimage. It optionally annotates regions with their labels.
        Additional customization options are provided through parameters such
        as subplot size, title, annotation properties, and Matplotlib configuration.

        Args:
            object_label (Optional[int]): A specific label to exclude from the
                overlay. If None, all objects are included.
            figsize (Tuple[int, int]): Size of the figure in inches as a tuple
                (width, height). If None, default size settings will be used.
            title (None|str): Title of the figure. If None, no title is displayed.
            annotate (bool): Whether to annotate object labels on the overlay.
                Defaults to False.
            annotation_params (None | dict): Additional parameters for customization of the
                object annotations. Defaults: size=12, color='white', facecolor='red'.
            ax (plt.Axes): Existing Matplotlib Axes object where the overlay will be
                plotted. If None, a new Axes object is created.
            overlay_params (None|dict): Additional parameters for the overlay
                generation. If None, default overlay settings will apply.
            imshow_params (None|dict): Additional Matplotlib imshow configuration parameters
                for customization. If None, default Matplotlib settings will apply.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib figure and
                axes where the overlay is displayed.
        """
        objmap = self._parent_image.objmap[:]
        if object_label is not None: objmap[objmap == object_label] = 0
        if annotation_params is None: annotation_params = {}

        fig, ax = self._plot_overlay(
            arr=self._parent_image.enh_matrix[:],
            objmap=objmap,
            figsize=figsize,
            title=title,
            ax=ax,
            overlay_params=overlay_params,
            imshow_params=imshow_params,

        )

        if annotate:
            ax = self._plot_annotations(
                ax=ax,
                color=annotation_params.get('color', 'white'),
                size=annotation_params.get('size', 12),
                facecolor=annotation_params.get('facecolor', 'red'),
                object_label=object_label,
            )

        return fig, ax
