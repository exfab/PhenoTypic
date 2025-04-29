from __future__ import annotations
from typing import TYPE_CHECKING


from typing import Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage.exposure import histogram

from phenotypic.core.accessors import ImageAccessor
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError
from phenotypic.util.constants_ import IMAGE_FORMATS
from warnings import warn


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
        Sets the value for a given key in the parent image's matrix. Changes are not reflected in the color matrix,
        and any objects detected are reset.

        Args:
            key: The key in the matrix to update.
            value: The new value to assign to the key. Must be an array of a compatible
                shape or a primitive type like int, float, or bool.

        Raises:
            ArrayKeyValueShapeMismatchError: If the shape of the value does not match
                the shape of the existing key in the parent image's matrix.
        """
        if isinstance(value, np.ndarray):
            if self._parent_image._matrix[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError
        elif value in {bool, int, float}:
            pass
        else:
            raise TypeError(f'Unsupported type for setting the matrix. Value should be scalar or a numpy array: {type(value)}')

        self._parent_image._matrix[key] = value
        self._parent_image.enh_matrix.reset()
        self._parent_image.objmap.reset()

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
        fig, axes[0] = self._plot(
            arr=self._parent_image.matrix[:],
            figsize=figsize,
            ax=axes[0],
            title=self._parent_image.name,
            cmap='gray',
            mpl_params=None,
        )

        hist_one, histc_one = histogram(self._parent_image.matrix[:])
        axes[1].plot(hist_one, histc_one, lw=2)
        axes[1].set_title('Grayscale Histogram')
        return fig, axes

    def show(self, ax: plt.Axes = None, figsize: Tuple[
        int, int] = None, cmap: str = 'gray', title: str = None, mpl_params: None | dict = None) -> (plt.Figure, plt.Axes):
        """
        Displays the image matrix using Matplotlib with optional customization for figure size,
        color map, title, and Matplotlib parameters.

        Args:
            ax (plt.Axes, optional): A Matplotlib Axes object on which to plot the image.
                If None, a new figure and axes are created. Defaults to None.
            figsize (Tuple[int, int], optional): Tuple defining the size of the figure
                in inches. Defaults to (6, 4) if not provided.
            cmap (str, optional): The colormap used for displaying the image. Defaults
                to 'gray'.
            title (str, optional): Title of the image plot. If None, no title is set. Defaults
                to None.
            mpl_params (None | dict, optional): Additional Matplotlib parameters for
                customizing the plot. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib Figure and Axes
                objects used for plotting.

        Raises:
            TypeError: If invalid types are provided for `ax`, `figsize`, `cmap`, or
                `mpl_params`.
            ValueError: If unexpected values are passed to the function arguments.

        """
        return self._plot(
            arr=self._parent_image.matrix[:],
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
            annotation_size: int = 12,
            annotation_color: str = 'white',
            annotation_facecolor: str = 'red',
            ax: plt.Axes = None,
            overlay_params: None | dict = None,
            mpl_params: None | dict = None,
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
            annotation_size (int): Font size of the annotations. Defaults to 12.
            annotation_color (str): Font color for annotations. Defaults to 'white'.
            annotation_facecolor (str): Background color for text annotations.
                Defaults to 'red'.
            ax (plt.Axes): Existing Matplotlib Axes object where the overlay will be
                plotted. If None, a new Axes object is created.
            overlay_params (None|dict): Additional parameters for the overlay
                generation. If None, default overlay settings will apply.
            mpl_params (None|dict): Additional Matplotlib configuration parameters
                for customization. If None, default Matplotlib settings will apply.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib figure and
                axes where the overlay is displayed.
        """
        objmap = self._parent_image.objmap[:]
        if object_label is not None: objmap[objmap == object_label] = 0

        fig, ax = self._plot_overlay(
            arr=self._parent_image.matrix[:],
            objmap=objmap,
            figsize=figsize,
            title=title,
            ax=ax,
            overlay_params=overlay_params,
            mpl_params=mpl_params,

        )

        if annotate:
            ax = self._plot_annotations(
                ax=ax,
                color=annotation_color,
                size=annotation_size,
                facecolor=annotation_facecolor,
                object_label=object_label,
            )

        return fig, ax
