from typing import Tuple, Optional

import numpy as np

from skimage.exposure import histogram
import matplotlib.pyplot as plt

from phenotypic.core._image_parts.accessor_abstracts import ImageArrDataAccessor


class ImageMatrixDataAccessor(ImageArrDataAccessor):
    """
    Handles interaction with Image 2-d matrix data by providing access to Image attributes and data.

    This class serves as a bridge for interacting with Image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    Image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessor`.

    Attributes:
        image (Any): Root Image object that this accessor is linked to.
        _main_arr (Any): Main array storing the Image-related data.
        _dtype (Any): Data type of the Image data stored in the target array.
    """

    def histogram(self, figsize: Tuple[int, int] = (10, 5)) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Generates a figure showing the matrix and its grayscale histogram.

        This method creates a subplot layout with 2 rows and 2 columns. The first subplot
        displays the parent image. The second subplot displays the grayscale histogram
        associated with the same image.

        Args:
            figsize (Tuple[int, int]): A tuple specifying the width and height of the created
                figure in inches. Default other_image is (10, 5).

        Returns:
            Tuple[plt.Figure, np.ndarray]: Returns a matplotlib Figure object containing
                the subplots and a NumPy array of axes for further customization.
        """
        fig, (imgAx, histAx) = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        fig, imgAx = self._plot(
            arr=self[:],
            figsize=figsize,
            ax=imgAx,
            title=self._root_image.name,
            cmap='gray',
            mpl_kwargs=None,
        )

        hist_one, histc_one = histogram(self[:])
        histAx.plot(hist_one, histc_one, lw=2)
        histAx.set_title('Grayscale Histogram')
        return fig, (imgAx, histAx)

    def show(self, ax: plt.Axes = None, figsize: Tuple[
        int, int] = None, cmap: str = 'gray', title: str = None, mpl_kwargs: None | dict = None) -> (plt.Figure, plt.Axes):
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
            mpl_kwargs (None | dict, optional): Additional Matplotlib parameters for
                customizing the plot. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib Figure and Axes
                objects used for plotting.

        Raises:
            TypeError: If invalid types are provided for `ax`, `figsize`, `cmap`, or
                `mpl_kwargs`.
            ValueError: If unexpected values are passed to the function arguments.

        """
        return self._plot(
            arr=self[:],
            figsize=figsize,
            ax=ax,
            title=title,
            cmap=cmap,
            mpl_kwargs=mpl_kwargs,
        )

    def show_overlay(
            self,
            object_label: Optional[int] = None,
            figsize: Tuple[int, int] | None = None,
            title: None | str = None,
            show_labels: bool = False,
            annotation_params: None | dict = None,
            ax: plt.Axes = None,
            overlay_kwargs: None | dict = None,
            mpl_kwargs: None | dict = None,
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
            show_labels (bool): Whether to show_labels object labels on the overlay.
                Defaults to False.
            annotation_params (None | dict): Additional parameters for customization of the
                object annotations. Defaults: size=12, color='white', facecolor='red'.
            ax (plt.Axes): Existing Matplotlib Axes object where the overlay will be
                plotted. If None, a new Axes object is created.
            overlay_kwargs (None|dict): Additional parameters for the overlay
                generation. If None, default overlay settings will apply.
            mpl_kwargs (None|dict): Additional Matplotlib imshow configuration parameters
                for customization. If None, default Matplotlib settings will apply.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib figure and
                axes where the overlay is displayed.
        """
        objmap = self._root_image.objmap[:]
        if object_label is not None: objmap[objmap == object_label] = 0
        if annotation_params is None: annotation_params = {}

        fig, ax = self._plot_overlay(
            arr=self[:],
            objmap=objmap,
            figsize=figsize,
            title=title,
            ax=ax,
            overlay_params=overlay_kwargs,
            mpl_kwargs=mpl_kwargs,

        )

        if show_labels:
            ax = self._plot_obj_labels(
                ax=ax,
                color=annotation_params.get('color', 'white'),
                size=annotation_params.get('size', 12),
                facecolor=annotation_params.get('facecolor', 'red'),
                object_label=object_label,
            )

        return fig, ax


