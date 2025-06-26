import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional

import skimage

from phenotypic.core.accessor_abstracts import ImageArrDataAccessor
from phenotypic.util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError, NoArrayError, EmptyImageError


class ImageArray(ImageArrDataAccessor):
    """An accessor for handling _root_image arrays with helper methods for accessing, modifying, visualizing, and analyzing the multichannel _root_image data.

    It relies on the parent _root_image handler object that serves as the bridge to the underlying _root_image
    array and associated metadata or attributes.

    The class allows users to interact with _root_image arrays intuitively while providing
    features such as advanced visualization (both for the raw images and their derived
    representations, like histograms or overlays). Through its properties and methods,
    users can explore, manipulate, and analyze the structural or geometrical attributes
    of the _root_image and its segmented objects.

    Key use cases for this class include displaying selected channels or the entire
    _root_image (including overlays and highlighted objects), generating channel-specific
    histograms, and accessing _root_image data attributes, such as shape.

    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Returns a copy of the elements at the subregion specified by the given key.

        This class provides a mechanism for extracting a specific subregion from
        the multichannel _root_image array. The extracted subregion is represented in the form of a
        NumPy array, and its indexable nature allows users to freely interact with the
        underlying array data.

        Returns:
            np.ndarray: A copy of the extracted subregion represented as a NumPy array.
        """
        if self.isempty():
            if self._root_image.matrix.isempty():
                raise EmptyImageError
            else:
                raise NoArrayError
        else:
            return self._root_image._data.array[key].copy()

    def __setitem__(self, key, value):
        """
        Sets a other_image for a given key in the parent _root_image array. The other_image must either be of
        type int, float, or bool, or it must match the shape of the corresponding key's other_image
        in the parent _root_image array. If the other_image's shape does not align with the required shape,
        an exception is raised.

        Note:
            If you want to change the entire _root_image array data, use Image.set_image() instead.

        Args:
            key: Index key specifying the location in the parent _root_image array to modify.
            value: The new other_image to assign to the specified key in the array. Can be of types
                int, float, or bool. If not, it must match the shape of the target array segment.

        Raises:
            ArrayKeyValueShapeMismatchError: If the other_image is an array and its shape does not match
        """
        if isinstance(value, (int, float, np.ndarray)):
            if isinstance(value, np.ndarray):
                if value.shape != self._root_image._data.array[key].shape:
                    raise ArrayKeyValueShapeMismatchError

            self._root_image._data.array[key] = value
            self._root_image._set_from_array(self._root_image._data.array, imformat=self._root_image.imformat)

        else:
            raise ValueError(f'Unsupported type for setting the array. Value should be scalar or a numpy array: {type(value)}')

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the parent _root_image's underlying array.

        Returns:
            tuple[int, ...]: A tuple representing the shape of the parent _root_image's
            underlying array.
        """
        return self._root_image._data.array.shape

    def copy(self) -> np.ndarray:
        """Creates and returns a deep copy of the parent _root_image array.

        Returns:
            np.ndarray: A copy of the parent _root_image array.
        """
        return self._root_image._data.array.copy()

    def histogram(self, figsize: tuple[int, int] = (10, 5), linewidth: int = 1) -> tuple[plt.Figure, plt.Axes]:
        """
        Generates histograms for each channel of the _root_image represented in the provided handler
        and visualizes them along with the original _root_image. It supports RGB and non-RGB _root_image
        schemas by adjusting the channel histograms accordingly. The method plots the original
        _root_image and histograms for three channels side by side, customizing titles to correspond
        to the _root_image schema (e.g., RGB or other channels).

        Args:
            figsize (tuple[int, int]): tuple representing the figure size (width, height) for
                the plot layout. Defaults to (10, 5).
            linewidth (int): Width of the lines used in the histogram plots. Defaults to 1.

        Returns:
            tuple[plt.Figure, np.ndarray]: A tuple where the first element is the figure
                containing the visualized plots (original _root_image and histograms). The second
                element is the array of axes represented by subplots.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes_ = axes.ravel()
        axes_[0].imshow(self._root_image._data.array)
        axes_[0].set_title(self._root_image.name)
        axes_[0].grid(False)
        axes_[0] = self._plot(
            arr=self._root_image._data.array,
            ax=axes_[0],
            title=self._root_image.name,
        )

        hist_one, histc_one = skimage.exposure.histogram(self._root_image._data.array[:, :, 0])
        axes_[1].plot(histc_one, hist_one, lw=linewidth)
        match self._root_image.imformat:
            case IMAGE_FORMATS.RGB:
                axes_[1].set_title("Red Histogram")
            case _:
                axes_[1].set_title("Channel 1 Histogram")

        hist_two, histc_two = skimage.exposure.histogram(self._root_image._data.array[:, :, 1])
        axes_[2].plot(histc_two, hist_two, lw=linewidth)
        match self._root_image.imformat:
            case IMAGE_FORMATS.RGB:
                axes_[2].set_title('Green Histogram')
            case _:
                axes_[2].set_title('Channel 2 Histogram')

        hist_three, histc_three = skimage.exposure.histogram(self._root_image._data.array[:, :, 2])
        axes_[3].plot(histc_three, hist_three, lw=linewidth)
        match self._root_image.imformat:
            case IMAGE_FORMATS.RGB:
                axes_[3].set_title('Blue Histogram')
            case _:
                axes_[3].set_title('Channel 3 Histogram')

        return fig, axes

    def show(self,
             channel: int | None = None,
             figsize: tuple[int, int] | None = None,
             title: str | None = None,
             ax: plt.Axes | None = None,
             mpl_params: dict | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays the _root_image array, either the full array or a specific channel, using matplotlib.

        Args:
            channel (int | None): Specifies the channel to display from the _root_image array. If None,
                the entire array is displayed. If an integer is provided, only the specified
                channel is displayed.
            figsize (None | tuple[int, int]): Optional tuple specifying the width and height of
                the figure in inches. If None, defaults to matplotlib's standard figure size.
            title (str | None): Title text for the plot. If None, no title will be displayed.
            ax (plt.Axes): Optional matplotlib Axes instance. If provided, the plot will be
                drawn on this axes object. If None, a new figure and axes will be created.
            mpl_params (dict | None): Optional dictionary of keyword arguments for customizing
                matplotlib parameters (e.g., color maps, fonts). If None, default settings will
                be used.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects
                associated with the plot. If `ax` is provided in the arguments, the returned
                tuple will include that Axes instance; otherwise, a new Figure and Axes pair
                will be returned.

        """

        if channel is None:
            return self._plot(arr=self._root_image.array[:], ax=ax, figsize=figsize, title=title, mpl_kwargs=mpl_params)

        else:
            title = f"{self._root_image.name} - Channel {channel}" if title is None else f'{title} - Channel {channel}'
            return self._plot(arr=self._root_image.array[:, :, channel], ax=ax, figsize=figsize, title=title, mpl_kwargs=mpl_params)

    def show_overlay(self, object_label: None | int = None,
                     figsize: tuple[int, int] | None = None,
                     title: str | None = None,
                     annotate: bool = False,
                     annotation_params: None | dict = None,
                     ax: plt.Axes = None,
                     overlay_params: None | dict = None,
                     imshow_params: None | dict = None,
                     ) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays an overlay of the object map on the parent _root_image with optional annotations.

        This method enables visualization by overlaying object regions on the parent _root_image. It
        provides options for customization, including the ability to show_labels specific objects
        and adjust visual styles like figure size, colors, and annotation properties.

        Args:
            object_label (None | int): Specific object label to be highlighted. If None,
                all objects are displayed.
            figsize (tuple[int, int]): Size of the figure in inches (width, height).
            title (None | str): Title for the plot. If None, the parent _root_image's name
                is used.
            annotate (bool): If True, displays annotations for object labels on the
                object centroids.
            annotation_params (None | dict): Additional parameters for customization of the
                object annotations. Defaults: size=12, color='white', facecolor='red'. Other kwargs
                are passed to the matplotlib.axes.text () method.
            ax (plt.Axes): Optional Matplotlib Axes object. If None, a new Axes is
                created.
            overlay_params (None | dict): Additional parameters for customization of the
                overlay.
            imshow_params (None|dict): Additional Matplotlib imshow configuration parameters
                for customization. If None, default Matplotlib settings will apply.

        Returns:
            tuple[plt.Figure, plt.Axes]: Matplotlib Figure and Axes objects containing
            the generated plot.

        """
        objmap = self._root_image.objmap[:]
        if object_label is not None: objmap[objmap != object_label] = 0
        if annotation_params is None: annotation_params = {}

        fig, ax = self._plot_overlay(
            arr=self._root_image.array[:],
            objmap=objmap,
            ax=ax,
            figsize=figsize,
            title=title,
            mpl_kwargs=imshow_params,
            overlay_params=overlay_params,
        )

        if annotate:
            ax = self._plot_obj_labels(
                ax=ax,
                color=annotation_params.get('color', 'white'),
                size=annotation_params.get('size', 12),
                facecolor=annotation_params.get('facecolor', 'red'),
                object_label=object_label,
            )
        return fig, ax

    def show_objects(self,
                     channel: int | None = None,
                     bg_color: int = 0,
                     cmap: str = 'gray',
                     figsize: tuple[int, int] = (10, 5),
                     title: str | None = None,
                     ax: plt.Axes | None = None,
                     mpl_params: dict | None = None,
                     ) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays the _root_image objects with customizable visualization parameters.

        This method provides a flexible way to display _root_image objects from a parent _root_image
        with optional customization of the visualization. By specifying a particular _root_image
        channel, background color, colormap, figure size, and other parameters, users can
        control how the objects are segmented. This is particularly useful for analysis or
        presentation purposes.

        Args:
            channel (Optional[int]): Specifies the _root_image channel to display. If None, the entire
                _root_image array is used.
            bg_color (int): Background color for non-object pixels. Non-object pixels will be
                set to this other_image.
            cmap (str): Colormap to be used for rendering the _root_image. Default is 'gray'.
            figsize (tuple[int, int]): Dimensions of the output figure in inches (width, height).
            title (str): Title of the plot. If None, uses the name of the parent _root_image.
            ax (plt.Axes): Pre-existing matplotlib Axes object to plot in. If None, a new Axes and
                Figure are created.
            mpl_params (None | dict): Additional matplotlib parameters for customization. If None,
                no extra parameters are applied.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the Figure and Axes objects of the plot.

        """
        # Set non-object pixels to zero
        if channel:
            display_arr = self._root_image.array[:, :, channel]
            display_arr[self._root_image.objmask[:] == 0] = bg_color
        else:
            display_arr = self._root_image.array[:]
            display_arr[np.dstack([self._root_image.objmask[:] for _ in range(display_arr.shape[2])]) == 0] = bg_color

        fig, ax = self._plot(
            arr=display_arr,
            ax=ax,
            figsize=figsize,
            title=title,
            cmap=cmap,
            mpl_kwargs=mpl_params,

        )

        return fig, ax

    def get_foreground(self, bg_color: int = 0) -> np.ndarray:
        """
        Returns the _root_image foreground based on the object mask, and set's the background to the specified background color.

        This method identifies and extracts all the connected components (objects) within the _root_image
        that are not of the specified background color. The operation relies on internal references
        to the parent _root_image and its associated mask processing capabilities.

        Args:
            bg_color (int): The background color to exclude when extracting objects.
                Pixels within the _root_image matching this color will be treated as background
                and ignored during the extraction process.

        Returns:
            np.ndarray: A Numpy array with non-object pixels set to the specified background color.
        """
        return self._root_image.objmask._extract_objects(self._root_image.array[:], bg_color=bg_color)
