import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional

import skimage

from phenotypic.core.accessors import ImageDataAccessor
from phenotypic.util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError, NoArrayError, EmptyImageError


class ImageArray(ImageDataAccessor):
    """An accessor for handling image arrays with helper methods for accessing, modifying, visualizing, and analyzing the multichannel image data.

    It relies on the parent image handler object that serves as the bridge to the underlying image
    array and associated metadata or attributes.

    The class allows users to interact with image arrays intuitively while providing
    features such as advanced visualization (both for the raw images and their derived
    representations, like histograms or overlays). Through its properties and methods,
    users can explore, manipulate, and analyze the structural or geometrical attributes
    of the image and its segmented objects.

    Key use cases for this class include displaying selected channels or the entire
    image (including overlays and highlighted objects), generating channel-specific
    histograms, and accessing image data attributes, such as shape.

    """

    def __getitem__(self, key) -> np.ndarray:
        """
        Returns a copy of the elements at the subregion specified by the given key.

        This class provides a mechanism for extracting a specific subregion from
        the multichannel image array. The extracted subregion is represented in the form of a
        NumPy array and its indexable nature allows users to freely interact with the
        underlying array data.

        Returns:
            np.ndarray: A copy of the extracted subregion represented as a NumPy array.
        """
        if self.isempty():
            if self._parent_image.matrix.isempty():
                raise EmptyImageError
            else:
                raise NoArrayError
        else:
            return self._parent_image._data.array[key].copy()

    def __setitem__(self, key, value):
        """
        Sets a value for a given key in the parent image array. The value must either be of
        type int, float, or bool, or it must match the shape of the corresponding key's value
        in the parent image array. If the value's shape does not align with the required shape,
        an exception is raised.

        Note:
            If you want to change the entire image array data, use Image.set_image() instead.

        Args:
            key: Index key specifying the location in the parent image array to modify.
            value: The new value to assign to the specified key in the array. Can be of types
                int, float, or bool. If not, it must match the shape of the target array segment.

        Raises:
            ArrayKeyValueShapeMismatchError: If the value is an array and its shape does not match
        """
        if isinstance(value, (int, float, np.ndarray)):
            if isinstance(value, np.ndarray):
                if value.shape != self._parent_image._data.array[key].shape:
                    raise ArrayKeyValueShapeMismatchError

            self._parent_image._data.array[key] = value
            self._parent_image._set_from_array(self._parent_image._data.array, imformat=self._parent_image.imformat)

        else:
            raise ValueError(f'Unsupported type for setting the array. Value should be scalar or a numpy array: {type(value)}')

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the parent image's underlying array.

        Returns:
            tuple[int, ...]: A tuple representing the shape of the parent image's
            underlying array.
        """
        return self._parent_image._data.array.shape

    def copy(self) -> np.ndarray:
        """Creates and returns a deep copy of the parent image array.

        Returns:
            np.ndarray: A copy of the parent image array.
        """
        return self._parent_image._data.array.copy()

    def histogram(self, figsize: tuple[int, int] = (10, 5), linewidth: int = 1) -> tuple[plt.Figure, plt.Axes]:
        """
        Generates histograms for each channel of the image represented in the provided handler
        and visualizes them along with the original image. It supports RGB and non-RGB image
        schemas by adjusting the channel histograms accordingly. The method plots the original
        image and histograms for three channels side by side, customizing titles to correspond
        to the image schema (e.g., RGB or other channels).

        Args:
            figsize (tuple[int, int]): tuple representing the figure size (width, height) for
                the plot layout. Defaults to (10, 5).
            linewidth (int): Width of the lines used in the histogram plots. Defaults to 1.

        Returns:
            tuple[plt.Figure, np.ndarray]: A tuple where the first element is the figure
                containing the visualized plots (original image and histograms). The second
                element is the array of axes represented by subplots.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes_ = axes.ravel()
        axes_[0].imshow(self._parent_image._data.array)
        axes_[0].set_title(self._parent_image.name)
        axes_[0].grid(False)
        axes_[0] = self._plot(
            arr=self._parent_image._data.array,
            ax=axes_[0],
            title=self._parent_image.name,
        )

        hist_one, histc_one = skimage.exposure.histogram(self._parent_image._data.array[:, :, 0])
        axes_[1].plot(histc_one, hist_one, lw=linewidth)
        match self._parent_image.imformat:
            case IMAGE_FORMATS.RGB:
                axes_[1].set_title("Red Histogram")
            case _:
                axes_[1].set_title("Channel 1 Histogram")

        hist_two, histc_two = skimage.exposure.histogram(self._parent_image._data.array[:, :, 1])
        axes_[2].plot(histc_two, hist_two, lw=linewidth)
        match self._parent_image.imformat:
            case IMAGE_FORMATS.RGB:
                axes_[2].set_title('Green Histogram')
            case _:
                axes_[2].set_title('Channel 2 Histogram')

        hist_three, histc_three = skimage.exposure.histogram(self._parent_image._data.array[:, :, 2])
        axes_[3].plot(histc_three, hist_three, lw=linewidth)
        match self._parent_image.imformat:
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
        Displays the image array, either the full array or a specific channel, using matplotlib.

        Args:
            channel (int | None): Specifies the channel to display from the image array. If None,
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
            return self._plot(arr=self._parent_image.array[:], ax=ax, figsize=figsize, title=title, mpl_params=mpl_params)

        else:
            title = f"{self._parent_image.name} - Channel {channel}" if title is None else f'{title} - Channel {channel}'
            return self._plot(arr=self._parent_image.array[:, :, channel], ax=ax, figsize=figsize, title=title, mpl_params=mpl_params)

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
        Displays an overlay of the object map on the parent image with optional annotations.

        This method enables visualization by overlaying object regions on the parent image. It
        provides options for customization, including the ability to annotate specific objects
        and adjust visual styles like figure size, colors, and annotation properties.

        Args:
            object_label (None | int): Specific object label to be highlighted. If None,
                all objects are displayed.
            figsize (tuple[int, int]): Size of the figure in inches (width, height).
            title (None | str): Title for the plot. If None, the parent image's name
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
        objmap = self._parent_image.objmap[:]
        if object_label is not None: objmap[objmap != object_label] = 0
        if annotation_params is None: annotation_params = {}

        fig, ax = self._plot_overlay(
            arr=self._parent_image.array[:],
            objmap=objmap,
            ax=ax,
            figsize=figsize,
            title=title,
            imshow_params=imshow_params,
            overlay_params=overlay_params,
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
        Displays the image objects with customizable visualization parameters.

        This method provides a flexible way to display image objects from a parent image
        with optional customization of the visualization. By specifying a particular image
        channel, background color, colormap, figure size, and other parameters, users can
        control how the objects are segmented. This is particularly useful for analysis or
        presentation purposes.

        Args:
            channel (Optional[int]): Specifies the image channel to display. If None, the entire
                image array is used.
            bg_color (int): Background color for non-object pixels. Non-object pixels will be
                set to this value.
            cmap (str): Colormap to be used for rendering the image. Default is 'gray'.
            figsize (tuple[int, int]): Dimensions of the output figure in inches (width, height).
            title (str): Title of the plot. If None, uses the name of the parent image.
            ax (plt.Axes): Pre-existing matplotlib Axes object to plot in. If None, a new Axes and
                Figure are created.
            mpl_params (None | dict): Additional matplotlib parameters for customization. If None,
                no extra parameters are applied.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the Figure and Axes objects of the plot.

        """
        # Set non-object pixels to zero
        if channel:
            display_arr = self._parent_image.array[:, :, channel]
            display_arr[self._parent_image.objmask[:] == 0] = bg_color
        else:
            display_arr = self._parent_image.array[:]
            display_arr[np.dstack([self._parent_image.objmask[:] for _ in range(display_arr.shape[2])]) == 0] = bg_color

        fig, ax = self._plot(
            arr=display_arr,
            ax=ax,
            figsize=figsize,
            title=title,
            cmap=cmap,
            mpl_params=mpl_params,

        )

        return fig, ax

    def get_foreground(self, bg_color: int = 0) -> np.ndarray:
        """
        Returns the image foreground based on the object mask, and set's the background to the specified background color.

        This method identifies and extracts all the connected components (objects) within the image
        that are not of the specified background color. The operation relies on internal references
        to the parent image and its associated mask processing capabilities.

        Args:
            bg_color (int): The background color to exclude when extracting objects.
                Pixels within the image matching this color will be treated as background
                and ignored during the extraction process.

        Returns:
            np.ndarray: A Numpy array with non-object pixels set to the specified background color.
        """
        return self._parent_image.objmask._extract_objects(self._parent_image.array[:], bg_color=bg_color)
