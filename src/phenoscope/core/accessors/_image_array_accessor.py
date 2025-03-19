from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from skimage.color import label2rgb
from skimage.exposure import histogram

from phenoscope.util.constants_ import IMAGE_FORMATS
from phenoscope.util.exceptions_ import IllegalElementAssignmentError


class ImageArray:
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

    Attributes:
        _parent_image: An internal attribute referring to the parent :class:`Image` object that mediates
            interactions with the underlying image array, properties (shape, name, schema),
            and associated object maps or labels.
    """

    def __init__(self, parent_image: Image):
        self._parent_image: Image = parent_image

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
        return self._parent_image._array[key].copy()

    def __setitem__(self, key, value):
        """Represents an exception that occurs when an illegal assignment is attempted on the images array. Use Image.set_image() to change image array data.

        Raises:
            IllegalElementAssignmentError: Raised when an illegal
                element assignment is attempted within the image array subhandler.
        """
        raise IllegalElementAssignmentError('Image.array')

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Returns the shape of the image"""
        return self._parent_image._array.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the image array"""
        return self._parent_image._array.copy()

    def histogram(self, figsize: Tuple[int, int] = (10, 5), linewidth: int = 1) -> Tuple[plt.Figure, np.ndarray]:
        """
        Generates histograms for each channel of the image represented in the provided handler
        and visualizes them along with the original image. It supports RGB and non-RGB image
        schemas by adjusting the channel histograms accordingly. The method plots the original
        image and histograms for three channels side by side, customizing titles to correspond
        to the image schema (e.g., RGB or other channels).

        Args:
            figsize (Tuple[int, int]): Tuple representing the figure size (width, height) for
                the plot layout. Defaults to (10, 5).
            linewidth (int): Width of the lines used in the histogram plots. Defaults to 1.

        Returns:
            Tuple[plt.Figure, np.ndarray]: A tuple where the first element is the figure
                containing the visualized plots (original image and histograms). The second
                element is the array of axes represented by subplots.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes_ = axes.ravel()
        axes_[0].imshow(self._parent_image._array)
        axes_[0].set_title(self._parent_image.name)
        axes_[0].grid(False)

        hist_one, histc_one = histogram(self._parent_image._array[:, :, 0])
        axes_[1].plot(histc_one, hist_one, lw=linewidth)
        match self._parent_image.schema:
            case IMAGE_FORMATS.RGB:
                axes_[1].set_title("Red Histogram")
            case _:
                axes_[1].set_title("Channel 1 Histogram")

        hist_two, histc_two = histogram(self._parent_image._array[:, :, 1])
        axes_[2].plot(histc_two, hist_two, lw=linewidth)
        match self._parent_image.schema:
            case IMAGE_FORMATS.RGB:
                axes_[2].set_title('Green Histogram')
            case _:
                axes_[2].set_title('Channel 2 Histogram')

        hist_three, histc_three = histogram(self._parent_image._array[:, :, 2])
        axes_[3].plot(histc_three, hist_three, lw=linewidth)
        match self._parent_image.schema:
            case IMAGE_FORMATS.RGB:
                axes_[3].set_title('Blue Histogram')
            case _:
                axes_[3].set_title('Channel 3 Histogram')

        return fig, axes

    def show(self,
             channel: Optional[int] = None,
             ax: plt.Axes = None,
             figsize: Tuple[int, int] = (10, 5),
             title: str = None) -> (plt.Figure, plt.Axes):
        """Display the image array with matplotlib.

        Args:
            channel: (Optional[int]) The channel number to display. If None, shows the combination of all channels
            ax: (plt.Axes) Axes object to use for plotting.
            figsize: (Tuple[int, int]): Figure size in inches.
            title: (str) a title for the plot

        Returns:
            tuple(plt.Figure, plt.Axes): matplotlib figure and axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot array
        if channel is None:
            ax.imshow(self._parent_image.array[:])
        else:
            ax.imshow(self._parent_image.array[:, :, channel])

        # Adjust ax settings
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(self._parent_image.name)
        ax.grid(False)

        return fig, ax

    def show_overlay(self, object_label: Optional[int] = None, ax: plt.Axes = None,
                     figsize: Tuple[int, int] = (10, 5),
                     annotate: bool = False,
                     annotation_size: int = 12,
                     annotation_color: str = 'white',
                     annotation_facecolor: str = 'red',
                     ) -> (plt.Figure, plt.Axes):
        """
        Displays an overlay visualization of a labeled object map overlaid on the original image.

        Allows for optional annotation of object labels in the visualization with configurable
        annotation styles. The visualization can be rendered on a new Matplotlib figure or an
        existing axis provided by the user.

        Args:
            object_label (Optional[int]): The specific label of the object to remove from
                the overlay visualization. If None, all labels will appear in the overlay.
            ax (plt.Axes, optional): The Matplotlib axis on which to render the visualization.
                If None, a new axis will be created.
            figsize (Tuple[int, int]): The size of the figure for the new axis, specified as a tuple.
                Ignored if `ax` is provided.
            annotate (bool): A flag indicating whether to annotate the object labels on the visualization.
            annotation_size (int): The font size of the annotation text displaying object labels.
            annotation_color (str): The color of the annotation text.
            annotation_facecolor (str): The background color of the annotation text boxes.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple where the first element is the Matplotlib figure
                and the second element is the axis used for the overlay visualization.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.grid(False)

        map_copy = self._parent_image.omap[:]
        if object_label is not None:
            map_copy[map_copy == object_label] = 0

        ax.imshow(label2rgb(label=map_copy, image=self._parent_image.array[:], saturation=1))

        if annotate:
            props = self._parent_image.objects.props
            for i, label in enumerate(self._parent_image.objects.labels):
                if object_label is None:
                    text_rr, text_cc = props[i].centroid
                    ax.text(
                        x=text_cc, y=text_rr,
                        s=f'{label}',
                        color=annotation_color,
                        fontsize=annotation_size,
                        bbox=dict(facecolor=annotation_facecolor, edgecolor='none', alpha=0.6, boxstyle='round')
                    )
                elif object_label == label:
                    text_rr, text_cc = props[i].centroid
                    ax.text(
                        x=text_cc, y=text_rr,
                        s=f'{label}',
                        color=annotation_color,
                        fontsize=annotation_size,
                        bbox=dict(facecolor=annotation_facecolor, edgecolor='none', alpha=0.6, boxstyle='round')
                    )

        return fig, ax

    def show_objects(self, channel: Optional[int] = None,
                     bg_color: int = 0, cmap: str = 'gray',
                     ax: plt.Axes = None, figsize: Tuple[int, int] = (10, 5),
                     title: str = None) -> (plt.Figure, plt.Axes):
        """
        Displays segmented objects in an image minus the background, enabling visual
        analysis of segmentation quality. The function provides customization options for displaying specific
        channels, modifying background color, and adjusting axes and figure settings. The
        segmented objects can be separated by channels with respective colormap, or the entire
        image can be displayed if no specific channel is provided. This visualization can
        help examine segmentations overlayed on image arrays.

        Args:
            channel: The specific channel of the image to display. If None, the entire
                image across all channels is displayed.
            bg_color: The background pixel value to be rendered for unsegmented areas in the
                displayed image.
            cmap: The colormap to apply to the image representation when a specific
                channel is displayed.
            ax: The axes object to draw the visualization. If None, a new matplotlib
                Axes object is created.
            figsize: The size of the figure in inches (width, height). Applicable when ax
                is None.
            title: Optional title for the plot displayed on the axes. If None, defaults
                to the name of the image handler.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and
                Axes objects used for rendering the visualization.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        display_arr = self._parent_image.array[:] if channel is None else self._parent_image.array[:, :, channel]
        display_arr[self._parent_image.omask[:] == 0] = bg_color

        ax.imshow(display_arr) if channel is None else ax.imshow(display_arr, cmap=cmap)

        bg_mask = self._parent_image.omask[:]
        if channel is None:
            display_arr = self._parent_image.omask[:]
            bg_mask = np.dstack(
                [bg_mask for _ in range(self._parent_image.array.shape[2])]
            )
            display_arr[bg_mask == 0] = bg_color
            ax.imshow(display_arr)
        else:
            display_arr = self._parent_image.omask[:, :, channel]
            display_arr[bg_mask == 0] = bg_color
            ax.imshow(display_arr, cmap=cmap)

        # Adjust ax settings
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(self._parent_image.name)

        ax.grid(False)
        return fig, ax

    def extract_objects(self, bg_color: int = 0) -> np.ndarray:
        """Extracts the objects from the image array. With the background elements set to 0"""
        return self._parent_image.omask._extract_objects(self._parent_image.array[:], bg_color=bg_color)
