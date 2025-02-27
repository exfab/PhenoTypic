import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from skimage.color import rgb2hsv, label2rgb
from skimage.exposure import histogram

from ...util.constants import C_ImageArraySubhandler, C_ImageFormats


class ImageArraySubhandler:
    """An immutable accessor image's multichannel information. Access image elements similar to a numpy array."""

    def __init__(self, handler):
        self._handler = handler

    def __getitem__(self, key) -> np.ndarray:
        return self._handler._array[key].copy()

    def __setitem__(self, key, value):
        raise C_ImageArraySubhandler.IllegalElementAssignmentError

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Returns the shape of the image"""
        return self._handler._array.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the image array"""
        return self._handler._array.copy()

    def histogram(self, figsize: Tuple[int, int] = (10, 5), linewidth: int = 1):
        """Returns a histogram of the image array.
        Args:
            figsize:

        Returns:

        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes_ = axes.ravel()
        axes_[0].imshow(self._handler._array)
        axes_[0].set_title(self._handler.name)

        hist_one, histc_one = histogram(self._handler._array[:, :, 0])
        axes_[1].plot(histc_one, hist_one, lw=linewidth)
        match self._handler.schema:
            case C_ImageFormats.RGB:
                axes_[1].set_title("Red Histogram")
            case _:
                axes_[1].set_title("Channel 1 Histogram")

        hist_two, histc_two = histogram(self._handler._array[:, :, 1])
        axes_[2].plot(histc_two, hist_two, lw=linewidth)
        match self._handler.schema:
            case C_ImageFormats.RGB:
                axes_[2].set_title('Green Histogram')
            case _:
                axes_[2].set_title('Channel 2 Histogram')

        hist_three, histc_three = histogram(self._handler._array[:, :, 2])
        axes_[3].plot(histc_three, hist_three, lw=linewidth)
        match self._handler.schema:
            case C_ImageFormats.RGB:
                axes_[3].set_title('Blue Histogram')
            case _:
                axes_[3].set_title('Channel 3 Histogram')

        return fig, axes

    def show(self, channel: Optional[int] = None, ax: plt.Axes = None, figsize: Tuple[int, int] = (10, 5),
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
            ax.imshow(self._handler.array[:])
        else:
            ax.imshow(self._handler.array[:, :, channel])

        # Adjust ax settings
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(self._handler.name)
        ax.grid(False)

        return fig, ax

    def show_overlay(self, object_label: Optional[int] = None, ax: plt.Axes = None,
                     figsize: Tuple[int, int] = (10, 5),
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

        ax.imshow(label2rgb(label=map_copy, image=self._handler.array[:], saturation=1))

        if annotate:
            props = self._handler.objects.props
            for i, label in enumerate(self._handler.objects.labels):
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

    def show_objects(self, channel: Optional[int] = None, bg_color: int = 0, ax: plt.Axes = None, figsize: Tuple[int, int] = (10, 5),
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
            ax.imshow(self._handler.obj_mask._extract_objects(self._handler.array[:], bg_color=bg_color))
        else:
            ax.imshow(np.ma.array(self._handler.array[:, :, channel], mask=~self._handler.obj_mask[:]))
            ax.imshow(self._handler.obj_mask[:], cmap='gray', alpha=0.5)

        # Adjust ax settings
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(self._handler.name)

        ax.grid(False)
        return fig, ax

    def extract_objects(self, bg_color: int = 0) -> np.ndarray:
        """Extracts the objects from the image array."""
        return self._handler.obj_mask._extract_objects(self._handler.array[:], bg_color=bg_color)