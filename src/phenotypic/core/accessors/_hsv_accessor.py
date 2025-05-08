from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from typing import Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.exposure import histogram

from phenotypic.core.accessors import ImageAccessor
from phenotypic.util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import IllegalAssignmentError


class HsvAccessor(ImageAccessor):
    """An accessor class to handle and analyze HSV (Hue, Saturation, Value) image data efficiently.

    This class provides functionality for accessing and processing HSV image data.
    Users can retrieve components (hue, saturation, brightness) of the image, generate
    visual histograms of color distributions, and measure specific object properties
    masked within the HSV image.

    Extensive visualization methods are also included, allowing display of HSV components
    and their masked variations. This class is ideal for image analysis tasks where color
    properties play a significant role.

    Attributes:
        _parent_image (Image): The parent Image object that manages image data and operations.
    """

    def __getitem__(self, key) -> np.ndarray:
        return self._parent_image._hsv[key].copy()

    def __setitem__(self, key, value):
        raise IllegalAssignmentError('HSV')

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Returns the shape of the image"""
        return self._parent_image._data.array.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the image array"""
        return self._parent_image._hsv.copy()

    def histogram(self, figsize: Tuple[int, int] = (10, 5), linewidth=1):
        """
        Generates and displays histograms for hue, saturation, and brightness components of an image,
        alongside the original image. The histograms depict the distribution of these components, and
        this analysis can aid in understanding the image's color properties.

        Args:
            figsize (Tuple[int, int]): The size of the figure that contains all subplots, specified as
                a tuple of width and height in inches.
            linewidth (int): The width of the lines used in the histograms.

        Returns:
            Tuple[Figure, ndarray]: A tuple containing the Matplotlib figure object and an ndarray
                of axes, where the axes correspond to the subplots.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes_ = axes.ravel()
        axes_[0].imshow(self._parent_image._data.array)
        axes_[0].set_title(self._parent_image.name)
        axes_[0].grid(False)

        hist_one, histc_one = histogram(self._parent_image._hsv[:, :, 0] * 360)
        axes_[1].plot(histc_one, hist_one, lw=linewidth)
        axes_[1].set_title('Hue')

        hist_two, histc_two = histogram(self._parent_image._hsv[:, :, 1])
        axes_[2].plot(histc_two, hist_two, lw=linewidth)
        axes_[2].set_title("Saturation")

        hist_three, histc_three = histogram(self._parent_image._hsv[:, :, 2])
        axes_[3].plot(histc_three, hist_three, lw=linewidth)
        axes_[3].set_title("Brightness")

        return fig, axes

    def show(self, figsize: Tuple[int, int] = (10, 8),
             title: str = None, shrink=0.2) -> (plt.Figure, plt.Axes):
        """
        Displays the Hue, Saturation, and Brightness (HSV components) of the given
        image data in a visualization using subplots. Each subplot corresponds to
        one of the HSV channels, and color bars are included to help interpret the
        values.

        A color map is used for better visual distinction, with 'hsv' for Hue,
        'viridis' for Saturation, and grayscale for Brightness. Provides an optional
        title for the entire figure and flexibility in the sizing and shrink factor
        of color bars.

        Args:
            figsize (Tuple[int, int]): Size of the figure in inches as a (width, height)
                tuple. Defaults to (10, 8).
            title (str): Title of the entire figure. If None, no title will be set.
            shrink (float): Shrink factor for the color bar size displayed next to
                subplots. Defaults to 0.6.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the created figure and axes
            for further customization or display.
        """
        fig, axes = plt.subplots(nrows=3, figsize=figsize)
        ax = axes.ravel()

        hue = ax[0].imshow(self._parent_image._hsv[:, :, 0] * 360, cmap='hsv', vmin=0, vmax=360)
        ax[0].set_title('Hue')
        ax[0].grid(False)
        fig.colorbar(mappable=hue, ax=ax[0], shrink=shrink)

        saturation = ax[1].imshow(self._parent_image._hsv[:, :, 1], cmap='viridis', vmin=0, vmax=1)
        ax[1].set_title('Saturation')
        ax[1].grid(False)
        fig.colorbar(mappable=saturation, ax=ax[1], shrink=shrink)

        brightness = ax[2].imshow(self._parent_image._hsv[:, :, 2], cmap='gray', vmin=0, vmax=1)
        ax[2].set_title('Brightness')
        ax[2].grid(False)
        fig.colorbar(mappable=brightness, ax=ax[2], shrink=shrink)

        # Adjust ax settings
        if title is not None: ax.set_title(title)

        return fig, ax

    def show_objects(self, figsize: Tuple[int, int] = (10, 8),
                     title: str = None, shrink=0.6) -> (plt.Figure, plt.Axes):
        """
        Displays the Hue, Saturation, and Brightness (HSV components) of the given
        image data in a visualization using subplots. Each subplot corresponds to
        one of the HSV channels, and color bars are included to help interpret the
        values.

        A color map is used for better visual distinction, with 'hsv' for Hue,
        'viridis' for Saturation, and grayscale for Brightness. Provides an optional
        title for the entire figure and flexibility in the sizing and shrink factor
        of color bars.

        Args:
            figsize (Tuple[int, int]): Size of the figure in inches as a (width, height)
                tuple. Defaults to (10, 8).
            title (str): Title of the entire figure. If None, no title will be set.
            shrink (float): Shrink factor for the color bar size displayed next to
                subplots. Defaults to 0.6.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the created figure and axes
            for further customization or display.
        """
        fig, axes = plt.subplots(nrows=3, figsize=figsize)
        ax = axes.ravel()

        hue = ax[0].imshow(np.ma.array(self._parent_image._hsv[:, :, 0] * 360, mask=~self._parent_image.objmask[:]),
                           cmap='hsv', vmin=0, vmax=360
                           )
        ax[0].set_title('Hue')
        ax[0].grid(False)
        fig.colorbar(mappable=hue, ax=ax[0], shrink=shrink)

        saturation = ax[1].imshow(np.ma.array(self._parent_image._hsv[:, :, 1], mask=~self._parent_image.objmask[:]),
                                  cmap='viridis', vmin=0, vmax=1
                                  )
        ax[1].set_title('Saturation')
        ax[1].grid(False)
        fig.colorbar(mappable=saturation, ax=ax[1], shrink=shrink)

        brightness = ax[2].imshow(np.ma.array(self._parent_image._hsv[:, :, 2], mask=~self._parent_image.objmask[:]),
                                  cmap='gray', vmin=0, vmax=1
                                  )
        ax[2].set_title('Brightness')
        ax[2].grid(False)
        fig.colorbar(mappable=brightness, ax=ax[2], shrink=shrink)

        # Adjust ax settings
        if title is not None: ax.set_title(title)

        return fig, ax

    def extract_obj_hue(self, bg_color: int = 0, normalized: bool = False):
        """Extracts the object hue from the HSV image.

        Note:
            - Unnormalized Range: 0-360 degrees.
            - Normalized Range: 0-1
        """
        return self._parent_image.objmask._extract_objects(
            self._parent_image._hsv[:, :, 0] if normalized else self._parent_image._hsv[:, :, 0] * 360,
            bg_color=bg_color
        )

    def extract_obj_saturation(self, bg_color: int = 0, normalized: bool = False):
        """Extracts the object saturation from the HSV image.

        Note:
            - Unnormalized Range: 0-255 (Same as OpenCV)
            - Normalized Range: 0-1

        """
        return self._parent_image.objmask._extract_objects(
            self._parent_image._hsv[:, :, 1] if normalized else self._parent_image._hsv[:, :, 1] * 255,
            bg_color=bg_color
        )

    def extract_obj_brightness(self, bg_color: int = 0, normalized: bool = False):
        """Extracts the object brightness from the HSV image.

        Note:
            - Unnormalized Range: 0-255 (Same as OpenCV)
            - Normalized Range: 0-1

        """
        return self._parent_image.objmask._extract_objects(
            self._parent_image._hsv[:, :, 2] if normalized else self._parent_image._hsv[:, :, 2] * 255,
            bg_color=bg_color
        )

    def extract_obj(self, bg_color: int = 0):
        """Extracts the object hue, saturation, and brightness from the HSV image. With the background elements set to 0"""
        return self._parent_image.objmask._extract_objects(self._parent_image._hsv[:, :, :], bg_color=bg_color)
