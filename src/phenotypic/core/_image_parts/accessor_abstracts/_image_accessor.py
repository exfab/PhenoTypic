from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Dict

if TYPE_CHECKING: from phenotypic import Image

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

from phenotypic.util.constants_ import MPL, METADATA_LABELS, IMAGE_FORMATS
from abc import ABC, abstractmethod


class ImageAccessorBase(ABC):
    """
    The base for classes that provides access to details and functionalities of a parent image.

    The ImageAccessorBase class serves as a base class for interacting with a parent image
    object. It requires an instance of the parent image for initialization to
    enable seamless operations on the image's properties and data.

    Attributes:
        image (Image): The parent image object that this accessor interacts
            with.
    """

    def __init__(self, parent_image: Image):
        self._root_image = parent_image

    @property
    @abstractmethod
    def _subject_arr(self) -> np.ndarray:
        """
        Abstract property representing a subject array. The subject array is expected to be a NumPy ndarray
        with a specific shape of (0, 0, 3), which can be used for various operations that require a structured
        multi-dimensional array.

        This property is abstract and must be implemented in any derived concrete class. The implementation
        should conform to the type signature and shape expectations as defined.

        Note: Read-only property. Changes should reference the specific array

        Returns:
            np.ndarray: A NumPy ndarray object with shape (0, 0, 3).
        """
        return np.empty(shape=(0, 0, 3))

    def histogram(self, figsize: Tuple[int, int] = (10, 5), *, linewidth=1, channel_names: list | None = None) -> Tuple[
        plt.Figure, plt.Axes]:
        """
        Plots the histogram of an image with an option to display across color channels. Supports both grayscale
        and multi-channel images. Creates subplots containing the image and its histogram(s). For grayscale images,
        a single histogram is generated. For multi-channel images, histograms are generated for each channel.

        Args:
            figsize (Tuple[int, int], optional): The size of the figure to be created. Default is (10, 5).
            linewidth (int, optional): The line width to be used for plotting the histogram. Default is 1.
            channel_names (list | None, optional): The names of the channels for multi-channel images. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the created matplotlib figure and axes.
        """
        match self._subject_arr.ndim:
            case 2:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
                axes = axes.ravel()
                axes[0] = self._plot(arr=self._subject_arr, figsize=figsize, title=self._root_image.name, cmap='gray', ax=axes[0])
                hist, histc = ski.exposure.histogram(image=self._subject_arr[:],
                                                     nbins=2 ** self._root_image.metadata[METADATA_LABELS.BIT_DEPTH])
                axes[1].plot(histc, hist, lw=linewidth)
            case 3:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
                axes[0] = self._plot(arr=self._subject_arr, figsize=figsize, title=self._root_image.name, ax=axes[0])
                for idx, ax in enumerate(axes.ravel()):
                    if idx == 0: continue
                    hist, histc = ski.exposure.histogram(image=self._subject_arr[:, :, idx],
                                                         nbins=2 ** self._root_image.metadata[METADATA_LABELS.BIT_DEPTH])
                    ax.plot(histc, hist, lw=linewidth)
                    ax.set_title(f'Channel-{channel_names[idx-1] if channel_names else idx}')

            case _:
                raise ValueError(f"Unsupported array dimension: {self._subject_arr.ndim}")
        return fig, axes

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the current image data.

        This method retrieves the dimensions of the array stored in the `_main_arr`
        attribute as a tuple, which indicates its size along each axis.

        Returns:
            Tuple[int, ...]: A tuple representing the dimensions of the `_main_arr`
            attribute.
        """
        return self._subject_arr.shape

    def copy(self) -> np.ndarray:
        return self._subject_arr.copy()

    def foreground(self):
        foreground = self._subject_arr.copy()
        foreground[self._root_image.objmask[:] == 0] = 0
        return foreground

    def _plot(self,
              arr: np.ndarray,
              figsize: Tuple[int, int] | None = None,
              title: str | bool | None = None,
              cmap: str = 'gray',
              ax: plt.Axes | None = None,
              mpl_kwargs: dict | None = None,
              ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots an image array using Matplotlib.

        This method is designed to render an image array using the `matplotlib.pyplot` module. It provides
        flexible options for color mapping, figure size, title customization, and additional Matplotlib
        parameters, which enable detailed control over the plot appearance.

        Args:
            arr (np.ndarray): The image data to plot. Can be 2D or 3D array representing the image.
            figsize ((int, int), optional): A tuple specifying the figure size. Defaults to (8, 6).
            title (None | str, optional): Plot title. If None, defaults to the name of the parent image. Defaults to None.
            cmap (str, optional): The colormap to be applied when the array is 2D. Defaults to 'gray'.
            ax (None | plt.Axes, optional): Existing Matplotlib axes to plot into. If None, a new figure is created. Defaults to None.
            mpl_kwargs (dict | None, optional): Additional Matplotlib keyword arguments for customization. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the created or passed Matplotlib `Figure` and `Axes` objects.

        """
        figsize = figsize if figsize else MPL.FIGSIZE
        fig, ax = (ax.get_figure(), ax) if ax else plt.subplots(figsize=figsize)

        mpl_kwargs = mpl_kwargs if mpl_kwargs else {}
        cmap = mpl_kwargs.get('cmap', cmap)

        # matplotlib.imshow can only handle ranges 0-1 or 0-255
        # this adds handling for higher bit-depth images
        max_val = arr.max()
        match max_val:
            case _ if max_val <= 255:
                arr = (arr.copy().astype(np.float32) / 255).clip(0, 1)
            case _ if max_val <= 65535:
                arr = (arr.copy().astype(np.float32) / 65535.0).clip(0, 1)
            case _:
                raise ValueError("Values exceed 16-bit range")

        ax.imshow(arr, cmap=cmap, **mpl_kwargs) if arr.ndim == 2 else ax.imshow(arr, **mpl_kwargs)

        ax.grid(False)
        if title is True:
            ax.set_title(self._root_image.name)
        elif title:
            ax.set_title(title)

        return fig, ax



    def _plot_obj_labels(self, ax: plt.Axes, color: str, size: int, facecolor: str, object_label: None | int, **kwargs):
        props = self._root_image.objects.props
        for i, label in enumerate(self._root_image.objects.labels):
            if object_label is None:
                text_rr, text_cc = props[i].centroid
                ax.text(
                    x=text_cc, y=text_rr,
                    s=f'{label}',
                    color=color,
                    fontsize=size,
                    bbox=dict(facecolor=facecolor, edgecolor='none', alpha=0.6, boxstyle='round'),
                    **kwargs,
                )
            elif object_label == label:
                text_rr, text_cc = props[i].centroid
                ax.text(
                    x=text_cc, y=text_rr,
                    s=f'{label}',
                    color=color,
                    fontsize=size,
                    bbox=dict(facecolor=facecolor, edgecolor='none', alpha=0.6, boxstyle='round'),
                    **kwargs,
                )
        return ax


