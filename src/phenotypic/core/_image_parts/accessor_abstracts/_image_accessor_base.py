from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING: from phenotypic import Image

import skimage as ski
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from phenotypic.util.constants_ import MPL, METADATA
from abc import ABC


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

    def __init__(self, root_image: Image):
        self._root_image = root_image

    @property
    def _subject_arr(self) -> np.ndarray:
        """
        Abstract property representing a image array. The image array is expected to be a NumPy ndarray
        with a specific shape of (0, 0, 3), which can be used for various operations that require a structured
        multi-dimensional array.

        This property is abstract and must be implemented in any derived concrete class. The implementation
        should conform to the type signature and shape expectations as defined.

        Note: Read-only property. Changes should reference the specific array

        Returns:
            np.ndarray: A NumPy ndarray object with shape (0, 0, 3).
        """
        raise NotImplementedError("This property is abstract and must be implemented in a derived class.")

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

    def isempty(self):
        return True if self.shape[0] == 0 else False

    def copy(self) -> np.ndarray:
        return self._subject_arr.copy()

    def foreground(self):
        foreground = self._subject_arr.copy()
        foreground[self._root_image.objmask[:] == 0] = 0
        return foreground

    def histogram(self, figsize: Tuple[int, int] = (10, 5), *, cmap='gray', linewidth=1,
                  channel_names: list | None = None) -> Tuple[
        plt.Figure, plt.Axes]:
        """
        Plots the histogram(s) of an image along with the image itself. The behavior depends on
        the dimensionality of the image array (2D or 3D). In the case of 2D, a single image and
        its histogram are produced. For 3D (multi-channel images), histograms for each channel
        are created alongside the image. This method supports customization such as figure size,
        colormap, line width of histograms, and labeling of channels.

        Args:
            figsize (Tuple[int, int]): The size of the figure to create. Default is (10, 5).
            cmap (str): Colormap used to render the image when the data is single channel. Default is 'gray'.
            linewidth (int): Line width of the plotted histograms. Default is 1.
            channel_names (list | None): Optional names for the channels in 3D data. These are
                used as titles for channel-specific histograms. If None, channels are instead
                labeled numerically.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The Matplotlib figure and axes objects representing the
            plotted image and its histograms.

        Raises:
            ValueError: If the dimensionality of the input image array is unsupported.
        """
        match self._subject_arr.ndim:
            case 2:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
                axes = axes.ravel()
                axes[0] = self._plot(arr=self._subject_arr, figsize=figsize, title=self._root_image.name, cmap=cmap,
                                     ax=axes[0])
                hist, histc = ski.exposure.histogram(image=self._subject_arr[:],
                                                     nbins=2 ** self._root_image.metadata[METADATA.BIT_DEPTH])
                axes[1].plot(histc, hist, lw=linewidth)
            case 3:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
                axes[0] = self._plot(arr=self._subject_arr, figsize=figsize, title=self._root_image.name, ax=axes[0])
                for idx, ax in enumerate(axes.ravel()):
                    if idx == 0: continue
                    hist, histc = ski.exposure.histogram(image=self._subject_arr[:, :, idx],
                                                         nbins=2 ** self._root_image.metadata[METADATA.BIT_DEPTH])
                    ax.plot(histc, hist, lw=linewidth)
                    ax.set_title(f'Channel-{channel_names[idx - 1] if channel_names else idx}')

            case _:
                raise ValueError(f"Unsupported array dimension: {self._subject_arr.ndim}")
        return fig, axes

    def _plot(self,
              arr: np.ndarray,
              figsize: Tuple[int, int] | None = None,
              title: str | bool | None = None,
              cmap: str = 'gray',
              ax: plt.Axes | None = None,
              mpl_settings: dict | None = None,
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
            mpl_settings (dict | None, optional): Additional Matplotlib keyword arguments for customization. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the created or passed Matplotlib `Figure` and `Axes` objects.

        """
        figsize = figsize if figsize else MPL.FIGSIZE
        fig, ax = (ax.get_figure(), ax) if ax else plt.subplots(figsize=figsize)

        mpl_settings = mpl_settings if mpl_settings else {}
        cmap = mpl_settings.get('cmap', cmap)

        # matplotlib.imshow can only handle ranges 0-1 or 0-255
        # this adds handling for higher bit-depth images
        max_val = arr.max()
        if 0 <= max_val <= 1:
            plot_arr = arr.copy().astype(np.float32)
        elif 1 < max_val <= 255:
            plot_arr = (arr.copy().astype(np.float32)/255).clip(0, 1)
        elif 255 < max_val <= 65535:
            plot_arr = (arr.copy().astype(np.float32)/65535.0).clip(0, 1)
        else:
            raise ValueError("Values exceed 16-bit range")

        ax.imshow(plot_arr, cmap=cmap, **mpl_settings) if plot_arr.ndim == 2 else ax.imshow(plot_arr, **mpl_settings)

        ax.grid(False)
        arr_shape = arr.shape

        if arr_shape[0] > 500:
            ax.yaxis.set_minor_locator(MultipleLocator(100))
            ax.yaxis.set_major_locator(MultipleLocator(500))

        if arr_shape[1] > 500:
            ax.xaxis.set_minor_locator(MultipleLocator(100))
            ax.xaxis.set_major_locator(MultipleLocator(500))

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

    def _plot_overlay(self,
                      arr: np.ndarray,
                      objmap: np.ndarray,
                      figsize: (int, int) = (8, 6),
                      title: str | bool | None = None,
                      cmap: str = 'gray',
                      ax: plt.Axes = None,
                      *,
                      overlay_settings: dict | None = None,
                      mpl_settings: dict | None = None,
                      ) -> (plt.Figure, plt.Axes):
        """
        Plots an array with optional object map overlay and customization options.

        Note:
            - If ax is None, a new figure and axes are created.

        Args:
            arr (np.ndarray): The primary array to be displayed as an image.
            objmap (np.ndarray, optional): An array containing labels for an object map to
                overlay on top of the image. Defaults to None.
            figsize (tuple[int, int], optional): The size of the figure as a tuple of
                (width, height). Defaults to (8, 6).
            title (str, optional): Title of the plot to be displayed. If not provided,
                defaults to the name of the self.image.
            cmap (str, optional): Colormap to apply to the image. Defaults to 'gray'. Only used if arr input_image is 2D.
            ax (plt.Axes, optional): An existing Matplotlib Axes instance for rendering
                the image. If None, a new figure and axes are created. Defaults to None.
            overlay_settings (dict | None, optional): Parameters passed to the
                `skimage.color.label2rgb` function for overlay customization.
                Defaults to None.
            mpl_settings (dict | None, optional): Additional parameters for the
                `ax.imshow` Matplotlib function to control image rendering.
                Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: The Matplotlib Figure and Axes objects used for
            the display. If an existing Axes is provided, its corresponding Figure is returned.
        """
        overlay_settings = overlay_settings if overlay_settings else {}
        overlay_alpha = overlay_settings.get('alpha', 0.15)
        overlay_arr = ski.color.label2rgb(label=objmap, image=arr, bg_label=0, alpha=overlay_alpha, **overlay_settings)

        fig, ax = self._plot(arr=overlay_arr, figsize=figsize, title=title, cmap=cmap, ax=ax, mpl_settings=mpl_settings)

        return fig, ax

    def show_overlay(self,
                     object_label: None | int = None,
                     figsize: tuple[int, int] | None = None,
                     title: str | None = None,
                     show_labels: bool = False,
                     ax: plt.Axes = None,
                     *,
                     label_settings: None | dict = None,
                     overlay_settings: None | dict = None,
                     imshow_settings: None | dict = None,
                     ) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays an overlay of the object map on the parent image with optional annotations.

        This method enables visualization by overlaying object regions on the parent image. It
                provides options for customization, including the ability to show_labels specific objects
        and adjust visual styles like figure size, colors, and annotation properties.

        Args:
            object_label (None | int): Specific object label to be highlighted. If None,
                all objects are displayed.
            figsize (tuple[int, int]): Size of the figure in inches (width, height).
            title (None | str): Title for the plot. If None, the parent image's name
                is used.
            show_labels (bool): If True, displays annotations for object labels on the
                object centroids.
            label_settings (None | dict): Additional parameters for customization of the
                object annotations. Defaults: size=12, color='white', facecolor='red'. Other kwargs
                are passed to the matplotlib.axes.text () method.
            ax (plt.Axes): Optional Matplotlib Axes object. If None, a new Axes is
                created.
            overlay_settings (None | dict): Additional parameters for customization of the
                overlay.
            imshow_settings (None|dict): Additional Matplotlib imshow configuration parameters
                for customization. If None, default Matplotlib settings will apply.

        Returns:
            tuple[plt.Figure, plt.Axes]: Matplotlib Figure and Axes objects containing
            the generated plot.

        """
        objmap = self._root_image.objmap[:]
        if object_label is not None: objmap[objmap != object_label] = 0
        if label_settings is None: label_settings = {}

        fig, ax = self._plot_overlay(
                arr=self._subject_arr,
                objmap=objmap,
                ax=ax,
                figsize=figsize,
                title=title,
                mpl_settings=imshow_settings,
                overlay_settings=overlay_settings,
        )

        if show_labels:
            ax = self._plot_obj_labels(
                    ax=ax,
                    color=label_settings.get('color', 'white'),
                    size=label_settings.get('size', 12),
                    facecolor=label_settings.get('facecolor', 'red'),
                    object_label=object_label,
            )
        return fig, ax

    def imsave(self, filepath:str|os.PathLike):
