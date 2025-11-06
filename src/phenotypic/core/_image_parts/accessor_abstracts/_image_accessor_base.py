from __future__ import annotations

from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: from phenotypic import Image

import skimage as ski
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from phenotypic.tools.constants_ import MPL, METADATA, IO
import warnings
from phenotypic.tools.funcs_ import normalize_rgb_bitdepth
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
        Abstract property representing an image array. The image array is expected to be a NumPy ndarray
        with a specific shape of (r, c, ...), which can be used for various operations that require a structured
        multi-dimensional array.

        This property is abc_ and must be implemented in any derived concrete class. The implementation
        should conform to the type signature and shape expectations as defined.

        Note: Read-only property. Changes should reference the specific array

        Returns:
            np.ndarray: A NumPy ndarray object with shape (r, c, ...).
        """
        raise NotImplementedError("This property is abc_ and must be implemented in a derived class.")

    def __array__(self, dtype=None, copy=None):
        """Implements the array interface for numpy compatibility.
        
        This allows numpy functions to operate directly on accessor objects.
        For example: np.sum(accessor), np.mean(accessor), etc.
        
        Args:
            dtype: Optional dtype to cast the array to
            copy: Optional copy parameter for NumPy 2.0+ compatibility
            
        Returns:
            np.ndarray: The underlying array data
        """
        arr = self._subject_arr
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        elif copy:
            arr = arr.copy()
        return arr

    def __len__(self) -> int:
        """
        Returns the length of the subject array.

        This method calculates and returns the total number of elements contained in the
        underlying array.

        Returns:
            int: The number of elements in the underlying array attribute.
        """
        return len(self._subject_arr)

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

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the underlying array.

        The `ndim` property provides access to the dimensionality of the array
        being encapsulated in the object. This value corresponds to the number
        of axes or dimensions the underlying array possesses. It can be useful
        for understanding the structure of the contained data.

        Returns:
            int: The number of dimensions of the underlying array.
        """
        return self._subject_arr.ndim

    @property
    def size(self) -> int:
        """
        Gets the size of the subject array.

        This property retrieves the total number of elements in the subject
        array. It is read-only.

        Returns:
            int: The total number of elements in the subject array.
        """
        return self._subject_arr.size

    def val_range(self) -> pd.Interval:
        """
        Return the closed interval [min, max] of the subject array values.

        Returns:
            pd.Interval: A single closed interval including both endpoints.
        """
        mn = self._subject_arr.min()
        mx = self._subject_arr.max()
        return pd.Interval(left=mn, right=mx, closed="both")

    @property
    def dtype(self):
        return self._subject_arr.dtype

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
        arr = self._subject_arr
        dtype = arr.dtype

        if np.issubdtype(dtype, np.floating):
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_min < 0.0 or arr_max > 1.0:
                raise ValueError(
                        f"Float image arrays must be within [0.0, 1.0]. Found range [{arr_min}, {arr_max}].")
            x_limits = (0.0, 1.0)
        elif np.issubdtype(dtype, np.bool_):
            x_limits = (0, 1)
        elif np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
            x_limits = (dtype_info.min, dtype_info.max)
        else:
            raise TypeError(f"Unsupported image dtype for histogram plotting: {dtype}")

        match self.ndim:
            case 2:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
                axes = axes.ravel()
                axes[0] = self._plot(arr=self._subject_arr, figsize=figsize, title=self._root_image.name, cmap=cmap,
                                     ax=axes[0])
                hist, histc = ski.exposure.histogram(image=self._subject_arr[:],
                                                     nbins=2 ** self._root_image.metadata[METADATA.BIT_DEPTH])
                axes[1].plot(histc, hist, lw=linewidth)
                axes[1].set_xlim(x_limits)

            case 3:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

                for idx, ax in enumerate(axes.flat):
                    if idx == 0:
                        self._plot(arr=self._subject_arr[:],
                                   figsize=figsize, title=self._root_image.name,
                                   ax=ax)
                    else:
                        hist, histc = ski.exposure.histogram(image=self._subject_arr[:, :, idx - 1],
                                                             nbins=2 ** self._root_image.metadata[METADATA.BIT_DEPTH])
                        ax.plot(histc, hist, lw=linewidth)
                        ax.set_title(f'Channel-{channel_names[idx - 1] if channel_names else idx}')
                        ax.set_xlim(x_limits)

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
        plot_arr = normalize_rgb_bitdepth(arr)

        ax.imshow(plot_arr, cmap=cmap, **mpl_settings) if plot_arr.ndim == 2 else ax.imshow(plot_arr, **mpl_settings)

        ax.grid(False)

        # arr_shape = arr.shape
        # if arr_shape[0] > 500:
        #     ax.yaxis.set_minor_locator(MultipleLocator(100))
        #
        # if arr_shape[1] > 500:
        #     ax.xaxis.set_minor_locator(MultipleLocator(100))

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
            cmap (str, optional): Colormap to apply to the image. Defaults to 'gray'. Only used if arr arr is 2D.
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

    def imsave(self, filepath: str | Path | None = None, ):
        filepath = Path(filepath)
        from PIL import Image as PIL_Image

        arr2save = self._subject_arr
        match filepath.suffix.lower():
            case x if x in IO.JPEG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8:
                        pass
                    case np.uint16:
                        warnings.warn(
                                'Saving a 16 bit array as a jpeg will result in information loss if the max value is higher than 255')
                        arr2save = ski.util.img_as_ubyte(arr2save)
                    case dt if np.issubdtype(dt, np.floating):
                        warnings.warn(
                                'Saving a float array as a jpeg will result in information loss if the max value is higher than 255')
                        arr2save = ski.util.img_as_ubyte(arr2save)
                PIL_Image.fromarray(arr2save).save(filepath, format=filepath.suffix.lower(), quality=100)
            case x if x in IO.PNG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8 | np.uint16:
                        pass
                    case dt if np.issubdtype(dt, np.floating):
                        warnings.warn(
                                '.png images only accept 8 bit ana 16 bit integer arrays. '
                                'Converting this array may cause information loss')
                        arr2save = ski.util.img_as_ubyte(arr2save) \
                            if self._root_image.bit_depth == 8 \
                            else ski.util.img_as_uint(arr2save)
                PIL_Image.fromarray(arr2save).save(filepath, format=filepath.suffix.lower(), optimize=True)
            case x if x in IO.TIFF_EXTENSIONS:
                PIL_Image.fromarray(arr2save).save(filepath, format=filepath.suffix.lower())
            case _:
                raise ValueError(f'unknown file extension for saving:{filepath.suffix}')
