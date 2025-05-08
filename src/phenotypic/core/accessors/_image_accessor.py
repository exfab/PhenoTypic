from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import skimage
import matplotlib.pyplot as plt
import numpy as np


class ImageAccessor:
    """
    The base for classes that provides access to details and functionalities of a parent image.

    The ImageAccessor class serves as a base class for interacting with a parent image
    object. It requires an instance of the parent image for initialization to
    enable seamless operations on the image's properties and data.

    Attributes:
        _parent_image (Image): The parent image object that this accessor interacts
            with.
    """

    def __init__(self, parent_image: Image):
        self._parent_image = parent_image

    def _plot(self,
              arr: np.ndarray,
              figsize: (int, int) = (8, 6),
              title: str | None = None,
              cmap: str = 'gray',
              ax: plt.Axes | None = None,
              mpl_params: dict | None = None,
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
            mpl_params (dict | None, optional): Additional Matplotlib keyword arguments for customization. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the created or passed Matplotlib `Figure` and `Axes` objects.

        """
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        else: fig = ax.get_figure()

        mpl_params = mpl_params if mpl_params else {}
        cmap = mpl_params.get('cmap', cmap)

        ax.imshow(arr, cmap=cmap, **mpl_params) if arr.ndim == 2 else ax.imshow(arr, **mpl_params)

        ax.grid(False)
        if title: ax.set_title(title)
        # ax.set_title(title) if title else ax.set_title(self._parent_image.name)

        return fig, ax

    def _plot_overlay(self,
                      arr: np.ndarray,
                      objmap: np.ndarray,
                      figsize: (int, int) = (8, 6),
                      title: str = None,
                      cmap: str = 'gray',
                      ax: plt.Axes = None,
                      overlay_params: dict | None = None,
                      imshow_params: dict | None = None,
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
                defaults to the name of the self._parent_image.
            cmap (str, optional): Colormap to apply to the image. Defaults to 'gray'. Only used if arr input is 2D.
            ax (plt.Axes, optional): An existing Matplotlib Axes instance for rendering
                the image. If None, a new figure and axes are created. Defaults to None.
            overlay_params (dict | None, optional): Parameters passed to the
                `skimage.color.label2rgb` function for overlay customization.
                Defaults to None.
            imshow_params (dict | None, optional): Additional parameters for the
                `ax.imshow` Matplotlib function to control image rendering.
                Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: The Matplotlib Figure and Axes objects used for
            the display. If an existing Axes is provided, its corresponding Figure is returned.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        overlay_params = overlay_params if overlay_params else {}

        imshow_params = imshow_params if imshow_params else {}
        cmap = imshow_params.get('cmap', cmap)

        imarray = skimage.color.label2rgb(label=objmap, image=arr, bg_label=0, **overlay_params)
        ax.imshow(imarray, cmap=cmap, **imshow_params) if imarray.ndim == 2 else ax.imshow(imarray, **imshow_params)

        ax.grid(False)
        if title: ax.set_title(title)

        return fig, ax

    def _plot_annotations(self, ax: plt.Axes, color: str, size: int, facecolor: str, object_label: None | int, **kwargs):
        props = self._parent_image.objects.props
        for i, label in enumerate(self._parent_image.objects.labels):
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
