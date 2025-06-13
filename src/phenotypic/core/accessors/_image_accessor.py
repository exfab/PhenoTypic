from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import skimage
import matplotlib.pyplot as plt
import numpy as np


class ImageAccessor:
    """
    The base for classes that provides access to details and functionalities of a parent _root_image.

    The ImageAccessor class serves as a base class for interacting with a parent _root_image
    object. It requires an instance of the parent _root_image for initialization to
    enable seamless operations on the _root_image's properties and data.

    Attributes:
        _parent_image (Image): The parent _root_image object that this accessor interacts
            with.
    """

    def __init__(self, parent_image: Image):
        self._root_image = parent_image

    def _plot(self,
              arr: np.ndarray,
              figsize: (int, int) = (8, 6),
              title: str | bool | None = None,
              cmap: str = 'gray',
              ax: plt.Axes | None = None,
              mpl_params: dict | None = None,
              ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots an _root_image array using Matplotlib.

        This method is designed to render an _root_image array using the `matplotlib.pyplot` module. It provides
        flexible options for color mapping, figure size, title customization, and additional Matplotlib
        parameters, which enable detailed control over the plot appearance.

        Args:
            arr (np.ndarray): The _root_image data to plot. Can be 2D or 3D array representing the _root_image.
            figsize ((int, int), optional): A tuple specifying the figure size. Defaults to (8, 6).
            title (None | str, optional): Plot title. If None, defaults to the name of the parent _root_image. Defaults to None.
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
