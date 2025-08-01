from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING: from phenotypic import Image

import skimage
import matplotlib.pyplot as plt
import numpy as np

from phenotypic.util.constants_ import MPL


class ImageAccessor:
    """
    The base for classes that provides access to details and functionalities of a parent image.

    The ImageAccessor class serves as a base class for interacting with a parent image
    object. It requires an instance of the parent image for initialization to
    enable seamless operations on the image's properties and data.

    Attributes:
        image (Image): The parent image object that this accessor interacts
            with.
    """

    def __init__(self, parent_image: Image):
        self._root_image = parent_image

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
