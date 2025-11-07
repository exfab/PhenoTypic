from abc import ABC
from typing import Tuple, Optional

import numpy as np

from skimage.exposure import histogram
import matplotlib.pyplot as plt

from phenotypic.core._image_parts.accessor_abstracts import ImageAccessorBase


class SingleChannelAccessor(ImageAccessorBase):
    """
    Handles interaction with Image 2-d gray data by providing access to Image attributes and data.

    This class serves as a bridge for interacting with Image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    Image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessorBase`.

    Attributes:
        image (Any): Root Image object that this accessor is linked to.
        _main_arr (Any): Main array storing the Image-related data.
        _dtype (Any): Data type of the Image data stored in the target array.
    """

    def show(self,
             figsize: tuple[int, int] | None = None,
             title: str | None = None,
             ax: plt.Axes | None = None,
             cmap: str | None = 'gray',
             foreground_only: bool = False,
             *,
             mpl_settings: dict | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays a visual representation of the current object using matplotlib.

        This method generates and displays an image or a plot of the object's data
        using matplotlib. It provides options to customize the figure size, title,
        color map, and other visual properties. It also allows focusing on specific
        foreground elements if desired.

        Args:
            figsize (tuple[int, int] | None): A tuple specifying the size of the
                matplotlib figure in inches (width, height). If None, default
                settings are used.
            title (str | None): The title of the plot. If None, no title is displayed.
            ax (plt.Axes | None): A matplotlib Axes object on which the plot will be
                drawn. If None, a new Axes object is created.
            cmap (str | None): The name of the colormap to use. Defaults to 'gray'.
            foreground_only (bool): A flag indicating whether to display only the
                foreground elements of the data. Defaults to False.
            mpl_settings (dict | None): A dictionary of matplotlib settings to apply
                to the figure or Axes. If None, no additional settings are applied.

        Returns:
            tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects
                containing the generated plot.
        """
        return self._plot(
                arr=self[:] if not foreground_only else self.foreground(),
                figsize=figsize,
                ax=ax,
                title=title,
                cmap=cmap,
                mpl_settings=mpl_settings,
        )
