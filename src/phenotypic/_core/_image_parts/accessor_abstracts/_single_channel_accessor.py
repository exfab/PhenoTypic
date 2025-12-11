from abc import ABC
from typing import Tuple, Optional, Literal, overload, TYPE_CHECKING

import numpy as np

from skimage.exposure import histogram
import matplotlib.pyplot as plt

from phenotypic._core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic._core._image_parts._plotting_backends import (
    PlotReturn,
    MatplotlibReturn,
    PlotlyReturn,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


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

    @overload
    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            ax: plt.Axes | None = None,
            cmap: str | None = "gray",
            foreground_only: bool = False,
            *,
            mpl_settings: dict | None = None,
            backend: Literal["matplotlib"] = "matplotlib",
    ) -> MatplotlibReturn:
        ...

    @overload
    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            ax: None = None,
            cmap: str | None = "gray",
            foreground_only: bool = False,
            *,
            mpl_settings: dict | None = None,
            backend: Literal["plotly"],
            plotly_settings: dict | None = None,
    ) -> PlotlyReturn:
        ...

    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            ax: plt.Axes | None = None,
            cmap: str | None = "gray",
            foreground_only: bool = False,
            *,
            mpl_settings: dict | None = None,
            backend: Literal["matplotlib", "plotly"] = "matplotlib",
            plotly_settings: dict | None = None,
    ) -> PlotReturn:
        """
        Display visual representation using matplotlib or plotly backend.

        Generates and displays an image or plot of the accessor's data with
        flexible backend selection and customization options.

        Args:
            figsize (tuple[int, int] | None, optional): Figure size in inches
                (width, height). If None, uses default settings. Defaults to None.
            title (str | None, optional): Plot title. If None, no title displayed.
                Defaults to None.
            ax (plt.Axes | None, optional): Matplotlib Axes object for rendering.
                Only valid for matplotlib backend. If None, new Axes created.
                Defaults to None.
            cmap (str | None, optional): Colormap name. Defaults to 'gray'.
            foreground_only (bool, optional): If True, display only foreground
                elements. Defaults to False.
            mpl_settings (dict | None, optional): Matplotlib settings. Only used
                with matplotlib backend. Defaults to None.
            backend (Literal["matplotlib", "plotly"], optional): Backend to use.
                Defaults to "matplotlib".
            plotly_settings (dict | None, optional): Plotly-specific settings.
                Only used with plotly backend. Defaults to None.

        Returns:
            PlotReturn:
                - If backend="matplotlib": tuple[plt.Figure, plt.Axes]
                - If backend="plotly": plotly.graph_objects.Figure

        Raises:
            ValueError: If backend invalid or ax with plotly backend.
            ImportError: If plotly requested but not installed.
        """
        return self._plot(
            arr=self[:] if not foreground_only else self.foreground(),
            figsize=figsize,
            ax=ax,
            title=title,
            cmap=cmap,
            mpl_settings=mpl_settings,
            backend=backend,
            plotly_settings=plotly_settings,
        )
