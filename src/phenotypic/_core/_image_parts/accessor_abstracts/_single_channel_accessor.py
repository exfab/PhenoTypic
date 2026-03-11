from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from phenotypic._core._image_parts.accessor_abstracts import ImageAccessorBase

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go


class SingleChannelAccessor(ImageAccessorBase, ABC):
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

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError

    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            cmap: str | None = "gray",
            foreground_only: bool = False,
            *,
            plotly_settings: dict | None = None,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Display the single-channel image data interactively.

        Uses Plotly when available, falling back to matplotlib otherwise.

        Args:
            figsize: Figure size in inches (width, height). If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, no title is displayed.
            cmap: Colormap name. Defaults to ``"gray"``.
            foreground_only: If True, display only foreground elements.
            plotly_settings: Additional Plotly layout settings.

        Returns:
            A ``plotly.graph_objects.Figure`` when plotly is installed,
            or a ``(plt.Figure, plt.Axes)`` tuple when using matplotlib
            fallback.
        """
        from phenotypic.tools_._plotly_helpers import PLOTLY_AVAILABLE

        arr = self[:] if not foreground_only else self.foreground()

        if not PLOTLY_AVAILABLE:
            return self._mpl_plot(
                arr=arr, figsize=figsize, title=title, cmap=cmap or "gray",
            )

        from phenotypic.tools_._plotly_helpers import plotly_imshow

        fig = plotly_imshow(arr=arr, figsize=figsize, title=title, cmap=cmap)
        if plotly_settings is not None:
            fig.update_layout(**plotly_settings)
        return fig
