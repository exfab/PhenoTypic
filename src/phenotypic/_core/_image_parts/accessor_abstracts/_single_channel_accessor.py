from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from phenotypic._core._image_parts.accessor_abstracts import ImageAccessorBase

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
else:  # runtime: resolvable stand-ins so get_type_hints works
    from phenotypic.sdk_._lazy_annotations import plt


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
            overlay: bool = True,
            *,
            ax: plt.Axes | None = None,
            show_overlay_notice: bool = True,
            object_label: int | None = None,
            show_labels: bool = False,
            show_grid: bool = True,
            label_settings: dict | None = None,
            overlay_settings: dict | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Display the single-channel image data using matplotlib.

        Args:
            figsize: Figure size in inches (width, height). If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, no title is displayed.
            cmap: Colormap name. Defaults to ``"gray"``.
            foreground_only: If True, display only foreground elements.
            overlay: If True, overlay the object map on the image.
                Falls back to plain image when no objects are detected.
            ax: Existing Matplotlib axes to plot into. If None, a new
                figure and axes are created.
            show_overlay_notice: If True, display an ``Overlay`` notice when
                an object overlay is rendered.
            object_label: Specific object label to highlight. If None,
                shows all detected objects. Only used when overlay is True.
            show_labels: If True, displays numeric labels at object centroids.
                Only used when overlay is True.
            show_grid: For GridImage only. If True, draws gridlines
                and colored section boxes. Only used when overlay is
                True. Ignored for regular Image.
            label_settings: Dict passed to text label rendering.
            overlay_settings: Dict passed to overlay composition.

        Returns:
            A ``(plt.Figure, plt.Axes)`` tuple.
        """
        arr = self[:] if not foreground_only else self.foreground()

        has_objects = self._root_image.num_objects > 0
        if overlay and has_objects:
            objmap = self._get_filtered_objmap(object_label)
            fig, ax = self._plot_overlay(
                    arr=arr, objmap=objmap, figsize=figsize, title=title,
                    ax=ax,
                    overlay_settings=overlay_settings,
            )
            self._decorate_mpl_overlay(
                    ax, has_objects=has_objects, object_label=object_label,
                    show_overlay_notice=(
                        show_overlay_notice and bool(objmap.any())
                    ),
                    show_labels=show_labels, show_grid=show_grid,
                    label_settings=label_settings,
            )
            return fig, ax

        return self._mpl_plot(
                arr=arr, figsize=figsize, title=title, cmap=cmap or "gray", ax=ax,
        )

    def dash(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            cmap: str | None = "gray",
            foreground_only: bool = False,
            overlay: bool = True,
            *,
            show_overlay_notice: bool = True,
            object_label: int | None = None,
            show_labels: bool = False,
            show_grid: bool = True,
            label_settings: dict | None = None,
            overlay_settings: dict | None = None,
            plotly_settings: dict | None = None,
    ) -> go.Figure:
        """Display the single-channel image data using Plotly.

        Args:
            figsize: Figure size in inches (width, height). If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, no title is displayed.
            cmap: Colormap name. Defaults to ``"gray"``.
            foreground_only: If True, display only foreground elements.
            overlay: If True, overlay the object map on the image.
                Falls back to plain image when no objects are detected.
            show_overlay_notice: If True, display an ``Overlay`` notice when
                an object overlay is rendered.
            object_label: Specific object label to highlight. If None,
                shows all detected objects. Only used when overlay is True.
            show_labels: If True, displays numeric labels at object centroids.
                Only used when overlay is True.
            show_grid: For GridImage only. If True, draws gridlines
                and colored section boxes. Only used when overlay is
                True. Ignored for regular Image.
            label_settings: Dict passed to text label rendering.
            overlay_settings: Dict passed to overlay composition.
            plotly_settings: Additional Plotly layout settings.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            ImportError: If plotly is not installed.
        """
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base_parents._accessor_dash_handler import (
            PLOTLY_AVAILABLE,
        )

        if not PLOTLY_AVAILABLE:
            raise ImportError(
                    "plotly is required for .dash(). "
                    "Install it with: pip install plotly"
            )

        arr = self[:] if not foreground_only else self.foreground()

        has_objects = self._root_image.num_objects > 0
        if overlay and has_objects:
            objmap = self._get_filtered_objmap(object_label)
            fig = self._plotly_overlay(
                    arr=arr, objmap=objmap, figsize=figsize, title=title,
                    overlay_settings=overlay_settings,
                    plotly_settings=plotly_settings,
            )
            self._decorate_plotly_overlay(
                    fig, has_objects=has_objects, object_label=object_label,
                    show_overlay_notice=(
                        show_overlay_notice and bool(objmap.any())
                    ),
                    show_labels=show_labels, show_grid=show_grid,
                    label_settings=label_settings,
            )
            return fig

        fig = self._plotly_imshow(arr=arr, figsize=figsize, title=title, cmap=cmap)
        if plotly_settings is not None:
            fig.update_layout(**plotly_settings)
        return fig
