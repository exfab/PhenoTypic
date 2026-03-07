"""Overlay visualization plotter for PhenoTypic images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import skimage as ski

from phenotypic.tools_._plotly_helpers import (
    add_plotly_gridlines,
    add_plotly_obj_labels,
    add_plotly_section_boxes,
    plotly_imshow,
)
from phenotypic.tools_.register import register_plotter

from ._base_plotter import BasePlotter

if TYPE_CHECKING:
    pass


@register_plotter
class OverlayPlotter(BasePlotter):
    """Provides overlay visualization methods for colony detection results.

    This class offers methods to visualize object detection overlays on images,
    with automatic support for grid-specific features when the root image is
    a GridImage instance. Grid features are automatically enabled when the
    image has a `grid_finder` attribute.

    The `overlay()` method creates color-coded visualizations of detected
    objects overlaid on the original image, supporting optional gridlines,
    section bounding boxes, and object labels.

    Examples:
        Basic overlay visualization:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> fig = image.plot.overlay()

        GridImage with grid features:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> grid_image = load_synth_yeast_plate()  # Returns GridImage
        >>> fig = grid_image.plot.overlay(
        ...     show_gridlines=True,
        ...     show_section_boxes=True,
        ...     show_labels=True
        ... )
    """

    call_name = "overlay"

    def overlay(
            self,
            object_label: int | None = None,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            show_labels: bool = False,
            *,
            show_gridlines: bool = True,
            show_section_boxes: bool = True,
            label_settings: dict | None = None,
            overlay_settings: dict | None = None,
            plotly_settings: dict | None = None,
    ) -> go.Figure:
        """Display object detection overlay with optional grid visualization.

        Creates a visualization showing detected objects overlaid on the image
        using color-coded labels. For GridImage instances, automatically adds
        grid lines and section bounding boxes when enabled.

        Args:
            object_label: Specific object label to highlight. If None,
                shows all detected objects. Useful for inspecting individual
                colonies in dense cultures.
            figsize: Figure size as (width, height) in inches. If None,
                auto-calculated from the image aspect ratio.
            title: Plot title. If None, uses the image name from metadata.
            show_labels: If True, displays numeric labels at object centroids.
                Useful for identifying specific colonies for downstream analysis.
            show_gridlines: For GridImage only. If True, draws cyan dashed
                gridlines at row/column boundaries with row/column labels.
                Ignored for regular Image.
            show_section_boxes: For GridImage only. If True, draws colored
                bounding boxes around each grid section using tab20 colormap.
                Useful for visualizing which objects belong to which wells.
                Ignored for regular Image.
            label_settings: Dict passed to text label rendering. Supported keys:
                - color (str): Text color. Default: 'white'
                - size (int): Font size. Default: 12
                - facecolor (str): Label background color. Default: 'red'
            overlay_settings: Dict passed to skimage.color.label2rgb. Common key:
                - alpha (float): Overlay transparency (0-1). Default: 0.15
            plotly_settings: Dict passed to plotly figure layout for additional
                display customization.

        Returns:
            A ``plotly.graph_objects.Figure`` containing the overlay visualization.

        Raises:
            EmptyImageError: If no image data is set.

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> fig = image.plot.overlay(show_labels=True)
        """
        # Determine which array to use (RGB preferred, fallback to gray)
        if not self._root_image.rgb.isempty():
            base_arr = self._root_image.rgb[:]
        else:
            base_arr = self._root_image.gray[:]

        # Get object map, optionally filtering to single object
        objmap = self._root_image.objmap[:]
        if object_label is not None:
            objmap = objmap.copy()
            objmap[objmap != object_label] = 0

        # Initialize settings
        if label_settings is None:
            label_settings = {}
        overlay_settings = dict(overlay_settings) if overlay_settings else {}

        # Create overlay array via label2rgb
        overlay_alpha = overlay_settings.pop("alpha", 0.15)
        overlay_arr = ski.color.label2rgb(
                label=objmap, image=base_arr, bg_label=0,
                alpha=overlay_alpha, **overlay_settings,
        )

        # Build plotly figure
        fig = plotly_imshow(arr=overlay_arr, figsize=figsize, title=title)

        if plotly_settings is not None:
            fig.update_layout(**plotly_settings)

        # Add object labels if requested
        if show_labels:
            add_plotly_obj_labels(
                    fig, self._root_image, object_label=object_label,
                    color=label_settings.get("color", "white"),
                    size=label_settings.get("size", 12),
                    bgcolor=label_settings.get("facecolor", "red"),
            )

        # Grid-specific features (duck typing check)
        is_grid_image = hasattr(self._root_image, 'grid_finder')

        if is_grid_image:
            if show_gridlines:
                col_edges = self._root_image.grid.get_col_edges()
                row_edges = self._root_image.grid.get_row_edges()
                add_plotly_gridlines(
                        fig, col_edges, row_edges,
                        self._root_image.ncols, self._root_image.nrows,
                )
            if show_section_boxes and self._root_image.num_objects > 0:
                add_plotly_section_boxes(fig, self._root_image)

        return fig


__all__ = ["OverlayPlotter"]
