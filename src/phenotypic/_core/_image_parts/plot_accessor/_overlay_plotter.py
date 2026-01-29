"""Overlay visualization plotter for PhenoTypic images."""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ._base_plotter import BasePlotter

if TYPE_CHECKING:
    from phenotypic import Image


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
        >>> fig, ax = image.plot.overlay()
        >>> plt.close(fig)  # Important: free memory

        GridImage with grid features:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> grid_image = load_synth_yeast_plate()  # Returns GridImage
        >>> fig, ax = grid_image.plot.overlay(
        ...     show_gridlines=True,
        ...     show_section_boxes=True,
        ...     show_labels=True
        ... )
        >>> plt.close(fig)
    """

    def overlay(
            self,
            object_label: int | None = None,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            show_labels: bool = False,
            ax: plt.Axes | None = None,
            *,
            show_gridlines: bool = True,
            show_section_boxes: bool = True,
            label_settings: dict | None = None,
            overlay_settings: dict | None = None,
            imshow_settings: dict | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Display object detection overlay with optional grid visualization.

        Creates a visualization showing detected objects overlaid on the image
        using color-coded labels. For GridImage instances, automatically adds
        grid lines and section bounding boxes when enabled.

        Args:
            object_label: Specific object label to highlight. If None,
                shows all detected objects. Useful for inspecting individual
                colonies in dense cultures.
            figsize: Figure size as (width, height) in inches. If None,
                uses default matplotlib sizing. For GridImage, (9, 10) is
                recommended to accommodate secondary axes.
            title: Plot title. If None, uses the image name from metadata.
            show_labels: If True, displays numeric labels at object centroids.
                Useful for identifying specific colonies for downstream analysis.
            ax: Existing matplotlib Axes to plot into. If None, creates a new
                figure and axes. Use this for subplot arrangements.
            show_gridlines: For GridImage only. If True, draws cyan dashed
                gridlines at row/column boundaries and adds secondary axes
                showing grid row/column numbers. Ignored for regular Image.
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
            imshow_settings: Dict passed to matplotlib ax.imshow() for additional
                image display customization.

        Returns:
            Tuple of (Figure, Axes) containing the overlay visualization.
            Caller should call plt.close(fig) after saving to free memory.

        Raises:
            EmptyImageError: If no image data is set.

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> fig, ax = image.plot.overlay(show_labels=True)
            >>> plt.close(fig)  # Free memory after use
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

        # Create base overlay using inherited method from ImageAccessorBase
        fig, ax = self._plot_overlay(
                arr=base_arr,
                objmap=objmap,
                ax=ax,
                figsize=figsize,
                title=title,
                mpl_settings=imshow_settings,
                overlay_settings=overlay_settings,
        )

        # Add object labels if requested
        if show_labels:
            ax = self._plot_obj_labels(
                    ax=ax,
                    color=label_settings.get("color", "white"),
                    size=label_settings.get("size", 12),
                    facecolor=label_settings.get("facecolor", "red"),
                    object_label=object_label,
            )

        # Grid-specific features (duck typing check)
        is_grid_image = hasattr(self._root_image, 'grid_finder')

        if is_grid_image and self._root_image.num_objects > 0:
            if show_gridlines:
                self._add_gridlines(ax)

            if show_section_boxes:
                self._add_section_boxes(ax)

        return fig, ax

    def _add_gridlines(self, ax: plt.Axes) -> None:
        """Add grid lines and secondary axes for GridImage.

        Draws cyan dashed gridlines at row/column boundaries and adds
        secondary axes on top and right showing grid row/column numbers.

        Args:
            ax: Matplotlib axes to draw gridlines on.
        """
        col_edges = self._root_image.grid.get_col_edges()
        row_edges = self._root_image.grid.get_row_edges()

        if len(col_edges) == 0 or len(row_edges) == 0:
            return

        # Secondary x-axis with column numbers
        secax_x = ax.secondary_xaxis("top")
        secax_x.set_xlabel("Grid Column Number")
        upper_col_edges = col_edges[1:]
        lower_col_edges = col_edges[:-1]
        col_centers = ((upper_col_edges - lower_col_edges) // 2) + lower_col_edges
        secax_x.set_xticks(col_centers)
        secax_x.set_xticklabels(np.arange(self._root_image.ncols))

        # Secondary y-axis with row numbers
        secax_y = ax.secondary_yaxis("right")
        secax_y.set_ylabel("Grid Row Number", rotation=270, labelpad=10)
        upper_row_edges = row_edges[1:]
        lower_row_edges = row_edges[:-1]
        row_centers = ((upper_row_edges - lower_row_edges) // 2) + lower_row_edges
        secax_y.set_yticks(row_centers)
        secax_y.set_yticklabels(np.arange(self._root_image.nrows))

        # Draw grid lines
        ax.vlines(
                x=col_edges,
                ymin=row_edges.min(),
                ymax=row_edges.max(),
                colors="c",
                linestyles="--",
        )
        ax.hlines(
                y=row_edges,
                xmin=col_edges.min(),
                xmax=col_edges.max(),
                color="c",
                linestyles="--",
        )

    def _add_section_boxes(self, ax: plt.Axes) -> None:
        """Add colored bounding boxes around grid sections.

        Draws colored rectangle patches around each grid section using
        the tab20 colormap. Useful for visualizing which objects belong
        to which wells in a plate layout.

        Args:
            ax: Matplotlib axes to draw section boxes on.
        """
        from phenotypic.measure import MeasureBounds
        from phenotypic.tools_.constants_ import BBOX

        cmap = plt.get_cmap("tab20")
        cmap_cycle = cycle(cmap(i) for i in range(cmap.N))

        img = self._root_image.copy()
        img.objmap = self._root_image.grid.get_section_map()
        gs_table = MeasureBounds().measure(img)

        for obj_label in gs_table.index.unique():
            subtable = gs_table.loc[obj_label, :]
            min_rr = subtable.loc[str(BBOX.MIN_RR)]
            max_rr = subtable.loc[str(BBOX.MAX_RR)]
            min_cc = subtable.loc[str(BBOX.MIN_CC)]
            max_cc = subtable.loc[str(BBOX.MAX_CC)]

            ax.add_patch(
                    Rectangle(
                            (min_cc, min_rr),
                            width=max_cc - min_cc,
                            height=max_rr - min_rr,
                            edgecolor=next(cmap_cycle),
                            facecolor="none",
                    ),
            )


__all__ = ["OverlayPlotter"]
