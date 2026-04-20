"""Plotly/Dash visualization layer for image accessors.

Provides interactive Plotly-based plotting and absorbs the helper
functions formerly in ``phenotypic.tools_._plotly_helpers``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._accessor_mpl_handler import AccessorMplHandler

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Mapping from matplotlib colormap names to plotly colorscale names
_MPL_TO_PLOTLY = {
    "gray": "gray",
    "viridis": "Viridis",
    "inferno": "Inferno",
    "plasma": "Plasma",
    "magma": "Magma",
    "hot": "Hot",
    "jet": "Jet",
    "RdBu": "RdBu",
    "coolwarm": "RdBu",
}

# Baseline container width (px) used to compute a figure height that matches
# the image aspect ratio when figsize is None. Chosen to approximate a
# typical Jupyter cell; actual width still autosizes to the container.
_AUTOSIZE_TARGET_WIDTH_PX = 800


class AccessorDashHandler(AccessorMplHandler):
    """Plotly/dash visualization layer with folded-in helper functions.

    Inherits overlay composition from :class:`AccessorMplHandler` and adds
    Plotly-specific rendering, decoration, and utility static methods.
    """

    PLOTLY_CONFIG: dict = {"scrollZoom": True}

    # ------------------------------------------------------------------
    # Plotly helper static methods (formerly in tools_/_plotly_helpers.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _require_plotly() -> None:
        """Raise ImportError if plotly is not installed."""
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly is required for interactive visualization. "
                "Install it with: pip install plotly>=6.0.0  "
                "or install the gui extras: pip install phenotypic[gui]"
            )

    @staticmethod
    def mpl_cmap_to_plotly(cmap_name: str) -> str | list:
        """Convert a matplotlib colormap name to a plotly colorscale.

        Args:
            cmap_name: Name of a matplotlib colormap.

        Returns:
            A plotly colorscale string for known names, or a list of
            ``[position, "rgb(r,g,b)"]`` pairs sampled from the matplotlib
            colormap for unknown names.
        """
        AccessorDashHandler._require_plotly()
        if cmap_name in _MPL_TO_PLOTLY:
            return _MPL_TO_PLOTLY[cmap_name]

        import matplotlib
        cmap = matplotlib.colormaps[cmap_name]
        n = 256
        scale = []
        for i in range(n):
            rgba = cmap(i / (n - 1))
            r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            scale.append([i / (n - 1), f"rgb({r},{g},{b})"])
        return scale

    @staticmethod
    def plotly_imshow(
        arr: np.ndarray,
        title: str | None = None,
        cmap: str = "gray",
        figsize: tuple[int, int] | None = None,
    ) -> go.Figure:
        """Render an image array as an interactive Plotly figure.

        Args:
            arr: 2D (grayscale) or 3D (RGB) image array. Supports uint8,
                uint16, and float dtypes.
            title: Figure title. Defaults to None.
            cmap: Matplotlib colormap name for 2D arrays. Ignored for 3D.
                Defaults to ``"gray"``.
            figsize: Figure size as ``(width_inches, height_inches)``.
                Converted to pixels at 100 dpi. If None, the figure
                autosizes its width to the container and sets its height
                from the array's aspect ratio (using
                ``_AUTOSIZE_TARGET_WIDTH_PX`` as the baseline) so the
                image fills the width without left/right whitespace.

        Returns:
            A ``plotly.graph_objects.Figure`` with zoom-friendly defaults.
        """
        AccessorDashHandler._require_plotly()

        if arr.ndim == 3:
            fig = px.imshow(arr, binary_string=True)
        else:
            colorscale = AccessorDashHandler.mpl_cmap_to_plotly(cmap)
            fig = px.imshow(
                arr, color_continuous_scale=colorscale, binary_string=True,
            )

        if figsize is not None:
            width_px = figsize[0] * 100
            height_px = figsize[1] * 100
            size_kwargs: dict = {"width": width_px, "height": height_px}
        else:
            h, w = arr.shape[:2]
            aspect_height = int(_AUTOSIZE_TARGET_WIDTH_PX * h / w)
            size_kwargs = {"autosize": True, "height": aspect_height}

        fig.update_layout(
            **size_kwargs,
            dragmode="zoom",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(
            showticklabels=False, showgrid=False,
            scaleanchor="x", constrain="domain", constraintoward="top",
        ),
            title=title,
        )

        return fig

    @staticmethod
    def add_plotly_gridlines(
        fig: go.Figure,
        col_edges: np.ndarray,
        row_edges: np.ndarray,
        ncols: int,
        nrows: int,
    ) -> None:
        """Add cyan dashed gridlines and row/column axis tick labels to a Plotly figure.

        Draws dashed cyan line shapes at each row/column edge and renders
        the row/column indices as axis tick labels (columns on the top
        axis, rows on the right axis). Batches updates into a single
        ``update_layout`` call.

        Args:
            fig: Plotly figure to modify in-place.
            col_edges: Column edge pixel positions.
            row_edges: Row edge pixel positions.
            ncols: Number of grid columns.
            nrows: Number of grid rows.
        """
        AccessorDashHandler._require_plotly()
        if len(col_edges) == 0 or len(row_edges) == 0:
            return

        y_min = float(row_edges.min())
        y_max = float(row_edges.max())
        x_min = float(col_edges.min())
        x_max = float(col_edges.max())

        line_style = dict(color="cyan", dash="dash", width=1)
        shapes = list(fig.layout.shapes or [])

        for x in col_edges:
            shapes.append(dict(
                type="line", x0=float(x), x1=float(x),
                y0=y_min, y1=y_max, line=line_style,
            ))
        for y in row_edges:
            shapes.append(dict(
                type="line", x0=x_min, x1=x_max,
                y0=float(y), y1=float(y), line=line_style,
            ))

        tickfont = dict(color="cyan", size=10)
        layout_kwargs: dict = {"shapes": shapes}

        if len(col_edges) > 1:
            col_centers = (col_edges[:-1] + col_edges[1:]) / 2
            n_col_labels = min(ncols, len(col_centers))
            layout_kwargs["xaxis"] = dict(
                tickmode="array",
                tickvals=[float(v) for v in col_centers[:n_col_labels]],
                ticktext=[str(i) for i in range(n_col_labels)],
                showticklabels=True,
                showgrid=False,
                side="top",
                automargin=True,
                tickfont=tickfont,
            )

        if len(row_edges) > 1:
            row_centers = (row_edges[:-1] + row_edges[1:]) / 2
            n_row_labels = min(nrows, len(row_centers))
            layout_kwargs["yaxis"] = dict(
                tickmode="array",
                tickvals=[float(v) for v in row_centers[:n_row_labels]],
                ticktext=[str(i) for i in range(n_row_labels)],
                showticklabels=True,
                showgrid=False,
                side="right",
                automargin=True,
                tickfont=tickfont,
                scaleanchor="x",
                constrain="domain",
                constraintoward="top",
            )

        fig.update_layout(**layout_kwargs)

    @staticmethod
    def add_plotly_section_boxes(
        fig: go.Figure,
        min_rr: np.ndarray,
        max_rr: np.ndarray,
        min_cc: np.ndarray,
        max_cc: np.ndarray,
    ) -> None:
        """Add colored bounding-box rectangles around objects in each grid section.

        Each rectangle hugs the union bounding box of all objects assigned
        to that grid section. Sections with no objects (NaN entries) are
        skipped. Colors cycle through tab20 indexed by section slot so the
        color-to-section mapping is stable across redraws.

        Args:
            fig: Plotly figure to modify in-place.
            min_rr: Per-section minimum row (pixel y) coordinate. Length
                ``nrows*ncols``. NaN marks an empty section.
            max_rr: Per-section maximum row coordinate. Same length/NaN
                convention as ``min_rr``.
            min_cc: Per-section minimum column (pixel x) coordinate.
            max_cc: Per-section maximum column coordinate.
        """
        AccessorDashHandler._require_plotly()

        if min_rr.size == 0:
            return

        import matplotlib
        tab20 = matplotlib.colormaps["tab20"]
        palette = [tab20(i) for i in range(tab20.N)]

        shapes = list(fig.layout.shapes or [])
        for i in range(len(min_rr)):
            if np.isnan(min_rr[i]):
                continue
            rgba = palette[i % len(palette)]
            color_str = (
                f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},"
                f"{int(rgba[2]*255)},{rgba[3]:.2f})"
            )
            shapes.append(dict(
                type="rect",
                x0=float(min_cc[i]), x1=float(max_cc[i]),
                y0=float(min_rr[i]), y1=float(max_rr[i]),
                line=dict(color=color_str, width=2),
            ))

        fig.update_layout(shapes=shapes)

    @staticmethod
    def add_plotly_obj_labels(
        fig: go.Figure,
        root_image,
        object_label: int | None = None,
        color: str = "white",
        size: int = 10,
        bgcolor: str = "black",
    ) -> None:
        """Add centroid labels for detected objects to a Plotly figure.

        Args:
            fig: Plotly figure to modify in-place.
            root_image: Image instance with ``objects`` accessor providing
                ``props`` and ``labels``.
            object_label: If set, only label this specific object.
                If None, label all objects.
            color: Label text color.
            size: Label font size.
            bgcolor: Label background color.
        """
        AccessorDashHandler._require_plotly()
        annotations = []
        for prop in root_image.objects.props:
            if object_label is not None and object_label != prop.label:
                continue
            rr, cc = prop.centroid
            annotations.append(dict(
                x=cc,
                y=rr,
                text=str(prop.label),
                showarrow=False,
                font=dict(color=color, size=size),
                bgcolor=bgcolor,
                opacity=0.6,
            ))
        existing = list(fig.layout.annotations or [])
        fig.update_layout(annotations=existing + annotations)

    # ------------------------------------------------------------------
    # Plotly overlay rendering
    # ------------------------------------------------------------------

    def _plotly_overlay(
        self,
        arr: np.ndarray,
        objmap: np.ndarray,
        figsize: tuple[int, int] | None = None,
        title: str | bool | None = None,
        *,
        overlay_settings: dict | None = None,
        plotly_settings: dict | None = None,
    ) -> go.Figure:
        """Plot an array with object map overlay using Plotly.

        Args:
            arr: The primary array to be displayed as an image.
            objmap: An array containing labels for an object map to
                overlay on top of the image.
            figsize: Figure size as (width, height) in inches. If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, defaults to the parent
                image name.
            overlay_settings: Parameters passed to
                ``skimage.color.label2rgb`` for overlay customization.
            plotly_settings: Additional Plotly layout settings.

        Returns:
            A ``plotly.graph_objects.Figure``.
        """
        overlay_arr = self._compose_overlay(arr, objmap, overlay_settings)
        fig = self.plotly_imshow(arr=overlay_arr, figsize=figsize, title=title)
        if plotly_settings is not None:
            fig.update_layout(**plotly_settings)

        return fig

    # ------------------------------------------------------------------
    # Plotly overlay decoration
    # ------------------------------------------------------------------

    def _decorate_plotly_overlay(
        self,
        fig: go.Figure,
        *,
        has_objects: bool,
        object_label: int | None = None,
        show_labels: bool = False,
        show_grid: bool = True,
        label_settings: dict | None = None,
    ) -> None:
        """Add labels, gridlines, and section boxes to a Plotly overlay.

        Args:
            fig: Plotly figure to decorate.
            has_objects: Whether the image has detected objects.
            object_label: Specific object label being highlighted.
            show_labels: Whether to add centroid labels.
            show_grid: Whether to add gridlines and section boxes
                (GridImage only).
            label_settings: Label rendering settings.
        """
        if label_settings is None:
            label_settings = {}
        if show_labels:
            self.add_plotly_obj_labels(
                fig=fig,
                root_image=self._root_image,
                object_label=object_label,
                color=label_settings.get("color", "white"),
                size=label_settings.get("size", 12),
                bgcolor=label_settings.get("facecolor", "red"),
            )
        if show_grid and hasattr(self._root_image, 'grid_finder'):
            col_edges = self._root_image.grid.get_col_edges()
            row_edges = self._root_image.grid.get_row_edges()
            self.add_plotly_gridlines(
                fig=fig, col_edges=col_edges, row_edges=row_edges,
                ncols=self._root_image.ncols, nrows=self._root_image.nrows,
            )
            if has_objects:
                min_rr, max_rr, min_cc, max_cc = (
                    self._root_image.grid._get_section_object_bounds_arrays()
                )
                self.add_plotly_section_boxes(
                    fig=fig, min_rr=min_rr, max_rr=max_rr,
                    min_cc=min_cc, max_cc=max_cc,
                )
