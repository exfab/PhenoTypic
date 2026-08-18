from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.patches import Rectangle

from phenotypic.schema import IMAGE
from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth

from ._accessor_io_handler import AccessorIOHandler

if TYPE_CHECKING:
    pass


class AccessorMplHandler(AccessorIOHandler):
    """Matplotlib plotting layer — histograms, overlays, grid decorations.

    Also hosts overlay-composition helpers (``_compose_overlay``,
    ``_fast_label_overlay``) that are shared with the Plotly/dash layer
    above in the MRO chain.
    """

    # Default overlay colors matching skimage.color.colorlabel.DEFAULT_COLORS
    _OVERLAY_COLORS: np.ndarray = np.array([
        [1.0, 0.0, 0.0],  # red
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 0.5, 0.0],  # green
        [0.294, 0.0, 0.510],  # indigo
        [1.0, 0.549, 0.0],  # darkorange
        [0.0, 1.0, 1.0],  # cyan
        [1.0, 0.753, 0.796],  # pink
        [0.604, 0.804, 0.196],  # yellowgreen
    ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Overlay composition (shared by mpl and plotly layers)
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_overlay(
        arr: np.ndarray,
        objmap: np.ndarray,
        overlay_settings: dict | None = None,
    ) -> np.ndarray:
        """Compose a label2rgb overlay array.

        Uses a fast direct blend instead of ``skimage.color.label2rgb`` to
        avoid expensive float64 conversions and unnecessary HSV round-trips.

        Args:
            arr: Base image array (2D grayscale or 3D RGB).
            objmap: Integer label array (same H x W as *arr*).
            overlay_settings: Dict with optional keys:

                * ``"alpha"`` (float) — overlay opacity, default 0.15.
                * ``"colors"`` (list) — list of RGB float triples.

                Any other keys are passed to ``skimage.color.label2rgb``
                as a fallback.

        Returns:
            A uint8 RGB overlay array.
        """
        overlay_settings = dict(overlay_settings) if overlay_settings else {}
        alpha = overlay_settings.pop("alpha", 0.15)
        colors = overlay_settings.pop("colors", None)

        # Fall back to skimage for non-standard options
        if overlay_settings:
            overlay_settings["alpha"] = alpha
            if colors is not None:
                overlay_settings["colors"] = colors
            return ski.color.label2rgb(
                label=objmap, image=arr, bg_label=0, **overlay_settings
            )

        return AccessorMplHandler._fast_label_overlay(
            arr, objmap, alpha=alpha, colors=colors,
        )

    @staticmethod
    def _fast_label_overlay(
        arr: np.ndarray,
        objmap: np.ndarray,
        alpha: float = 0.15,
        colors: list | np.ndarray | None = None,
    ) -> np.ndarray:
        """Blend colored labels onto an image without color-space conversions.

        Args:
            arr: Base image, 2D (H, W) or 3D (H, W, 3). uint8/uint16/float.
            objmap: Integer label map, shape (H, W). 0 = background.
            alpha: Label overlay opacity.
            colors: Nx3 array of RGB floats in [0, 1]. Defaults to the
                standard 10-color cycle.

        Returns:
            uint8 RGB array (H, W, 3).
        """
        if colors is None:
            color_lut = AccessorMplHandler._OVERLAY_COLORS
        else:
            color_lut = np.asarray(colors, dtype=np.float32)

        # Normalise image to float32 RGB in [0, 1]
        if arr.ndim == 2:
            img_f = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
        else:
            img_f = arr.astype(np.float32)

        if img_f.max() > 1.0:
            scale = 255.0 if arr.dtype == np.uint8 else 65535.0
            img_f *= 1.0 / scale

        # Build per-pixel color layer via LUT indexing
        n_colors = len(color_lut)
        fg_mask = objmap > 0
        if not np.any(fg_mask):
            return (img_f * 255).astype(np.uint8)

        # Map labels → color indices (1-based labels, mod into LUT)
        color_idx = (objmap[fg_mask] - 1) % n_colors
        color_layer = color_lut[color_idx]  # (N_fg, 3)

        # Blend only foreground pixels
        result = img_f.copy()
        result[fg_mask] = (
            color_layer * alpha + result[fg_mask] * (1.0 - alpha)
        )

        return (result * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def histogram(
        self,
        figsize: Tuple[int, int] = (10, 5),
        *,
        cmap="gray",
        linewidth=1,
        channel_names: list | None = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the histogram(s) of an image along with the image itself. The behavior depends on
        the dimensionality of the image array (2D or 3D). In the case of 2D, a single image and
        its histogram are produced. For 3D (multi-channel images), histograms for each channel
        are created alongside the image. This method supports customization such as figure size,
        colormap, line width of histograms, and labeling of channels.

        Args:
            figsize (Tuple[int, int]): The size of the figure to create. Default is (10, 5).
            cmap (str): Colormap used to render the image when the data is single channel. Default is 'gray'.
            linewidth (int): Line width of the plotted histograms. Default is 1.
            channel_names (list | None): Optional names for the channels in 3D data. These are
                used as titles for channel-specific histograms. If None, channels are instead
                labeled numerically.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The Matplotlib figure and axes objects representing the
            plotted image and its histograms.

        Raises:
            ValueError: If the dimensionality of the input image array is unsupported.

        Notes:
            This method uses `skimage.exposure.histogram <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram>`_
            for computing the histogram data.
        """
        arr = self._subject_arr
        dtype = arr.dtype

        if np.issubdtype(dtype, np.floating):
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_min < 0.0 or arr_max > 1.0:
                raise ValueError(
                    f"Float image arrays must be within [0.0, 1.0]. Found range [{arr_min}, {arr_max}]."
                )
            x_limits = (0.0, 1.0)
        elif np.issubdtype(dtype, np.bool_):
            x_limits = (0, 1)
        elif np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
            x_limits = (dtype_info.min, dtype_info.max)
        else:
            raise TypeError(f"Unsupported image dtype for histogram plotting: {dtype}")

        match self.ndim:
            case 2:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
                axes = axes.ravel()
                axes[0] = self._mpl_plot(
                    arr=self._subject_arr,
                    figsize=figsize,
                    title=self._root_image.name,
                    cmap=cmap,
                    ax=axes[0],
                )
                hist, histc = ski.exposure.histogram(
                    image=self._subject_arr[:],
                    nbins=2 ** self._root_image.metadata[IMAGE.BIT_DEPTH],
                )
                axes[1].plot(histc, hist, lw=linewidth)
                axes[1].set_xlim(x_limits)

            case 3:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

                for idx, ax in enumerate(axes.flat):
                    if idx == 0:
                        self._mpl_plot(
                            arr=self._subject_arr[:],
                            figsize=figsize,
                            title=self._root_image.name,
                            ax=ax,
                        )
                    else:
                        hist, histc = ski.exposure.histogram(
                            image=self._subject_arr[:, :, idx - 1],
                            nbins=2 ** self._root_image.metadata[IMAGE.BIT_DEPTH],
                        )
                        ax.plot(histc, hist, lw=linewidth)
                        ax.set_title(
                            f"Channel-{channel_names[idx - 1] if channel_names else idx}"
                        )
                        ax.set_xlim(x_limits)

            case _:
                raise ValueError(
                    f"Unsupported array dimension: {self._subject_arr.ndim}"
                )
        return fig, axes

    # ------------------------------------------------------------------
    # Core matplotlib rendering
    # ------------------------------------------------------------------

    def _mpl_plot(
        self,
        arr: np.ndarray,
        figsize: Tuple[int, int] | None = None,
        title: str | bool | None = None,
        cmap: str = "gray",
        ax: plt.Axes | None = None,
        mpl_settings: dict | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots an image array using Matplotlib.

        This method is designed to render an image array using the `matplotlib.pyplot` module. It provides
        flexible options for color mapping, figure size, title customization, and additional Matplotlib
        parameters, which enable detailed control over the plot appearance.

        Args:
            arr (np.ndarray): The image data to plot. Can be 2D or 3D array representing the image.
            figsize ((int, int), optional): A tuple specifying the figure size in inches. If None, the figure size
                is automatically calculated as integer dimensions in [6, 30] that best match the image aspect ratio.
            title (None | str, optional): Plot title. If None, defaults to the name of the parent image. Defaults to None.
            cmap (str, optional): The colormap to be applied when the array is 2D. Defaults to 'gray'.
            ax (None | plt.Axes, optional): Existing Matplotlib axes to plot into. If None, a new figure is created. Defaults to None.
            mpl_settings (dict | None, optional): Additional Matplotlib keyword arguments for customization. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the created or passed Matplotlib `Figure` and `Axes` objects.

        """
        fig, ax = (ax.get_figure(), ax) if ax else plt.subplots(figsize=figsize)

        mpl_settings = mpl_settings if mpl_settings else {}
        cmap = mpl_settings.pop("cmap", cmap)

        # matplotlib.imshow can only handle ranges 0-1 or 0-255
        # this adds handling for higher bit-depth images
        plot_arr = normalize_rgb_bitdepth(arr) if arr.ndim == 3 else arr
        if np.issubdtype(plot_arr.dtype, np.integer):
            vmax = np.iinfo(plot_arr.dtype).max
        elif np.issubdtype(plot_arr.dtype, np.floating):
            vmax = 1.0
        else:
            vmax = 1
        vmax = mpl_settings.pop("vmax", vmax)

        ax.imshow(
            plot_arr, cmap=cmap, **mpl_settings
        ) if plot_arr.ndim == 2 else ax.imshow(plot_arr, vmax=vmax, **mpl_settings)

        ax.grid(False)

        if title is True:
            ax.set_title(self._root_image.name)
        elif title:
            ax.set_title(title)

        return fig, ax

    # ------------------------------------------------------------------
    # Object label rendering (matplotlib)
    # ------------------------------------------------------------------

    def _plot_obj_labels(
        self,
        ax: plt.Axes,
        color: str,
        size: int,
        facecolor: str,
        object_label: None | int,
        **kwargs,
    ):
        """
        Adds labels to objects in an image plot. This method overlays numerical labels onto
        the visual representation of segmented objects (e.g., microbe colonies) on a solid
        media agar. These labels typically correspond to unique identifiers from the object's
        segmentation process, helping in visually associating each object with its properties.

        This functionality is particularly useful in microbiology image analysis where
        different colonies need to be identified and studied individually. Adjusting certain
        attributes impacts the clarity, visibility, and interpretability of labels, aiding
        in downstream qualitative and quantitative analyses.

        Args:
            ax (plt.Axes): The matplotlib Axes object to plot on. This canvas will display
                the overlaid labels and is intended to correspond to a plot of the segmented
                agar plate.
            color (str): The color of the label text. Altering this influences the contrast
                and visibility of the text against the image, which might be critical when
                distinguishing labels on different background or media types used.
            size (int): The font size of the label text. Larger values make the labels more
                prominent, useful for densely populated plates or distant views, whereas smaller
                values add discretion and are better for crowded colonies or finer details.
            facecolor (str): The background color of the label's text box. This can help to
                enhance text contrast, especially when visualizing colonies with similar colors
                as the text. An opaque background makes labels clearer when overlapping colonies.
            object_label (None | int): If `None`, all objects are labeled. Setting a specific
                integer labels only the corresponding object. Modifying this allows targeted
                labeling, which simplifies results for cases focusing on individual colonies
                with unique interest.
            **kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.text()`
                that control text rendering properties such as rotation, alignment, or weight,
                providing flexibility in presentation.
        """
        props = self._root_image.objects.props
        for i, label in enumerate(self._root_image.objects.labels):
            if object_label is None:
                text_rr, text_cc = props[i].centroid
                ax.text(
                    x=text_cc,
                    y=text_rr,
                    s=f"{label}",
                    color=color,
                    fontsize=size,
                    bbox=dict(
                        facecolor=facecolor,
                        edgecolor="none",
                        alpha=0.6,
                        boxstyle="round",
                    ),
                    **kwargs,
                )
            elif object_label == label:
                text_rr, text_cc = props[i].centroid
                ax.text(
                    x=text_cc,
                    y=text_rr,
                    s=f"{label}",
                    color=color,
                    fontsize=size,
                    bbox=dict(
                        facecolor=facecolor,
                        edgecolor="none",
                        alpha=0.6,
                        boxstyle="round",
                    ),
                    **kwargs,
                )
        return ax

    # ------------------------------------------------------------------
    # Overlay plotting (matplotlib)
    # ------------------------------------------------------------------

    def _plot_overlay(
        self,
        arr: np.ndarray,
        objmap: np.ndarray,
        figsize: tuple[int, int] | None = None,
        title: str | bool | None = None,
        *,
        ax: plt.Axes | None = None,
        overlay_settings: dict | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot an array with object map overlay using matplotlib.

        Args:
            arr: The primary array to be displayed as an image.
            objmap: An array containing labels for an object map to
                overlay on top of the image.
            figsize: Figure size as (width, height) in inches. If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, defaults to the parent
                image name.
            ax: Existing Matplotlib axes to plot into. If None, a new
                figure and axes are created.
            overlay_settings: Parameters passed to
                ``skimage.color.label2rgb`` for overlay customization.

        Returns:
            A ``(plt.Figure, plt.Axes)`` tuple.
        """
        overlay_arr = self._compose_overlay(arr, objmap, overlay_settings)
        return self._mpl_plot(
            arr=overlay_arr, figsize=figsize, title=title, ax=ax
        )

    # ------------------------------------------------------------------
    # Matplotlib overlay decoration
    # ------------------------------------------------------------------

    def _decorate_mpl_overlay(
        self,
        ax: plt.Axes,
        *,
        has_objects: bool,
        object_label: int | None = None,
        show_overlay_notice: bool = True,
        show_labels: bool = False,
        show_grid: bool = True,
        label_settings: dict | None = None,
    ) -> None:
        """Add labels, gridlines, and section boxes to a matplotlib overlay.

        Args:
            ax: Matplotlib axes to decorate.
            has_objects: Whether the image has detected objects.
            object_label: Specific object label being highlighted.
            show_overlay_notice: Whether to display the overlay notice.
            show_labels: Whether to add centroid labels.
            show_grid: Whether to add gridlines and section boxes
                (GridImage only).
            label_settings: Label rendering settings.
        """
        if label_settings is None:
            label_settings = {}
        if show_overlay_notice:
            self._add_mpl_overlay_notice(ax)
        if show_labels:
            self._plot_obj_labels(
                ax=ax,
                color=label_settings.get("color", "white"),
                size=label_settings.get("size", 12),
                facecolor=label_settings.get("facecolor", "red"),
                object_label=object_label,
            )
        if show_grid and hasattr(self._root_image, 'grid_finder'):
            self._add_gridlines(ax)
            if has_objects:
                self._add_section_boxes(ax)

    @staticmethod
    def _add_mpl_overlay_notice(ax: plt.Axes) -> None:
        """Add a compact notice that the displayed image includes an overlay."""
        ax.text(
            0.01,
            0.99,
            "Overlay",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            zorder=100,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#1a1a1a",
                "edgecolor": "none",
                "alpha": 1.0,
            },
        )

    # ------------------------------------------------------------------
    # Grid visualisation (matplotlib)
    # ------------------------------------------------------------------

    def _add_gridlines(self, ax: plt.Axes) -> None:
        """Add grid lines and secondary axes for GridImage.

        Draws cyan dashed gridlines at row/column boundaries and adds
        secondary axes on top and right showing grid row/column numbers.

        Args:
            ax: Matplotlib axes to draw gridlines on.
        """
        col_edges = self._root_image.grid.get_col_edges()  # type: ignore[attr-defined]
        row_edges = self._root_image.grid.get_row_edges()  # type: ignore[attr-defined]

        if len(col_edges) == 0 or len(row_edges) == 0:
            return

        # Secondary x-axis with column numbers
        secax_x = ax.secondary_xaxis("top")
        secax_x.set_xlabel("Grid Column Number")
        upper_col_edges = col_edges[1:]
        lower_col_edges = col_edges[:-1]
        col_centers = ((upper_col_edges - lower_col_edges) // 2) + lower_col_edges
        secax_x.set_xticks(col_centers)
        secax_x.set_xticklabels(np.arange(self._root_image.ncols))  # type: ignore[attr-defined]

        # Secondary y-axis with row numbers
        secax_y = ax.secondary_yaxis("right")
        secax_y.set_ylabel("Grid Row Number", rotation=270, labelpad=10)
        upper_row_edges = row_edges[1:]
        lower_row_edges = row_edges[:-1]
        row_centers = ((upper_row_edges - lower_row_edges) // 2) + lower_row_edges
        secax_y.set_yticks(row_centers)
        secax_y.set_yticklabels(np.arange(self._root_image.nrows))  # type: ignore[attr-defined]

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
        """Add colored bounding boxes hugging objects in each grid section.

        Draws colored rectangle patches around the union bounding box of
        the objects assigned to each grid section, using the tab20
        colormap. Empty sections (no detected objects) are skipped.
        Color-to-section mapping is stable across redraws because colors
        are indexed by the flattened section slot.

        Args:
            ax: Matplotlib axes to draw section boxes on.
        """
        min_rr, max_rr, min_cc, max_cc = (
            self._root_image.grid._get_section_object_bounds_arrays()  # type: ignore[attr-defined]
        )

        if min_rr.size == 0:
            return

        cmap = plt.get_cmap("tab20")
        palette = [cmap(i) for i in range(cmap.N)]

        for i in range(len(min_rr)):
            if np.isnan(min_rr[i]):
                continue
            ax.add_patch(
                Rectangle(
                    (float(min_cc[i]), float(min_rr[i])),
                    width=float(max_cc[i] - min_cc[i]),
                    height=float(max_rr[i] - min_rr[i]),
                    edgecolor=palette[i % len(palette)],
                    facecolor="none",
                ),
            )
