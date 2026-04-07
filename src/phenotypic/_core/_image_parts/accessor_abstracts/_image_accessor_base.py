from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from abc import ABC, abstractmethod
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as ski
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image

import phenotypic
from phenotypic.tools_.constants_ import IO, METADATA
from phenotypic.tools_.funcs_ import normalize_rgb_bitdepth

if TYPE_CHECKING:
    import napari
    import plotly.graph_objects as go
    from phenotypic._core._image import Image

import importlib.util

_HAS_NAPARI = importlib.util.find_spec("napari") is not None

# Global napari viewer instance for persistent Jupyter notebook workflows
_global_napari_viewer: napari.Viewer | None = None


def _viewer_is_alive(viewer: napari.Viewer | None) -> bool:
    """Return True if *viewer* is open and its Qt window still exists."""
    if viewer is None:
        return False
    try:
        window = getattr(viewer, "window", None)
        if window is None:
            return False
        # Access the underlying Qt widget to verify it hasn't been deleted.
        # After viewer.close(), the Qt C++ object may be garbage-collected,
        # causing RuntimeError or AttributeError on attribute access.
        qt_window = window._qt_window
        return qt_window is not None and qt_window.isVisible()
    except (RuntimeError, AttributeError):
        # Qt C++ object has been deleted (user closed window)
        return False


class ImageAccessorBase(ABC):
    """
    Provides an abstract base class for image accessor operations.

    The `ImageAccessorBase` class serves as a foundational abstract base class
    to standardize the handling and manipulation of image data. It provides
    attributes and methods for image loading, property enforcement, and shape
    management for image data, supporting both access and processing functionalities.

    This class is particularly useful for processing and analyzing images of
    microbe colonies grown on solid media agar. By ensuring consistent handling
    of image formats, metadata validation, and structured array management, this
    class helps streamline image analysis workflows, maintain metadata integrity,
    and enable reproducible results.

    Attributes:
        _root_image (Image): The root image object from which operations derive.
            Modifying this can change the basis of calculations or operations
            performed within accessor methods. For example, a grayscale `_root_image`
            might yield entirely different results when compared to an RGB image
            for colony segmentation or measurement.
    """

    @property
    @abstractmethod
    def _accessor_property_name(self) -> str:
        """Name of the Image property that surfaces this accessor."""
        raise NotImplementedError

    @classmethod
    def _accessor_property_name_value(cls) -> str:
        """Retrieve accessor property name from the subclass' property without instantiation."""
        return cls._accessor_property_name.fget(object.__new__(cls))  # type: ignore[attr-defined]

    @property
    @abstractmethod
    def _subject_arr(self) -> np.ndarray:
        raise NotImplementedError

    def __init__(self, root_image: Image):
        self._root_image = root_image

    @classmethod
    def load(cls, filepath: str | Path) -> np.ndarray:
        """Load an image array from file and verify it was saved from this accessor type.

        Checks if the image contains PhenoTypic metadata indicating it was saved
        from the same accessor type (e.g., Image.gray, Image.rgb). If metadata
        doesn't match or is missing, a warning is raised but the array is still loaded.

        Args:
            filepath: Path to the image file to load.

        Returns:
            np.ndarray: The loaded image array.

        Warns:
            UserWarning: If metadata is missing or indicates the image was saved
                from a different accessor type.

        Examples:
            Load a grayscale image from file:

            >>> from phenotypic import Image
            >>> image = Image(arr)
            >>> # load an object map you saved or hand-graded
            >>> image.objmap.load("path/to/map.png")
        """
        filepath = Path(filepath)
        expected_property = f"Image.{cls._accessor_property_name_value()}"

        # Load the array using cv2 for reliable uint16 round-trip
        import cv2

        arr = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(
                f"Could not read image file: {filepath}. "
                "File may not exist, be corrupt, or be in an "
                "unsupported format."
            )
        # cv2 loads colour images as BGR/BGRA; convert to RGB/RGBA
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        # Try to extract and verify PhenoTypic metadata
        phenotypic_data = cls._extract_phenotypic_metadata(filepath)

        if phenotypic_data is None:
            warnings.warn(
                f"No PhenoTypic metadata found in '{filepath.name}'. "
                f"Cannot verify this image was saved from {expected_property}. "
                "Loading anyway, but this may lead to undefined behavior.",
                UserWarning,
            )
        else:
            saved_property = phenotypic_data.get("phenotypic_image_property", "unknown")
            if saved_property != expected_property:
                warnings.warn(
                    f"Metadata mismatch: Image was saved from '{saved_property}' "
                    f"but being loaded as '{expected_property}'. "
                    "This may lead to undefined behavior.",
                    UserWarning,
                )

        return arr

    @classmethod
    def _extract_phenotypic_metadata(cls, filepath: Path) -> dict | None:
        """Extract PhenoTypic metadata from an image file.

        Args:
            filepath: Path to the image file.

        Returns:
            dict or None: The PhenoTypic metadata dict if found, None otherwise.
        """
        suffix = filepath.suffix.lower()

        try:
            if suffix in IO.PNG_FILE_EXTENSIONS:
                with PIL_Image.open(filepath) as img:
                    phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
                    if phenotypic_json:
                        return json.loads(phenotypic_json)

            elif suffix in IO.JPEG_FILE_EXTENSIONS:
                # Try exiftool for JPEG UserComment
                if shutil.which("exiftool"):
                    result = subprocess.run(
                        ["exiftool", "-json", "-UserComment", str(filepath)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        exif_data = json.loads(result.stdout)
                        user_comment = (
                            exif_data[0].get("UserComment") if exif_data else None
                        )
                        if user_comment:
                            data = json.loads(user_comment)
                            if "phenotypic_version" in data:
                                return data

            elif suffix in IO.TIFF_EXTENSIONS:
                with PIL_Image.open(filepath) as img:
                    desc = img.tag_v2.get(270) if hasattr(img, "tag_v2") else None
                    if desc:
                        try:
                            data = json.loads(desc)
                            if "phenotypic_version" in data:
                                return data
                        except json.JSONDecodeError:
                            pass

        except Exception:
            pass

        return None

    @property
    def _subject_arr(self) -> np.ndarray:
        """
        Abstract property representing an image array. The image array is expected to be a NumPy ndarray
        with a specific shape of (r, c, ...), which can be used for various operations that require a structured
        multi-dimensional array.

        This property is abc_ and must be implemented in any derived concrete class. The implementation
        should conform to the type signature and shape expectations as defined.

        Note: Read-only property. Changes should reference the specific array

        Returns:
            np.ndarray: A NumPy ndarray object with shape (r, c, ...).
        """
        raise NotImplementedError(
            "This property is abc_ and must be implemented in a derived class."
        )

    def __array__(self, dtype=None, copy=None):
        """Implements the array interface for numpy compatibility.

        This allows numpy functions to operate directly on accessor objects.
        For example: np.sum(accessor), np.mean(accessor), etc.

        Args:
            dtype: Optional dtype to cast the array to
            copy: Optional copy parameter for NumPy 2.0+ compatibility

        Returns:
            np.ndarray: The underlying array data
        """
        arr = self._subject_arr
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        elif copy:
            arr = arr.copy()
        return arr

    def __len__(self) -> int:
        """
        Returns the length of the subject array.

        This method calculates and returns the total number of elements contained in the
        underlying array.

        Returns:
            int: The number of elements in the underlying array attribute.
        """
        return len(self._subject_arr)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the current image data.

        This method retrieves the dimensions of the array stored in the `_main_arr`
        attribute as a tuple, which indicates its size along each axis.

        Returns:
            Tuple[int, ...]: A tuple representing the dimensions of the `_main_arr`
            attribute.
        """
        return self._subject_arr.shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the underlying array.

        The `ndim` property provides access to the dimensionality of the array
        being encapsulated in the object. This value corresponds to the number
        of axes or dimensions the underlying array possesses. It can be useful
        for understanding the structure of the contained data.

        Returns:
            int: The number of dimensions of the underlying array.
        """
        return self._subject_arr.ndim

    @property
    def size(self) -> int:
        """
        Gets the size of the subject array.

        This property retrieves the total number of elements in the subject
        array. It is read-only.

        Returns:
            int: The total number of elements in the subject array.
        """
        return self._subject_arr.size

    def val_range(self) -> pd.Interval:
        """
        Return the closed interval [min, max] of the subject array values.

        Returns:
            pd.Interval: A single closed interval including both endpoints.
        """
        mn = self._subject_arr.min()
        mx = self._subject_arr.max()
        return pd.Interval(left=mn, right=mx, closed="both")

    @property
    def dtype(self):
        return self._subject_arr.dtype

    def isempty(self):
        return True if self.shape[0] == 0 else False

    def copy(self) -> np.ndarray:
        return self._subject_arr.copy()

    def foreground(self):
        """
        Extracts and returns the foreground of the image by masking out the background.

        This method generates a foreground image by applying the object mask
        stored in the Image to the current array representation.
        Pixels outside the object mask are set to zero in the resulting foreground
        image. This is useful in image processing tasks to isolate the region
        of interest in the image, such as microbe colonies on an agar plate.

        Returns:
            numpy.ndarray: A numpy array containing only the foreground portion
            of the image, with all non-foreground pixels set to zero.
        """
        foreground = self._subject_arr.copy()
        foreground[self._root_image.objmask[:] == 0] = 0
        return foreground

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
                    nbins=2 ** self._root_image.metadata[METADATA.BIT_DEPTH],
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
                            nbins=2 ** self._root_image.metadata[METADATA.BIT_DEPTH],
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

        # arr_shape = arr.shape
        # if arr_shape[0] > 500:
        #     ax.yaxis.set_minor_locator(MultipleLocator(100))
        #
        # if arr_shape[1] > 500:
        #     ax.xaxis.set_minor_locator(MultipleLocator(100))

        if title is True:
            ax.set_title(self._root_image.name)
        elif title:
            ax.set_title(title)

        return fig, ax

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

    def _get_filtered_objmap(self, object_label: int | None = None) -> np.ndarray:
        """Fetch the object map, optionally filtering to a single label.

        Args:
            object_label: If provided, zero out all labels except this one.

        Returns:
            A dense object map array (always a fresh copy from sparse).
        """
        objmap = self._root_image.objmap[:]
        if object_label is not None:
            objmap[objmap != object_label] = 0
        return objmap

    @staticmethod
    def _compose_overlay(
        arr: np.ndarray,
        objmap: np.ndarray,
        overlay_settings: dict | None = None,
    ) -> np.ndarray:
        """Compose a label2rgb overlay array.

        Args:
            arr: Base image array.
            objmap: Object map to overlay.
            overlay_settings: Parameters passed to
                ``skimage.color.label2rgb``.

        Returns:
            The blended overlay array.
        """
        overlay_settings = dict(overlay_settings) if overlay_settings else {}
        overlay_alpha = overlay_settings.pop("alpha", 0.15)
        return ski.color.label2rgb(
            label=objmap, image=arr, bg_label=0, alpha=overlay_alpha, **overlay_settings
        )

    def _plot_overlay(
        self,
        arr: np.ndarray,
        objmap: np.ndarray,
        figsize: tuple[int, int] | None = None,
        title: str | bool | None = None,
        *,
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
            overlay_settings: Parameters passed to
                ``skimage.color.label2rgb`` for overlay customization.

        Returns:
            A ``(plt.Figure, plt.Axes)`` tuple.
        """
        overlay_arr = self._compose_overlay(arr, objmap, overlay_settings)
        return self._mpl_plot(arr=overlay_arr, figsize=figsize, title=title)

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
        from phenotypic.tools_._plotly_helpers import plotly_imshow

        overlay_arr = self._compose_overlay(arr, objmap, overlay_settings)
        fig = plotly_imshow(arr=overlay_arr, figsize=figsize, title=title)
        if plotly_settings is not None:
            fig.update_layout(**plotly_settings)

        return fig

    def _decorate_mpl_overlay(
        self,
        ax: plt.Axes,
        *,
        has_objects: bool,
        object_label: int | None = None,
        show_labels: bool = False,
        show_gridlines: bool = True,
        show_section_boxes: bool = True,
        label_settings: dict | None = None,
    ) -> None:
        """Add labels, gridlines, and section boxes to a matplotlib overlay.

        Args:
            ax: Matplotlib axes to decorate.
            has_objects: Whether the image has detected objects.
            object_label: Specific object label being highlighted.
            show_labels: Whether to add centroid labels.
            show_gridlines: Whether to add gridlines (GridImage only).
            show_section_boxes: Whether to add section boxes (GridImage only).
            label_settings: Label rendering settings.
        """
        if label_settings is None:
            label_settings = {}
        if show_labels:
            self._plot_obj_labels(
                ax=ax,
                color=label_settings.get("color", "white"),
                size=label_settings.get("size", 12),
                facecolor=label_settings.get("facecolor", "red"),
                object_label=object_label,
            )
        if hasattr(self._root_image, 'grid_finder'):
            if show_gridlines:
                self._add_gridlines(ax)
            if show_section_boxes and has_objects:
                self._add_section_boxes(ax)

    def _decorate_plotly_overlay(
        self,
        fig: go.Figure,
        *,
        has_objects: bool,
        object_label: int | None = None,
        show_labels: bool = False,
        show_gridlines: bool = True,
        show_section_boxes: bool = True,
        label_settings: dict | None = None,
    ) -> None:
        """Add labels, gridlines, and section boxes to a Plotly overlay.

        Args:
            fig: Plotly figure to decorate.
            has_objects: Whether the image has detected objects.
            object_label: Specific object label being highlighted.
            show_labels: Whether to add centroid labels.
            show_gridlines: Whether to add gridlines (GridImage only).
            show_section_boxes: Whether to add section boxes (GridImage only).
            label_settings: Label rendering settings.
        """
        if label_settings is None:
            label_settings = {}
        if show_labels:
            from phenotypic.tools_._plotly_helpers import add_plotly_obj_labels
            add_plotly_obj_labels(
                fig=fig,
                root_image=self._root_image,
                object_label=object_label,
                color=label_settings.get("color", "white"),
                size=label_settings.get("size", 12),
                bgcolor=label_settings.get("facecolor", "red"),
            )
        if hasattr(self._root_image, 'grid_finder'):
            if show_gridlines:
                from phenotypic.tools_._plotly_helpers import add_plotly_gridlines
                col_edges = self._root_image.grid.get_col_edges()
                row_edges = self._root_image.grid.get_row_edges()
                add_plotly_gridlines(
                    fig=fig, col_edges=col_edges, row_edges=row_edges,
                    ncols=self._root_image.ncols, nrows=self._root_image.nrows,
                )
            if show_section_boxes and has_objects:
                from phenotypic.tools_._plotly_helpers import add_plotly_section_boxes
                add_plotly_section_boxes(fig=fig, root_image=self._root_image)

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
        """Add colored bounding boxes around grid sections.

        Draws colored rectangle patches around each grid section using
        the tab20 colormap. Useful for visualizing which objects belong
        to which wells in a plate layout.

        Args:
            ax: Matplotlib axes to draw section boxes on.
        """
        from phenotypic.measure import MeasureBounds
        from phenotypic.tools_.measurement_info_ import BBOX

        cmap = plt.get_cmap("tab20")
        cmap_cycle = cycle(cmap(i) for i in range(cmap.N))

        img = self._root_image.copy()
        img.objmap = self._root_image.grid.get_section_map()  # type: ignore[attr-defined]
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

    def _generate_overlay_array(
        self,
        overlay_alpha: float = 0.3,
        bg_label: int = 0,
        colors: list | None = None,
        **label2rgb_kwargs,
    ) -> np.ndarray:
        """Generate a full-resolution overlay array blending objmap with the subject image.

        Creates an RGB overlay by blending the object map labels with the underlying
        image data using skimage.color.label2rgb. Unlike show(overlay=True) which
        returns a matplotlib figure, this returns the raw array suitable for pixel-level
        inspection and high-resolution saving.

        Args:
            overlay_alpha: Alpha value for label overlay (0.0 = transparent,
                1.0 = opaque). Higher values make the colored labels more prominent.
                Defaults to 0.3.
            bg_label: Label value to treat as background (will be transparent).
                Defaults to 0.
            colors: List of RGB colors to use for labels. If None, uses default
                label2rgb colormap.
            **label2rgb_kwargs: Additional keyword arguments passed to
                skimage.color.label2rgb.

        Returns:
            np.ndarray: 8-bit RGB array (dtype uint8, shape H x W x 3) containing
                the blended overlay image.
        """
        arr = self._subject_arr
        objmap = self._root_image.objmap[:]

        # Handle grayscale images: normalize and convert to 3-channel for label2rgb
        if arr.ndim == 2:
            if np.issubdtype(arr.dtype, np.floating):
                arr_norm = arr
            else:
                arr_norm = arr.astype(np.float64) / np.iinfo(arr.dtype).max
            # Stack to create 3-channel grayscale image
            arr_rgb = np.stack([arr_norm] * 3, axis=-1)
        else:
            # RGB image: normalize to [0, 1] for label2rgb
            if np.issubdtype(arr.dtype, np.floating):
                arr_rgb = arr
            else:
                arr_rgb = arr.astype(np.float64) / np.iinfo(arr.dtype).max

        # Build label2rgb kwargs
        kwargs = {
            "label": objmap,
            "image": arr_rgb,
            "bg_label": bg_label,
            "alpha": overlay_alpha,
        }
        if colors is not None:
            kwargs["colors"] = colors
        kwargs.update(label2rgb_kwargs)

        # Generate overlay using label2rgb
        overlay_arr = ski.color.label2rgb(**kwargs)

        # Convert to 8-bit uint8 for saving
        overlay_uint8 = (overlay_arr * 255).astype(np.uint8)

        return overlay_uint8

    def _build_phenotypic_metadata(self) -> dict:
        """Build PhenoTypic metadata dictionary for embedding in saved images.

        Returns:
            Dictionary containing phenotypic version, source property, and metadata.
        """
        # Filter out None values and convert to JSON-serializable types
        protected = {}
        for key, value in self._root_image._metadata.protected.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                protected[str(key)] = value

        public = {}
        for key, value in self._root_image._metadata.public.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                public[str(key)] = value

        return {
            "phenotypic_version": phenotypic.__version__,
            "phenotypic_image_property": f"Image.{self._accessor_property_name}",
            "protected": protected,
            "public": public,
        }

    @staticmethod
    def _write_jpeg_metadata(filepath: Path, pil_image, metadata_json: str) -> None:
        """Write metadata to JPEG file using EXIF UserComment tag via exiftool.

        Args:
            filepath: Path to save the JPEG file.
            pil_image: PIL Image object to save.
            metadata_json: JSON string of PhenoTypic metadata.
        """
        # First save the image
        pil_image.save(filepath, quality=100)

        # Then add metadata using exiftool if available
        if shutil.which("exiftool"):
            try:
                subprocess.run(
                    [
                        "exiftool",
                        "-overwrite_original",
                        f"-UserComment={metadata_json}",
                        str(filepath),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                warnings.warn(f"Failed to write EXIF metadata to JPEG: {e}")
        else:
            warnings.warn(
                "exiftool not found. JPEG metadata will not be saved. "
                "Install exiftool for full metadata support."
            )

    @staticmethod
    def _inject_png_text_chunk(
        filepath: Path, key: str, value: str
    ) -> None:
        """Inject a tEXt metadata chunk into an existing PNG file.

        Inserts the chunk immediately after IHDR without re-encoding
        pixel data.

        Args:
            filepath: Path to the PNG file.
            key: Metadata key (latin-1 encodable, max 79 chars).
            value: Metadata value (latin-1 encodable).
        """
        import struct
        import zlib

        with open(filepath, "rb") as f:
            data = f.read()

        # PNG: 8-byte signature + IHDR chunk
        # (4 len + 4 type + 13 data + 4 CRC = 25 bytes)
        ihdr_end = 33

        chunk_data = (
            key.encode("latin-1") + b"\x00" + value.encode("latin-1")
        )
        chunk_type = b"tEXt"
        chunk = (
            struct.pack(">I", len(chunk_data))
            + chunk_type
            + chunk_data
            + struct.pack(
                ">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            )
        )

        with open(filepath, "wb") as f:
            f.write(data[:ihdr_end] + chunk + data[ihdr_end:])

    @staticmethod
    def _write_png_cv2(
        filepath: Path,
        arr: np.ndarray,
        metadata_json: str | None,
    ) -> None:
        """Save a uint16 array as a 16-bit PNG using OpenCV.

        Args:
            filepath: Destination path.
            arr: uint16 array (2-D grayscale or 3-D RGB).
            metadata_json: Optional JSON metadata to embed as a
                tEXt chunk.
        """
        import cv2

        # cv2 expects BGR for colour images
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, ::-1]

        cv2.imwrite(str(filepath), arr)

        if metadata_json:
            ImageAccessorBase._inject_png_text_chunk(
                filepath, IO.PHENOTYPIC_METADATA_KEY, metadata_json
            )

    @staticmethod
    def _write_png_metadata(filepath: Path, pil_image, metadata_json: str) -> None:
        """Write metadata to PNG file using tEXt chunk.

        Args:
            filepath: Path to save the PNG file.
            pil_image: PIL Image object to save.
            metadata_json: JSON string of PhenoTypic metadata.
        """
        from PIL import PngImagePlugin

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text(IO.PHENOTYPIC_METADATA_KEY, metadata_json)
        pil_image.save(filepath, optimize=True, pnginfo=pnginfo)

    @staticmethod
    def _write_tiff_tifffile(
        filepath: Path,
        arr: np.ndarray,
        metadata_json: str | None,
    ) -> None:
        """Save a uint16 array as a 16-bit TIFF using tifffile.

        Uses tifffile for lossless uint16 TIFF writing with metadata
        support. This avoids PIL's limitation with multi-channel uint16
        arrays.

        Args:
            filepath: Destination path.
            arr: uint16 array (2-D grayscale or 3-D RGB).
            metadata_json: Optional JSON metadata to embed as TIFF
                ImageDescription tag.
        """
        import tifffile

        photometric = "rgb" if arr.ndim == 3 and arr.shape[2] >= 3 else "minisblack"
        tifffile.imwrite(
            filepath,
            arr,
            description=metadata_json if metadata_json else None,
            photometric=photometric,
        )

    @staticmethod
    def _write_tiff_metadata(filepath: Path, pil_image, metadata_json: str) -> None:
        """Write metadata to TIFF file using ImageDescription tag.

        Args:
            filepath: Path to save the TIFF file.
            pil_image: PIL Image object to save.
            metadata_json: JSON string of PhenoTypic metadata.
        """
        # TIFF ImageDescription tag is 270
        pil_image.save(filepath, tiffinfo={270: metadata_json})

    def _save_image(
        self,
        filepath: Path,
        arr: np.ndarray,
        bit_depth: Literal[8, 16],
        metadata_json: str | None,
    ) -> None:
        """Save an image array to disk with embedded PhenoTypic metadata.

        Args:
            filepath: Destination file path including extension.
            arr: Image data to save.
            bit_depth: Target bit depth used when coercing float arrays for PNG.
            metadata_json: JSON string containing PhenoTypic metadata to embed.

        Raises:
            ValueError: If the file extension is not supported.

        Warns:
            UserWarning: When saving arrays that require downcasting and may lose
                information (e.g., float or 16-bit arrays to JPEG, float arrays to PNG).
        """
        filepath = Path(filepath)
        arr2save = arr
        suffix = filepath.suffix.lower()

        match suffix:
            case x if x in IO.JPEG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8:
                        pass
                    case np.uint16:
                        warnings.warn(
                            "Saving a 16 bit array as a jpeg will potentially "
                            "result in information loss during conversion"
                        )
                        arr2save = ski.util.img_as_ubyte(arr2save)
                    case dt if np.issubdtype(dt, np.floating):
                        warnings.warn(
                            "Saving a float array as a jpeg will potentially"
                            "result in information loss during conversion"
                        )
                        arr2save = ski.util.img_as_ubyte(arr2save)
                pil_img = PIL_Image.fromarray(arr2save)
                if metadata_json:
                    self._write_jpeg_metadata(filepath, pil_img, metadata_json)
                else:
                    pil_img.save(filepath)

            case x if x in IO.PNG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8:
                        pass
                    case np.uint16:
                        pass  # preserve uint16 for 16-bit PNG
                    case dt if np.issubdtype(dt, np.floating):
                        warnings.warn(
                            ".png images only accept 8 bit and 16 bit "
                            "integer arrays. Converting this array may cause "
                            "information loss"
                        )
                        arr2save = (
                            ski.util.img_as_ubyte(arr2save)
                            if bit_depth == 8
                            else ski.util.img_as_uint(arr2save)
                        )

                if arr2save.dtype == np.uint16:
                    self._write_png_cv2(filepath, arr2save, metadata_json)
                else:
                    pil_img = PIL_Image.fromarray(arr2save)
                    if metadata_json:
                        self._write_png_metadata(
                            filepath, pil_img, metadata_json
                        )
                    else:
                        pil_img.save(filepath)

            case x if x in IO.TIFF_EXTENSIONS:
                if arr2save.dtype == np.uint16:
                    self._write_tiff_tifffile(filepath, arr2save, metadata_json)
                else:
                    pil_img = PIL_Image.fromarray(arr2save)
                    if metadata_json:
                        self._write_tiff_metadata(filepath, pil_img, metadata_json)
                    else:
                        pil_img.save(filepath)

            case _:
                raise ValueError(f"unknown file extension for saving:{filepath.suffix}")

    def imsave(
        self, filepath: str | Path | None = None, bit_depth: Literal[8, 16] | None = None
    ) -> None:
        """
        Saves an array representing a microbe colony image to a specified file format while preserving or adjusting
        metadata and pixel depth as needed. Supports JPEG, PNG, and TIFF formats.

        The behavior of the function is context-sensitive based on the
        file format's restrictions and array properties. Proper file format selection
        and bit depth adjustment can have an impact on the accuracy of image analysis
        and preservation of data integrity.

        Args:
            filepath (str | Path | None): The destination file path where the image will be saved. The extension of the
                file path determines the image format (e.g., .jpeg, .png, .tiff). Changing the file format influences how
                the image data is handled during saving:
                    1. `.jpeg`: Compression or loss of data may occur. Maximal value limit (255) for uint8 pixel
                       depth affects the fidelity of rich intensity details in microbe colonies.
                    2. `.png`: Retains high-quality output but supports only 8-bit or 16-bit images. Conversions may
                       occur if the array has a different data type, which could result in data loss.
                    3. `.tiff`: Ideal for high-bit-depth precision and analysis preservation; best for maintaining
                       intricate morphological details of microbial colonies.

            bit_depth (Literal[8, 16] | None, optional): Specifies the bit depth of the saved image (either 8-bit or
                16-bit). The provided bit depth must align with the file format's capabilities. Misalignment could
                trigger conversion with possible data truncation or rounding. For example:
                    - 8-bit: Useful for efficiently representing intensity when detail is moderate, suitable for JPEG
                      or simple PNG outputs.
                    - 16-bit: Allows for higher intensity ranges, especially valuable for preserving subtle
                      morphological gradient differentiation when analyzing colonies.

        Raises:
            ValueError: An error occurs when an unsupported file extension is provided in `filepath`.

        Warns:
            UserWarnings: Warnings are issued under the following conditions:
                - Saving a 16-bit or floating-point array as JPEG, as these conversions may cause information loss due
                  to format restrictions.
                - Saving a floating-point array as PNG when conversions to 8-bit or 16-bit integers might lead to truncated
                  or altered pixel intensity values.
        """
        bit_depth = self._check_bit_depth(bit_depth)

        filepath = Path(filepath)

        arr2save = self._subject_arr

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=True)

        self._save_image(
            filepath=filepath,
            arr=arr2save,
            bit_depth=bit_depth,
            metadata_json=metadata_json,
        )

    def _check_bit_depth(self, bit_depth: int | None) -> Literal[8, 16]:
        if bit_depth is None:
            bit_depth = self._root_image.bit_depth
        elif bit_depth not in [8, 16]:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        return bit_depth

    def save_overlay(
        self,
        filepath: str | Path,
        overlay_alpha: float = 0.3,
        bg_label: int = 0,
        colors: list | None = None,
        show_gridlines: bool = True,
        gridline_color: tuple[int, int, int] = (0, 255, 255),
        show_section_boxes: bool = True,
        section_box_colors: list[tuple[int, int, int]] | None = None,
        **label2rgb_kwargs,
    ) -> None:
        """Save a full-resolution overlay image blending objmap with the subject array.

        Creates an RGB overlay by blending the object map labels with the underlying
        image data and saves it to disk. Unlike show(overlay=True) which produces a
        matplotlib figure, this method saves the raw pixel data at full resolution,
        suitable for pixel-level quality validation of detection results.

        For GridImage objects, gridlines and section boxes are automatically drawn
        when their respective show_ flags are True. The line widths scale dynamically
        with image size.

        Args:
            filepath: Destination file path. Should have .png or .jpeg extension.
            overlay_alpha: Alpha value for label overlay (0.0 = transparent,
                1.0 = opaque). Defaults to 0.3.
            bg_label: Label value to treat as background. Defaults to 0.
            colors: List of RGB colors to use for labels. If None, uses default
                colormap.
            show_gridlines: Whether to draw gridlines on overlay for GridImage
                objects. Ignored for regular Image objects. Defaults to True.
            gridline_color: RGB color tuple for gridlines. Defaults to cyan
                (0, 255, 255).
            show_section_boxes: Whether to draw colored bounding boxes around
                each grid section's detected objects. Only applies to GridImage.
                Defaults to True.
            section_box_colors: List of RGB tuples for cycling through section
                box colors. Defaults to tab20 colormap colors.
            **label2rgb_kwargs: Additional keyword arguments for label2rgb.

        Raises:
            ValueError: If the file extension is not supported.

        Examples:
            Save full-resolution overlay:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> image.rgb.save_overlay("overlay_rgb.png", overlay_alpha=0.4)
        """
        filepath = Path(filepath)

        # Generate full-resolution overlay array
        overlay_arr = self._generate_overlay_array(
            overlay_alpha=overlay_alpha,
            bg_label=bg_label,
            colors=colors,
            **label2rgb_kwargs,
        )

        # For GridImage, draw gridlines if requested (duck typing check)
        if show_gridlines and hasattr(self._root_image, "_draw_gridlines_on_overlay"):
            overlay_arr = self._root_image._draw_gridlines_on_overlay(
                overlay_arr, gridline_color
            )

        # For GridImage, draw section boxes if requested (duck typing check)
        if show_section_boxes and hasattr(
            self._root_image, "_draw_section_boxes_on_overlay"
        ):
            overlay_arr = self._root_image._draw_section_boxes_on_overlay(
                overlay_arr, section_box_colors
            )

        # Save using existing _save_image infrastructure (no metadata for overlays)
        self._save_image(
            filepath=filepath,
            arr=overlay_arr,
            bit_depth=8,  # Overlays are always 8-bit
            metadata_json=None,  # No phenotypic metadata for overlay images
        )

    @property
    def nbytes(self) -> int:
        return self._subject_arr.nbytes

    def napari(
        self,
        name: str | None = None,
        reset: bool = False,
        *,
        viewer: napari.Viewer | None = None,
        layer_name: str | None = None,
    ) -> napari.Viewer:
        """Add image to a persistent global napari viewer for Jupyter workflows.

        Creates or reuses a single napari viewer instance that persists across
        multiple method calls. This is particularly useful in Jupyter notebooks
        where multiple accessors can contribute layers to the same viewer,
        enabling interactive comparison of different image transformations
        (e.g., grayscale, RGB, binary masks) on the same data.

        The viewer is automatically displayed in Jupyter environments and
        recreated if it has been closed externally.

        Args:
            name: Optional custom name for the image layer. If provided, the layer
                will be named ``{accessor}_{name}``. If not provided, defaults to
                using the image's name attribute.
            reset: If True, closes the current napari viewer and creates a fresh
                one. This is useful for starting a new visualization session
                without lingering layers from previous calls. Defaults to False.
            viewer: Optional external napari viewer instance to use instead of the
                global viewer. When provided, global viewer management (creation,
                reset, smart-grid installation) is bypassed entirely. Defaults to
                None.
            layer_name: Optional full layer name to use instead of the auto-generated
                ``{accessor}_{image_name}`` pattern. Defaults to None.

        Returns:
            napari.Viewer: The global napari viewer instance with the current
                image added as a new layer.

        Raises:
            ImportError: If napari is not installed. Install with
                ``pip install phenotypic[gui]``.

        Examples:
            View multiple image transformations in one viewer:

            >>> from phenotypic import Image
            >>> img = Image(arr)
            >>> # Add grayscale version to viewer
            >>> viewer = img.gray.napari()
            >>> # Add RGB version to same viewer
            >>> viewer = img.rgb.napari()
            >>> # Add binary segmentation with custom name
            >>> viewer = img.objmask.napari(name="segmentation_v2")

            Using custom names for comparison:

            >>> viewer = img.gray.napari(name="raw_grayscale")
            >>> viewer = img.objmask.napari(name="segmentation_v2")

            Resetting the viewer for a fresh session:

            >>> viewer = img.gray.napari()
            >>> viewer = img.rgb.napari()  # Same viewer, added layer
            >>> viewer = img.gray.napari(reset=True)  # Fresh viewer, old layers gone

        Note:
            Layers are named using the pattern ``{accessor}_{image_name}`` to
            ensure descriptive identification. If a layer with the same name
            already exists, it is replaced with the new image data. This allows
            for easy updates and comparison of different processing stages.
        """
        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[gui]"
            )
        import napari as _napari

        # Determine active viewer
        if viewer is not None:
            active_viewer = viewer
        else:
            global _global_napari_viewer

            # Reset viewer if requested
            if reset and _viewer_is_alive(_global_napari_viewer):
                _global_napari_viewer.close()
                _global_napari_viewer = None

            # Create new viewer if needed
            if not _viewer_is_alive(_global_napari_viewer):
                _global_napari_viewer = _napari.Viewer()
                from phenotypic.gui._smart_grid import install_smart_grid
                install_smart_grid(_global_napari_viewer)

            active_viewer = _global_napari_viewer

        # Generate descriptive layer name
        if layer_name is not None:
            resolved_layer_name = layer_name
        elif name is not None:
            image_name = name
            resolved_layer_name = f"{self._accessor_property_name}_{image_name}"
        else:
            image_name = getattr(self._root_image, "name", "image")
            resolved_layer_name = f"{self._accessor_property_name}_{image_name}"

        # Replace layer if it exists, otherwise add new layer

        imdata = self._subject_arr
        if imdata.ndim == 3:
            imdata = normalize_rgb_bitdepth(imdata)
        try:
            existing_layer = active_viewer.layers[resolved_layer_name]
            existing_layer.data = imdata
        except KeyError:
            active_viewer.add_image(
                imdata, name=resolved_layer_name,
                contrast_limits=(0, int(np.iinfo(imdata.dtype).max))
                    if np.issubdtype(imdata.dtype, np.integer)
                    else (float(imdata.min()), float(imdata.max())),
                gamma=1.0,
            )

        return active_viewer
