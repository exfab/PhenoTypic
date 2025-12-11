from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Tuple, Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as ski
from PIL import Image as PIL_Image
from matplotlib.ticker import MultipleLocator

import phenotypic
from phenotypic.settings_ import MPL
from phenotypic.tools.constants_ import IO, METADATA
from .._plotting_backends import (
    validate_backend,
    plot_image_matplotlib,
    plot_image_plotly,
    plot_overlay_plotly,
    add_scatter_annotations_plotly,
    PlotReturn,
    MatplotlibReturn,
    PlotlyReturn,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from phenotypic import Image


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
        return cls._accessor_property_name.fget(
                object.__new__(cls))  # type: ignore[attr-defined]

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
            .. dropdown:: Load a grayscale image from file

                >>> from phenotypic._core._image_parts.accessors import Grayscale
                >>> arr = Grayscale.load("my_gray_image.png")
        """
        filepath = Path(filepath)
        expected_property = f"Image.{cls._accessor_property_name_value()}"

        # Load the array
        arr = ski.io.imread(str(filepath))

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

    @overload
    def histogram(
            self,
            figsize: Tuple[int, int] = (10, 5),
            *,
            cmap: str = "gray",
            linewidth: int = 1,
            channel_names: list | None = None,
            backend: Literal["matplotlib"] = "matplotlib",
    ) -> MatplotlibReturn:
        ...

    @overload
    def histogram(
            self,
            figsize: Tuple[int, int] = (10, 5),
            *,
            cmap: str = "gray",
            linewidth: int = 1,
            channel_names: list | None = None,
            backend: Literal["plotly"],
    ) -> PlotlyReturn:
        ...

    def histogram(
            self,
            figsize: Tuple[int, int] = (10, 5),
            *,
            cmap: str = "gray",
            linewidth: int = 1,
            channel_names: list | None = None,
            backend: Literal["matplotlib", "plotly"] = "matplotlib",
    ) -> PlotReturn:
        """
        Plot histogram(s) of image alongside image data using selected backend.

        Creates side-by-side visualization of image and intensity distribution(s).
        Behavior depends on array dimensionality (2D vs 3D).

        Args:
            figsize (Tuple[int, int], optional): Figure size in inches.
                Defaults to (10, 5).
            cmap (str, optional): Colormap for single-channel images.
                Defaults to 'gray'.
            linewidth (int, optional): Line width for histogram plots.
                Defaults to 1.
            channel_names (list | None, optional): Names for 3D channels.
                If None, uses numeric labels. Defaults to None.
            backend (Literal["matplotlib", "plotly"], optional): Backend to use.
                Defaults to "matplotlib".

        Returns:
            PlotReturn:
                - If backend="matplotlib": tuple[plt.Figure, plt.Axes]
                - If backend="plotly": plotly.graph_objects.Figure

        Raises:
            ValueError: If unsupported array dimensionality or backend.
            TypeError: If unsupported image dtype.
            ImportError: If plotly requested but not installed.

        Notes:
            Uses skimage.exposure.histogram for computing histogram data.
        """
        backend = validate_backend(backend)

        arr = self._subject_arr
        dtype = arr.dtype

        if np.issubdtype(dtype, np.floating):
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_min < 0.0 or arr_max > 1.0:
                raise ValueError(
                    f"Float image arrays must be within [0.0, 1.0]. "
                    f"Found range [{arr_min}, {arr_max}]."
                )
            x_limits = (0.0, 1.0)
        elif np.issubdtype(dtype, np.bool_):
            x_limits = (0, 1)
        elif np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
            x_limits = (dtype_info.min, dtype_info.max)
        else:
            raise TypeError(
                f"Unsupported image dtype for histogram plotting: {dtype}"
            )

        if backend == "matplotlib":
            match self.ndim:
                case 2:
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
                    axes = axes.ravel()
                    self._plot(
                        arr=self._subject_arr,
                        figsize=figsize,
                        title=self._root_image.name,
                        cmap=cmap,
                        ax=axes[0],
                        backend="matplotlib",
                    )
                    hist, histc = ski.exposure.histogram(
                        image=self._subject_arr[:],
                        nbins=2
                        ** self._root_image.metadata[METADATA.BIT_DEPTH],
                    )
                    axes[1].plot(histc, hist, lw=linewidth)
                    axes[1].set_xlim(x_limits)

                case 3:
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

                    for idx, ax in enumerate(axes.flat):
                        if idx == 0:
                            self._plot(
                                arr=self._subject_arr[:],
                                figsize=figsize,
                                title=self._root_image.name,
                                ax=ax,
                                backend="matplotlib",
                            )
                        else:
                            hist, histc = ski.exposure.histogram(
                                image=self._subject_arr[:, :, idx - 1],
                                nbins=2
                                ** self._root_image.metadata[
                                    METADATA.BIT_DEPTH
                                ],
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

        else:  # backend == "plotly"
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            match self.ndim:
                case 2:
                    # 1x2 subplot: image + histogram
                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=(self._root_image.name, "Histogram"),
                    )

                    # Add image (as heatmap for 2D)
                    fig.add_trace(
                        go.Heatmap(
                            z=self._subject_arr,
                            colorscale=translate_colormap(cmap, "plotly"),
                            showscale=False,
                        ),
                        row=1,
                        col=1,
                    )

                    # Add histogram
                    hist, histc = ski.exposure.histogram(
                        image=self._subject_arr[:],
                        nbins=2
                        ** self._root_image.metadata[METADATA.BIT_DEPTH],
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=histc,
                            y=hist,
                            mode="lines",
                            line=dict(width=linewidth),
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )

                case 3:
                    # 2x2 subplot: image + 3 channel histograms
                    fig = make_subplots(
                        rows=2,
                        cols=2,
                        subplot_titles=(
                            self._root_image.name,
                            f"Channel-{channel_names[0] if channel_names else 0}",
                            f"Channel-{channel_names[1] if channel_names else 1}",
                            f"Channel-{channel_names[2] if channel_names else 2}",
                        ),
                    )

                    # Add RGB image
                    plot_arr = self._subject_arr
                    if plot_arr.dtype in (np.float32, np.float64):
                        plot_arr = (plot_arr * 255).astype(np.uint8)
                    fig.add_trace(go.Image(z=plot_arr), row=1, col=1)

                    # Add channel histograms
                    positions = [(1, 2), (2, 1), (2, 2)]
                    for channel_idx, (row, col) in enumerate(positions):
                        hist, histc = ski.exposure.histogram(
                            image=self._subject_arr[:, :, channel_idx],
                            nbins=2
                            ** self._root_image.metadata[
                                METADATA.BIT_DEPTH
                            ],
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=histc,
                                y=hist,
                                mode="lines",
                                line=dict(width=linewidth),
                                showlegend=False,
                            ),
                            row=row,
                            col=col,
                        )

                case _:
                    raise ValueError(
                        f"Unsupported array dimension: {self._subject_arr.ndim}"
                    )

            # Configure layout
            width_px = int(figsize[0] * 100)
            height_px = int(figsize[1] * 100)
            fig.update_layout(width=width_px, height=height_px, showlegend=False)

            return fig

    @overload
    def _plot(
            self,
            arr: np.ndarray,
            figsize: Tuple[int, int] | None = None,
            title: str | bool | None = None,
            cmap: str = "gray",
            ax: plt.Axes | None = None,
            mpl_settings: dict | None = None,
            *,
            backend: Literal["matplotlib"] = "matplotlib",
    ) -> MatplotlibReturn:
        ...

    @overload
    def _plot(
            self,
            arr: np.ndarray,
            figsize: Tuple[int, int] | None = None,
            title: str | bool | None = None,
            cmap: str = "gray",
            ax: None = None,
            mpl_settings: dict | None = None,
            *,
            backend: Literal["plotly"],
            plotly_settings: dict | None = None,
    ) -> PlotlyReturn:
        ...

    def _plot(
            self,
            arr: np.ndarray,
            figsize: Tuple[int, int] | None = None,
            title: str | bool | None = None,
            cmap: str = "gray",
            ax: plt.Axes | None = None,
            mpl_settings: dict | None = None,
            *,
            backend: Literal["matplotlib", "plotly"] = "matplotlib",
            plotly_settings: dict | None = None,
    ) -> PlotReturn:
        """
        Plot an image array using matplotlib or plotly backend.

        Renders an image array with flexible backend selection for both static
        (matplotlib) and interactive (plotly) visualization.

        Args:
            arr (np.ndarray): The image data to plot. Can be 2D or 3D array.
            figsize ((int, int), optional): Figure size in inches.
                Defaults to settings_.MPL.FIGSIZE.
            title (None | str | bool, optional): Plot title. If True, uses
                parent image name. If str, uses that string. If None/False,
                no title. Defaults to None.
            cmap (str, optional): Colormap for 2D arrays. Defaults to 'gray'.
            ax (None | plt.Axes, optional): Existing matplotlib axes. Only
                valid for matplotlib backend. Defaults to None.
            mpl_settings (dict | None, optional): Additional matplotlib
                imshow kwargs. Only used with matplotlib backend.
                Defaults to None.
            backend (Literal["matplotlib", "plotly"], optional): Backend to use.
                Defaults to "matplotlib".
            plotly_settings (dict | None, optional): Additional plotly trace
                kwargs. Only used with plotly backend. Defaults to None.

        Returns:
            PlotReturn:
                - If backend="matplotlib": tuple[plt.Figure, plt.Axes]
                - If backend="plotly": plotly.graph_objects.Figure

        Raises:
            ValueError: If backend not supported or ax provided with plotly.
            ImportError: If plotly requested but not installed.

        Examples:
            >>> # Matplotlib backend (default)
            >>> fig, ax = image.gray._plot(image.gray[:])

            >>> # Plotly backend
            >>> fig = image.gray._plot(image.gray[:], backend="plotly")
            >>> fig.show()
        """
        # Validate backend
        backend = validate_backend(backend)

        # Handle title
        if title is True:
            title = self._root_image.name
        elif title is False:
            title = None

        # Set default figsize
        figsize = figsize if figsize else MPL.FIGSIZE

        # Route to appropriate backend
        if backend == "matplotlib":
            return plot_image_matplotlib(
                arr=arr,
                figsize=figsize,
                title=title,
                cmap=cmap,
                ax=ax,
                mpl_settings=mpl_settings,
            )

        else:  # backend == "plotly"
            if ax is not None:
                raise ValueError(
                    "The 'ax' parameter is not supported with plotly backend. "
                    "Plotly creates standalone figures and cannot plot on existing "
                    "matplotlib axes. To use existing axes, switch to backend='matplotlib'."
                )

            return plot_image_plotly(
                arr=arr,
                figsize=figsize,
                title=title,
                cmap=cmap,
                plotly_settings=plotly_settings,
            )

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

    @overload
    def _plot_overlay(
            self,
            arr: np.ndarray,
            objmap: np.ndarray,
            figsize: Tuple[int, int] = (8, 6),
            title: str | bool | None = None,
            cmap: str = "gray",
            ax: plt.Axes | None = None,
            *,
            overlay_settings: dict | None = None,
            mpl_settings: dict | None = None,
            backend: Literal["matplotlib"] = "matplotlib",
    ) -> MatplotlibReturn:
        ...

    @overload
    def _plot_overlay(
            self,
            arr: np.ndarray,
            objmap: np.ndarray,
            figsize: Tuple[int, int] = (8, 6),
            title: str | bool | None = None,
            cmap: str = "gray",
            ax: None = None,
            *,
            overlay_settings: dict | None = None,
            mpl_settings: dict | None = None,
            backend: Literal["plotly"],
            plotly_settings: dict | None = None,
    ) -> PlotlyReturn:
        ...

    def _plot_overlay(
            self,
            arr: np.ndarray,
            objmap: np.ndarray,
            figsize: Tuple[int, int] = (8, 6),
            title: str | bool | None = None,
            cmap: str = "gray",
            ax: plt.Axes | None = None,
            *,
            overlay_settings: dict | None = None,
            mpl_settings: dict | None = None,
            backend: Literal["matplotlib", "plotly"] = "matplotlib",
            plotly_settings: dict | None = None,
    ) -> PlotReturn:
        """
        Plot array with optional object map overlay using selected backend.

        Renders an image with colored object labels overlaid on top, supporting
        both matplotlib and plotly backends.

        Args:
            arr (np.ndarray): The primary array to be displayed as image.
            objmap (np.ndarray): Array containing labels for object map overlay.
            figsize (Tuple[int, int], optional): Figure size (width, height) in inches.
                Defaults to (8, 6).
            title (str | bool | None, optional): Plot title. If True, uses parent
                image name. Defaults to None.
            cmap (str, optional): Colormap for 2D arrays. Defaults to 'gray'.
            ax (plt.Axes | None, optional): Existing matplotlib axes. Only valid
                for matplotlib backend. Defaults to None.
            overlay_settings (dict | None, optional): Parameters for
                skimage.color.label2rgb overlay customization. Defaults to None.
            mpl_settings (dict | None, optional): Additional matplotlib imshow
                kwargs. Only used with matplotlib backend. Defaults to None.
            backend (Literal["matplotlib", "plotly"], optional): Backend to use.
                Defaults to "matplotlib".
            plotly_settings (dict | None, optional): Additional plotly trace kwargs.
                Only used with plotly backend. Defaults to None.

        Returns:
            PlotReturn:
                - If backend="matplotlib": tuple[plt.Figure, plt.Axes]
                - If backend="plotly": plotly.graph_objects.Figure

        Raises:
            ValueError: If backend invalid or ax with plotly backend.
            ImportError: If plotly requested but not installed.
        """
        backend = validate_backend(backend)

        overlay_settings = overlay_settings if overlay_settings else {}
        overlay_alpha = overlay_settings.get("beta", 0.15)

        if backend == "matplotlib":
            overlay_arr = ski.color.label2rgb(
                label=objmap, image=arr, bg_label=0, alpha=overlay_alpha,
                **overlay_settings
            )

            return self._plot(
                arr=overlay_arr,
                figsize=figsize,
                title=title,
                cmap=cmap,
                ax=ax,
                mpl_settings=mpl_settings,
                backend="matplotlib",
            )

        else:  # backend == "plotly"
            if ax is not None:
                raise ValueError(
                    "The 'ax' parameter is not supported with plotly backend."
                )

            return plot_overlay_plotly(
                arr=arr,
                objmap=objmap,
                figsize=figsize,
                title=title,
                overlay_alpha=overlay_alpha,
                plotly_settings=plotly_settings,
            )

    @overload
    def show_overlay(
            self,
            object_label: None | int = None,
            figsize: Tuple[int, int] | None = None,
            title: str | None = None,
            show_labels: bool = False,
            ax: plt.Axes | None = None,
            *,
            label_settings: None | dict = None,
            overlay_settings: None | dict = None,
            imshow_settings: None | dict = None,
            backend: Literal["matplotlib"] = "matplotlib",
    ) -> MatplotlibReturn:
        ...

    @overload
    def show_overlay(
            self,
            object_label: None | int = None,
            figsize: Tuple[int, int] | None = None,
            title: str | None = None,
            show_labels: bool = False,
            ax: None = None,
            *,
            label_settings: None | dict = None,
            overlay_settings: None | dict = None,
            imshow_settings: None | dict = None,
            backend: Literal["plotly"],
            plotly_settings: None | dict = None,
    ) -> PlotlyReturn:
        ...

    def show_overlay(
            self,
            object_label: None | int = None,
            figsize: Tuple[int, int] | None = None,
            title: str | None = None,
            show_labels: bool = False,
            ax: plt.Axes | None = None,
            *,
            label_settings: None | dict = None,
            overlay_settings: None | dict = None,
            imshow_settings: None | dict = None,
            backend: Literal["matplotlib", "plotly"] = "matplotlib",
            plotly_settings: None | dict = None,
    ) -> PlotReturn:
        """
        Display overlay of object map on parent image with optional annotations.

        Visualizes object regions overlaid on the parent image with customization
        options for figure size, colors, and annotation properties. Supports both
        matplotlib and plotly backends.

        Args:
            object_label (None | int, optional): Specific object label to highlight.
                If None, all objects displayed. Defaults to None.
            figsize (Tuple[int, int] | None, optional): Figure size in inches
                (width, height). Defaults to None.
            title (str | None, optional): Plot title. If None, uses parent image
                name. Defaults to None.
            show_labels (bool, optional): If True, display annotations for object
                labels at centroids. Defaults to False.
            ax (plt.Axes | None, optional): Existing matplotlib axes. Only valid
                for matplotlib backend. Defaults to None.
            label_settings (dict | None, optional): Label annotation customization.
                Defaults: size=12, color='white', facecolor='red'. Defaults to None.
            overlay_settings (dict | None, optional): Overlay customization params.
                Defaults to None.
            imshow_settings (dict | None, optional): Matplotlib imshow kwargs.
                Only used with matplotlib backend. Defaults to None.
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

        Examples:
            >>> # Matplotlib overlay with labels
            >>> fig, ax = image.gray.show_overlay(show_labels=True)

            >>> # Plotly interactive overlay
            >>> fig = image.gray.show_overlay(
            ...     backend="plotly",
            ...     show_labels=True
            ... )
            >>> fig.show()
        """
        backend = validate_backend(backend)

        objmap = self._root_image.objmap[:]
        if object_label is not None:
            objmap[objmap != object_label] = 0
        if label_settings is None:
            label_settings = {}

        if backend == "matplotlib":
            fig, ax = self._plot_overlay(
                arr=self._subject_arr,
                objmap=objmap,
                ax=ax,
                figsize=figsize,
                title=title,
                mpl_settings=imshow_settings,
                overlay_settings=overlay_settings,
                backend="matplotlib",
            )

            if show_labels:
                ax = self._plot_obj_labels(
                    ax=ax,
                    color=label_settings.get("color", "white"),
                    size=label_settings.get("size", 12),
                    facecolor=label_settings.get("facecolor", "red"),
                    object_label=object_label,
                )
            return fig, ax

        else:  # backend == "plotly"
            if ax is not None:
                raise ValueError(
                    "The 'ax' parameter is not supported with plotly backend."
                )

            fig = self._plot_overlay(
                arr=self._subject_arr,
                objmap=objmap,
                figsize=figsize,
                title=title,
                overlay_settings=overlay_settings,
                backend="plotly",
                plotly_settings=plotly_settings,
            )

            if show_labels:
                # Extract centroids and labels for annotation
                props = self._root_image.objects.props
                labels = self._root_image.objects.labels

                if object_label is None:
                    centroids = [prop.centroid for prop in props]
                    label_values = labels
                else:
                    idx = np.where(labels == object_label)[0]
                    if len(idx) > 0:
                        centroids = [props[idx[0]].centroid]
                        label_values = [object_label]
                    else:
                        centroids = []
                        label_values = []

                fig = add_scatter_annotations_plotly(
                    fig=fig,
                    labels=label_values,
                    centroids=centroids,
                    color=label_settings.get("color", "white"),
                    size=label_settings.get("size", 12),
                )

            return fig

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
            "phenotypic_version"       : phenotypic.__version__,
            "phenotypic_image_property": f"Image.{self._accessor_property_name}",
            "protected"                : protected,
            "public"                   : public,
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
            metadata_json: str,
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
        arr2save = arr
        suffix = filepath.suffix.lower()

        match suffix:
            case x if x in IO.JPEG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8:
                        pass
                    case np.uint16:
                        warnings.warn(
                                "Saving a 16 bit array as a jpeg will result in information loss if the max value is higher than 255"
                        )
                        arr2save = ski.util.img_as_ubyte(arr2save)
                    case dt if np.issubdtype(dt, np.floating):
                        warnings.warn(
                                "Saving a float array as a jpeg will result in information loss if the max value is higher than 255"
                        )
                        arr2save = ski.util.img_as_ubyte(arr2save)
                pil_img = PIL_Image.fromarray(arr2save)
                self._write_jpeg_metadata(filepath, pil_img, metadata_json)

            case x if x in IO.PNG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8 | np.uint16:
                        pass
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
                pil_img = PIL_Image.fromarray(arr2save)
                self._write_png_metadata(filepath, pil_img, metadata_json)

            case x if x in IO.TIFF_EXTENSIONS:
                pil_img = PIL_Image.fromarray(arr2save)
                self._write_tiff_metadata(filepath, pil_img, metadata_json)

            case _:
                raise ValueError(f"unknown file extension for saving:{filepath.suffix}")

    def imsave(self,
               filepath: str | Path | None = None,
               bit_depth: Literal[8, 16] | None = None) -> None:
        """
        Saves an array representing a microbe colony image to a specified file format while preserving or adjusting
        metadata and pixel depth as needed. Supports JPEG, PNG, and TIFF formats.

        The behavior of the function is context-sensitive based on the file format's restrictions and array properties.
        For microbe colony images on agar media, proper file format selection and bit depth adjustment can have an impact
        on the accuracy of image analysis and preservation of data integrity.

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
        if bit_depth is None:
            bit_depth = self._root_image.bit_depth
        elif bit_depth not in [8, 16]:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        filepath = Path(filepath)

        arr2save = self._subject_arr

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=False)

        self._save_image(
                filepath=filepath,
                arr=arr2save,
                bit_depth=bit_depth,
                metadata_json=metadata_json,
        )
