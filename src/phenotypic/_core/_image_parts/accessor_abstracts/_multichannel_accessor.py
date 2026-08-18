from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import numpy as np
import skimage as ski
from PIL import Image as PIL_Image
from abc import ABC, abstractmethod
from phenotypic._core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic.schema import IMAGE
from phenotypic.sdk_.constants_ import IO

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go


class MultiChannelAccessor(ImageAccessorBase, ABC):
    """
    Handles interaction with Image data by providing access to Image attributes and data.

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

    def imsave(
            self,
            filepath: str | Path,
            bit_depth: Literal[8, 16] | None = None,
    ) -> None:
        """Save the multichannel image array to a file with PhenoTypic metadata embedded.

        Metadata is embedded in format-specific locations:
        - JPEG: EXIF UserComment tag
        - PNG: tEXt chunk with key 'phenotypic'
        - TIFF: ImageDescription tag (270)

        Args:
            filepath: Path to save the image file. Extension determines format.
            bit_depth: Target bit depth (8 or 16). If None, uses image's bit depth.

        Raises:
            AttributeError: If bit depth is not 8 or 16.
        """
        filepath = Path(filepath)
        arr = self._subject_arr.copy()

        effective_bit_depth = (
            bit_depth if bit_depth is not None
            else self._root_image.metadata[IMAGE.BIT_DEPTH]
        )

        # Convert to appropriate bit depth
        if arr.dtype not in (np.uint8, np.uint16):
            match effective_bit_depth:
                case 8:
                    arr = ski.util.img_as_ubyte(arr)
                case 16:
                    arr = ski.util.img_as_uint(arr)
                case _:
                    raise AttributeError(
                            f"Unsupported bit depth: {effective_bit_depth}"
                    )

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=True)

        suffix = filepath.suffix.lower()

        if suffix in IO.JPEG_FILE_EXTENSIONS:
            # Convert 16-bit to 8-bit for JPEG
            if arr.dtype == np.uint16:
                warnings.warn(
                        "Saving 16-bit RGB as JPEG will result in information loss"
                )
                arr = ski.util.img_as_ubyte(arr)
            pil_img = PIL_Image.fromarray(arr)
            self._write_jpeg_metadata(filepath, pil_img, metadata_json)

        elif suffix in IO.PNG_FILE_EXTENSIONS:
            if arr.dtype == np.uint16:
                self._write_png_cv2(filepath, arr, metadata_json)
            else:
                pil_img = PIL_Image.fromarray(arr)
                self._write_png_metadata(filepath, pil_img, metadata_json)

        elif suffix in IO.TIFF_EXTENSIONS:
            if arr.dtype == np.uint16:
                self._write_tiff_tifffile(filepath, arr, metadata_json)
            else:
                pil_img = PIL_Image.fromarray(arr)
                self._write_tiff_metadata(filepath, pil_img, metadata_json)

        else:
            # Fallback to skimage without metadata
            ski.io.imsave(fname=filepath, arr=arr, check_contrast=False)

    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            channel: int | None = None,
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
        """Display the multichannel image data using matplotlib.

        Args:
            figsize: Figure size in inches (width, height). If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, a default title is
                generated based on the image and channel.
            channel: Specific channel index to plot. If None, all
                channels are displayed as RGB.
            foreground_only: If True, only foreground is displayed.
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

        if channel is not None:
            title = (
                f"{self._root_image.name} - Channel {channel}"
                if title is None
                else f"{title} - Channel {channel}"
            )

        has_objects = self._root_image.num_objects > 0
        if overlay and has_objects:
            plot_arr = arr if channel is None else arr[:, :, channel]
            objmap = self._get_filtered_objmap(object_label)
            fig, ax = self._plot_overlay(
                    arr=plot_arr, objmap=objmap, figsize=figsize, title=title,
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

        if channel is None:
            return self._mpl_plot(
                    arr=arr, figsize=figsize, title=title, ax=ax,
            )
        return self._mpl_plot(
                arr=arr[:, :, channel], figsize=figsize, title=title,
                cmap="gray", ax=ax,
        )

    def dash(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            channel: int | None = None,
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
        """Display the multichannel image data using Plotly.

        Args:
            figsize: Figure size in inches (width, height). If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, a default title is
                generated based on the image and channel.
            channel: Specific channel index to plot. If None, all
                channels are displayed as RGB.
            foreground_only: If True, only foreground is displayed.
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

        if channel is not None:
            title = (
                f"{self._root_image.name} - Channel {channel}"
                if title is None
                else f"{title} - Channel {channel}"
            )

        has_objects = self._root_image.num_objects > 0
        if overlay and has_objects:
            plot_arr = arr if channel is None else arr[:, :, channel]
            objmap = self._get_filtered_objmap(object_label)
            fig = self._plotly_overlay(
                    arr=plot_arr, objmap=objmap, figsize=figsize, title=title,
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

        if channel is None:
            fig = self._plotly_imshow(arr=arr, figsize=figsize, title=title)
        else:
            fig = self._plotly_imshow(
                    arr=arr[:, :, channel],
                    figsize=figsize,
                    title=title,
                    cmap="gray",
            )
        if plotly_settings is not None:
            fig.update_layout(**plotly_settings)
        return fig
