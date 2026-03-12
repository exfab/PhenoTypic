from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import skimage as ski
from PIL import Image as PIL_Image
from abc import ABC, abstractmethod
from phenotypic._core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic.tools_.constants_ import METADATA, IO

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

    def imsave(self, fname: str | Path) -> None:
        """Save the multichannel image array to a file with PhenoTypic metadata embedded.

        Metadata is embedded in format-specific locations:
        - JPEG: EXIF UserComment tag
        - PNG: tEXt chunk with key 'phenotypic'
        - TIFF: ImageDescription tag (270)

        Args:
            fname: Path to save the image file. Extension determines format.

        Raises:
            AttributeError: If bit depth is not 8 or 16.
        """
        fname = Path(fname)
        arr = self._subject_arr.copy()

        # Convert to appropriate bit depth
        if arr.dtype not in (np.uint8, np.uint16):
            match self._root_image.metadata[METADATA.BIT_DEPTH]:
                case 8:
                    arr = ski.util.img_as_ubyte(arr)
                case 16:
                    arr = ski.util.img_as_uint(arr)
                case _:
                    raise AttributeError(
                            f"Unsupported bit depth: {self._root_image.metadata[METADATA.BIT_DEPTH]}"
                    )

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=False)

        suffix = fname.suffix.lower()

        if suffix in IO.JPEG_FILE_EXTENSIONS:
            # Convert 16-bit to 8-bit for JPEG
            if arr.dtype == np.uint16:
                warnings.warn(
                        "Saving 16-bit RGB as JPEG will result in information loss"
                )
                arr = ski.util.img_as_ubyte(arr)
            pil_img = PIL_Image.fromarray(arr)
            self._write_jpeg_metadata(fname, pil_img, metadata_json)

        elif suffix in IO.PNG_FILE_EXTENSIONS:
            if arr.dtype == np.uint16:
                self._write_png_cv2(fname, arr, metadata_json)
            else:
                pil_img = PIL_Image.fromarray(arr)
                self._write_png_metadata(fname, pil_img, metadata_json)

        elif suffix in IO.TIFF_EXTENSIONS:
            if arr.dtype == np.uint16:
                self._write_tiff_tifffile(fname, arr, metadata_json)
            else:
                pil_img = PIL_Image.fromarray(arr)
                self._write_tiff_metadata(fname, pil_img, metadata_json)

        else:
            # Fallback to skimage without metadata
            ski.io.imsave(fname=fname, arr=arr, check_contrast=False)

    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            channel: int | None = None,
            foreground_only: bool = False,
            *,
            plotly_settings: dict | None = None,
    ) -> go.Figure | tuple[plt.Figure, plt.Axes]:
        """Display the multichannel image data interactively.

        Uses Plotly when available, falling back to matplotlib otherwise.

        Args:
            figsize: Figure size in inches (width, height). If None,
                auto-calculated from array aspect ratio.
            title: Title of the plot. If None, a default title is
                generated based on the image and channel.
            channel: Specific channel index to plot. If None, all
                channels are displayed as RGB.
            foreground_only: If True, only foreground is displayed.
            plotly_settings: Additional Plotly layout settings.

        Returns:
            A ``plotly.graph_objects.Figure`` when plotly is installed,
            or a ``(plt.Figure, plt.Axes)`` tuple when using matplotlib
            fallback.
        """
        from phenotypic.tools_._plotly_helpers import PLOTLY_AVAILABLE

        arr = self[:] if not foreground_only else self.foreground()

        if channel is not None:
            title = (
                f"{self._root_image.name} - Channel {channel}"
                if title is None
                else f"{title} - Channel {channel}"
            )

        if not PLOTLY_AVAILABLE:
            if channel is None:
                return self._mpl_plot(
                    arr=arr, figsize=figsize, title=title,
                )
            return self._mpl_plot(
                arr=arr[:, :, channel], figsize=figsize, title=title,
                cmap="gray",
            )

        from phenotypic.tools_._plotly_helpers import plotly_imshow

        if channel is None:
            fig = plotly_imshow(arr=arr, figsize=figsize, title=title)
        else:
            fig = plotly_imshow(
                    arr=arr[:, :, channel],
                    figsize=figsize,
                    title=title,
                    cmap="gray",
            )
        if plotly_settings is not None:
            fig.update_layout(**plotly_settings)
        return fig
