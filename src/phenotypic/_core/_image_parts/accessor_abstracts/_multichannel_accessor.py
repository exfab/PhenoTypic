import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Literal, overload, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from PIL import Image as PIL_Image
from PIL import PngImagePlugin

import phenotypic
from phenotypic._core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic._core._image_parts._plotting_backends import (
    PlotReturn,
    MatplotlibReturn,
    PlotlyReturn,
)
from phenotypic.tools.constants_ import METADATA, IO

if TYPE_CHECKING:
    import plotly.graph_objects as go


class MultiChannelAccessor(ImageAccessorBase):
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
            pil_img = PIL_Image.fromarray(arr)
            self._write_png_metadata(fname, pil_img, metadata_json)

        elif suffix in IO.TIFF_EXTENSIONS:
            pil_img = PIL_Image.fromarray(arr)
            self._write_tiff_metadata(fname, pil_img, metadata_json)

        else:
            # Fallback to skimage without metadata
            ski.io.imsave(fname=fname, arr=arr, check_contrast=False)

    @overload
    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            ax: plt.Axes | None = None,
            channel: int | None = None,
            foreground_only: bool = False,
            *,
            mpl_settings: dict | None = None,
            backend: Literal["matplotlib"] = "matplotlib",
    ) -> MatplotlibReturn:
        ...

    @overload
    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            ax: None = None,
            channel: int | None = None,
            foreground_only: bool = False,
            *,
            mpl_settings: dict | None = None,
            backend: Literal["plotly"],
            plotly_settings: dict | None = None,
    ) -> PlotlyReturn:
        ...

    def show(
            self,
            figsize: tuple[int, int] | None = None,
            title: str | None = None,
            ax: plt.Axes | None = None,
            channel: int | None = None,
            foreground_only: bool = False,
            *,
            mpl_settings: dict | None = None,
            backend: Literal["matplotlib", "plotly"] = "matplotlib",
            plotly_settings: dict | None = None,
    ) -> PlotReturn:
        """
        Display image with optional backend selection and customization.

        Visualizes image data with flexible options for channel selection,
        foreground filtering, and backend choice.

        Args:
            figsize (tuple[int, int] | None, optional): Figure size in inches
                (width, height). If None, uses default. Defaults to None.
            title (str | None, optional): Plot title. If None, auto-generated
                from image name and channel. Defaults to None.
            ax (plt.Axes | None, optional): Matplotlib Axes object. Only valid
                for matplotlib backend. If None, new axis created. Defaults to None.
            channel (int | None, optional): Specific channel index to plot.
                If None, all channels displayed. Defaults to None.
            foreground_only (bool, optional): If True, display only foreground.
                If False, show entire image. Defaults to False.
            mpl_settings (dict | None, optional): Matplotlib settings. Only used
                with matplotlib backend. Defaults to None.
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
        """
        arr = self[:] if not foreground_only else self.foreground()
        if channel is None:
            return self._plot(
                arr=arr,
                ax=ax,
                figsize=figsize,
                title=title,
                mpl_settings=mpl_settings,
                backend=backend,
                plotly_settings=plotly_settings,
            )

        else:
            title = (
                f"{self._root_image.name} - Channel {channel}"
                if title is None
                else f"{title} - Channel {channel}"
            )
            return self._plot(
                arr=arr[:, :, channel],
                ax=ax,
                figsize=figsize,
                title=title,
                mpl_settings=mpl_settings,
                backend=backend,
                plotly_settings=plotly_settings,
            )
