import json
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from PIL import Image as PIL_Image
from PIL import PngImagePlugin

import phenotypic
from phenotypic.core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic.tools.constants_ import METADATA, IO


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
                    raise AttributeError(f"Unsupported bit depth: {self._root_image.metadata[METADATA.BIT_DEPTH]}")

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=False)

        suffix = fname.suffix.lower()

        if suffix in IO.JPEG_FILE_EXTENSIONS:
            # Convert 16-bit to 8-bit for JPEG
            if arr.dtype == np.uint16:
                warnings.warn('Saving 16-bit RGB as JPEG will result in information loss')
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

    def show(self,
             figsize: tuple[int, int] | None = None,
             title: str | None = None,
             ax: plt.Axes | None = None,
             channel: int | None = None,
             foreground_only: bool = False,
             *,
             mpl_settings: dict | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays the image data, with the option to customize its visualization
        and plot settings.

        Args:
            figsize (tuple[int, int] | None): Size of the figure in inches (width, height).
                If None, a default size is used.
            title (str | None): Title of the plot. If None, a default title is
                generated based on the image and channel.
            ax (plt.Axes | None): Matplotlib Axes object. If provided, the image
                is plotted on this axis. If None, a new axis is created.
            channel (int | None): Specific channel index to plot. If None, all
                channels in the image are displayed.
            foreground_only (bool): If True, only the foreground portion of the
                image is displayed. If False, the entire image is shown.
            mpl_settings (dict | None): Optional Matplotlib settings. Allows
                customization of plot parameters.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib Figure
            and Axes objects used for plotting the image.
        """
        arr = self[:] if not foreground_only else self.foreground()
        if channel is None:
            return self._plot(arr=arr, ax=ax, figsize=figsize, title=title, mpl_settings=mpl_settings)

        else:
            title = f"{self._root_image.name} - Channel {channel}" if title is None else f'{title} - Channel {channel}'
            return self._plot(arr=arr[:, :, channel], ax=ax, figsize=figsize, title=title, mpl_settings=mpl_settings)
