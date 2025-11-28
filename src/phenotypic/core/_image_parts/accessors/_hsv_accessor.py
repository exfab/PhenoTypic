from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import tifffile
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from skimage.exposure import histogram
import skimage.io

import phenotypic
from phenotypic.core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic.tools.constants_ import IMAGE_MODE, IO
from phenotypic.tools.exceptions_ import IllegalAssignmentError


class HsvAccessor(ImageAccessorBase):
    """An accessor class to handle and analyze HSB (Hue, Saturation, Brightness) image data efficiently.

    This class provides functionality for accessing and processing HSV image data.
    Users can retrieve components (hue, saturation, brightness) of the image, generate
    visual histograms of color distributions, and measure specific object properties
    masked within the HSV image.

    Extensive visualization methods are also included, allowing display of HSV components
    and their masked variations. This class is ideal for image analysis tasks where color
    properties play a significant role.

    Attributes:
        image (Image): The parent Image object that manages image data and operations.
    """

    _accessor_property_name: str = "color.hsv"

    @classmethod
    def load(cls, filepath: str | os.PathLike | Path) -> np.ndarray:
        """Load an HSV array from a TIFF file and verify it was saved from this accessor type.

        HSV arrays are stored as float32 TIFF files. This method checks if the
        image contains PhenoTypic metadata indicating it was saved from the HSV
        accessor. If metadata doesn't match or is missing, a warning is raised
        but the array is still loaded.

        Args:
            filepath: Path to the TIFF file to load.

        Returns:
            np.ndarray: The loaded HSV array (float32) with shape (H, W, 3).

        Raises:
            ValueError: If file extension is not .tif or .tiff.

        Warns:
            UserWarning: If metadata is missing or indicates the image was saved
                from a different accessor type.

        Example:
            >>> from phenotypic.core._image_parts.accessors import HsvAccessor
            >>> hsv_arr = HsvAccessor.load("my_hsv_image.tif")
        """
        filepath = Path(filepath)
        expected_property = f"Image.{cls._accessor_property_name}"

        if filepath.suffix.lower() not in IO.TIFF_EXTENSIONS:
            raise ValueError(
                'HSV arrays can only be loaded from TIFF format (.tif, .tiff). '
                f'File extension is: {filepath.suffix.lower()}'
            )

        # Load using tifffile for float array support
        with tifffile.TiffFile(filepath) as tif:
            arr = tif.asarray()
            desc = tif.pages[0].description if tif.pages else None

        # Check metadata
        phenotypic_data = None
        if desc:
            try:
                data = json.loads(desc)
                if 'phenotypic_version' in data:
                    phenotypic_data = data
            except json.JSONDecodeError:
                pass

        if phenotypic_data is None:
            warnings.warn(
                f"No PhenoTypic metadata found in '{filepath.name}'. "
                f"Cannot verify this image was saved from {expected_property}. "
                "Loading anyway, but this may lead to undefined behavior.",
                UserWarning
            )
        else:
            saved_property = phenotypic_data.get('phenotypic_image_property', 'unknown')
            if saved_property != expected_property:
                warnings.warn(
                    f"Metadata mismatch: Image was saved from '{saved_property}' "
                    f"but being loaded as '{expected_property}'. "
                    "This may lead to undefined behavior.",
                    UserWarning
                )

        return arr

    @property
    def _subject_arr(self) -> np.ndarray:
        if self._root_image.rgb.isempty():
            raise AttributeError('HSV is not available for grayscale images')
        else:
            return rgb2hsv(self._root_image.rgb[:])

    def __getitem__(self, key) -> np.ndarray:
        view = self._subject_arr[key]
        view.flags.writeable = False
        return view

    def __setitem__(self, key, value):
        raise IllegalAssignmentError('HSV')

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Returns the shape of the image"""
        return self._root_image._data.rgb.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the image array"""
        return self._subject_arr.copy()

    def histogram(self, figsize: Tuple[int, int] = (10, 5), linewidth=1,
                  hue_bins: int = 1, hue_offset: float = 0.0):
        """
        Generates and displays histograms for hue, saturation, and brightness components of an image,
        alongside the original image. The hue histogram is displayed as a radial plot with colored bins,
        while saturation and brightness histograms remain as traditional line plots.

        Args:
            figsize (Tuple[int, int]): The size of the figure that contains all subplots, specified as
                a tuple of width and height in inches.
            linewidth (int): The width of the lines used in the histograms.
            hue_bins (int): The bin size for the hue histogram in degrees. Default is 1 degree.
            hue_offset (float): Offset to apply to hue values in degrees. Default is 0.0.

        Returns:
            Tuple[Figure, ndarray]: A tuple containing the Matplotlib figure object and an ndarray
                of axes, where the axes correspond to the subplots.
        """
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                                 subplot_kw={'projection': None})
        axes_ = axes.ravel()

        # Original image
        axes_[0].imshow(self._root_image.rgb[:])
        axes_[0].set_title(self._root_image.name)
        axes_[0].grid(False)

        # Hue radial histogram
        axes_[1].remove()  # Remove the regular axes
        axes_[1] = fig.add_subplot(2, 2, 2, projection='polar')

        # Get hue data and apply offset
        hue_data = (self._subject_arr[:, :, 0]*360 + hue_offset)%360

        # Create bins
        bin_edges = np.arange(0, 360 + hue_bins, hue_bins)
        hist_counts, _ = np.histogram(hue_data.flatten(), bins=bin_edges)

        # Convert bin edges to radians and get bin centers
        bin_centers_deg = (bin_edges[:-1] + bin_edges[1:])/2
        bin_centers_rad = np.deg2rad(bin_centers_deg)
        bin_width_rad = np.deg2rad(hue_bins)

        # Create colors for each bin based on hue value
        # Convert hue to HSV then to RGB for coloring
        colors = []
        for hue_deg in bin_centers_deg:
            # Create HSV color (hue/360, saturation=1, value=1)
            hsv_color = np.array([hue_deg/360, 1.0, 1.0])
            # Convert to RGB
            rgb_color = mcolors.hsv_to_rgb(hsv_color)
            colors.append(rgb_color)

        # Create the radial histogram
        bars = axes_[1].bar(bin_centers_rad, hist_counts,
                            width=bin_width_rad, color=colors, alpha=0.8)

        # Set radial gridlines for count values
        max_count = np.max(hist_counts) if len(hist_counts) > 0 else 1
        # Create 5 evenly spaced grid lines
        grid_values = np.linspace(0, max_count, 6)[1:]  # Exclude 0
        axes_[1].set_ylim(0, max_count)
        axes_[1].set_rticks(grid_values)
        axes_[1].set_rlabel_position(45)  # Position radial labels at 45 degrees

        # Set angular ticks for hue degrees
        axes_[1].set_theta_zero_location('N')  # 0 degrees at top
        axes_[1].set_theta_direction(-1)  # Clockwise
        axes_[1].set_thetagrids(np.arange(0, 360, 30))  # Every 30 degrees

        axes_[1].set_title('Hue (Radial)', pad=20)
        axes_[1].grid(True, alpha=0.3)

        # Saturation histogram (unchanged)
        hist_two, histc_two = histogram(self._subject_arr[:, :, 1])
        axes_[2].plot(histc_two, hist_two, lw=linewidth)
        axes_[2].set_title("Saturation")

        # Brightness histogram (unchanged)
        hist_three, histc_three = histogram(self._subject_arr[:, :, 2])
        axes_[3].plot(histc_three, hist_three, lw=linewidth)
        axes_[3].set_title("Brightness")

        return fig, axes

    def show(self, figsize: Tuple[int, int] = (10, 8),
             title: str = None, shrink=0.2) -> (plt.Figure, plt.Axes):
        """
        Displays the Hue, Saturation, and Brightness (HSV components) of the given
        image data in a visualization using subplots. Each subplot corresponds to
        one of the HSV channels, and color bars are included to help interpret the
        values.

        A color map is used for better visual distinction, with 'hsb' for Hue,
        'viridis' for Saturation, and grayscale for Brightness. Provides an optional
        title for the entire figure and flexibility in the sizing and shrink factor
        of color bars.

        Args:
            figsize (Tuple[int, int]): Size of the figure in inches as a (width, height)
                tuple. Defaults to (10, 8).
            title (str): Title of the entire figure. If None, no title will be set.
            shrink (float): Shrink factor for the color bar size displayed next to
                subplots. Defaults to 0.6.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the created figure and axes
            for further customization or display.
        """
        fig, axes = plt.subplots(nrows=3, figsize=figsize)
        ax = axes.ravel()

        hue = ax[0].imshow(self._subject_arr[:, :, 0]*360, cmap='hsb', vmin=0, vmax=360)
        ax[0].set_title('Hue')
        ax[0].grid(False)
        fig.colorbar(mappable=hue, ax=ax[0], shrink=shrink)

        saturation = ax[1].imshow(self._subject_arr[:, :, 1], cmap='viridis', vmin=0, vmax=1)
        ax[1].set_title('Saturation')
        ax[1].grid(False)
        fig.colorbar(mappable=saturation, ax=ax[1], shrink=shrink)

        brightness = ax[2].imshow(self._subject_arr[:, :, 2], cmap='gray', vmin=0, vmax=1)
        ax[2].set_title('Brightness')
        ax[2].grid(False)
        fig.colorbar(mappable=brightness, ax=ax[2], shrink=shrink)

        # Adjust ax settings
        if title is not None: ax.set_title(title)

        return fig, ax

    def show_objects(self, figsize: Tuple[int, int] = (10, 8),
                     title: str = None, shrink=0.6) -> (plt.Figure, plt.Axes):
        """
        Displays the Hue, Saturation, and Brightness (HSV components) of the given
        image data in a visualization using subplots. Each subplot corresponds to
        one of the HSV channels, and color bars are included to help interpret the
        values.

        A color map is used for better visual distinction, with 'hsb' for Hue,
        'viridis' for Saturation, and grayscale for Brightness. Provides an optional
        title for the entire figure and flexibility in the sizing and shrink factor
        of color bars.

        Args:
            figsize (Tuple[int, int]): Size of the figure in inches as a (width, height)
                tuple. Defaults to (10, 8).
            title (str): Title of the entire figure. If None, no title will be set.
            shrink (float): Shrink factor for the color bar size displayed next to
                subplots. Defaults to 0.6.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the created figure and axes
            for further customization or display.
        """
        fig, axes = plt.subplots(nrows=3, figsize=figsize)
        ax = axes.ravel()

        hue = ax[0].imshow(np.ma.array(self._subject_arr[:, :, 0]*360, mask=~self._root_image.objmask[:]),
                           cmap='hsb', vmin=0, vmax=360,
                           )
        ax[0].set_title('Hue')
        ax[0].grid(False)
        fig.colorbar(mappable=hue, ax=ax[0], shrink=shrink)

        saturation = ax[1].imshow(np.ma.array(self._subject_arr[:, :, 1], mask=~self._root_image.objmask[:]),
                                  cmap='viridis', vmin=0, vmax=1,
                                  )
        ax[1].set_title('Saturation')
        ax[1].grid(False)
        fig.colorbar(mappable=saturation, ax=ax[1], shrink=shrink)

        brightness = ax[2].imshow(np.ma.array(self._subject_arr[:, :, 2], mask=~self._root_image.objmask[:]),
                                  cmap='gray', vmin=0, vmax=1,
                                  )
        ax[2].set_title('Brightness')
        ax[2].grid(False)
        fig.colorbar(mappable=brightness, ax=ax[2], shrink=shrink)

        # Adjust ax settings
        if title is not None: ax.set_title(title)

        return fig, ax

    def imsave(self, filepath: str | os.PathLike | Path) -> None:
        """Save HSV color space data to TIFF file with PhenoTypic metadata embedded.

        HSV arrays can only be saved in TIFF format due to their floating-point nature.
        Metadata is embedded in the ImageDescription tag.

        Args:
            filepath: Path to save the TIFF file.

        Raises:
            ValueError: If file extension is not .tif or .tiff.
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() not in IO.TIFF_EXTENSIONS:
            raise ValueError(
                'HSV arrays can only be saved in TIFF format (.tif, .tiff). '
                f'File extension is: {filepath.suffix.lower()}'
            )

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=False)

        # Get array and ensure it's float32 for TIFF compatibility
        arr = self._subject_arr
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        # Use tifffile directly for float array support
        tifffile.imwrite(
            filepath,
            arr,
            description=metadata_json,
            compression='zlib',
            photometric='minisblack'
        )
