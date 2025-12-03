from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from phenotypic import Image

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
    """Access and analyze HSV (Hue, Saturation, Value) color space data from image objects.

    This accessor class provides comprehensive functionality for working with the HSV color
    space, which decomposes color information into three independent channels. HSV is
    particularly useful for color-based image analysis because it separates color (hue)
    from luminosity (value), making it well-suited for object segmentation and analysis
    based on color properties.

    The HSV color space is computed on-demand from the parent image's RGB data using
    scikit-image's rgb2hsv conversion, which assumes input RGB values in the range [0, 1]
    and produces HSV output where:
    - Hue (H) ranges from 0 to 1 (representing 0 to 360 degrees)
    - Saturation (S) ranges from 0 to 1 (0% to 100%)
    - Value (V) ranges from 0 to 1 (brightness from dark to bright)

    The class provides methods for:
    - Accessing individual HSV channels and entire HSV arrays
    - Generating histograms of color distributions with radial hue visualization
    - Displaying HSV components with color-mapped subplots
    - Viewing masked HSV data for segmented objects only
    - Saving and loading HSV arrays with embedded PhenoTypic metadata

    Attributes:
        _root_image (Image): The parent Image object containing RGB data and metadata.
        _accessor_property_name (str): The property name "color.hsv" for metadata tracking.

    Note:
        This accessor is only available for RGB images. For grayscale images,
        accessing HSV data raises an AttributeError.
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

        Examples:
            .. dropdown:: Loading an HSV array from a TIFF file

                >>> from phenotypic.core._image_parts.accessors import HsvAccessor
                >>> hsv_arr = HsvAccessor.load("my_hsv_image.tif")
        """
        filepath = Path(filepath)
        expected_property = f"Image.{cls._accessor_property_name}"

        if filepath.suffix.lower() not in IO.TIFF_EXTENSIONS:
            raise ValueError(
                "HSV arrays can only be loaded from TIFF format (.tif, .tiff). "
                f"File extension is: {filepath.suffix.lower()}"
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
                if "phenotypic_version" in data:
                    phenotypic_data = data
            except json.JSONDecodeError:
                pass

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

    @property
    def _subject_arr(self) -> np.ndarray:
        """Compute and return the HSV representation of the parent image's RGB data.

        This property computes the HSV color space representation on-demand from the
        parent image's RGB data. The conversion assumes RGB values in the range [0, 1]
        and produces an HSV array with shape (height, width, 3) where each channel
        contains normalized values in [0, 1].

        Returns:
            np.ndarray: HSV array with shape (H, W, 3) where:
                - [:, :, 0] contains hue values (0 to 1 representing 0-360 degrees)
                - [:, :, 1] contains saturation values (0 to 1)
                - [:, :, 2] contains value/brightness (0 to 1)

        Raises:
            AttributeError: If the parent image contains only grayscale data
                and has no RGB channel available.
        """
        if self._root_image.rgb.isempty():
            raise AttributeError("HSV is not available for grayscale images")
        else:
            return rgb2hsv(self._root_image.rgb[:])

    def __getitem__(self, key) -> np.ndarray:
        """Retrieve a subset of HSV data using NumPy-style indexing.

        Returns a read-only view of the HSV array with the specified indexing
        applied. This enables NumPy-like slicing operations (e.g., [:, :, 0]
        to extract hue channel, [10:50, 20:70] for spatial subsets).

        Args:
            key: NumPy-style index or slice. Examples:
                - `:, :, 0` to get hue channel
                - `10:50, 20:70` for spatial region
                - `[row, col]` for single pixel

        Returns:
            np.ndarray: Read-only view of the indexed HSV data.
        """
        view = self._subject_arr[key]
        view.flags.writeable = False
        return view

    def __setitem__(self, key, value):
        """Prevent direct assignment to HSV data.

        HSV data is computed on-demand from RGB data and cannot be directly modified.
        To change HSV properties, modify the underlying RGB data in the parent image.

        Args:
            key: Index specification (unused).
            value: Value to assign (unused).

        Raises:
            IllegalAssignmentError: Always raised to prevent data modification.
        """
        raise IllegalAssignmentError("HSV")

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Return the shape of the HSV image array.

        Returns the dimensions of the HSV array as a tuple (height, width, 3).
        The third dimension always has size 3, representing the three HSV channels.

        Returns:
            Optional[tuple[int, ...]]: A tuple (H, W, 3) representing image height,
                width, and number of channels. Returns None if the parent image
                is empty (has no RGB data).
        """
        return self._root_image._data.rgb.shape

    def copy(self) -> np.ndarray:
        """Create and return an independent copy of the HSV array.

        Returns a deep copy of the HSV data that can be safely modified without
        affecting the original parent image's HSV representation. This is useful
        for performing HSV-based operations that require array manipulation.

        Returns:
            np.ndarray: A copy of the HSV array with shape (H, W, 3) and dtype float64.
        """
        return self._subject_arr.copy()

    def histogram(
        self,
        figsize: Tuple[int, int] = (10, 5),
        linewidth=1,
        hue_bins: int = 1,
        hue_offset: float = 0.0,
    ):
        """Generate and display histograms for HSV channels with specialized hue visualization.

        Creates a comprehensive visualization with four subplots:
        1. Original RGB image (for reference)
        2. Hue histogram as a polar/radial plot with color-coded bins
        3. Saturation histogram as a standard line plot
        4. Brightness (Value) histogram as a standard line plot

        The hue histogram uses a polar coordinate system where each bin is colored
        according to its hue angle, providing intuitive color distribution visualization.
        Saturation and brightness use traditional Cartesian histograms.

        Args:
            figsize (Tuple[int, int]): Size of the figure in inches as (width, height).
                Defaults to (10, 5).
            linewidth (int, optional): Width of lines in saturation and brightness histograms.
                Defaults to 1.
            hue_bins (int, optional): Bin size for hue histogram in degrees. Each bin
                represents an angular range. Defaults to 1 (one degree per bin).
                Use larger values (e.g., 5, 10, 30) for coarser binning and faster
                rendering with large datasets.
            hue_offset (float, optional): Rotation offset to apply to all hue values
                in degrees. Useful for adjusting the starting angle of the polar plot.
                Defaults to 0.0.

        Returns:
            Tuple[plt.Figure, np.ndarray]: A tuple of:
                - plt.Figure: The Matplotlib figure containing all subplots
                - np.ndarray: Flattened array of Axes objects (axes[0] = image,
                  axes[1] = hue polar, axes[2] = saturation, axes[3] = brightness)

        Note:
            The radial hue histogram uses 0 degrees at the top (north) with clockwise
            direction. Angular gridlines appear every 30 degrees. Radial gridlines
            show histogram bin counts.
        """
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=figsize, subplot_kw={"projection": None}
        )
        axes_ = axes.ravel()

        # Original image
        axes_[0].imshow(self._root_image.rgb[:])
        axes_[0].set_title(self._root_image.name)
        axes_[0].grid(False)

        # Hue radial histogram
        axes_[1].remove()  # Remove the regular axes
        axes_[1] = fig.add_subplot(2, 2, 2, projection="polar")

        # Get hue data and apply offset
        hue_data = (self._subject_arr[:, :, 0] * 360 + hue_offset) % 360

        # Create bins
        bin_edges = np.arange(0, 360 + hue_bins, hue_bins)
        hist_counts, _ = np.histogram(hue_data.flatten(), bins=bin_edges)

        # Convert bin edges to radians and get bin centers
        bin_centers_deg = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_rad = np.deg2rad(bin_centers_deg)
        bin_width_rad = np.deg2rad(hue_bins)

        # Create colors for each bin based on hue value
        # Convert hue to HSV then to RGB for coloring
        colors = []
        for hue_deg in bin_centers_deg:
            # Create HSV color (hue/360, saturation=1, value=1)
            hsv_color = np.array([hue_deg / 360, 1.0, 1.0])
            # Convert to RGB
            rgb_color = mcolors.hsv_to_rgb(hsv_color)
            colors.append(rgb_color)

        # Create the radial histogram
        bars = axes_[1].bar(
            bin_centers_rad, hist_counts, width=bin_width_rad, color=colors, alpha=0.8
        )

        # Set radial gridlines for count values
        max_count = np.max(hist_counts) if len(hist_counts) > 0 else 1
        # Create 5 evenly spaced grid lines
        grid_values = np.linspace(0, max_count, 6)[1:]  # Exclude 0
        axes_[1].set_ylim(0, max_count)
        axes_[1].set_rticks(grid_values)
        axes_[1].set_rlabel_position(45)  # Position radial labels at 45 degrees

        # Set angular ticks for hue degrees
        axes_[1].set_theta_zero_location("N")  # 0 degrees at top
        axes_[1].set_theta_direction(-1)  # Clockwise
        axes_[1].set_thetagrids(np.arange(0, 360, 30))  # Every 30 degrees

        axes_[1].set_title("Hue (Radial)", pad=20)
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

    def show(
        self, figsize: Tuple[int, int] = (10, 8), title: str = None, shrink=0.2
    ) -> (plt.Figure, plt.Axes):
        """Display HSV channels as color-mapped images with colorbars.

        Creates a visualization of all three HSV channels stacked vertically, with
        appropriate colormaps and colorbars for each channel:
        - Hue: HSB colormap showing color wheel (0-360 degrees)
        - Saturation: Viridis colormap showing saturation level (0-1)
        - Brightness: Grayscale colormap showing luminosity (0-1)

        This is the primary visualization method for exploring HSV color distributions
        across the entire image without segmentation mask constraints.

        Args:
            figsize (Tuple[int, int]): Size of the figure in inches as (width, height).
                Defaults to (10, 8).
            title (str, optional): Title for the entire figure. If None, no title is set.
                Defaults to None.
            shrink (float, optional): Shrink factor for colorbar width relative to
                subplot size. Smaller values (e.g., 0.2) create narrower colorbars,
                larger values (e.g., 0.8) create wider ones. Defaults to 0.2.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple of:
                - plt.Figure: The Matplotlib figure object
                - np.ndarray: Array of three Axes objects for hue, saturation,
                  and brightness subplots respectively
        """
        fig, axes = plt.subplots(nrows=3, figsize=figsize)
        ax = axes.ravel()

        hue = ax[0].imshow(
            self._subject_arr[:, :, 0] * 360, cmap="hsb", vmin=0, vmax=360
        )
        ax[0].set_title("Hue")
        ax[0].grid(False)
        fig.colorbar(mappable=hue, ax=ax[0], shrink=shrink)

        saturation = ax[1].imshow(
            self._subject_arr[:, :, 1], cmap="viridis", vmin=0, vmax=1
        )
        ax[1].set_title("Saturation")
        ax[1].grid(False)
        fig.colorbar(mappable=saturation, ax=ax[1], shrink=shrink)

        brightness = ax[2].imshow(
            self._subject_arr[:, :, 2], cmap="gray", vmin=0, vmax=1
        )
        ax[2].set_title("Brightness")
        ax[2].grid(False)
        fig.colorbar(mappable=brightness, ax=ax[2], shrink=shrink)

        # Adjust ax settings
        if title is not None:
            ax.set_title(title)

        return fig, ax

    def show_objects(
        self, figsize: Tuple[int, int] = (10, 8), title: str = None, shrink=0.6
    ) -> (plt.Figure, plt.Axes):
        """Display HSV channels for segmented objects only, masked by object mask.

        Creates a visualization of all three HSV channels stacked vertically, with
        appropriate colormaps and colorbars for each channel. Unlike show(), this
        method applies the parent image's object mask, displaying HSV data only for
        pixels belonging to segmented objects. Background pixels are masked out.

        This method is useful for analyzing color properties of individual objects
        (e.g., colonies on an agar plate) without interference from background variation.

        Args:
            figsize (Tuple[int, int]): Size of the figure in inches as (width, height).
                Defaults to (10, 8).
            title (str, optional): Title for the entire figure. If None, no title is set.
                Defaults to None.
            shrink (float, optional): Shrink factor for colorbar width relative to
                subplot size. Smaller values (e.g., 0.2) create narrower colorbars,
                larger values (e.g., 0.8) create wider ones. Defaults to 0.6.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple of:
                - plt.Figure: The Matplotlib figure object
                - np.ndarray: Array of three Axes objects for masked hue, saturation,
                  and brightness subplots respectively

        Note:
            Uses np.ma.array (masked arrays) to suppress visualization of background
            pixels. Only pixels where the object mask is non-zero are displayed.
        """
        fig, axes = plt.subplots(nrows=3, figsize=figsize)
        ax = axes.ravel()

        hue = ax[0].imshow(
            np.ma.array(
                self._subject_arr[:, :, 0] * 360, mask=~self._root_image.objmask[:]
            ),
            cmap="hsb",
            vmin=0,
            vmax=360,
        )
        ax[0].set_title("Hue")
        ax[0].grid(False)
        fig.colorbar(mappable=hue, ax=ax[0], shrink=shrink)

        saturation = ax[1].imshow(
            np.ma.array(self._subject_arr[:, :, 1], mask=~self._root_image.objmask[:]),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax[1].set_title("Saturation")
        ax[1].grid(False)
        fig.colorbar(mappable=saturation, ax=ax[1], shrink=shrink)

        brightness = ax[2].imshow(
            np.ma.array(self._subject_arr[:, :, 2], mask=~self._root_image.objmask[:]),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax[2].set_title("Brightness")
        ax[2].grid(False)
        fig.colorbar(mappable=brightness, ax=ax[2], shrink=shrink)

        # Adjust ax settings
        if title is not None:
            ax.set_title(title)

        return fig, ax

    def imsave(self, filepath: str | os.PathLike | Path) -> None:
        """Save HSV array to file with PhenoTypic metadata embedded.

        HSV arrays are saved exclusively in TIFF format because their floating-point
        values (range 0.0-1.0) require lossless compression. The method computes HSV
        on-demand from RGB data, converts to float32 if needed for compatibility, and
        embeds PhenoTypic metadata (version, source property, image metadata) in the
        TIFF ImageDescription tag for later verification via load().

        Args:
            filepath (str | os.PathLike | Path): Path for the output TIFF file.
                The file extension must be .tif or .tiff (case-insensitive).

        Raises:
            ValueError: If file extension is not .tif or .tiff. The error message
                specifies the invalid extension that was provided.

        Note:
            - Uses zlib compression for efficient storage of floating-point data
            - Automatically converts float64 arrays to float32 if necessary
            - Creates ImageDescription TIFF tag with JSON-formatted metadata
            - The TIFF photometric interpretation is set to 'minisblack'

        Examples:
            .. dropdown:: Saving HSV data and verifying metadata

                >>> image.color.hsv.imsave("output_hsv.tif")
                >>> loaded_hsv = HsvAccessor.load("output_hsv.tif")  # Verify metadata
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() not in IO.TIFF_EXTENSIONS:
            raise ValueError(
                "HSV arrays can only be saved in TIFF format (.tif, .tiff). "
                f"File extension is: {filepath.suffix.lower()}"
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
            compression="zlib",
            photometric="minisblack",
        )
