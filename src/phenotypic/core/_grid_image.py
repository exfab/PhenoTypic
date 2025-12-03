from os import PathLike
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from phenotypic.abc_ import GridFinder
from ._image import Image
from ._image_parts._image_grid_handler import ImageGridHandler


class GridImage(ImageGridHandler):
    """A specialized Image object that supports grid-based processing and overlay visualization.

    This class extends ImageGridHandler to provide an intuitive interface for analyzing
    arrayed samples on agar plates or other gridded microbe cultures. It combines complete
    image processing capabilities with grid-aware operations, enabling well-level detection,
    measurement, and visualization with grid overlays. This is useful for high-throughput
    phenotyping workflows where colonies are arranged in regular arrays (e.g., 96-well or
    384-well plates).

    The class automatically manages grid detection and alignment, supports grid-based
    slicing to extract individual well images, and provides overlay visualizations with
    gridlines, well labels, and measurements aligned to the detected grid structure.

    Attributes:
        grid_finder (Optional[GridFinder]): Object responsible for detecting and optimizing
            the grid layout. If None, an AutoGridFinder is created with specified nrows/ncols.
        nrows (int): Number of rows in the grid structure. Default 8 (standard for 96-well).
        ncols (int): Number of columns in the grid structure. Default 12 (standard for 96-well).
    """

    def __init__(
        self,
        arr: np.ndarray | Image | PathLike | Path | str | None = None,
        name: str | None = None,
        grid_finder: Optional[GridFinder] = None,
        nrows: int = 8,
        ncols: int = 12,
        bit_depth: Literal[8, 16] | None = None,
        illuminant: str | None = "D65",
        gamma_encoding: Literal["sRGB"] | None = "sRGB",
    ):
        """Initialize a GridImage with grid-based processing capabilities.

        Creates a new GridImage instance with support for grid detection, well-level
        analysis, and grid-aligned visualization. Inherits all image processing capabilities
        from the parent hierarchy while adding grid-specific features.

        Args:
            arr (np.ndarray | Image | PathLike | Path | str | None): Initial image data.
                Can be a NumPy array (2-D grayscale or 3-D RGB), an Image instance,
                a file path string, or None for an empty image. Defaults to None.
            name (str | None): Human-readable name for the image. If None, uses the image UUID.
                Defaults to None.
            grid_finder (Optional[GridFinder]): Custom grid detection algorithm. If None,
                an AutoGridFinder is instantiated with the specified nrows and ncols.
                Defaults to None.
            nrows (int): Number of rows in the grid structure. Used only if grid_finder
                is None. Typical values: 8 (96-well), 16 (384-well), 32 (1536-well).
                Defaults to 8.
            ncols (int): Number of columns in the grid structure. Used only if grid_finder
                is None. Typical values: 12 (96-well), 24 (384-well), 48 (1536-well).
                Defaults to 12.
            bit_depth (Literal[8, 16] | None): Bit depth of the image (8 or 16 bits).
                If None, automatically inferred from arr dtype. Defaults to None.
            illuminant (str | None): Reference illuminant for color calculations.
                'D65' (standard daylight) or 'D50' (imaging illuminant).
                Defaults to 'D65'.
            gamma_encoding (Literal["sRGB"] | None): Gamma encoding for color correction.
                'sRGB' for gamma-corrected images, None for linear RGB.
                Defaults to 'sRGB'.

        Raises:
            ValueError: If illuminant is not 'D65' or 'D50'.
            ValueError: If gamma_encoding is not 'sRGB' or None.
            TypeError: If arr is provided but is not a valid image type.

        Examples:
            .. dropdown:: Create from a plate image file

                .. code-block:: python

                    from phenotypic import GridImage

                    # Load plate image with 96-well grid (8 rows x 12 cols)
                    grid_img = GridImage('plate_scan.jpg', nrows=8, ncols=12)
                    grid_img.show_overlay(show_gridlines=True)

            .. dropdown:: Create with custom grid finder

                .. code-block:: python

                    from phenotypic import GridImage
                    from phenotypic.grid import AutoGridFinder

                    finder = AutoGridFinder(nrows=16, ncols=24)  # 384-well plate
                    grid_img = GridImage('plate_384.jpg', grid_finder=finder)
                    print(grid_img.nrows, grid_img.ncols)  # Output: 16 24
        """
        super().__init__(
            arr=arr,
            name=name,
            grid_finder=grid_finder,
            nrows=nrows,
            ncols=ncols,
            bit_depth=bit_depth,
            illuminant=illuminant,
            gamma_encoding=gamma_encoding,
        )
