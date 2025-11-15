from os import PathLike
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from phenotypic.abc_ import GridFinder
from ._image import Image
from ._image_parts._image_grid_handler import ImageGridHandler


class GridImage(ImageGridHandler):
    """
    A specialized Image object that supports grid-based processing and overlay visualization.

    This class extends the base `Image` class functionality to include grid handling,
    grid-based slicing, and advanced visualization capabilities such as displaying overlay information
    with gridlines and annotations. It interacts with the provided grid handling utilities
    to determine grid structure and assign/overlay it effectively on the image.

    Args:
            arr (Optional[Union[np.ndarray, Type[Image]]]): The input_image
                image, which can be a NumPy rgb or an image-like object. If
                this parameter is not provided, it defaults to None.
            grid_finder (Optional[GridFinder]): An optional GridFinder instance
                for defining grids on the image. If not provided, it defaults to
                a center grid setter.
            nrows (int): An integer passed to the grid setter to specify the number of nrows in the grid
                (Defaults to 8).
            ncols (int): An integer passed to the grid setter to specify the number of columns in the grid
                (Defaults to 12).

    """

    def __init__(self,
                 arr: np.ndarray | Image | PathLike | Path | str | None = None,
                 name: str | None = None,
                 grid_finder: Optional[GridFinder] = None,
                 nrows: int = 8, ncols: int = 12,
                 bit_depth: Literal[8, 16] | None = None,
                 illuminant: str | None = 'D65',
                 gamma_encoding='sRGB',
                 observer='CIE 1931 2 Degree Standard Observer',
                 ):
        super().__init__(
                arr=arr,
                name=name,
                grid_finder=grid_finder,
                nrows=nrows, ncols=ncols,
                bit_depth=bit_depth,
                illuminant=illuminant,
                gamma_encoding=gamma_encoding,
                observer=observer,
        )
