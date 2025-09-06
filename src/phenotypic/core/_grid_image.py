from ._image import Image

from typing import Optional
import numpy as np
from ._image_parts import ImageGridHandler

from os import PathLike
from pathlib import Path
from phenotypic.abstract import GridFinder


class GridImage(ImageGridHandler):
    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | Path | str | None = None,
                 imformat: str | None = None,
                 name: str | None = None,
                 grid_finder: Optional[GridFinder] = None,
                 nrows: int = 8, ncols: int = 12,
                 illuminant: str | None = 'D65',
                 color_profile='sRGB',
                 observer='CIE 1931 2 Degree Standard Observer',
                 ):
        super().__init__(
            input_image=input_image,
            imformat=imformat,
            name=name,
            grid_finder=grid_finder,
            nrows=nrows, ncols=ncols,
            illuminant=illuminant,
            color_profile=color_profile,
            observer=observer
        )


GridImage.__doc__ = ImageGridHandler.__doc__
