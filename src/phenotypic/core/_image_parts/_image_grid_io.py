from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

from os import PathLike

try:
    import rawpy
except ImportError:
    rawpy = None
import phenotypic.abc_
from phenotypic.grid import AutoGridFinder
from ._image_grid_handler import ImageGridHandler
from phenotypic.tools.constants_ import IMAGE_FORMATS


class ImageGridIO(ImageGridHandler):
    @classmethod
    def imread(cls,
               filepath: PathLike,
               rawpy_params: dict | None = None,
               **kwargs) -> GridImage:
        image = super().imread(
                filepath=filepath,
                rawpy_params=rawpy_params,
        )

        import phenotypic as pt

        return pt.GridImage(image, **kwargs)
