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
               grid_finder: phenotypic.abstract.GridFinder | None = None,
               nrows: int = None,
               ncols: int = None,
               *,
               gamma: Tuple[int, int] = None,
               demosaic_algorithm: rawpy.DemosaicAlgorithm = None,
               use_camera_wb: bool = False,
               median_filter_passes: int = 0,
               rawpy_params: dict | None = None,
               **kwargs) -> GridImage:
        image = super().imread(
                filepath=filepath,
                gamma=gamma,
                demosaic_algorithm=demosaic_algorithm,
                use_camera_wb=use_camera_wb,
                median_filter_passes=median_filter_passes,
                rawpy_params=rawpy_params,
                **kwargs,
        )
        if grid_finder is None:
            grid_finder = AutoGridFinder(
                    nrows=nrows if nrows else 8,
                    ncols=ncols if ncols else 12,
            )
        import phenotypic as pt

        return pt.GridImage(image, grid_finder=grid_finder)
