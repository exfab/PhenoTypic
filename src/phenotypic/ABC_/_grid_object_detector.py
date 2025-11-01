from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

from phenotypic.ABC_ import ObjectDetector, GridOperation
from phenotypic.tools.funcs_ import validate_operation_integrity
from phenotypic.tools.exceptions_ import GridImageInputError
from abc import ABC


class GridObjectDetector(ObjectDetector, GridOperation, ABC):
    """GridObjectDetectors are a type of ObjectDetector that use a grid to detect objects in an image. They change the image object mask and map."""

    @validate_operation_integrity('image.array', 'image.matrix', 'image.enh_matrix')
    def apply(self, image: GridImage, inplace=False) -> GridImage:
        from phenotypic import GridImage

        if not isinstance(image, GridImage): raise GridImageInputError
        return super().apply(image=image, inplace=inplace)

    def _operate(self, image: GridImage) -> GridImage:
        return image
