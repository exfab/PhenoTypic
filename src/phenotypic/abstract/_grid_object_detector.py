from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

from phenotypic.abstract import ObjectDetector, GridOperation
from phenotypic.util.funcs_ import validate_operation_integrity
from phenotypic.util.exceptions_ import GridImageInputError


class GridObjectDetector(ObjectDetector, GridOperation):
    """GridObjectDetectors are a type of ObjectDetector that use a grid to detect objects in an image. They change the image object mask and map."""

    @validate_operation_integrity('image.array', 'image.matrix', 'image.enh_matrix')
    def apply(self, image: GridImage, inplace=False) -> GridImage:
        from phenotypic import GridImage
        if not isinstance(image, GridImage): raise GridImageInputError
        return super().apply(image=image, inplace=inplace)

    def _operate(self, image: GridImage) -> GridImage:
        return image
