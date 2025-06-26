from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Literal

from ...util.exceptions_ import NoObjectsError, IllegalAssignmentError

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from ..accessors import ObjectsAccessor
from ._image_handler import ImageHandler


class ImageObjectsHandler(ImageHandler):
    """Adds the ability to isolate and work with specific objects from an image."""

    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | None = None,
                 imformat: str | None = None,
                 name: str | None = None, bit_depth: Literal[8, 16, 32] = 16):
        super().__init__(input_image=input_image, imformat=imformat, name=name, bit_depth=bit_depth)
        self._accessors.objects = ObjectsAccessor(self)

    @property
    def objects(self) -> ObjectsAccessor:
        """Returns an acessor to the objects in an _root_image and perform operations on them, such as measurement calculations.

        This method provides access to `ImageObjects`.

        Returns:
            ObjectsAccessor: The subhandler instance that manages _root_image-related objects.

        Raises:
            NoObjectsError: If no objects are targeted in the _root_image. Apply an ObjectDetector first.
        """
        if self.num_objects == 0:
            raise NoObjectsError(self.name)
        else:
            return self._accessors.objects

    @objects.setter
    def objects(self, objects):
        raise IllegalAssignmentError('objects')