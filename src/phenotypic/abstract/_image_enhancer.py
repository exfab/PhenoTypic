from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import InterfaceError, DataIntegrityError, OperationFailedError
from phenotypic.util.constants_ import IMAGE_FORMATS


class ImageEnhancer(ImageOperation):
    def __init__(self):
        pass

    def apply(self, image: Image, inplace: bool = False) -> Image:
        try:
            # Make a copy for post checking
            imcopy = image.copy()

            # Reset the object map in case the inherited class forgets to do so
            image.objmap.reset()

            if inplace:
                output = self._operate(image)
            else:
                output = self._operate(image.copy())

            if image._image_format.is_array():
                if not np.array_equal(output.array[:], imcopy.array[:]):
                    raise DataIntegrityError(component='array', operation=self.__class__.__name__)

            if not np.array_equal(output.matrix[:], imcopy.matrix[:]):
                raise DataIntegrityError(component='matrix', operation=self.__class__.__name__)

            return output
        except Exception as e:
            raise OperationFailedError(operation=self.__class__.__name__,
                                       image_name=image.name,
                                       err_type=type(e),
                                       message=str(e)
                                       )

    def _operate(self, image: Image) -> Image:
        raise InterfaceError
