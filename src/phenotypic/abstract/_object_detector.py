from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import OperationFailedError, DataIntegrityError, InterfaceError


# <<Interface>>
class ObjectDetector(ImageOperation):
    """ObjectDetectors are for detecting objects in an image. They change the image object mask and map."""

    def __init__(self):
        pass

    def apply(self, image: Image, inplace: bool = False) -> Image:
        try:
            imcopy = image.copy()

            if inplace:
                output = self._operate(image)
            else:
                output = self._operate(image.copy())

            # Post Operation Checks
            if not np.array_equal(imcopy.matrix[:], output.matrix[:]): raise DataIntegrityError(component='matrix',
                                                                                                operation=self.__class__.__name__,
                                                                                                image_name=image.name
                                                                                                )
            if not np.array_equal(imcopy.enh_matrix[:], output.enh_matrix[:]): raise DataIntegrityError(component='enh_matrix',
                                                                                                        operation=self.__class__.__name__,
                                                                                                        image_name=image.name
                                                                                                        )
            output.objmap.relabel()
            return output
        except DataIntegrityError as e:
            raise e
        except Exception as e:
            raise OperationFailedError(operation=self.__class__.__name__,
                                       image_name=image.name,
                                       err_type=type(e),
                                       message=str(e)
                                       )

    def _operate(self, image: Image) -> Image:
        raise InterfaceError
