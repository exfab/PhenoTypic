from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from ._image_operation import ImageOperation
from ..util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import OperationFailedError, InterfaceError, DataIntegrityError


# <<Interface>>
class MapModifier(ImageOperation):
    """Map modifiers edit the object map and are used for removing, combining, and re-ordering objects."""

    def apply(self, image: Image, inplace: bool = False) -> Image:
        try:
            imcopy = image.copy()

            if inplace:
                output = self._operate(image)
            else:
                output = self._operate(image.copy())

            # TODO: Fix this check
            if output._image_format.is_array():
                if not np.array_equal(imcopy.array[:], output.array[:]): raise DataIntegrityError(
                    component='array', operation=self.__class__.__name__, image_name=image.name
                )

            # TODO: Fix this check
            # if not np.array_equal(imcopy.matrix[:], output.matrix[:]): raise DataIntegrityError(
            #     component='matrix', operation=self.__class__.__name__, image_name=image.name
            # )

            if not np.array_equal(imcopy.enh_matrix[:], output.enh_matrix[:]): raise DataIntegrityError(
                component='enh_matrix', operation=self.__class__.__name__, image_name=image.name
            )
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
