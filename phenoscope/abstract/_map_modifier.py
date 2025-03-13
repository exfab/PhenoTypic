import numpy as np

from ._image_operation import ImageOperation

from .. import Image
from ..util.constants import INTERFACE_ERROR_MSG, C_ImageOperation, C_ImageFormats


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
            if output.schema in C_ImageFormats.MATRIX_FORMATS:
                if not np.array_equal(imcopy.array[:], output.array[:]): raise C_ImageOperation.IntegrityError(
                    component='array', operation=self.__class__.__name__, image_name=image.name
                )

            if not np.array_equal(imcopy.matrix[:], output.matrix[:]): raise C_ImageOperation.IntegrityError(
                component='matrix', operation=self.__class__.__name__, image_name=image.name
            )

            if not np.array_equal(imcopy.enh_matrix[:], output.enh_matrix[:]): raise C_ImageOperation.IntegrityError(
                component='enh_matrix', operation=self.__class__.__name__, image_name=image.name
            )
            return output

        except Exception as e:
            raise C_ImageOperation.OperationError(operation=self.__class__.__name__,
                                                  image_name=image.name,
                                                  err_type=type(e),
                                                  message=str(e)
                                                  )

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
