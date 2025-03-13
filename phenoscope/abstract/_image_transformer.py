from typing import Union, Dict

from ._image_operation import ImageOperation
from ..util.constants import INTERFACE_ERROR_MSG, C_ImageOperation
from .. import Image


class ImageTransformer(ImageOperation):
    """ImageTransformers are for general operations that alter every image component such as rotating. These have no integrity checks.

    """

    def __init__(self):
        pass

    def apply(self, image: Image, inplace=False) -> Union[Image, Dict[str, Image]]:
        try:
            if inplace:
                output = self._operate(image)
            else:
                output = self._operate(image.copy())
            return output
        except Exception as e:
            raise C_ImageOperation.OperationError(operation=self.__class__.__name__,
                                                  image_name=image.name,
                                                  err_type=type(e),
                                                  message=str(e)
                                                  )

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
