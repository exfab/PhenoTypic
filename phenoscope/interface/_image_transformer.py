from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG
from .. import Image


class ImageTransformer(ImageOperation):
    def __init__(self):
        pass

    def transform(self, image: Image, inplace=False) -> Image:
        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())
        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
