from ..util.constants import INTERFACE_ERROR_MSG

from .. import Image
from ._docstring_metaclass import ApplyDocstringMeta

class ImageOperation(metaclass=ApplyDocstringMeta):
    """Represents an abstract base class for image operations.

    This class provides a common abstract for applying transformations or
    operations to images. It defines a method to apply the operation and
    enforces the implementation of the specific operation in a subclass.
    Users can apply operations either in-place or on a copy of the image.

    Methods:
        apply(image, inplace=False): Applies the operation to the given image.
        _operate(image): Abstract method for the specific operation implementation
                         that must be implemented in a subclass.
    """
    def apply(self, image, inplace=False) -> Image:
        if inplace:
            return self._operate(image)
        else:
            return self._operate(image.copy())

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
