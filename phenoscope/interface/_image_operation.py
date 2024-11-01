from ..util.error_message import INTERFACE_ERROR_MSG

from .. import Image

class ImageOperation:

    def _operate(self, image:Image)->Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)