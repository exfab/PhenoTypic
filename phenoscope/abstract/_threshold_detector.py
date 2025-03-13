from ._object_detector import ObjectDetector

from .. import Image
from ..util.constants import INTERFACE_ERROR_MSG


# <<Interface>>
class ThresholdDetector(ObjectDetector):
    """ThresholdDetectors are a type of ObjectDetector that use a threshold, such as Otsu's threshold, to detect objects in an image. They change the image object mask and map."""
    pass
