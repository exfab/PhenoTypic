from ._object_detector import ObjectDetector
from abc import ABC


# <<Interface>>
class ThresholdDetector(ObjectDetector, ABC):
    """ThresholdDetectors are a type of ObjectDetector that use a threshold, such as Otsu's threshold, to detect objects in an image. They change the image object mask and map."""
    pass
