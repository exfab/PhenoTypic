from skimage.filters import threshold_triangle

from ..interface import ThresholdDetector
from .. import Image


class TriangleDetector(ThresholdDetector):
    def _operate(self, image: Image) -> Image:
        image.object_mask = image.enhanced_array >= threshold_triangle(image.enhanced_array)
        return image
