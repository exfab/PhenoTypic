import numpy as np
from skimage.morphology import disk, square, white_tophat, cube, ball

from ..interface import ImagePreprocessor
from .. import Image


class WhiteTophatPreprocessor(ImagePreprocessor):
    def __init__(self, footprint_shape='disk', footprint_radius: int = None):
        self._footprint_shape = footprint_shape
        self._footprint_radius = footprint_radius

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
                image.enhanced_array,
                footprint=self._get_footprint(
                        self._get_footprint_radius(array=image.enhanced_array)
                )
        )
        image.enhanced_array = image.enhanced_array - white_tophat_results

        return image

    def _get_footprint_radius(self, array: np.ndarray) -> int:
        if self._footprint_radius is None:
            return int(np.min(array.shape) * 0.004)
        else:
            return self._footprint_radius

    def _get_footprint(self, radius: int) -> np.ndarray:
        match self._footprint_shape:
            case 'disk':
                return disk(radius=radius)
            case 'square':
                return square(radius * 2)
            case 'sphere':
                return ball(radius)
            case 'cube':
                return cube(radius * 2)
