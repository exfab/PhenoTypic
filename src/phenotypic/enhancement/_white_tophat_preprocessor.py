import numpy as np
from skimage.morphology import disk, square, white_tophat, cube, ball

from ..abstract import ImageEnhancer
from .. import Image


class WhiteTophatEnhancer(ImageEnhancer):
    def __init__(self, footprint_shape='disk', footprint_radius: int = None):
        self._footprint_shape = footprint_shape
        self._footprint_radius = footprint_radius

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
            image.enh_matrix[:],
                footprint=self._get_footprint(
                        self._get_footprint_radius(detection_matrix=image.enh_matrix[:])
                )
        )
        image.enh_matrix[:] = image.enh_matrix[:] - white_tophat_results

        return image

    def _get_footprint_radius(self, detection_matrix: np.ndarray) -> int:
        if self._footprint_radius is None:
            return int(np.min(detection_matrix.shape) * 0.004)
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
