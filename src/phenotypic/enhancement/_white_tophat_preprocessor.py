import numpy as np
from skimage.morphology import disk, square, white_tophat, cube, ball, diamond

from ..abstract import ImageEnhancer
from .. import Image


class WhiteTophatEnhancer(ImageEnhancer):
    def __init__(self, shape:str='diamond', radius: int = None):
        self.shape = shape
        self.radius = radius

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
            image.enh_matrix[:],
            footprint=self._get_footprint(
                self._get_footprint_radius(detection_matrix=image.enh_matrix[:]),
            ),
        )
        image.enh_matrix[:] = image.enh_matrix[:] - white_tophat_results

        return image

    def _get_footprint_radius(self, detection_matrix: np.ndarray) -> int:
        if self.radius is None:
            return int(np.min(detection_matrix.shape) * 0.004)
        else:
            return self.radius

    def _get_footprint(self, radius: int) -> np.ndarray:
        match self.shape:
            case 'disk':
                return disk(radius=radius)
            case 'square':
                return square(width=radius * 2)
            case 'diamond':
                return diamond(radius=radius)
            case 'sphere':
                return ball(radius)
            case 'cube':
                return cube(radius * 2)
