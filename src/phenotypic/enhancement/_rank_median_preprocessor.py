import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk, square, cube, ball
from skimage.util import img_as_ubyte

from .. import Image
from ..abstract import ImageEnhancer


class RankMedianEnhancer(ImageEnhancer):
    def __init__(self, footprint_shape: str = 'square', footprint_radius: int = None, shift_x=0, shift_y=0):
        if footprint_shape not in ['disk', 'square', 'sphere', 'cube']:
            raise ValueError(f'footprint shape {footprint_shape} is not supported')

        self.footprint_shape = footprint_shape
        self.footprint_radius = footprint_radius
        self.shift_x = shift_x
        self.shift_y = shift_y

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = median(
                image=img_as_ubyte(image.enh_matrix[:]),
                footprint=self._get_footprint(self._get_footprint_radius(image.enh_matrix[:]))
        )
        return image

    def _get_footprint_radius(self, det_matrix: np.ndarray) -> int:
        if self.footprint_radius is None:
            return int(np.min(det_matrix.shape) * 0.002)
        else:
            return self.footprint_radius

    def _get_footprint(self, radius: int) -> np.ndarray:
        match self.footprint_shape:
            case 'disk':
                return disk(radius=radius)
            case 'square':
                return square(int(radius * 2))
            case 'ball':
                return ball(radius)
            case 'cube':
                return cube(int(radius * 2))
