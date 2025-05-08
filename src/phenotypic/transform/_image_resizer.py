from typing import Optional, Tuple
import skimage as ski
import cv2

from phenotypic import Image
from phenotypic.abstract import ImageCorrector
import numpy as np

SUPPORTED_PAD_MODE_LIST = [None, 'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect',
                           'symmetric', 'wrap', 'empty']


class ImageResizer(ImageCorrector):
    def __init__(
            self,
            target_size: Tuple,
            preserve_aspect_ratio: bool = True,
            pad_mode: str = 'constant',
            padding_value: int = 0,
            anti_aliasing: bool = True,
            order: Optional[int] = 3,
            anti_aliasing_sigma: Optional[float] = None,
    ):

        if len(target_size) != 2:
            raise ValueError(f'target_dimensions should be a length 2 tuple with a target width & height ')

        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio

        if pad_mode not in SUPPORTED_PAD_MODE_LIST: raise ValueError(f'pad_mode {pad_mode} is not supported.')
        self.pad_mode = pad_mode
        self.padding_value = padding_value
        self.anti_aliasing = anti_aliasing
        self.order = order
        self.anti_aliasing_sigma = anti_aliasing_sigma

        self.new_width = self.new_height = self.left_pad = self.right_pad = None

    def _operate(self, image: Image) -> Image:
        """
        Resizes an image and all its components that are not None using hybrid padding. Currently, any labels of objects with no direct connection will be reset.
        :param image (Image): image to be resized
        :return: Image: resized image
        """
        original_height, original_width = image.shape[:2]

        if self.preserve_aspect_ratio and self.pad_mode is not None:

            # Computer scale factor
            scale_w = self.target_size[0] / original_width
            scale_h = self.target_size[1] / original_height

            # Use the smaller scale factor to preserve aspect ratio
            scale = min(scale_w, scale_h)

            # Compute the new dimensions
            self.new_width = int(original_width * scale)
            self.new_height = int(original_height * scale)

            # Compute image padding
            self.left_pad = (self.target_size[1] - self.new_width) // 2
            if ((self.target_size[1] - self.new_width) % 2) != 0:
                self.right_pad = self.left_pad + 1
            else:
                self.right_pad = self.left_pad

            self.top_pad = (self.target_size[0] - self.new_height) // 2
            if ((self.target_size[0] - self.new_height) % 2) != 0:
                self.bottom_pad = self.top_pad + 1
            else:
                self.bottom_pad = self.top_pad

        # Resize the image while preserving aspect ratio

        # Resize 2D array
        new_array = self._skimage_resize(array=image.matrix)
        if self.preserve_aspect_ratio and self.pad_mode is not None:
            new_array = self._create_padded_image(array=new_array)

        # Resize enhanced array
        new_enhanced_array = self._skimage_resize(array=image.enhanced_matrix)
        if self.preserve_aspect_ratio:
            new_enhanced_array = self._create_padded_image(array=new_enhanced_array)

        # Resize mask array
        if image.objmask is not None:
            new_mask = self._cv_resize(array=image.objmask[:].astype(int))
            new_mask = new_mask != 0  # Ensure binary values in mask array

            if self.preserve_aspect_ratio and self.pad_mode is not None:
                new_mask = self._create_padded_image(array=new_mask)
        else:
            new_mask = None

        # Resize map array
        if image.objmap is not None:
            new_map = self._cv_resize(array=image.objmap)

            if self.preserve_aspect_ratio and self.pad_mode is not None:
                new_map = self._create_padded_image(array=new_map)
        else:
            new_map = None

        # Resize color array if it exist
        if image.array is not None:
            new_rgb_array = self._skimage_resize(array=image.array)

            if self.preserve_aspect_ratio and self.pad_mode is not None:
                new_rgb_array = self._create_padded_image(new_rgb_array)

            image.array = new_rgb_array

        image.matrix = new_array
        image.enhanced_matrix = new_enhanced_array
        image.objmask = new_mask
        image.objmap = new_map
        return image

    def _skimage_resize(self, array: np.ndarray, ) -> np.ndarray:
        """
        This resizes the actual image arrays using skimage resizing algorithm. This algorithm maintains the most detail for analysis purposes
        :param array:
        :return:
        """
        if self.preserve_aspect_ratio:
            return ski.transform.resize(image=array,
                                        output_shape=(self.new_height, self.new_width),
                                        order=self.order,
                                        anti_aliasing=self.anti_aliasing,
                                        anti_aliasing_sigma=self.anti_aliasing_sigma,
                                        mode=self.pad_mode,
                                        cval=0,
                                        )
        else:
            return ski.transform.resize(image=array,
                                        output_shape=self.target_size,
                                        order=self.order,
                                        anti_aliasing=self.anti_aliasing,
                                        anti_aliasing_sigma=self.anti_aliasing_sigma,
                                        mode=self.pad_mode,
                                        cval=0
                                        )

    def _cv_resize(self, array: np.ndarray):
        """
        This allows the binary mask and object map to be resized while maintaining the integer values needed to track objects.
        :param array:
        :return:
        """
        if self.preserve_aspect_ratio:
            return cv2.resize(array, dsize=(self.new_width, self.new_height), interpolation=cv2.INTER_NEAREST)
        else:
            return cv2.resize(array, dsize=self.target_size, interpolation=cv2.INTER_NEAREST)

    def _create_padded_image(self, array: np.ndarray) -> np.ndarray:
        """
        pad the image along each dimension

        :param array:
        :return:
        """
        if array.ndim == 2:
            return np.pad(array=array,
                          pad_width=((self.top_pad, self.bottom_pad), (self.left_pad, self.right_pad)),
                          mode=self.pad_mode, constant_values=self.padding_value)
        elif array.ndim == 3:
            return np.pad(array=array,
                          pad_width=((self.top_pad, self.right_pad), (self.left_pad, self.right_pad), (0, 0)),
                          mode=self.pad_mode, constant_values=self.padding_value)
        else:
            raise ValueError(f'Array dimension {array.ndim} is not supported.')


