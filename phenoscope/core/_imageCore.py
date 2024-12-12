from typing import Union, Optional
from typing_extensions import Self

import numpy as np
from skimage.measure import label
from skimage.color import rgb2gray

from ..util.type_checks import is_binary_mask
from ..util.error_message import INVALID_MASK_SHAPE_MSG, INVALID_MAP_SHAPE_MSG, NO_IMAGE_DATA_ERROR_MSG


# TODO: Import Sparse Matrix
class ImageCore:
    def __init__(self, image: Optional[Union[np.ndarray, Self]] = None):
        if image is None:  # Create an empty image object
            self.__array = None
            self.__image_matrix = None
            self.__enhanced_image_matrix = None
            self.__object_mask = None
            self.__object_map = None
        else:  # Load image array into object
            if type(image) is np.ndarray:
                if image.ndim == 3:
                    self.__array = image
                    self.__image_matrix = rgb2gray(image)
                else:
                    self.__array = None
                    self.__image_matrix: np.ndarray = image
                self.__enhanced_image_matrix = self.__image_matrix
                self.__object_mask: Optional[np.ndarray] = None
                self.__object_map: Optional[np.ndarray] = None

            elif issubclass(type(image), ImageCore):
                self.__array = image.array
                self.__image_matrix: np.ndarray = image.matrix
                self.__enhanced_image_matrix: np.ndarray = image.enhanced_matrix
                self.__object_mask: Optional[np.ndarray] = image.object_mask
                self.__object_map: Optional[np.ndarray] = image.object_map
            else:
                raise ValueError(
                    f'Unsupported input for image class constructor - Input: {type(image)} - Accepted Input:{np.ndarray} or {self.__class__}'
                )

    def __getitem__(self, index):
        if self.__image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)
        if len(index) != 2: raise ValueError(
            'Image objects only support 2-dimensional slicing. RGB images will be sliced evenly across each channel.')
        if self.__array is not None:
            new_img = self.__class__(self.__array[index])
            new_img.matrix = self.__image_matrix[index]
            new_img.enhanced_matrix = self.__enhanced_image_matrix[index]
        else:
            new_img = self.__class__(self.__image_matrix[index])
            new_img.enhanced_matrix = self.__enhanced_image_matrix[index]

        if self.__object_mask is not None:
            new_img.object_mask = self.__object_mask[index]

        if self.__object_map is not None:
            new_img.object_map = self.__object_map[index]

        return new_img

    @property
    def shape(self) -> tuple:
        if self.__image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)
        return self.__image_matrix.shape

    @property
    def ndim(self) -> int:
        if self.__image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)
        return self.__image_matrix.ndim

    # The color representation of the image. This is blank if a grayscale image is set as the image
    @property
    def array(self) -> Optional[np.ndarray]:
        if self.__image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)
        return self.__array

    @array.setter
    def array(self, image: np.ndarray):
        if image.ndim != 3: raise ValueError('The input image array was not an RGB array.')
        self.__array = image
        self.matrix = rgb2gray(image)

    # The 2-dimensional representation of the image
    @property
    def matrix(self) -> np.ndarray:
        if self.__image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)
        return np.copy(self.__image_matrix)

    @matrix.setter
    def matrix(self, image: np.ndarray) -> None:
        self.__image_matrix = image
        self.__enhanced_image_matrix = image
        self.__object_mask = None
        self.__object_map = None

    # The 2-dimensional enhanced representation of the image that is fed into the detection algorithm
    @property
    def enhanced_matrix(self) -> np.ndarray:
        if self.__enhanced_image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)
        return np.copy(self.__enhanced_image_matrix)

    @enhanced_matrix.setter
    def enhanced_matrix(self, enhanced_image: np.ndarray) -> None:
        self.__enhanced_image_matrix = enhanced_image
        self.__object_mask = None
        self.__object_map = None

    # Object Mask: A binary array that represents detected objects
    @property
    def object_mask(self) -> Union[np.ndarray, None]:
        """
        The object mask is a boolean array indicating the indices of the objects in the image.
        :return:
        """
        if self.__image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)

        if self.__object_mask is not None:
            return np.copy(self.__object_mask)
        else:
            return None

    @object_mask.setter
    def object_mask(self, mask: np.ndarray) -> None:
        if mask is not None:
            if is_binary_mask(mask) is False:
                raise ValueError("Mask must be a binary array.")
            if not np.array_equal(mask.shape, self.__enhanced_image_matrix.shape): raise ValueError(
                INVALID_MASK_SHAPE_MSG)
            self.__object_mask = mask
            self.__object_map = label(self.__object_mask)
        else:
            self.__object_mask = None

    @property
    def object_map(self) -> Union[np.ndarray, None]:
        """
        The object map is a numpy array with integer values that represent the different objects within the mask
        :return:
        """
        if self.__image_matrix is None: raise AttributeError(NO_IMAGE_DATA_ERROR_MSG)

        if self.__object_map is not None:
            return np.copy(self.__object_map)
        else:
            return None

    @object_map.setter
    def object_map(self, object_map: np.ndarray) -> None:
        if object_map is not None:
            if not np.array_equal(object_map.shape, self.__enhanced_image_matrix.shape): raise ValueError(
                INVALID_MAP_SHAPE_MSG)
            self.__object_map = object_map
            self.__object_mask = self.__object_map > 0
        else:
            self.__object_map = None

    def get_object_labels(self) -> np.ndarray[Optional[int]]:
        """
        Returns all the object labels present in the image map.
        :return:
        """
        if self.__object_map is None:
            return np.array([None])
        else:
            labels = np.unique(self.__object_map)
            return labels[np.nonzero(self.__object_map)]

    def copy(self):
        if self.array is not None:
            new_image = self.__class__(self.__array)
            new_image.matrix = self.__image_matrix
        else:
            new_image = self.__class__(self.__image_matrix)

        new_image.enhanced_matrix = self.__enhanced_image_matrix
        new_image.object_mask = self.__object_mask
        new_image.object_map = self.__object_map
        return new_image

    def reset(self):
        self.__enhanced_image_matrix = self.matrix
        self.__object_mask = None
        self.__object_map = None

    def shape_integrity_check(self):
        rr_len = self.__image_matrix.shape[0]
        cc_len = self.__image_matrix.shape[1]

        if self.__enhanced_image_matrix.shape[0] != rr_len or self.__enhanced_image_matrix.shape[1] != cc_len:
            raise RuntimeError(
                'Detected enhanced image shape do not match the image shape. Ensure that enhanced image array shape is not changed during execution.')

        if self.__object_mask is not None:
            if self.__object_map.shape[0] != rr_len or self.__object_map.shape[1] != cc_len:
                raise RuntimeError(
                    'Detected object map shape do not match the image shape. Ensure that the object map array shape is not changed during runtime.')

        if self.__object_map is not None:
            if self.__object_mask.shape[0] != rr_len or self.__object_mask.shape[1] != cc_len:
                raise RuntimeError(
                    'Detected object mask shape do not match the image shape. Ensure that the object mask array shape is not changed during runtime.')

        return True
