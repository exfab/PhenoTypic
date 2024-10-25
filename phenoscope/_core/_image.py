from typing import Union, Optional
from typing_extensions import Self

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.color import label2rgb

from ..util.type_checks import is_binary_mask
from ..util.error_message import INVALID_MASK_SHAPE_MSG, INVALID_MAP_SHAPE_MSG


class Image:
    def __init__(self, image: Union[np.ndarray, Self]):
        if type(image) is np.ndarray:
            self.__image_array: np.ndarray = image
            self.__enhanced_image_array = image
            self.__object_mask: Optional[np.ndarray] = None
            self.__object_map: Optional[np.ndarray] = None
        elif hasattr(image, 'array') and hasattr(image, 'enhanced_array') and hasattr(image, 'object_mask') and hasattr(image,
                                                                                                                        'object_map'):
            self.__image_array: np.ndarray = image.array
            self.__enhanced_image_array: np.ndarray = image.enhanced_array
            self.__object_mask: Optional[np.ndarray] = image.object_mask
            self.__object_map: Optional[np.ndarray] = image.object_map
        else:
            raise ValueError('Unsupported input for image class constructor')

    def __getitem__(self, index):
        return self.__class__(self.__image_array[index])

    @property
    def shape(self) -> tuple:
        return self.__image_array.shape

    @property
    def ndim(self) -> int:
        return self.__image_array.ndim

    @property
    def array(self) -> np.ndarray:
        return np.copy(self.__image_array)

    @array.setter
    def array(self, image: np.ndarray) -> None:
        self.__image_array = image
        self.__enhanced_image_array = image
        self.__object_mask = None
        self.__object_map = None

    @property
    def enhanced_array(self) -> np.ndarray:
        return np.copy(self.__enhanced_image_array)

    @enhanced_array.setter
    def enhanced_array(self, image: np.ndarray) -> None:
        self.__enhanced_image_array = image
        self.__object_mask = None
        self.__object_map = None

    @property
    def object_mask(self) -> np.ndarray:
        """
        The object mask is a boolean array indicating the indices of the objects in the image.
        :return:
        """
        return np.copy(self.__object_mask)

    @object_mask.setter
    def object_mask(self, mask: np.ndarray) -> None:
        if mask is not None:
            if is_binary_mask(mask) is False:
                raise ValueError("Mask must be a binary array.")
            if not np.array_equal(mask.shape, self.__enhanced_image_array.shape): raise ValueError(INVALID_MASK_SHAPE_MSG)
            self.__object_mask = mask
            self.__object_map = label(self.__object_mask)
        else:
            self.__object_mask = None

    @property
    def object_map(self) -> np.ndarray:
        return np.copy(self.__object_map)

    @object_map.setter
    def object_map(self, object_map: np.ndarray) -> None:
        if object_map is not None:
            if not np.array_equal(object_map.shape, self.__enhanced_image_array.shape): raise ValueError(INVALID_MAP_SHAPE_MSG)
            self.__object_map = object_map
            self.__object_mask = self.__object_map > 0
        else:
            self.__object_map = None

    def copy(self):
        new_image = self.__class__(self.array)
        new_image.enhanced_array = self.__enhanced_image_array
        new_image.object_mask = self.__object_mask
        new_image.object_map = self.__object_map
        return new_image

    def show(self, ax=None, cmap='gray', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)
        if len(self.array.shape) == 2:
            func_ax.imshow(self.array, cmap=cmap)
        else:
            func_ax.imshow(self.array)

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_enhanced(self, ax=None, cmap='gray', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)
        if len(self.enhanced_array.shape) == 2:
            func_ax.imshow(self.enhanced_array, cmap=cmap)
        else:
            func_ax.imshow(self.enhanced_array)

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_mask(self, ax=None, cmap='gray', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)
        if len(self.object_mask.shape) == 2:
            func_ax.imshow(self.object_mask, cmap=cmap)
        else:
            func_ax.imshow(self.object_mask)

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_map(self, ax=None, cmap='tab20', figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)
        if len(self.object_map.shape) == 2:
            func_ax.imshow(self.object_map, cmap=cmap)
        else:
            func_ax.imshow(self.object_map)

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_overlay(self, use_enhanced=False, ax=None, figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.object_map, image=self.enhanced_array))
        else:
            func_ax.imshow(label2rgb(label=self.object_map, image=self.array))

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def shape_integrity_check(self):
        rr_len = self.__image_array.shape[0]
        cc_len = self.__image_array.shape[1]

        if self.__enhanced_image_array.shape[0] != rr_len or self.__enhanced_image_array.shape[1] != cc_len:
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
