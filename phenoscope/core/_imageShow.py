import matplotlib.pyplot as plt
from skimage.color import label2rgb
from typing import Optional, Tuple
from math import gcd, log10

from tinycss2.ast import AtRule

from ._imageCore import ImageCore


class ImageShow(ImageCore):
    def show_array(self,
                   ax: plt.Axes = None,
                   figsize: Tuple[int] = None) -> (plt.Figure, plt.Axes):
        if figsize is None: figsize = self.__get_default_figsize()

        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)
        func_ax.imshow(self.array)

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_matrix(self, ax: plt.Axes = None, cmap: str = 'gray', figsize: Tuple[int] = None) -> (
            plt.Figure, plt.Axes):
        if figsize is None: figsize = self.__get_default_figsize()
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)
        if len(self.matrix.shape) == 2:
            func_ax.imshow(self.matrix, cmap=cmap)
        else:
            raise AttributeError('The image matrix is not 2D.')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_enhanced(self, ax: plt.Axes = None, cmap: str = 'gray', figsize: Tuple[int] = None) -> (
            plt.Figure, plt.Axes):
        if figsize is None: self.__get_default_figsize()
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)
        if len(self.enhanced_matrix.shape) == 2:
            func_ax.imshow(self.enhanced_matrix, cmap=cmap)
        else:
            func_ax.imshow(self.enhanced_matrix)

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_mask(self, ax: plt.Axes = None, cmap: str = 'gray', figsize: Tuple[int] = None) -> (plt.Figure, plt.Axes):
        if figsize is None: figsize = self.__get_default_figsize()
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

    def show_map(self, object_label: Optional[int] = None, ax: plt.Axes = None, cmap: str = 'tab20',
                 figsize: Tuple[int] = None) -> (
            plt.Figure, plt.Axes):
        if figsize is None: figsize = self.__get_default_figsize()
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        # Optionally isolate a certain object
        obj_map = self.object_map
        if object_label is not None:
            obj_map[obj_map != object_label] = 0

        func_ax.imshow(obj_map, cmap=cmap)

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    def show_overlay(self,
                     object_label: Optional[int] = None,
                     use_enhanced=False,
                     ax=None,
                     figsize=(9, 10),
                     alpha=0.2) -> (
            plt.Figure, plt.Axes):
        if figsize is None: figsize = self.__get_default_figsize()
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        # Create an obj map and isolate an object if user-specified
        obj_map = self.object_map
        if object_label is not None:
            obj_map[obj_map != object_label] = 0

        # Plot image
        if use_enhanced:
            func_ax.imshow(label2rgb(label=obj_map, image=self.enhanced_matrix, alpha=alpha))
        else:
            func_ax.imshow(label2rgb(label=obj_map, image=self.matrix, alpha=alpha))

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    # TODO: Fix default figsize function
    def __get_default_figsize(self) -> Tuple[int, int]:
        """
        Calculate the aspect ratio of an image to maintain its information
        :return:
        """
        # height, width = self.shape[0], self.shape[1]
        #
        # def round_to_base(x):
        #     magnitude = int(log10(x))
        #     return round(x ,-magnitude)
        #
        # height = round_to_base(height)
        # width = round_to_base(width)
        #
        # divisor = gcd(width, height)
        # width_ratio = width // divisor
        # height_rato = height // divisor
        # return width_ratio, height_rato
        return None
