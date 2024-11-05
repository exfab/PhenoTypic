import matplotlib.pyplot as plt
from skimage.color import label2rgb

from ._imageCore import ImageCore

class ImageShow(ImageCore):
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