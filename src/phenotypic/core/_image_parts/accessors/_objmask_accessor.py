from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: pass

from skimage.measure import label
import matplotlib.pyplot as plt
import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import ImageArrDataAccessor
from phenotypic.util.exceptions_ import InvalidMaskValueError, InvalidMaskScalarValueError, ArrayKeyValueShapeMismatchError


class ObjectMask(ImageArrDataAccessor):
    """Represents a binary object mask linked to a parent image.

    This class allows for manipulation and analysis of a binary object mask associated with a parent image. It provides
    functionality to access, modify, display, and extract object regions of the mask. The object mask is tightly linked
    to the parent image, which is used as the source for the binary map.

    Note:
        - Changes to the object mask will reset the labeling of the object map.
    """

    def __getitem__(self, key):
        """Returns a copy of the binary object mask in array form"""
        return (self._root_image.objmap[key] > 0).astype(int)

    def __setitem__(self, key, value: np.ndarray):
        """Sets values of the object mask to other_image and resets the labeling in the map"""
        mask = self._root_image.objmap[:] > 0

        # Check to make sure the section of the mask the key accesses is the same as the other_image
        if isinstance(value, (int, bool)):
            try:
                value = 1 if value != 0 else 0
                mask[key] = value
            except TypeError:
                raise InvalidMaskScalarValueError
        elif isinstance(value, np.ndarray):
            # Check input_image and section have matching shape
            if mask[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError

            # Sets the section of the binary mask to the other_image array
            mask[key] = (value > 0)
        else:
            raise InvalidMaskValueError(type(value))

        # Relabel the mask and set the underlying csc matrix to the new mask
        # Where the reset of labeling occurs. May eventually add a way to sync without a label reset in the future
        self._root_image.objmap[:] = label(mask)

    @property
    def shape(self):
        """
        Represents the shape of a parent image's omap property.

        This property is a getter for retrieving the shape of the `omap` attribute
        of the associated parent image.

        Returns:
            The shape of the object map
        """
        return self._root_image.objmap.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the binary object mask"""
        return self._root_image.objmask[:].copy()

    def reset(self):
        """
        Resets the overlay map (omap) tied to the parent image. This function interacts with
        the `omap` object contained within the parent image, delegating the reset operation
        to it.

        """
        self._root_image.objmap.reset()

    def show(self, ax: plt.Axes | None = None,
             figsize: str | None = None,
             cmap: str = 'gray',
             title: str | None = None
             ) -> (plt.Figure, plt.Axes):
        """Display the boolean object mask with matplotlib.

        Calls object_map linked by the image handler

        Args:
            ax: (plt.Axes) Axes object to use for plotting.
            figsize: (Tuple[int, int]): Figure size in inches.
            cmap: (str, optional) Colormap to use.
            title: (str) a title for the plot

        Returns:
            tuple(plt.Figure, plt.Axes): matplotlib figure and axes object
        """
        return self._plot(arr=self._root_image.objmap[:] > 0, figsize=figsize, ax=ax, title=title, cmap=cmap)

    def _create_foreground(self, array: np.ndarray, bg_label: int = 0) -> np.ndarray:
        """Returns a copy of the array with every non-object pixel set to 0. Equivalent to np.ma.array.filled(bg_label)"""
        mask = self._root_image.objmap[:] > 0
        if array.ndim == 3: mask = np.dstack([(mask > 0) for _ in range(array.shape[-1])])

        array[~mask] = bg_label
        return array
