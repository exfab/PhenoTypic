from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

import numpy as np

from scipy.sparse import csc_matrix, coo_matrix
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table, label

from phenoscope.util.exceptions_ import UnknownError, ArrayKeyValueShapeMismatchError, InvalidMapValueError


class ObjectMap:
    """Manages an object map for labeled regions in an image.

    This class provides a mechanism to manipulate and access labeled object maps
    within a given image. It is tightly coupled with the parent image object and
    provides methods for accessing sparse and dense representations, relabeling,
    resetting, and visualization.

    Args:
        parent_image (ImageHandler): The image handler that the ObjectMap belongs to.

    Attributes:
        _parent_image (ImageHandler): Reference to the parent image handler.
    """

    def __init__(self, parent_image:Image):
        """

        Args:
            parent_image: (ImageHandler) The image handler that the ObjectMap belongs to.
        """
        self._parent_image = parent_image

    @property
    def _num_objects(self):
        return len(self._labels)

    @property
    def _labels(self):
        """Returns the labels in the image.

               We considered using a simple numpy.unique() call on the object map, but wanted to guarantee that the labels will always be consistent
               with any skimage version outputs.

               """
        return regionprops_table(label_image=self._parent_image._sparse_object_map.toarray(), properties=['label'], cache=False)['label']

    def __getitem__(self, key):
        """Returns a copy of the object_map of the image. If there are no objects, this is a matrix with all values set to 1 and the same shape as the iamge matrix."""
        if self._num_objects > 0:
            return self._parent_image._sparse_object_map.toarray()[key]
        elif self._num_objects == 0:
            return np.full(self._parent_image._sparse_object_map.toarray()[key].shape, fill_value=1, dtype=np.uint32)
        else:
            raise RuntimeError(UnknownError)

    def __setitem__(self, key, value):
        """Uncompresses the csc array & changes the values at the specified coordinates before recompressing the object map array."""
        dense = self._parent_image._sparse_object_map.toarray()

        if type(value) == np.ndarray:
            value = value.astype(self._parent_image._sparse_object_map.dtype)
            if dense[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError
            elif dense.dtype != value.dtype:
                raise ArrayKeyValueShapeMismatchError

            dense[key] = value
        elif type(value) == int:
            dense[key] = value
        else:
            raise InvalidMapValueError

        self._parent_image._sparse_object_map = self._dense_to_sparse(dense)

    @property
    def shape(self) -> tuple[int, int]:
        return self._parent_image._sparse_object_map.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the object_map."""
        return self._parent_image._sparse_object_map.toarray().copy()

    def to_csc(self) -> csc_matrix:
        """Returns a copy of the object map as a compressed sparse column matrix"""
        return self._parent_image._sparse_object_map.tocsc()

    def to_coo(self) -> coo_matrix:
        """Returns a copy of the object map in COOrdinate format or ijv matrix"""
        return self._parent_image._sparse_object_map.tocoo()

    def show(self, ax=None, figsize=None, cmap='gray', title=None) -> (plt.Figure, plt.Axes):
        """Display the object_map with matplotlib.

        Args:
            ax: (plt.Axes) Axes object to use for plotting.
            figsize: (Tuple[int, int]): Figure size in inches.
            cmap: (str, optional) Colormap to use.
            title: (str) a title for the plot

        Returns:
            tuple(plt.Figure, plt.Axes): matplotlib figure and axes object
        """
        if figsize is None: figsize = (6, 4)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.imshow(self._parent_image._sparse_object_map.toarray(), cmap=cmap)
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    def reset(self) -> None:
        """Resets the object_map to an empty map array with no objects in it."""
        if self._parent_image.isempty():
            self._parent_image._sparse_object_map = None
        else:
            self._parent_image._sparse_object_map = self._dense_to_sparse(self._parent_image.matrix.shape)

    def relabel(self):
        """Relables all the objects based on their connectivity"""
        self._dense_to_sparse(label(self._parent_image.omask[:]))


    @staticmethod
    def _dense_to_sparse(arg) -> csc_matrix:
        """Constructs a sparse array from the arg parameter. Used so that the underlying sparse matrix can be changed

        Args:
            arg: either the dense object array or the shape

        Returns:

        """
        sparse = csc_matrix(arg, dtype=np.uint32)
        sparse.eliminate_zeros()
        return sparse
