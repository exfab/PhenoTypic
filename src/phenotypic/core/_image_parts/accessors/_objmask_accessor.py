from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: pass

from skimage.measure import label
import matplotlib.pyplot as plt
import numpy as np

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import InvalidMaskValueError, InvalidMaskScalarValueError, \
    ArrayKeyValueShapeMismatchError


class ObjectMask(SingleChannelAccessor):
    """Represents a binary object mask linked to a parent image.

    This class allows for manipulation and analysis of a binary object mask associated with a parent image. It provides
    functionality to access, modify, display, and extract object regions of the mask. The object mask is tightly linked
    to the parent image, which is used as the source for the binary map.

    Note:
        - Changes to the object mask will reset the labeling of the object map.
    """
    
    @property
    def _backend(self):
        """Returns the current sparse backend reference from the object map.
        
        This ensures consistency with ObjectMap and avoids redundant conversions.
        """
        return self._root_image._data.sparse_object_map
    
    def __array__(self, dtype=None, copy=None):
        """Implements the array interface for numpy compatibility.
        
        This allows numpy functions to operate on the dense form of the binary mask.
        For example: np.sum(objmask), np.count_nonzero(objmask), etc.
        
        Args:
            dtype: Optional dtype to cast the array to
            copy: Optional copy parameter for NumPy 2.0+ compatibility
            
        Returns:
            Dense binary numpy array (0s and 1s) representation of the object mask
        """
        arr = (self._backend.toarray() > 0).astype(int)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        elif copy:
            arr = arr.copy()
        return arr

    def __getitem__(self, key):
        """Returns a slice of the binary object mask as if it were a dense array.
        
        The slicing behavior matches numpy arrays, converting the sparse
        representation to dense and then to binary (0s and 1s).
        """
        return (self._backend.toarray()[key] > 0).astype(int)

    def __setitem__(self, key, value: np.ndarray):
        """Sets values of the object mask as if it were a dense array.
        
        This operation converts to dense, applies the update, relabels with scikit-image,
        and updates the backend. The mask is automatically relabeled to maintain
        consistent object IDs.
        
        Args:
            key: Index or slice for the mask
            value: Binary value(s) to set (int, bool, or ndarray)
        """
        # Get current mask as dense array (convert once)
        mask = self._backend.toarray() > 0

        # Apply the value based on type
        if isinstance(value, (int, bool)):
            try:
                value = 1 if value != 0 else 0
                mask[key] = value
            except TypeError:
                raise InvalidMaskScalarValueError
        elif isinstance(value, np.ndarray):
            # Check arr and section have matching shape
            if mask[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError

            # Sets the section of the binary mask to the value array
            mask[key] = (value > 0)
        else:
            raise InvalidMaskValueError(type(value))

        # Relabel the mask and update the backend atomically
        # This is where the relabeling occurs to maintain consistent object IDs
        relabeled = label(mask)
        new_sparse = self._root_image.objmap._dense_to_sparse(relabeled)
        new_sparse.eliminate_zeros()
        self._root_image._data.sparse_object_map = new_sparse

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
        return (self._backend.toarray() > 0).astype(int).copy()

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
        return self._plot(arr=(self._backend.toarray() > 0), figsize=figsize, ax=ax, title=title, cmap=cmap)

    def _create_foreground(self, array: np.ndarray, bg_label: int = 0) -> np.ndarray:
        """Returns a copy of the array with every non-object pixel set to 0. Equivalent to np.ma.array.filled(bg_label)"""
        mask = self._backend.toarray() > 0
        if array.ndim == 3: 
            mask = np.dstack([mask for _ in range(array.shape[-1])])

        array[~mask] = bg_label
        return array

    @property
    def _subject_arr(self) -> np.ndarray:
        return (self._backend.toarray() > 0).astype(int)
