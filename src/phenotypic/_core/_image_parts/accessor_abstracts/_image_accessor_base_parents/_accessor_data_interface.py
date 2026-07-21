from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class AccessorDataInterface(ABC):
    """Abstract base providing the numpy array interface and data-access helpers.

    This is the foundational layer of the accessor MRO chain.  It declares
    the two abstract properties every concrete accessor must implement
    (``_accessor_property_name`` and ``_subject_arr``), stores the reference
    back to the owning :class:`Image`, and exposes a NumPy-compatible
    interface (shape, dtype, ``__array__``, etc.).

    Attributes:
        _root_image: The :class:`Image` instance this accessor belongs to.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _accessor_property_name(self) -> str:
        """Name of the Image property that surfaces this accessor."""
        raise NotImplementedError

    @classmethod
    def _accessor_property_name_value(cls) -> str:
        """Retrieve accessor property name from the subclass' property without instantiation."""
        return cls._accessor_property_name.fget(object.__new__(cls))  # type: ignore[attr-defined]

    @property
    @abstractmethod
    def _subject_arr(self) -> np.ndarray:
        raise NotImplementedError

    # Concrete override retained for subclasses that provide a non-array
    # data interface. The original monolithic class had
    # this same pattern — an @abstractmethod declaration followed by a
    # second @property that shadowed it on the same class.
    @_subject_arr.getter  # type: ignore[attr-defined]
    def _subject_arr(self) -> np.ndarray:
        """
        Abstract property representing an image array. The image array is expected to be a NumPy ndarray
        with a specific shape of (r, c, ...), which can be used for various operations that require a structured
        multi-dimensional array.

        This property is abstract and must be implemented in any derived concrete class. The implementation
        should conform to the type signature and shape expectations as defined.

        Note: Read-only property. Changes should reference the specific array

        Returns:
            np.ndarray: A NumPy ndarray object with shape (r, c, ...).
        """
        raise NotImplementedError(
            "This property is abstract and must be implemented in a derived class."
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, root_image: Image):
        self._root_image = root_image

    # ------------------------------------------------------------------
    # NumPy array interface
    # ------------------------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        """Implements the array interface for numpy compatibility.

        This allows numpy functions to operate directly on accessor objects.
        For example: np.sum(accessor), np.mean(accessor), etc.

        Args:
            dtype: Optional dtype to cast the array to
            copy: Optional copy parameter for NumPy 2.0+ compatibility

        Returns:
            np.ndarray: The underlying array data
        """
        arr = self._subject_arr
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        elif copy:
            arr = arr.copy()
        return arr

    def __len__(self) -> int:
        """
        Returns the length of the subject array.

        This method calculates and returns the total number of elements contained in the
        underlying array.

        Returns:
            int: The number of elements in the underlying array attribute.
        """
        return len(self._subject_arr)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the current image data.

        This method retrieves the dimensions of the array stored in the `_main_arr`
        attribute as a tuple, which indicates its size along each axis.

        Returns:
            Tuple[int, ...]: A tuple representing the dimensions of the `_main_arr`
            attribute.
        """
        return self._subject_arr.shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the underlying array.

        The `ndim` property provides access to the dimensionality of the array
        being encapsulated in the object. This value corresponds to the number
        of axes or dimensions the underlying array possesses. It can be useful
        for understanding the structure of the contained data.

        Returns:
            int: The number of dimensions of the underlying array.
        """
        return self._subject_arr.ndim

    @property
    def size(self) -> int:
        """
        Gets the size of the subject array.

        This property retrieves the total number of elements in the subject
        array. It is read-only.

        Returns:
            int: The total number of elements in the subject array.
        """
        return self._subject_arr.size

    def val_range(self) -> pd.Interval:
        """
        Return the closed interval [min, max] of the subject array values.

        Returns:
            pd.Interval: A single closed interval including both endpoints.
        """
        mn = self._subject_arr.min()
        mx = self._subject_arr.max()
        return pd.Interval(left=mn, right=mx, closed="both")

    @property
    def dtype(self):
        return self._subject_arr.dtype

    def isempty(self):
        return True if self.shape[0] == 0 else False

    def copy(self) -> np.ndarray:
        return self._subject_arr.copy()

    @property
    def nbytes(self) -> int:
        return self._subject_arr.nbytes

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def foreground(self):
        """
        Extracts and returns the foreground of the image by masking out the background.

        This method generates a foreground image by applying the object mask
        stored in the Image to the current array representation.
        Pixels outside the object mask are set to zero in the resulting foreground
        image. This is useful in image processing tasks to isolate the region
        of interest in the image, such as microbe colonies on an agar plate.

        Returns:
            numpy.ndarray: A numpy array containing only the foreground portion
            of the image, with all non-foreground pixels set to zero.
        """
        foreground = self._subject_arr.copy()
        foreground[self._root_image.objmask[:] == 0] = 0
        return foreground

    def _get_filtered_objmap(self, object_label: int | None = None) -> np.ndarray:
        """Fetch the object map, optionally filtering to a single label.

        Args:
            object_label: If provided, zero out all labels except this one.

        Returns:
            A dense object map array (always a fresh copy from sparse).
        """
        objmap = self._root_image.objmap[:]
        if object_label is not None:
            objmap[objmap != object_label] = 0
        return objmap
