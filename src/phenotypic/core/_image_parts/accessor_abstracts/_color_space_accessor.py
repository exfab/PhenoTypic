from __future__ import annotations

from pathlib import Path

import numpy as np

from phenotypic.tools.constants_ import IO
from phenotypic.tools.exceptions_ import IllegalAssignmentError
from ._image_accessor_base import ImageAccessorBase


class ColorSpaceAccessor(ImageAccessorBase):
    """Base class for color space accessors.
    
    Provides read-only access to color space transformations of the parent image.
    Color space accessors compute transformed representations on-the-fly and prevent
    direct modification to maintain data integrity.
    
    Attributes:
        _root_image (Image): The parent image object that this accessor transforms.
    """

    def __getitem__(self, key) -> np.ndarray:
        """Access color space data by index.
        
        Args:
            key: Index or slice for array access.
            
        Returns:
            np.ndarray: Copy of the requested color space data.
        """
        return self._subject_arr[key].copy()

    def __setitem__(self, key, value):
        """Prevent direct modification of color space data.
        
        Args:
            key: Index or slice for array access.
            value: Value to assign (not allowed).
            
        Raises:
            IllegalAssignmentError: Always raised as color space data is read-only.
        """
        raise IllegalAssignmentError(self.__class__.__name__)

    def imsave(self, filepath: str | os.PathLike | Path):
        """Save color space data to file."""
        filepath = Path(filepath)
        from PIL import Image as PIL_Image

        if filepath.suffix.lower() in IO.TIFF_EXTENSIONS:
            PIL_Image.fromarray(self._subject_arr).save(fp=filepath, format=filepath.suffix.lower())
        else:
            raise ValueError(
                    'Color space arrays can only be saved in TIFF format (.tif, .tiff). File extension is: '
                    + filepath.suffix.lower())
