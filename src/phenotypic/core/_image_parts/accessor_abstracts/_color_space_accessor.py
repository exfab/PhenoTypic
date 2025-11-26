from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import tifffile

import phenotypic
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
        _accessor_property_name (str): Name of the property on Image that returns this accessor.
    """

    _accessor_property_name: str = "color.unknown"

    @classmethod
    def load(cls, filepath: str | os.PathLike | Path) -> np.ndarray:
        """Load a color space array from a TIFF file and verify it was saved from this accessor type.

        Color space arrays are stored as float32 TIFF files. This method checks
        if the image contains PhenoTypic metadata indicating it was saved from
        the same color space accessor type. If metadata doesn't match or is missing,
        a warning is raised but the array is still loaded.

        Args:
            filepath: Path to the TIFF file to load.

        Returns:
            np.ndarray: The loaded color space array (float32).

        Raises:
            ValueError: If file extension is not .tif or .tiff.

        Warns:
            UserWarning: If metadata is missing or indicates the image was saved
                from a different accessor type.

        Example:
            >>> from phenotypic.core._image_parts.color_space_accessors import CieLabAccessor
            >>> lab_arr = CieLabAccessor.load("my_lab_image.tif")
        """
        filepath = Path(filepath)
        expected_property = f"Image.{cls._accessor_property_name}"

        if filepath.suffix.lower() not in IO.TIFF_EXTENSIONS:
            raise ValueError(
                'Color space arrays can only be loaded from TIFF format (.tif, .tiff). '
                f'File extension is: {filepath.suffix.lower()}'
            )

        # Load using tifffile for float array support
        with tifffile.TiffFile(filepath) as tif:
            arr = tif.asarray()
            desc = tif.pages[0].description if tif.pages else None

        # Check metadata
        phenotypic_data = None
        if desc:
            try:
                data = json.loads(desc)
                if 'phenotypic_version' in data:
                    phenotypic_data = data
            except json.JSONDecodeError:
                pass

        if phenotypic_data is None:
            warnings.warn(
                f"No PhenoTypic metadata found in '{filepath.name}'. "
                f"Cannot verify this image was saved from {expected_property}. "
                "Loading anyway, but this may lead to undefined behavior.",
                UserWarning
            )
        else:
            saved_property = phenotypic_data.get('phenotypic_image_property', 'unknown')
            if saved_property != expected_property:
                warnings.warn(
                    f"Metadata mismatch: Image was saved from '{saved_property}' "
                    f"but being loaded as '{expected_property}'. "
                    "This may lead to undefined behavior.",
                    UserWarning
                )

        return arr

    def __getitem__(self, key) -> np.ndarray:
        """Access color space data by index, returning a non-writeable view."""
        view = self._subject_arr[key]
        view.flags.writeable = False
        return view

    def __setitem__(self, key, value):
        """Prevent direct modification of color space data.
        
        Args:
            key: Index or slice for rgb access.
            value: Value to assign (not allowed).
            
        Raises:
            IllegalAssignmentError: Always raised as color space data is read-only.
        """
        raise IllegalAssignmentError(self.__class__.__name__)

    def imsave(self, filepath: str | os.PathLike | Path) -> None:
        """Save color space data to TIFF file with PhenoTypic metadata embedded.

        Color space arrays can only be saved in TIFF format due to their
        floating-point nature. Float arrays are converted to float32 for
        compatibility. Metadata is embedded in the ImageDescription tag.

        Args:
            filepath: Path to save the TIFF file.

        Raises:
            ValueError: If file extension is not .tif or .tiff.
        """
        import skimage.io

        filepath = Path(filepath)

        if filepath.suffix.lower() not in IO.TIFF_EXTENSIONS:
            raise ValueError(
                'Color space arrays can only be saved in TIFF format (.tif, .tiff). '
                f'File extension is: {filepath.suffix.lower()}'
            )

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=False)

        # Get array and ensure it's float32 for TIFF compatibility
        arr = self._subject_arr
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        # Use tifffile directly for float array support with metadata
        tifffile.imwrite(
            filepath,
            arr,
            description=metadata_json,
            compression='zlib',
            photometric='minisblack'
        )
