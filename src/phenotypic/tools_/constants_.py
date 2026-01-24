"""
PhenoTypic Constants

This module contains constant values and enumerations used throughout the PhenoTypic library.
Constants are organized by module and functionality.

Note: Class names are defined in ALL_CAPS to avoid namespace conflicts with actual classes
    in the codebase (e.g., GRID_DEP vs an actual Grid class). When importing, use the format:
        from PhenoTypic.tools_.constants import IMAGE_MODE, OBJECT
"""

from .measurement_info_ import BBOX, PIPE_STATUS, GRID, METADATA
import phenotypic
from enum import Enum
from packaging.version import Version
from pathlib import Path


# Image format constants
class IMAGE_MODE(Enum):
    """Constants for supported image formats."""

    NONE = None
    GRAYSCALE = "GRAYSCALE"
    GRAYSCALE_SINGLE_CHANNEL = "Grayscale (single channel)"
    HSV = "HSV"
    RGB_OR_BGR = "RGB/BGR (ambiguous)"
    RGBA_OR_BGRA = "RGBA/BGRA (ambiguous)"
    RGB = "RGB"
    LINEAR_RGB = "LINEAR RGB"
    RGBA = "RGBA"
    BGR = "BGR"
    BGRA = "BGRA"
    SUPPORTED_FORMATS = (RGB, RGBA, GRAYSCALE, BGR, BGRA)
    MATRIX_FORMATS = (GRAYSCALE, GRAYSCALE_SINGLE_CHANNEL)
    AMBIGUOUS_FORMATS = (RGB_OR_BGR, RGBA_OR_BGRA)

    def is_matrix(self):
        return self in {IMAGE_MODE.GRAYSCALE, IMAGE_MODE.GRAYSCALE_SINGLE_CHANNEL}

    def is_array(self):
        return self in {
            IMAGE_MODE.RGB,
            IMAGE_MODE.RGBA,
            IMAGE_MODE.BGR,
            IMAGE_MODE.BGRA,
            IMAGE_MODE.LINEAR_RGB,
        }

    def is_ambiguous(self):
        return self in {IMAGE_MODE.RGB_OR_BGR, IMAGE_MODE.RGBA_OR_BGRA}

    def is_none(self):
        return self is IMAGE_MODE.NONE

    CHANNELS_DEFAULT = 3
    DEFAULT_SCHEMA = RGB


# Object information constants
class OBJECT:
    """Constants for object information properties."""

    LABEL = "ObjectLabel"


class IO:
    RAW_FILE_EXTENSIONS = (".cr3", ".CR3")
    PNG_FILE_EXTENSIONS = (".png", ".PNG")
    JPEG_FILE_EXTENSIONS = (".jpeg", ".JPEG", ".jpg")
    TIFF_EXTENSIONS = (".tif", ".tiff")
    ACCEPTED_FILE_EXTENSIONS = (
            PNG_FILE_EXTENSIONS + JPEG_FILE_EXTENSIONS + TIFF_EXTENSIONS
    )

    # Key used for PhenoTypic metadata container in image files
    PHENOTYPIC_METADATA_KEY = "phenotypic"

    if Version(phenotypic.__version__) < Version("0.7.1"):
        SINGLE_IMAGE_HDF5_PARENT_GROUP = Path("phenotypic/")
    else:
        SINGLE_IMAGE_HDF5_PARENT_GROUP = "/phenotypic/images/"

    IMAGE_SET_HDF5_PARENT_GROUP = "/phenotypic/image_sets/"

    IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY = "measurements"
    IMAGE_STATUS_SUBGROUP_KEY = "status"


# Feature extraction constants
# TODO: Fix this constant access pattern
class GRID_LINREG_STATS_EXTRACTOR:
    """Constants for grid linear regression statistics extractor."""

    ROW_LINREG_M, ROW_LINREG_B = "RowLinReg_M", "RowLinReg_B"
    COL_LINREG_M, COL_LINREG_B = "ColLinReg_M", "ColLinReg_B"
    PRED_RR, PRED_CC = "RowLinReg_PredRR", "ColLinReg_PredCC"
    RESIDUAL_ERR = "LinReg_ResidualError"


class IMAGE_TYPES(Enum):
    """The string labels for different types of images generated when accessing subimages of a parent image."""

    BASE = "Image"
    CROP = "Crop"
    OBJECT = "Object"
    GRID = "GridImage"
    GRID_SECTION = "GridSection"

    def __str__(self):
        return self.value
