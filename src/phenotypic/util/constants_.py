"""
PhenoTypic Constants

This module contains constant values and enumerations used throughout the PhenoTypic library.
Constants are organized by module and functionality.

Note: Class names are defined in ALL_CAPS to avoid namespace conflicts with actual classes 
    in the codebase (e.g., GRID vs an actual Grid class). When importing, use the format:
        from PhenoTypic.util.constants import IMAGE_FORMATS, OBJECT
"""

from enum import Enum

DEFAULT_MPL_IMAGE_FIGSIZE = (8,6)

# Image format constants
class IMAGE_FORMATS(Enum):
    """Constants for supported _root_image formats."""
    NONE = None
    GRAYSCALE = 'GRAYSCALE'
    GRAYSCALE_SINGLE_CHANNEL = 'Grayscale (single channel)'
    HSV = 'HSV'
    RGB_OR_BGR = 'RGB/BGR (ambiguous)'
    RGBA_OR_BGRA = 'RGBA/BGRA (ambiguous)'
    RGB = 'RGB'
    RGBA = 'RGBA'
    BGR = 'BGR'
    BGRA = 'BGRA'
    SUPPORTED_FORMATS = (RGB, RGBA, GRAYSCALE, BGR, BGRA)
    MATRIX_FORMATS = (GRAYSCALE, GRAYSCALE_SINGLE_CHANNEL)
    AMBIGUOUS_FORMATS = (RGB_OR_BGR, RGBA_OR_BGRA)

    def is_matrix(self):
        return self in {IMAGE_FORMATS.GRAYSCALE, IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL}

    def is_array(self):
        return self in {IMAGE_FORMATS.RGB, IMAGE_FORMATS.RGBA, IMAGE_FORMATS.BGR, IMAGE_FORMATS.BGRA}

    def is_ambiguous(self):
        return self in {IMAGE_FORMATS.RGB_OR_BGR, IMAGE_FORMATS.RGBA_OR_BGRA}

    def is_none(self):
        return self is IMAGE_FORMATS.NONE

    CHANNELS_DEFAULT = 3
    DEFAULT_SCHEMA = RGB


# Object information constants
class OBJECT:
    """Constants for object information properties."""
    LABEL = 'ObjectLabel'
    CENTER_RR = 'Bbox_CenterRR'
    MIN_RR = 'Bbox_MinRR'
    MAX_RR = 'Bbox_MaxRR'
    CENTER_CC = 'Bbox_CenterCC'
    MIN_CC = 'Bbox_MinCC'
    MAX_CC = 'Bbox_MaxCC'


class IMAGE_IO:
    H5PY_MODULE_SIGNATURE = 'phenotypic'
    H5PY_IMAGE_GROUP = '_root_image'
    H5PY_IMAGE_SET_GROUP = 'image_set'


# Grid constants
class GRID:
    """
    Constants for grid structure in the PhenoTypic module.

    This class defines grid-related configurations, such as the number of rows and columns 
    in the grid, intervals between these rows and columns, and grid section information 
    like section number and index.
    """
    GRID_ROW_NUM = 'Grid_RowNum'
    GRID_ROW_INTERVAL = 'Grid_RowInterval'
    GRID_COL_NUM = 'Grid_ColNum'
    GRID_COL_INTERVAL = 'Grid_ColInterval'
    GRID_SECTION_NUM = 'Grid_SectionNum'
    GRID_SECTION_IDX = 'Grid_SectionIndex'


# Feature extraction constants
class GRID_LINREG_STATS_EXTRACTOR:
    """Constants for grid linear regression statistics extractor."""
    ROW_LINREG_M, ROW_LINREG_B = 'RowLinReg_M', 'RowLinReg_B'
    COL_LINREG_M, COL_LINREG_B = 'ColLinReg_M', 'ColLinReg_B'
    PRED_RR, PRED_CC = 'RowLinReg_PredRR', 'ColLinReg_PredCC'
    RESIDUAL_ERR = 'LinReg_ResidualError'


# Metadata constants
class METADATA_LABELS:
    """Constants for metadata labels."""
    UUID = 'UUID'
    IMAGE_NAME = 'ImageName'
    PARENT_IMAGE_NAME = 'ParentImageName'
    PARENT_UUID = 'ParentUUID'
    IMFORMAT = 'ImageFormat'
    SUBIMAGE_TYPE = 'SubimageType'


class SUBIMAGE_TYPES:
    """The string labels for different types of subimages generated when accessing subimages of a parent _root_image."""
    ORIGINAL = 'Original'
    CROP = 'Crop'
    OBJECT = 'Object'
    GRID = 'Grid'


