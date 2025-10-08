"""
PhenoTypic Constants

This module contains constant values and enumerations used throughout the PhenoTypic library.
Constants are organized by module and functionality.

Note: Class names are defined in ALL_CAPS to avoid namespace conflicts with actual classes 
    in the codebase (e.g., GRID_DEP vs an actual Grid class). When importing, use the format:
        from PhenoTypic.util.constants import IMAGE_FORMATS, OBJECT
"""

from phenotypic._shared_modules._measurement_info import MeasurementInfo
import phenotypic
from enum import Enum
from packaging.version import Version
from pathlib import Path

DEFAULT_MPL_IMAGE_FIGSIZE = (8, 6)


class MPL:
    """Holds defaults for matplotlib parameters"""
    FIGSIZE = (8, 6)


# Image format constants
class IMAGE_FORMATS(Enum):
    """Constants for supported image formats."""
    NONE = None
    GRAYSCALE = 'GRAYSCALE'
    GRAYSCALE_SINGLE_CHANNEL = 'Grayscale (single channel)'
    HSV = 'HSV'
    RGB_OR_BGR = 'RGB/BGR (ambiguous)'
    RGBA_OR_BGRA = 'RGBA/BGRA (ambiguous)'
    RGB = 'RGB'
    LINEAR_RGB = 'LINEAR RGB'
    RGBA = 'RGBA'
    BGR = 'BGR'
    BGRA = 'BGRA'
    SUPPORTED_FORMATS = (RGB, RGBA, GRAYSCALE, BGR, BGRA)
    MATRIX_FORMATS = (GRAYSCALE, GRAYSCALE_SINGLE_CHANNEL)
    AMBIGUOUS_FORMATS = (RGB_OR_BGR, RGBA_OR_BGRA)

    def is_matrix(self):
        return self in {IMAGE_FORMATS.GRAYSCALE, IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL}

    def is_array(self):
        return self in {IMAGE_FORMATS.RGB,
                        IMAGE_FORMATS.RGBA,
                        IMAGE_FORMATS.BGR,
                        IMAGE_FORMATS.BGRA,
                        IMAGE_FORMATS.LINEAR_RGB}

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


class BBOX(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return 'Bbox'

    CENTER_RR = 'CenterRR', 'The row coordinate of the center of the bounding box.'
    MIN_RR = 'MinRR', 'The smallest row coordinate of the bounding box.'
    MAX_RR = 'MaxRR', 'The largest row coordinate of the bounding box.'
    CENTER_CC = 'CenterCC', ' The column coordinate of the center of the bounding box.'
    MIN_CC = 'MinCC', ' The smallest column coordinate of the bounding box.'
    MAX_CC = 'MaxCC', ' The largest column coordinate of the bounding box.'


class IO:
    RAW_FILE_EXTENSIONS = ('.cr3', '.CR3')
    PNG_FILE_EXTENSIONS = ('.png', '.PNG')
    JPEG_FILE_EXTENSIONS = ('.jpeg', '.JPEG')
    TIFF_EXTENSIONS = ('.tif', '.tiff')
    ACCEPTED_FILE_EXTENSIONS = PNG_FILE_EXTENSIONS + JPEG_FILE_EXTENSIONS + TIFF_EXTENSIONS

    if Version(phenotypic.__version__) < Version("0.7.1"):
        SINGLE_IMAGE_HDF5_PARENT_GROUP = Path(f'phenotypic/')
    else:
        SINGLE_IMAGE_HDF5_PARENT_GROUP = f'/phenotypic/images/'

    IMAGE_SET_HDF5_PARENT_GROUP = f'/phenotypic/image_sets/'

    IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY = 'measurements'
    IMAGE_STATUS_SUBGROUP_KEY = "status"


class PIPE_STATUS(MeasurementInfo):
    """Constants for image set status."""

    @classmethod
    def category(cls) -> str:
        return 'Status'

    PROCESSED = 'Processed', "Whether the image has been processed successfully."
    MEASURED = 'Measured', "Whether the image has been measured successfully."
    # ERROR = 'Error', "Whether the image has encountered an error during processing."
    # INVALID_ANALYSIS = (
    #     'AnalysisInvalid',
    #     'Whether the image measurements are considered invalid. '
    #     'This can be set during measurement extraction or post-processing.'
    # )
    # INVALID_SEGMENTATION = 'SegmentationInvalid', "Whether the image segmentation is considered valid."


class GRID(MeasurementInfo):
    """Constants for grid structure in the PhenoTypic module."""

    @classmethod
    def category(cls) -> str:
        return 'Grid'

    ROW_NUM = 'RowNum', 'The row idx of the object'
    ROW_INTERVAL_START = 'RowIntervalStart', 'The start of the row interval of the object'
    ROW_INTERVAL_END = 'RowIntervalEnd', 'The end of the row interval of the object'
    ROW_INTERVAL = 'RowInterval', 'The row interval of the object'

    COL_NUM = 'ColNum', 'The column idx of the object'
    COL_INTERVAL_START = 'ColIntervalStart', 'The start of the column interval of the object'
    COL_INTERVAL_END = 'ColIntervalEnd', 'The end of the column interval of the object'
    COL_INTERVAL = 'ColInterval', 'The column interval of the object'

    SECTION_NUM = 'SectionNum', 'The section number of the object. Ordered left to right, top to bottom'
    SECTION_IDX = 'SectionIndex', 'The section index of the object'


# Feature extraction constants
# TODO: Fix this constant access pattern
class GRID_LINREG_STATS_EXTRACTOR:
    """Constants for grid linear regression statistics extractor."""
    ROW_LINREG_M, ROW_LINREG_B = 'RowLinReg_M', 'RowLinReg_B'
    COL_LINREG_M, COL_LINREG_B = 'ColLinReg_M', 'ColLinReg_B'
    PRED_RR, PRED_CC = 'RowLinReg_PredRR', 'ColLinReg_PredCC'
    RESIDUAL_ERR = 'LinReg_ResidualError'


# Metadata constants
class METADATA(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return 'Metadata'

    # Metadata values are not prepended with the category
    def __new__(cls, label: str, desc: str | None = None):
        full = f"{label}"
        obj = str.__new__(cls, full)
        obj._value_ = label
        obj.label = label
        obj.desc = desc or label
        obj.pair = (label, obj.desc)
        return obj

    """Constants for metadata labels."""
    UUID = 'UUID', 'The unique identifier of the image.'
    IMAGE_NAME = 'ImageName', 'The name of the image.'
    PARENT_IMAGE_NAME = 'ParentImageName', 'The name of the parent image.'
    PARENT_UUID = 'ParentUUID', 'The UUID of the parent image.'
    IMFORMAT = 'ImageFormat', 'The format of the image.'
    IMAGE_TYPE = 'ImageType', 'The type of the image.'
    BIT_DEPTH = 'BitDepth', 'The bit depth of the image.'


class IMAGE_TYPES(Enum):
    """The string labels for different types of images generated when accessing subimages of a parent image."""
    BASE = 'Base'
    CROP = 'Crop'
    OBJECT = 'Object'
    GRID = 'GridImage'
    GRID_SECTION = 'GridSection'

    def __str__(self):
        return self.value
