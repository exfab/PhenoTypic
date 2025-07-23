"""
PhenoTypic Constants

This module contains constant values and enumerations used throughout the PhenoTypic library.
Constants are organized by module and functionality.

Note: Class names are defined in ALL_CAPS to avoid namespace conflicts with actual classes 
    in the codebase (e.g., GRID vs an actual Grid class). When importing, use the format:
        from PhenoTypic.util.constants import IMAGE_FORMATS, OBJECT
"""
from textwrap import dedent

import phenotypic
from enum import Enum
from packaging.version import Version
from pathlib import Path


class MeasurementInfo(Enum):
    """
    The labels and descriptions of the measurement information. This class helps with module consistency and documentation automation.

    Note:
        - Overwrite the CATEGORY label and add other measurement info here

    """

    @property
    def CATEGORY(self) -> str:
        """Overwrite this in inherited classes; should return a string with the category name"""
        raise NotImplementedError

    def __init__(self, label, desc=None):
        self.label, self.desc = label, desc

    def __str__(self):
        return f'{self.CATEGORY}_{self.label}'

    @classmethod
    def iter_labels(cls):
        """Yield all measurement info members except CATEGORY."""
        return (member for member in cls if member is not cls.CATEGORY)

    @classmethod
    def get_headers(cls):
        """Return full measurement info labels for use in pandas dataframe columns."""
        return [f'{x}' for x in cls.iter_labels() if cls.name.endswith('_') is False]

    @classmethod
    def rst_table(
            cls,
            *,
            title: str | None = None,
            header: tuple[str, str] = ("Label", "Description"),
    ) -> str:
        """
        Generates an RST table in the "list-table" format with the specified title and
        header. Includes rows based on the class's iterable members that provide labels
        and descriptions.

        Args:
            title: Optional title for the table. If none is provided, the name of the
                class is used as the default title.
            header: A tuple containing the header labels for the table. Defaults to
                ("Label", "Description").

        Returns:
            str: A string containing the formatted RST table.
        """
        title = title or cls.__name__
        left, right = header

        lines: list[str] = [
            f".. list-table:: {title}",
            "   :header-rows: 1",
            "",
            f"   * - {left}",
            f"     - {right}",
        ]

        for member in cls.iter_labels():
            lines.extend(
                [
                    f"   * - ``{member.label}``",
                    f"     - {member.desc}",
                ],
            )
        return dedent("\n".join(lines))

    @classmethod
    def append_rst_to_doc(cls, module) -> str:
        """
        returns a string with the RST table appended to the module docstring.
        """
        if isinstance(module, str):
            return module + "\n\n" + cls.rst_table()
        else:
            return module.__doc__ + "\n\n" + cls.rst_table()


DEFAULT_MPL_IMAGE_FIGSIZE = (8, 6)

if Version(phenotypic.__version__) < Version("0.7.1"):
    SINGLE_IMAGE_HDF5_PARENT_GROUP = Path(f'phenotypic/')
else:
    SINGLE_IMAGE_HDF5_PARENT_GROUP = Path(f'phenotypic/Image/')

IMAGE_SET_PARENT_GROUP = f'phenotypic/ImageSet/'


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


class BBOX(MeasurementInfo):
    @property
    def CATEGORY(self) -> str:
        return 'Bbox'

    CENTER_RR = 'CenterRR', 'The row coordinate of the center of the bounding box.'
    MIN_RR = 'MinRR', 'The smallest row coordinate of the bounding box.'
    MAX_RR = 'MaxRR', 'The largest row coordinate of the bounding box.'
    CENTER_CC = 'CenterCC', ' The column coordinate of the center of the bounding box.'
    MIN_CC = 'MinCC', ' The smallest column coordinate of the bounding box.'
    MAX_CC = 'MaxCC', ' The largest column coordinate of the bounding box.'


class IO:
    ACCEPTED_FILE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    if Version(phenotypic.__version__) < Version("0.7.1"):
        SINGLE_IMAGE_HDF5_PARENT_GROUP = Path(f'phenotypic/')
    else:
        SINGLE_IMAGE_HDF5_PARENT_GROUP = f'/phenotypic/images/'

    IMAGE_SET_HDF5_PARENT_GROUP = f'/phenotypic/image_sets/'

    IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY = 'measurements'
    IMAGE_STATUS_SUBGROUP_KEY = "status"


class SET_STATUS(MeasurementInfo):
    """Constants for image set status."""
    # should be placed in <image_name>/{SUBGROUP_KEY}
    SUBGROUP_KEY_ = "status"

    @property
    def CATEGORY(self) -> str:
        return 'Status'

    PROCESSED = 'Processed', "Whether the image has been processed successfully."
    MEASURED = 'Measured', "Whether the image has been measured successfully."
    ERROR = 'Error', "Whether the image has encountered an error during processing."
    VALID_ANALYSIS = (
        'AnalysisValid',
        'Whether the image measurements are considered valid. '
        'This can be set during measurement extraction or post-processing.'
    )
    VALID_SEGMENTATION = 'SegmentationValid', "Whether the image segmentation is considered valid."


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
    IMAGE_TYPE = 'ImageType'


class IMAGE_TYPES(Enum):
    """The string labels for different types of images generated when accessing subimages of a parent image."""
    BASE = 'Base'
    CROP = 'Crop'
    OBJECT = 'Object'
    GRID = 'GridImage'
    GRID_SECTION = 'GridSection'

    def __str__(self):
        return self.value
