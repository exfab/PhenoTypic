INTERFACE_ERROR_MSG = "An abstract method was called when it was not supposed to be. Make sure any inherited classes properly overload this method."
NO_IMAGE_DATA_ERROR_MSG = 'No image has been loaded into this class. Use an io method or set the color_array or array equal to an image data array.'

NO_OUTPUT_ERROR_MSG = 'No output was returned in this operation'
OUTPUT_NOT_IMAGE_MSG = "This method's output is not a phenoscope Image object even though it should be."
OUTPUT_NOT_GRIDDED_IMAGE_MSG = "This method's output is not a phenoscope GriddedImage object even though it should be."
OUTPUT_NOT_TABLE_MSG = "This method's output is not a pandas DataFrame even though it should be."

INVALID_MASK_SHAPE_MSG = 'Object Mask shape should be the same as the image shape.'
INVALID_MAP_SHAPE_MSG = 'Object objects should be the same as the image shape.'

ARRAY_CHANGE_ERROR_MSG = 'The image array of the input was changed. This operation should not change the image array of the input.'
ENHANCED_ARRAY_CHANGE_ERROR_MSG = 'The enhanced image array of the input was changed. This operation should not change the enhanced image array of the input'
MASK_CHANGE_ERROR_MSG = 'The object mask of the input was changed. This operation should not change the object mask of the input'
MAP_CHANGE_ERROR_MSG = ' The object objects of the input was changed. This operation should not change the object objects of the input'

MISSING_MASK_ERROR_MSG = 'This image is missing an object mask. This operation requires the image to have an object mask. Run a detector on the image first.'
MISSING_MAP_ERROR_MSG = 'This image is missing an object mask. This operation requires the image to have an object mask. Run a detector on the image first.'

GRID_SERIES_INPUT_IMAGE_ERROR_MSG = 'For GridOperation classes with the exception of GridExtractor objects, the input must be an instance of the GriddedImage object type.'

from types import SimpleNamespace


class C_PhenoScopeModule:
    class UnknownError(Exception):
        def __init__(self):
            super().__init__('An unknown error occurred.')


"""
Image Operation
"""


class C_ImageOperation(C_PhenoScopeModule):
    """The Constants for ImageOperation classes."""
    class IntegrityError(AttributeError):
        def __init__(self, component, operation, image_name):
            super().__init__(
                f'The {operation} operation attempted to change the component {component} for image {image_name}. This operation should not change the component {component}.'
            )

    class OperationError(Exception):
        def __init__(self, operation, image_name, err_type, message):
            super().__init__(f'The operation: {operation} failed on image: {image_name}. {err_type}: {message}.')


"""
Image Handler
"""


class C_ImageHandler(C_PhenoScopeModule):
    """The Constants for ImageHandler classes."""
    class IllegalAssignmentError(AttributeError):
        def __init__(self, attr):
            super().__init__(
                f'The {attr} attribute should not directly assigned to a new object. If trying to change array elemennts use Image.{attr}[:]=value instead. If trying to change the image being represented use Image.set_image(new_image).'
            )

    class UuidAssignmentError(AttributeError):
        def __init__(self):
            super().__init__('The Image uuid should not be changed as this can lead to errors with data integrity')

    class NoArrayError(AttributeError):
        def __init__(self):
            super().__init__(
                "No array found. Either Input image was 2-D and had no array form. Set a multi-channel image or use a FormatConverter"
            )

    class NoObjectsError(AttributeError):
        def __init__(self, image_name):
            super().__init__(f'No objects currently in image "{image_name}". Apply a `Detector` to the Image object first or access image-wide information using Image.props')

    class EmptyImageError(AttributeError):
        def __init__(self):
            super().__init__(f'No image data loaded. Use Image.set_image(new_image) to load data.')

    class UnsupportedFileType(ValueError):
        def __init__(self, suffix):
            super().__init__(f'Image.imread() does not support file type: {suffix}')


class C_ImmutableImageComponent(C_PhenoScopeModule):
    """The Constants for ImmutableImageComponent classes."""
    class IllegalElementAssignmentError(AttributeError):
        """Exception raiased when trying to change the array/matrix elements directly. The User should use Image.set_image(new_image) instead."""

        def __init__(self, component_name):
            super().__init__(
                f'{component_name} components should not be changed directly. Change the {component_name} elements by using Image.set_image(new_image).'
            )


class C_ImageArraySubhandler(C_ImmutableImageComponent):
    """The Constants for ImageArraySubhandler classes."""
    class NoArrayError(AttributeError):
        def __init__(self):
            super().__init__(
                "No array form found. Either Input image was 2-D and had no array form. Set a multi-channel image or use a FormatConverter"
            )

    class InvalidSchemaHsv(TypeError):
        def __init__(self, schema):
            super().__init__(
                f'To be converted to HSV format, the schema should be RGB, but got {schema}'
            )


class C_ImageMatrixSubhandler(C_ImmutableImageComponent): pass


class C_MutableImageComponent(C_PhenoScopeModule):
    ARRAY_KEY_VALUE_SHAPE_MISMATCH = (
        "The shape of the value trying to be set is not the same as the section indicated by the key"
    )

    class ArrayKeyValueShapeMismatchError(AttributeError):
        def __init__(self):
            super().__init__(
                'The shape of the array being set does not match the shape of the section indicated being accessed'
            )

    class InputShapeMismatchError(AttributeError):
        def __init__(self, param_name):
            super().__init__(f'The shape of {param_name} must be the same shape as the Image.matrix')


class C_ObjectMask(C_MutableImageComponent):
    class InvalidValueTypeError(ValueError):
        def __init__(self, value_type):
            super().__init__(
                f'The mask array section was trying to be set with an array of type {value_type} and could not be cast to a boolean array.'
            )

    class InvalidScalarValueError(ValueError):
        def __init__(self):
            super().__init__(
                'The scalar value could not be converted to a boolean value. If value is an integer, it should be either 0 or 1.'
            )


class C_ImageDetectionMatrix(C_MutableImageComponent): pass


class C_ObjectMap(C_MutableImageComponent):
    class InvalidValueTypeError(ValueError):
        def __init__(self, value_type):
            super().__init__(
                f'ObjectMap elements were attempted to be set with {value_type}, but should only be set to an array of integers or an integer'
            )


class C_Metadata(C_PhenoScopeModule):
    LABELS = SimpleNamespace(
        UUID='UUID',
        IMAGE_NAME='ImageName',
        PARENT_IMAGE_NAME='ParentImageName',
        PARENT_UUID='ParentUUID',
        SCHEMA='Schema',
    )

    class UUIDReassignmentError(ValueError):
        def __init__(self):
            super().__init__('The uuid metadata should not be changed to preserve data integrity.')

    class MetadataKeyTypeError(ValueError):
        def __init__(self, type_recieved):
            super().__init__(f'The metadata key type must be a string, but got type {type_recieved}.')

    class MetadataKeySpacesError(ValueError):
        def __init__(self):
            super().__init__('The metadata keys should not have spaces in them.')

    class MetadataValueNonScalarError(ValueError):
        def __init__(self, type_value):
            super().__init__(f'The metadata values should be scalar values. Got type {type_value}.')


class C_HSVHandler(C_ImageArraySubhandler): pass


"""
Image Formats
"""


class C_ImageFormats:
    GRAYSCALE = 'Grayscale'
    GRAYSCALE_SINGLE_CHANNEL = 'Grayscale (single channel)'
    HSV = 'HSV'
    RGB_OR_BGR = 'RGB/BGR (ambiguous)'
    RGBA_OR_BGRA = 'RGBA/BGRA (ambiguous)'
    RGB = 'RGB'
    RGBA = 'RGBA'
    BGR = 'BGR'
    BGRA = 'BGRA'
    SUPPORTED_FORMATS = [RGB, RGBA, GRAYSCALE, BGR, BGRA]
    MATRIX_FORMATS = [GRAYSCALE, GRAYSCALE_SINGLE_CHANNEL]

    class UnsupportedFormatError(ValueError):
        def __init__(self, input_format):
            super().__init__(
                f"input image format {input_format} is not supported.  Accepted formats are ['RGB', 'RGBA','Grayscale','BGR','BGRA']"
            )


"""
Objects
"""


class C_ImageObjects(C_ImmutableImageComponent):
    class InvalidObjectLabel(ValueError):
        def __init__(self, label):
            super().__init__(
                f'The object with label {label} is not in the object map. If you meant to access the object by index use Image.objects.at() instead'
            )


class C_ObjectInfo(C_PhenoScopeModule):
    OBJECT_LABELS = 'ObjectLabel'

    CENTER_RR = 'Bbox_CenterRR'
    MIN_RR = 'Bbox_MinRR'
    MAX_RR = 'Bbox_MaxRR'

    CENTER_CC = 'Bbox_CenterCC'
    MIN_CC = 'Bbox_MinCC'
    MAX_CC = 'Bbox_MaxCC'


"""
Grid
"""


class C_Grid(C_PhenoScopeModule):
    """Represents a grid structure in the PhenoScope module.

    This class is designed to manage and define grid-related configurations,
    such as the number of rows and columns in the grid, intervals between these
    rows and columns, and grid section information like section number and
    index. The primary purpose of this class is to provide a structured and
    encapsulated way to handle grid parameters and their usage within the
    PhenoScope framework.

    Attributes:
        GRID_ROW_NUM (str): Key for the number of rows in the grid.
        GRID_ROW_INTERVAL (str): Key for the interval between rows in the grid.
        GRID_COL_NUM (str): Key for the number of columns in the grid.
        GRID_COL_INTERVAL (str): Key for the interval between columns in the grid.
        GRID_SECTION_NUM (str): Key for the number of sections in the grid.
        GRID_SECTION_IDX (str): Key for the index of a specific section in the grid.
    """
    GRID_ROW_NUM = 'Grid_RowNum'
    GRID_ROW_INTERVAL = 'Grid_RowInterval'

    GRID_COL_NUM = 'Grid_ColNum'
    GRID_COL_INTERVAL = 'Grid_ColInterval'

    GRID_SECTION_NUM = 'Grid_SectionNum'
    GRID_SECTION_IDX = 'Grid_SectionIndex'


"""
Features
"""


class C_GridLinRegStatsExtractor:
    ROW_LINREG_M, ROW_LINREG_B = 'RowLinReg_M', 'RowLinReg_B'
    COL_LINREG_M, COL_LINREG_B = 'ColLinReg_M', 'LinReg_B'
    PRED_RR, PRED_CC = 'RowLinReg_PredRR', 'ColLinReg_PredCC'
    RESIDUAL_ERR = 'LinReg_ResidualError'
