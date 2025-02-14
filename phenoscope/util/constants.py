INTERFACE_ERROR_MSG = "An interface method was called when it was not supposed to be. Make sure any inherited classes properly overload this method."
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


class C_PhenoScopeModule:
    class UnknownError(Exception):
        def __init__(self):
            super().__init__('An unknown error occurred.')


"""
Image Handler
"""


class C_ImageHandler(C_PhenoScopeModule):
    class IllegalAssignmentError(AttributeError):
        def __init__(self, attr):
            super().__init__(
                f'The {attr} attribute should not directly assigned to a new object. If trying to change array elemennts use Image.{attr}[:]=value instead. If trying to change the image being represented use Image.set_image(new_image).'
            )

    class uuidAssignmentError(AttributeError):
        def __init__(self):
            super().__init__('The Image uuid should not be changed as this can lead to errors with data integrity')

    class NoArrayError(AttributeError):
        def __init__(self):
            super().__init__(
                "No array form found. Either Input image was 2-D and had no array form. Push image through a FormatConvertor in order to add an array component."
                )

    class EmptyImageError(AttributeError):
        def __init__(self):
            super().__init__('No array or matrix form was found. Use Image.set_image(new_image) to load data.')

    AMBIGUOUS_IMAGE_FORMAT = 'Input image array has an ambiguous image format that could be either RGB or BGR. RGB was assumed. To suppress warning, specify image format constructor or Image.set_image()'


class C_ImmutableImageComponent(C_PhenoScopeModule):
    class IllegalElementAssignmentError(AttributeError):
        """Exception raiased when trying to change the array/matrix elements directly. The User should use Image.set_image(new_image) instead."""

        def __init__(self, component_name):
            super().__init__(
                f'{component_name} elements should not be changed directly. Change the {component_name} elements by using Image.set_image(new_image).'
            )


class C_ImageArray(C_ImmutableImageComponent): pass


class C_ImageMatrix(C_ImmutableImageComponent): pass


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


class C_ImageDetectionMatrix(C_MutableImageComponent): pass


class ConstObjectMask(C_MutableImageComponent):
    class InvalidValueTypeError(AttributeError):
        def __init__(self, value_type):
            super().__init__(
                f'The mask array slice was trying to be set with an array of type {value_type} and could not be cast to a boolean array.'
                )


class ConstObjectMap(C_MutableImageComponent): pass


class C_MetadataContainer(C_PhenoScopeModule):
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
    SUPPORTED_FORMATS = [RGB, RGBA, GRAYSCALE, BGR, BGRA, HSV]

    class UnsupportedFormatError(ValueError):
        def __init__(self, input_format):
            super().__init__(
                f"input image format {input_format} is not supported.  Accepted formats are ['RGB', 'RGBA','Grayscale','BGR','BGRA','HSV']"
            )


"""
Objects
"""


class C_ObjectInfo(C_PhenoScopeModule):
    OBJECT_MAP_ID = 'ObjectLabel'

    CENTER_RR = 'Bbox_CenterRR'
    MIN_RR = 'Bbox_MinRR'
    MAX_RR = 'Bbox_MaxRR'

    CENTER_CC = 'Bbox_CenterCC'
    MIN_CC = 'Bbox_MinCC'
    MAX_CC = 'Bbox_MaxCC'
