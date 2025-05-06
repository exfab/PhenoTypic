"""
PhenoTypic Exceptions

This module contains all exceptions used throughout the PhenoTypic library.
Exceptions are organized by module and functionality.
"""


# Base exception class for all PhenoTypic exceptions
class PhenoTypicError(Exception):
    """Base exception class for all PhenoTypic errors."""
    pass


# General exceptions
class UnknownError(PhenoTypicError):
    """Exception raised when an unknown error occurs."""

    def __init__(self, message="An unknown error occurred."):
        super().__init__(message)


class InterfaceError(NotImplementedError, PhenoTypicError):
    """Exception raised when an abstract method is called when it's not supposed to be."""

    def __init__(self):
        super().__init__(
            "An abstract method was called when it was not supposed to be. Make sure any inherited classes properly overload this method."
            )


class NoOutputError(PhenoTypicError):
    """Exception raised when no output was returned in an operation."""

    def __init__(self):
        super().__init__("No output was returned in this operation")


class OutputValueError(PhenoTypicError):
    """Exception raised when output is not of the expected type."""

    def __init__(self, expected_type):
        super().__init__(f"This method's output is not a {expected_type} object even though it should be.")


class UnsupportedSchemaError(ValueError):
    """Represents an error when an unsupported imformat format is detected.

    This error is raised when a specified imformat format is not supported by the
    application or library that utilizes this exception class.

    Args:
        schema (str): The imformat format that triggered this exception.
    """

    def __init__(self, imformat):
        super().__init__(f"The format {schema} is not supported.")


# Image-related exceptions
class NoImageDataError(AttributeError):
    """Exception raised when no image data is loaded."""

    def __init__(self):
        super().__init__(
            "No image has been loaded into this class. Use an io method or set the color_array or array equal to an image data array."
            )


class InvalidShapeError(ValueError):
    """Exception raised when a shape mismatch occurs."""

    def __init__(self, component):
        super().__init__(f"Object {component} shape should be the same as the image shape.")


class DataIntegrityError(AttributeError):
    """Exception raised when data integrity is compromised."""

    def __init__(self, component, operation, image_name=None):
        image_str = f" for image {image_name}" if image_name else ""
        super().__init__(
            f"The {component} of the input was changed{image_str} by operation: {operation}. This operation should not change the {component} of the input."
            )


class NoComponentError(AttributeError):
    """Exception raised when a required object is missing."""

    def __init__(self, component):
        super().__init__(f"This image does not have a {component}.")


# Image Operation exceptions
class ImageOperationError(PhenoTypicError):
    """Base exception for image operation errors."""
    pass


class OperationIntegrityError(AttributeError):
    """Exception raised when an operation attempts to change a component it shouldn't."""

    def __init__(self, component, operation, image_name=None):
        image_str = f" for image {image_name}" if image_name else ""
        super().__init__(
            f"The {operation} operation attempted to change the component {component}{image_str}. This operation should not change the component {component}."
            )


class OperationFailedError(ImageOperationError):
    """Exception raised when an operation fails."""

    def __init__(self, operation, image_name, err_type, message):
        super().__init__(f"The operation: {operation} failed on image: {image_name}. {err_type}: {message}.")


# Image Handler exceptions
class IllegalAssignmentError(ValueError):
    """Exception raised when trying to directly assign a new object to a protected attribute."""

    def __init__(self, attr):
        super().__init__(
            f"The {attr} attribute should not be directly assigned to a new object. If trying to change array elements use Image.{attr}[:]=value instead. If trying to change the image being represented use Image.set_image(new_image)."
            )


class UuidAssignmentError(AttributeError):
    """Exception raised when trying to change the UUID of an image."""

    def __init__(self):
        super().__init__("The Image uuid should not be changed as this can lead to errors with data integrity")


class NoArrayError(AttributeError):
    """Exception raised when no array is found."""

    def __init__(self):
        super().__init__(
            "No array form found. Either input image was 2-D and had no array form. Set a multi-channel image or use a FormatConverter"
            )


class NoObjectsError(AttributeError):
    """Exception raised when no objects are found in an image."""

    def __init__(self, image_name=None):
        image_str = f' "{image_name}"' if image_name else ""
        super().__init__(
            f"No objects currently in image:{image_str}. Apply a `Detector` to the Image object first or access image-wide information using Image.props"
            )


class EmptyImageError(AttributeError):
    """Exception raised when no image data is loaded."""

    def __init__(self):
        super().__init__("No image data loaded. Use Image.set_image(new_image) to load data.")


class UnsupportedFileTypeError(ValueError):
    """Exception raised when trying to read an unsupported file type."""

    def __init__(self, suffix):
        super().__init__(f"Image.imread() does not support file type: {suffix}")


# Component exceptions
class ImmutableComponentError(AttributeError):
    """Base exception for immutable component errors."""
    pass


class IllegalElementAssignmentError(ImmutableComponentError):
    """Exception raised when trying to change array/matrix elements directly."""

    def __init__(self, component_name):
        super().__init__(
            f"{component_name} components should not be changed directly. Change the {component_name} elements by using Image.set_image(new_image)."
            )


class InvalidHsvSchemaError(AttributeError):
    """Exception raised when trying to convert to HSV from a non-RGB imformat."""

    def __init__(self, imformat):
        super().__init__(f"To be converted to HSV format, the imformat should be RGB, but got {schema}")


# Mutable component exceptions
class ArrayKeyValueShapeMismatchError(ValueError):
    """Exception raised when the shape of the value doesn't match the key's section."""

    def __init__(self):
        super().__init__("The shape of the array being set does not match the shape of the section indicated being accessed")


class InputShapeMismatchError(ValueError):
    """Exception raised when input shape doesn't match the image matrix."""

    def __init__(self, param_name):
        super().__init__(f"The shape of {param_name} must be the same shape as the Image.matrix")


# Mask exceptions
class InvalidMaskValueError(ValueError):
    """Exception raised when trying to set mask with invalid value type."""

    def __init__(self, value_type):
        super().__init__(
            f"The mask array section was trying to be set with an array of type {value_type} and could not be cast to a boolean array."
            )


class InvalidMaskScalarValueError(ValueError):
    """Exception raised when trying to set mask with invalid scalar value."""

    def __init__(self):
        super().__init__("The scalar value could not be converted to a boolean value. If value is an integer, it should be either 0 or 1.")


# Object map exceptions
class InvalidMapValueError(ValueError):
    """Exception raised when trying to set object map with invalid value type."""

    def __init__(self, value_type):
        super().__init__(
            f"ObjectMap elements were attempted to be set with {value_type}, but should only be set to an array of integers or an integer"
            )


# Metadata exceptions
class UUIDReassignmentError(AttributeError):
    """Exception raised when trying to change the UUID metadata."""

    def __init__(self):
        super().__init__("The uuid metadata should not be changed to preserve data integrity.")


class MetadataKeyValueError(ValueError):
    """Exception raised when metadata key is not a string."""

    def __init__(self, type_received):
        super().__init__(f"The metadata key type must be a string, but got type {type_received}.")


class MetadataKeySpacesError(TypeError):
    """Exception raised when metadata key contains spaces."""

    def __init__(self):
        super().__init__("The metadata keys should not have spaces in them.")


class MetadataValueNonScalarError(TypeError):
    """Exception raised when metadata value is not scalar."""

    def __init__(self, type_value):
        super().__init__(f"The metadata values should be scalar values. Got type {type_value}.")


# Object-related exceptions
class ObjectNotFoundError(AttributeError):
    """Exception raised when an object with a specified label is not found."""

    def __init__(self, label):
        super().__init__(
            f"The object with label {label} is not in the object map. If you meant to access the object by index use Image.objects.at() instead"
            )


# Grid exceptions
class GridImageInputError(ValueError):
    """Exception raised when a non-GriddedImage object is provided as input."""

    def __init__(self):
        super().__init__(
            "For GridOperation classes with the exception of GridExtractor objects, the input must be an instance of the GriddedImage object type."
            )
