from .handlers._image_io_handler import ImageIOHandler


class Image(ImageIOHandler):
    """A comprehensive class for handling _root_image processing, including manipulation, information sync, metadata management, and format conversion.

    The `Image` class is designed to load, process, and manage _root_image data using different
    representation formats (e.g., arrays and matrices). This class allows for metadata editing,
    schema definition, and subcomponent handling to streamline _root_image processing tasks.

    Note:
        - If the input_image is 2-D, the ImageHandler leaves the array form as empty
        - If the input_image is 3-D, the ImageHandler will automatically set the matrix component to the grayscale representation.
        - Added in v0.5.0, HSV handling support
    """
    pass