from ._image import Image
from pathlib import Path


def imread(filepath: str, imparams: dict = None):
    """
    Reads an image from the specified file path using the `Image().imread` method.

    Args:
        filepath (str): The path to the image file to be read.

    Returns:
        Image: An `Image` object containing the data read from the specified file.
    """
    filepath = Path(filepath)
    imparams = imparams or {}
    return Image.imread(filepath, **imparams)
