import skimage
from ._image import Image
from pathlib import Path


def imread(filepath: str):
    filepath = Path(filepath)
    return Image().imread(filepath)
