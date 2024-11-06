import skimage
from ._image import Image
from pathlib import Path


def imread(filepath: str):
    return Image(skimage.io.imread(Path(filepath)))
