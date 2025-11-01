__version__ = "0.9.1"

# Configure matplotlib to use non-interactive backend
import matplotlib

matplotlib.use('Agg')

from .core._grid_image import GridImage
from .core._image import Image
from .core._image_pipeline import ImagePipeline
from .core._image_set import ImageSet
from .core._imread import imread

from . import (ABC_,
               analysis,
               correction,
               data,
               detection,
               enhance,
               grid,
               measure,
               objedit,
               tools,
               prefab)

__all__ = [
    "Image",  # Class imported from core
    "imread",  # Function imported from core
    "GridImage",  # Class imported from core
    "ImagePipeline",
    "ImageSet",
    "ABC_",
    "analysis",
    "data",
    "detection",
    "measure",
    "grid",
    "objedit",
    "prefab",
    "correction",
    "enhance",
    "tools",
]
