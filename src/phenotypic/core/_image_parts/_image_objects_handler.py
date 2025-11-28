from __future__ import annotations

from typing import TYPE_CHECKING

from ...tools.exceptions_ import IllegalAssignmentError, NoObjectsError

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from phenotypic.core._image_parts.accessors import ObjectsAccessor
from ._image_handler import ImageHandler


class ImageObjectsHandler(ImageHandler):
    """Adds the ability to isolate and work with specific objects in an image.

    This class extends ImageHandler with object-oriented functionality, enabling:
    - Access to detected/segmented objects through the objects accessor
    - Individual object measurement calculations
    - Object filtering and manipulation
    - Seamless integration with object detection and analysis workflows

    The class provides a unified interface for working with image objects while
    maintaining consistency with the underlying image data representations.

    Attributes:
        _accessors (SimpleNamespace): Extended accessor container including an
            ObjectsAccessor instance for object-specific operations.
    """

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 name: str | None = None,
                 bit_depth: int | None = None, ):
        """Initialize ImageObjectsHandler with object detection support.

        Args:
            arr (np.ndarray | Image | None): Optional initial image data. Defaults to None.
            name (str | None): Optional image name. Defaults to None.
            bit_depth (int | None): Optional bit depth (8 or 16). Defaults to None.
        """
        super().__init__(arr=arr, name=name, bit_depth=bit_depth)
        self._accessors.objects = ObjectsAccessor(self)

    @property
    def objects(self) -> ObjectsAccessor:
        """Accessor for performing operations on detected objects in the image.

        Provides access to individual or grouped objects detected in the image,
        enabling measurement calculations, filtering, and object-specific analyses.
        Objects are identified through the object map (objmap) component which stores
        integer labels for each detected object.

        Returns:
            ObjectsAccessor: An accessor instance that manages object-specific operations
                and measurements.

        Raises:
            NoObjectsError: If no objects are present in the image. This occurs when
                num_objects == 0, indicating that either no object detection has been
                performed yet, or the detection found no objects. Apply an ObjectDetector
                first to identify and label objects.

        Examples:
            >>> img = Image.imread('sample.jpg')
            >>> detector = ObjectDetector()
            >>> detector.detect(img)
            >>> obj_accessor = img.objects
            >>> measurements = img.objects.measure.area()
        """
        if self.num_objects == 0:
            raise NoObjectsError(self.name)
        else:
            return self._accessors.objects

    def info(self, include_metadata: bool = True):
        return self.objects.info(include_metadata=include_metadata)

    @objects.setter
    def objects(self, objects):
        raise IllegalAssignmentError('objects')
