import numpy as np

from typing_extensions import Self
from typing import Optional, Union

from ._imageMetadata import ImageMetadata

from core._image_components._measurement_container import MeasurementContainer

class ImageMeasurements(ImageMetadata):
    """
    This class adds the ability to track image measurements
    """

    def __init__(self, image: Optional[Union[np.ndarray, Self]] = None, name: Optional[str] = None):
        super().__init__(image=image, name=name)

        self.__measurements: MeasurementContainer = MeasurementContainer()

    @property
    def measurements(self)->MeasurementContainer:
        return self.__measurements

    @measurements.setter
    def measurements(self, value: MeasurementContainer):
        if type(value) is not MeasurementContainer:
            raise TypeError("measurements must be a MeasurementContainer")
        else:
            self.__measurements.clear()
            for key in value.keys():
                self.__measurements[key] = value[key].copy()

    def copy(self):
        new_img = super().copy()
        new_img.__measurements = self.__measurements.copy()
        return new_img

