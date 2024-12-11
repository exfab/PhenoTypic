import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from typing_extensions import Self
from typing import Optional, Union

from ._imageShow import ImageShow

IMAGE_COUNT = 0


class ImageMetadata(ImageShow):
    """
    This class adds the ability to handle image metadata in a way that it can be inserted into extracted measurements.
    """

    LABEL_IMAGE_NAME = 'ImageName'

    def __init__(self, image: Optional[Union[np.ndarray, Self]] = None, name: Optional[str] = None):
        super().__init__(image=image)

        global IMAGE_COUNT
        self.IMAGE_NUMBER = IMAGE_COUNT
        IMAGE_COUNT = IMAGE_COUNT + 1

        if name is None:
            self.__metadata = {self.LABEL_IMAGE_NAME: f'{self.IMAGE_NUMBER:02}'}
        else:
            self.__metadata = {self.LABEL_IMAGE_NAME: name}

    @property
    def name(self):
        return self.__metadata[self.LABEL_IMAGE_NAME]

    @name.setter
    def name(self, value: str):
        self.__metadata[self.LABEL_IMAGE_NAME] = value

    def copy(self):
        new_img = super().copy()
        for key in self.__metadata.keys():
            new_img.set_metadata(key, self.__metadata[key])
        return new_img

    def set_metadata(self, key, value) -> None:
        try:
            str(value)
        except Exception:
            raise Exception('Value provided to function could not be represented as a string. Metadata values must be able to be represented as a string.')

        self.__metadata[key] = value

    def get_metadata_keys(self) -> list:
        return [self.__metadata.keys()]

    def get_metadata(self, key):
        return self.__metadata[key]

    def get_metadata_table(self) -> pd.DataFrame:
        if not isinstance(self.__metadata, dict):
            raise AttributeError(
                'The metadata attribute was changed previously from a dictionary. In order for functions to work properly, it must be a dict.')

        # Create a pandas dataframe that has all the object labels
        table = pd.DataFrame(regionprops_table(label_image=self.object_map, properties=['label']))

        # Insert metadata
        for key in self.__metadata.keys():
            table.insert(loc=1, column=f'Metadata_{key}', value=self.__metadata[key])

        # Set the labels as index to match other feature extraction convenction
        table.set_index(keys='label', inplace=True)
        return table
