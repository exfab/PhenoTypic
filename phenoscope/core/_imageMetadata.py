import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from typing_extensions import Self
from typing import Optional, Union

from ._imageIO import ImageIO

IMAGE_COUNT = 0


class ImageMetadata(ImageIO):
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
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, value: dict):
        raise ValueError('The metadata dictionary itself should not be changed. The values itself are mutable using dictionary accessors.')

    @property
    def name(self):
        return self.__metadata[self.LABEL_IMAGE_NAME]

    @name.setter
    def name(self, value: str):
        self.__metadata[self.LABEL_IMAGE_NAME] = value

    def copy(self):
        new_img = super().copy()
        for key in self.metadata.keys():
            new_img.metadata[key] = self.metadata[key]
        return new_img

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
