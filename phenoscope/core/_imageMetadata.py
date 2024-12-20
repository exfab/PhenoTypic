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

        self._metadata = {}
        self._metadata_dtype = {}

        if name is None:
            self.set_metadata(self.LABEL_IMAGE_NAME, f'{IMAGE_COUNT:02d}')
        else:
            self.set_metadata(self.LABEL_IMAGE_NAME, name)

    @property
    def name(self):
        return self._metadata[self.LABEL_IMAGE_NAME]

    @name.setter
    def name(self, value: str):
        self.set_metadata(self.LABEL_IMAGE_NAME, value)

    def copy(self):
        new_img = super().copy()
        for key in self._metadata.keys():
            new_img.set_metadata(key, self._metadata[key])
        return new_img

    def set_metadata(self, key, value) -> None:
        acceptable_dtypes = (int, float, str, bool)
        if not isinstance(value, acceptable_dtypes): raise ValueError(f'value must be of types {acceptable_dtypes}')

        self._metadata[key] = value
        self._metadata_dtype[key] = f'{type(value).__name__}'

    def get_metadata_keys(self) -> list:
        return list(self._metadata.keys())

    def get_metadata(self, key):
        return self._metadata[key]

    def _get_metadata_dtype_name(self, key)->str:
        """
        Returns the name of the metadata dtype.
        :param key:
        :return: (str) The name of the dtype
        """
        return self._metadata_dtype[key]

    def validate_metadata_dtype(self):
        """
        Validates the dtypes of the metadata values to match the expected dtypes. This helps ensure the saving/loading
        of Image functions properly, but can also have other use cases.
        :return:
        """
        for key in self._metadata.keys():
            value = self._metadata[key]
            match self._get_metadata_dtype_name(key):
                case 'str': self._metadata[key] = str(value)
                case 'int': self._metadata[key] = int(value)
                case 'float': self._metadata[key] = float(value)
                case 'bool': self._metadata[key] = bool(value)
                case _: self._metadata[key] = value

    def get_metadata_table(self) -> pd.DataFrame:
        if not isinstance(self._metadata, dict):
            raise AttributeError(
                'The metadata attribute was changed previously from a dictionary. In order for functions to work properly, it must be a dict.')

        # Create a pandas dataframe that has all the object labels
        table = pd.DataFrame(regionprops_table(label_image=self.object_map, properties=['label']))

        # Insert metadata
        for key in self._metadata.keys():
            table.insert(loc=1, column=f'Metadata_{key}', value=self._metadata[key])

        # Set the labels as index to match other feature extraction convenction
        table.set_index(keys='label', inplace=True)
        return table
