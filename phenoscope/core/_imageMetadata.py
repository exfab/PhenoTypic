import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from typing_extensions import Self
from typing import Optional, Union

from ._imageShow import ImageShow

from ._image_components._metadata_container import MetadataContainer
from ..util.labels.metadata import IMAGE_NAME


class ImageMetadata(ImageShow):
    """
    This class adds the ability to handle image metadata in a way that it can be inserted into extracted measurements.
    """

    def __init__(self, image: Optional[Union[np.ndarray, Self]] = None, name: Optional[str] = None):
        super().__init__(image=image)

        self.__metadata: MetadataContainer = MetadataContainer()

        if name is None:
            self.__metadata[IMAGE_NAME] = self.uuid
        else:
            self.__metadata[IMAGE_NAME] = name

    @property
    def name(self):
        return self.__metadata[IMAGE_NAME]

    @name.setter
    def name(self, value: str):
        if type(value) is not str:
            raise TypeError("name must be a string")
        self.__metadata[IMAGE_NAME] = value

    @property
    def metadata(self)->MetadataContainer:
        return self.__metadata

    @metadata.setter
    def metadata(self, value: MetadataContainer):
        if type(value) is not MetadataContainer:
            raise TypeError("metadata must be a MetadataContainer")
        else:
            self.__metadata.clear()
            for key in value.keys():
                self.__metadata[key] = value[key]

    def copy(self):
        new_img = super().copy()
        new_img.__metadata = self.__metadata.copy()
        return new_img

    def get_metadata_table(self) -> pd.DataFrame:
        # Create a pandas dataframe that has all the object labels
        table = pd.DataFrame(regionprops_table(label_image=self.object_map, properties=['label']))

        # Insert metadata
        for key in self.metadata.keys():
            table.insert(loc=1, column=f'Metadata_{key}', value=self._metadata[key])

        # Set the labels as index to match other feature extraction convenction
        table.set_index(keys='label', inplace=True)
        return table
