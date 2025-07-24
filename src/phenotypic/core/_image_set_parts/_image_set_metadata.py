from __future__ import annotations
from typing import TYPE_CHECKING, Literal, List

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from os import PathLike
from ._image_set_measurements import ImageSetMeasurements
from ._image_set_accessors._image_set_metadata_accessor import ImageSetMetadataAccessor

class ImageSetMetadata(ImageSetMeasurements):

    def __init__(self,
                 name: str,
                 image_template: Image | None = None,
                 image_list: List[Image] | None = None,
                 src_path: PathLike | None = None,
                 out_path: PathLike | None = None,
                 overwrite: bool = False, ):
        super().__init__(name=name, image_template=image_template,
                         image_list=image_list, src_path=src_path,
                         out_path=out_path, overwrite=overwrite)
        self._metadata_accessor = ImageSetMetadataAccessor(self)


    @property
    def metadata(self) -> ImageSetMetadataAccessor:
        return self._metadata_accessor