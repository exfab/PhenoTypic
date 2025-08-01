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
                 src: List[Image] | PathLike | None = None,
                 outpath: PathLike | None = None,
                 overwrite: bool = False, ):
        super().__init__(name=name, image_template=image_template,
                         src=src, outpath=outpath, overwrite=overwrite)
        self._metadata_accessor = ImageSetMetadataAccessor(self)


    @property
    def metadata(self) -> ImageSetMetadataAccessor:
        return self._metadata_accessor