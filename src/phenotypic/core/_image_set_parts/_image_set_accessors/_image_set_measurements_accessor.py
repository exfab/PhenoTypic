from __future__ import annotations
from typing import TYPE_CHECKING, Dict

import h5py
import numpy as np

if TYPE_CHECKING: from phenotypic import ImageSet

import pandas as pd
from pandas.api.types import pandas_dtype, is_extension_array_dtype, is_integer_dtype, is_bool_dtype
from phenotypic.tools.constants_ import IO


# TODO: Not fully integrated yet
class SetMeasurementAccessor:

    def __init__(self, image_set: ImageSet):
        self._image_set = image_set

    def table(self) -> pd.DataFrame:
        measurements = []
        with self._image_set.hdf_.swmr_reader() as reader:
            images = self._image_set.hdf_.get_data_group(reader)
            for image_name in images.keys():
                image_group = images[image_name]
                if self._image_set.hdf_.IMAGE_MEASUREMENT_SUBGROUP_KEY in image_group:
                    measurements.append(self._load_dataframe_from_hdf5_group(group=image_group,
                                                                             measurement_key=self._image_set.hdf_.IMAGE_MEASUREMENT_SUBGROUP_KEY))
        return pd.concat(measurements) if measurements else pd.DataFrame()
