from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Literal, List

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from ._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
from phenotypic.util.constants_ import SET_STATUS
from ._image_set_status import ImageSetStatus


class ImageSetMeasurements(ImageSetStatus):
    """
    This class adds measurement handling to the ImageSetStatus class.
    """
    def __init__(self,
                 name: str,
                 image_template: Image | None = None,
                 src: List[Image] | PathLike | None = None,
                 out_path: PathLike | None = None,
                 overwrite: bool = False, ):
        super().__init__(name=name, image_template=image_template,
                         src=src, out_path=out_path, overwrite=overwrite)
        self._measurement_accessor = SetMeasurementAccessor(self)

    @property
    def measurements(self) -> SetMeasurementAccessor:
        return self._measurement_accessor

    def get_measurement(self, image_names: List[str] | str | None = None) -> pd.DataFrame:
        if image_names is None:
            image_names = self.get_image_names()
        else:
            assert isinstance(image_names, (str, list)), 'image_names must be a list of image names or a str.'
            if isinstance(image_names, str):
                image_names = [image_names]

        with self._hdf.reader() as handle:
            measurements = []

            # iterate over each image
            for name in image_names:
                image_group = self._hdf.get_image_data_group(handle=handle, image_name=name)
                status_group = self._hdf.get_image_status_subgroup(handle=handle, image_name=name)
                if (self._hdf.IMAGE_MEASUREMENT_SUBGROUP_KEY in image_group
                        and status_group.attrs[SET_STATUS.PROCESSED.label]
                        and status_group.attrs[SET_STATUS.MEASURED.label]
                        and not status_group.attrs[SET_STATUS.ERROR.label]
                ):
                    measurements.append(
                        SetMeasurementAccessor._load_dataframe_from_hdf5_group(image_group[self._hdf.IMAGE_MEASUREMENT_SUBGROUP_KEY]),
                    )
                else:
                    measurements.append(pd.DataFrame())

        return pd.concat(measurements, axis=0)