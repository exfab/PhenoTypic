from __future__ import annotations
from typing import TYPE_CHECKING, Literal, List

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from phenotypic.abstract import Measurements
from phenotypic.util.constants_ import SET_STATUS
from ._image_set_status import ImageSetStatus


class ImageSetMeasurements(ImageSetStatus):
    """
    This class adds measurement handling to the ImageSetStatus class.
    """

    def get_measurement(self, image_names: List[str] | str | None = None) -> pd.DataFrame:
        if image_names is None:
            image_names = self.get_image_names()
        else:
            assert isinstance(image_names, (str, list)), 'image_names must be a list of image names or a str.'
            if isinstance(image_names, str):
                image_names = [image_names]

        with self._main_hdf.reader() as handle:
            measurements = []

            # iterate over each image
            for name in image_names:
                image_group = self._main_hdf.get_images_subgroup(handle=handle, image_name=name)
                status_group = self._main_hdf.get_image_status_subgroup(handle=handle, image_name=name)
                if (self._main_hdf.IMAGE_MEASUREMENT_SUBGROUP_KEY in image_group
                        and status_group.attrs[SET_STATUS.PROCESSED.label]
                        and status_group.attrs[SET_STATUS.MEASURED.label]
                        and not status_group.attrs[SET_STATUS.ERROR.label]
                ):
                    measurements.append(
                        Measurements._load_dataframe_from_hdf5_group(image_group[self._main_hdf.IMAGE_MEASUREMENT_SUBGROUP_KEY]),
                    )
                else:
                    measurements.append(pd.DataFrame())

        return pd.concat(measurements, axis=0)