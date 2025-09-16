from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, List

from phenotypic.abstract import GridFinder

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from ._image_set_accessors._image_set_measurements_accessor import SetMeasurementAccessor
from phenotypic.util.constants_ import PIPE_STATUS
from ._image_set_status import ImageSetStatus


class ImageSetMeasurements(ImageSetStatus):
    """
    This class adds measurement handling to the ImageSetStatus class.
    """

    def __init__(self,
                 name: str,
                 grid_finder: GridFinder | None = None,
                 src: List[Image] | PathLike | None = None,
                 outpath: PathLike | None = None,
                 overwrite: bool = False, ):
        """
        Initializes the instance with the specified parameters.

        This constructor initializes an instance with the given name, grid finder,
        source, output path, and overwrite settings. The initialization involves
        calling the superclass constructor with the provided arguments and setting
        up the measurement accessor using the SetMeasurementAccessor class.

        Args:
            name (str): A unique name identifying the instance.
            grid_finder (GridFinder | None): An instance of GridFinder to help with
                grid identification. Defaults to None.
            src (List[Image] | PathLike | None): A list of image objects, a path-like
                object, or None, representing the source of the data. Defaults to None.
            outpath (PathLike | None): A path-like object specifying the output path
                for processed or generated data. Defaults to None.
            overwrite (bool): A flag indicating whether to overwrite existing files or
                data at the output path. Defaults to False.
        """
        super().__init__(name=name, grid_finder=grid_finder,
                         src=src, outpath=outpath, overwrite=overwrite)
        self._measurement_accessor = SetMeasurementAccessor(self)

    @property
    def measurements(self) -> SetMeasurementAccessor:
        return self._measurement_accessor

    def get_measurement(self, image_names: List[str] | str | None = None) -> pd.DataFrame:
        import logging
        logger = logging.getLogger(f"ImageSet.get_measurement")

        if image_names is None:
            image_names = self.get_image_names()
        else:
            assert isinstance(image_names, (str, list)), 'image_names must be a list of image names or a str.'
            if isinstance(image_names, str):
                image_names = [image_names]

        logger.debug(
            f"🔍 get_measurement: Retrieving measurements for {len(image_names)} images: {image_names[:3]}{'...' if len(image_names) > 3 else ''}")

        with self.hdf_.reader() as handle:
            measurements = []

            # iterate over each image
            for name in image_names:
                logger.debug(f"🔍 get_measurement: Processing image '{name}'")
                image_group = self.hdf_.get_image_group(handle=handle, image_name=name)
                logger.debug(f"🔍 get_measurement: Image group contents for '{name}': {list(image_group.keys())}")

                # Check if measurements exist
                measurement_key = self.hdf_.IMAGE_MEASUREMENT_SUBGROUP_KEY
                logger.debug(f"🔍 get_measurement: Looking for measurement key '{measurement_key}' in image group")

                status_subgroup = self.hdf_.get_status_subgroup(handle=handle, image_name=name)
                logger.debug(f'Image group status attrs for "{name}": {status_subgroup.attrs.keys()}')
                # TODO: Finish implementing measurement aggregator
                # Validate that the measurements were sucessfully taken by the pipeline
                if ((status_subgroup.attrs[PIPE_STATUS.PROCESSED])
                        and (status_subgroup.attrs[PIPE_STATUS.MEASURED])
                        and (measurement_key in image_group)):
                    df = self.hdf_.load_frame(group=self.hdf_.get_image_measurement_subgroup(handle=handle, image_name=name), )

                    prot_metadata_group = self.hdf_.get_protected_metadata_subgroup(handle=handle, image_name=name)
                    pub_metadata_group = self.hdf_.get_public_metadata_subgroup(handle=handle, image_name=name)
                    for name in prot_metadata_group.attrs:
                        if name.startswith('Metadata_') is False:
                            colname = f'Metadata_{name}'
                        else:
                            colname = name
                        logger.debug(f'Inserting protected metadata: {colname}')
                        df.insert(loc=0, column=colname, value=prot_metadata_group.attrs[name])

                    for name in pub_metadata_group.attrs:
                        if name.startswith('Metadata_') is False:
                            colname = f'Metadata_{name}'
                        else:
                            colname = name
                        logger.debug(f'Inserting public metadata: {colname}')
                        df.insert(loc=0, column=colname, value=pub_metadata_group.attrs[name])

                    measurements.append(df)

        total_rows = sum(len(df) for df in measurements)
        logger.debug(f"🔍 get_measurement: Concatenating {len(measurements)} DataFrames with total {total_rows} rows")
        result = pd.concat(measurements, axis=0) if measurements else pd.DataFrame()
        logger.debug(f"🔍 get_measurement: Final result shape: {result.shape}")
        return result
