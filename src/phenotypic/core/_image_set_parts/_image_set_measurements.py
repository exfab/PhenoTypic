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
                 outpath: PathLike | None = None,
                 overwrite: bool = False, ):
        super().__init__(name=name, image_template=image_template,
                         src=src, outpath=outpath, overwrite=overwrite)
        self._measurement_accessor = SetMeasurementAccessor(self)

    @property
    def measurements(self) -> SetMeasurementAccessor:
        return self._measurement_accessor

    def get_measurement(self, image_names: List[str] | str | None = None) -> pd.DataFrame:
        import logging
        logger = logging.getLogger(f"{__name__}.get_measurement")
        
        if image_names is None:
            image_names = self.get_image_names()
        else:
            assert isinstance(image_names, (str, list)), 'image_names must be a list of image names or a str.'
            if isinstance(image_names, str):
                image_names = [image_names]

        logger.debug(f"🔍 get_measurement: Retrieving measurements for {len(image_names)} images: {image_names[:3]}{'...' if len(image_names) > 3 else ''}")
        
        with self.hdf_.reader() as handle:
            measurements = []

            # iterate over each image
            for name in image_names:
                logger.debug(f"🔍 get_measurement: Processing image '{name}'")
                image_group = self.hdf_.get_image_group(handle=handle, image_name=name)
                logger.debug(f"🔍 get_measurement: Image group contents for '{name}': {list(image_group.keys())}")
                
                # Check if measurements exist - more robust than checking status groups
                measurement_key = self.hdf_.IMAGE_MEASUREMENT_SUBGROUP_KEY
                logger.debug(f"🔍 get_measurement: Looking for measurement key '{measurement_key}' in image group")
                
                if measurement_key in image_group:
                    logger.debug(f"🔍 get_measurement: Found measurement group for '{name}', attempting to load")
                    
                    # Check if this is SWMR format (has 'index' and 'values' groups) or legacy format
                    meas_group = image_group[measurement_key]
                    has_swmr_format = 'index' in meas_group and 'values' in meas_group
                    has_legacy_format = 'columns' in meas_group and 'index' in meas_group
                    
                    logger.debug(f"🔍 get_measurement: Measurement group contents: {list(meas_group.keys())}")
                    logger.debug(f"🔍 get_measurement: SWMR format detected: {has_swmr_format}, Legacy format detected: {has_legacy_format}")
                    
                    try:
                        if has_swmr_format:
                            logger.debug(f"🔍 get_measurement: Using SWMR-compatible loader for '{name}'")
                            df = SetMeasurementAccessor._load_dataframe_from_hdf5_group_swmr(image_group)
                        else:
                            logger.debug(f"🔍 get_measurement: Using legacy loader for '{name}'")
                            df = SetMeasurementAccessor._load_dataframe_from_hdf5_group(image_group)
                        
                        logger.debug(f"🔍 get_measurement: Successfully loaded DataFrame for '{name}', shape: {df.shape}")
                        measurements.append(df)
                    except Exception as e:
                        logger.error(f"🔍 get_measurement: Failed to load measurements for '{name}': {e}")
                        # If loading fails, add empty DataFrame
                        measurements.append(pd.DataFrame())
                else:
                    logger.debug(f"🔍 get_measurement: No measurement group found for '{name}', adding empty DataFrame")
                    measurements.append(pd.DataFrame())

        total_rows = sum(len(df) for df in measurements)
        logger.debug(f"🔍 get_measurement: Concatenating {len(measurements)} DataFrames with total {total_rows} rows")
        result = pd.concat(measurements, axis=0) if measurements else pd.DataFrame()
        logger.debug(f"🔍 get_measurement: Final result shape: {result.shape}")
        return result