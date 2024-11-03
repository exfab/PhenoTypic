import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import numpy as np

from ._imaging_pipeline import ImagingPipeline
from ...util.type_checks import is_binary_mask


class ThresholdDetectorPipeline(ImagingPipeline):
    def detect(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if type(image) is not np.ndarray:
            raise ValueError("Input of a ThresholdPipeline must be a digitized image array")

        output = self._execute_pipeline(input=image, inplace=inplace)

        if is_binary_mask(output):
            return output
        else:
            raise ValueError("Final output of a ThresholdPipeline must be a binary object mask array")
