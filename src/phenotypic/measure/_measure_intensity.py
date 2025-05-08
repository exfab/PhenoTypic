from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd

from phenotypic.abstract import FeatureMeasure

from phenotypic.util.constants_ import INTENSITY_LABELS as C

# TODO: Add more measurements
class MeasureIntensity(FeatureMeasure):
    """Calculates various intensity measures of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the intensity measures of the objects in the image.

    Notes:
        Integrated Intensity: Sum of all pixel values in the object's grayscale footprint

    """

    def _operate(self, image: Image) -> pd.DataFrame:
        measurements = {
            str(C.INTEGRATED_INTENSITY): []
        }

        for obj_props in image.objects.props:
            measurements[str(C.INTEGRATED_INTENSITY)].append(obj_props.intensity_image.sum())

        return pd.DataFrame(measurements, index=image.objects.get_labels_series())