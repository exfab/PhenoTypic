from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic.tools.constants_ import OBJECT

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import MeasurementInfo, MeasureFeatures


class SIZE(MeasurementInfo):
    """The labels and descriptions of the size measurements."""

    @classmethod
    def category(cls):
        return 'Size'

    AREA = ('Area',
            "Total number of pixels occupied by the microbial colony. Represents colony biomass and growth extent on agar plates. Larger areas typically indicate more robust growth or longer incubation times.")
    INTEGRATED_INTENSITY = ('IntegratedIntensity', 'The sum of the object\'s grayscale pixels')


class MeasureSize(MeasureFeatures):
    """Calculates basic size measurements of the objects in the image.

    Returns:
        pd.DataFrame: A dataframe containing the size measurements of the objects in the image.

    """

    def _operate(self, image: Image) -> pd.DataFrame:
        # Create empty numpy arrays to store measurements
        measurements = {str(feature): np.zeros(shape=image.num_objects) for feature in SIZE if
                        feature != SIZE.CATEGORY}

        # Calculate integrated intensity using the sum calculation method from base class
        intensity_matrix = image.gray[:].copy()
        objmap = image.objmap[:].copy()
        measurements[str(SIZE.INTEGRATED_INTENSITY)] = self._calculate_sum(array=intensity_matrix, labels=objmap)

        # Calculate area from object properties
        obj_props = image.objects.props
        for idx, obj_image in enumerate(image.objects):
            current_props = obj_props[idx]
            measurements[str(SIZE.AREA)][idx] = current_props.area

        measurements = pd.DataFrame(measurements)
        measurements.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return measurements


MeasureSize.__doc__ = SIZE.append_rst_to_doc(MeasureSize)

