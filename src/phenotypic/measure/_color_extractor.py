from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import mahotas as mh
import numpy as np
import pandas as pd

from phenotypic.abstract import FeatureMeasure
from phenotypic.util.constants_ import OBJECT_INFO

OBJ_LABEL = 'ObjLabel'
AREA = 'Area'

HUE = 'Hue'
SATURATION = 'Saturation'
BRIGHTNESS = 'Brightness'

ANGULAR_SECOND_MOMENT = 'Angular Second Moment'
CONTRAST = 'Contrast'
CORRELATION = 'Correlation'
VARIANCE = 'Haralick Variance'
INVERSE_DIFFERENCE_MOMENT = 'Inverse Difference Moment'
SUM_AVERAGE = 'Sum Average'
SUM_VARIANCE = 'Sum Variance'
SUM_ENTROPY = 'Sum Entropy'
ENTROPY = 'Entropy'
DIFFERENCE_VARIANCE = 'Difference Variance'
DIFFERENCE_ENTROPY = 'Difference Entropy'
IMC1 = 'Information Measure of Correlation 1'
IMC2 = 'Information measure of Correlation 2'

MEDIAN = 'Median'
MEAN = 'Mean'
STDDEV = 'Standard Deviation'
COEFF_VARIANCE = 'Coefficient Variance'


class ColorMeasure(FeatureMeasure):
    """
    Represents a feature extractor for color-based texture analysis.

    This class is a specialized image feature extractor that calculates texture metrics
    based on the hue, saturation, and brightness components from an input image. The
    extracted features are useful for texture and object-based analysis in image
    processing tasks. The 'measure' method converts the extracted texture metrics into
    a DataFrame suitable for further analysis and usage.

    """

    def _operate(self, image: Image):
        hue_texture = self._compute_matrix_texture(image, image.hsv.extract_obj_hue())
        hue_texture = {f'{HUE} {key}': value for key, value in hue_texture.items()}

        saturation_texture = self._compute_matrix_texture(image, image.hsv.extract_obj_saturation())
        saturation_texture = {f'{SATURATION} {key}': value for key, value in saturation_texture.items()}

        brightness_texture = self._compute_matrix_texture(image, image.hsv.extract_obj_brightness())
        brightness_texture = {f'{BRIGHTNESS} {key}': value for key, value in brightness_texture.items()}

        return pd.DataFrame({OBJECT_INFO.OBJECT_LABELS: image.objects.labels,
                             **hue_texture, **saturation_texture, **brightness_texture}
                            ).set_index(OBJECT_INFO.OBJECT_LABELS)

    @staticmethod
    def _compute_matrix_texture(image: Image, foreground_array: np.ndarray):
        """
          Computes texture metrics from input image data and a binary foreground mask.

          This function processes gridded image objects and calculates various texture
          features using Haralick descriptors across segmented objects. The calculated
          texture metrics include statistical data and Haralick texture features, which
          are useful in descriptive and diagnostic analyses for image processing applications.

          Args:
              image (Image): The PhenoTypic Image object containing the image data and objects information
              foreground_array (numpy.ndarray): A matrix array with all background pixels set
                  to 0, defining the binary mask.

          Returns:
              dict: A dictionary containing calculated measurements, including object
                  labels, statistical data (e.g., area, mean, standard deviation), and
                  multiple Haralick texture metrics (e.g., contrast, entropy).
          """

        measurements = {
            MEAN: [],
            STDDEV: [],
            MEDIAN: [],
            COEFF_VARIANCE: [],
            ANGULAR_SECOND_MOMENT: [],
            CONTRAST: [],
            CORRELATION: [],
            VARIANCE: [],
            INVERSE_DIFFERENCE_MOMENT: [],
            SUM_AVERAGE: [],
            SUM_VARIANCE: [],
            SUM_ENTROPY: [],
            ENTROPY: [],
            DIFFERENCE_VARIANCE: [],
            DIFFERENCE_ENTROPY: [],
            IMC1: [],
            IMC2: [],
        }
        for i, label in enumerate(image.objects.labels):
            slices = image.objects.props[i].slice
            obj_extracted = foreground_array[slices]

            # In case there's more than one object in the crop
            obj_extracted[image.objmap[slices] != label] = 0

            measurements[MEAN].append(np.mean(obj_extracted[obj_extracted.nonzero()]))
            measurements[MEDIAN].append(np.median(obj_extracted[obj_extracted.nonzero()]))
            measurements[STDDEV].append(np.std(obj_extracted[obj_extracted.nonzero()]))
            measurements[COEFF_VARIANCE].append(
                np.std(obj_extracted[obj_extracted.nonzero()]) / np.mean(obj_extracted[obj_extracted.nonzero()])
            )

            try:
                haralick_features = mh.features.haralick(obj_extracted.astype(np.uint8),
                                                         distance=5,
                                                         ignore_zeros=True,
                                                         )
            except Exception as e:
                haralick_features = np.full((4, 13), np.nan, dtype=np.float64)

            measurements[ANGULAR_SECOND_MOMENT].append(np.mean(haralick_features[:, 0]))
            measurements[CONTRAST].append(np.mean(haralick_features[:, 1]))
            measurements[CORRELATION].append(np.mean(haralick_features[:, 2]))
            measurements[VARIANCE].append(np.mean(haralick_features[:, 3]))
            measurements[INVERSE_DIFFERENCE_MOMENT].append(np.mean(haralick_features[:, 4]))
            measurements[SUM_AVERAGE].append(np.mean(haralick_features[:, 5]))
            measurements[SUM_VARIANCE].append(np.mean(haralick_features[:, 6]))
            measurements[SUM_ENTROPY].append(np.mean(haralick_features[:, 7]))
            measurements[ENTROPY].append(np.mean(haralick_features[:, 8]))
            measurements[DIFFERENCE_VARIANCE].append(np.mean(haralick_features[:, 9]))
            measurements[DIFFERENCE_ENTROPY].append(np.mean(haralick_features[:, 10]))
            measurements[IMC1].append(np.mean(haralick_features[:, 11]))
            measurements[IMC2].append(np.mean(haralick_features[:, 12]))
        return measurements

# Set documentation of measure to match operate
ColorMeasure.measure.__doc__ = ColorMeasure._operate.__doc__