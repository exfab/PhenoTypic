from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import mahotas as mh
import numpy as np
import pandas as pd

from phenotypic.abstract import FeatureMeasure
from phenotypic.util.constants_ import OBJECT_INFO

AREA = 'Area'

HUE = 'Hue'
SATURATION = 'Saturation'
BRIGHTNESS = 'Brightness'

MEDIAN = 'Median'
MEAN = 'Mean'
STDDEV = 'StdDev'
COEFF_VARIANCE = 'CoefficientVariance'


class MeasureColor(FeatureMeasure):
    """
    Represents a feature extractor for color-based texture analysis.

    This class is a specialized image feature extractor that calculates texture metrics
    based on the hue, saturation, and brightness components from an input_image image. The
    extracted features are useful for texture and object-based analysis in image
    processing tasks. The 'measure' method converts the extracted texture metrics into
    a DataFrame suitable for further analysis and usage.

    """

    def _operate(self, image: Image):
        hue_texture = self._compute_matrix_texture(image, image.hsv.extract_obj_hue())
        hue_texture = {f'{HUE}_{key}': value for key, value in hue_texture.items()}

        saturation_texture = self._compute_matrix_texture(image, image.hsv.extract_obj_saturation())
        saturation_texture = {f'{SATURATION}_{key}': value for key, value in saturation_texture.items()}

        brightness_texture = self._compute_matrix_texture(image, image.hsv.extract_obj_brightness())
        brightness_texture = {f'{BRIGHTNESS}_{key}': value for key, value in brightness_texture.items()}

        return pd.DataFrame({OBJECT_INFO.OBJECT_LABELS: image.objects.labels,
                             **hue_texture, **saturation_texture, **brightness_texture}
                            ).set_index(OBJECT_INFO.OBJECT_LABELS)

    @staticmethod
    def _compute_matrix_texture(image: Image, foreground_array: np.ndarray):
        """
          Computes texture metrics from input_image image data and a binary foreground mask.

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

        return measurements
