from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd

from phenotypic.abstract import MeasureFeatures

AREA = 'Area'

HUE = 'Hue'
SATURATION = 'Saturation'
BRIGHTNESS = 'Brightness'

MEDIAN = 'Median'
MEAN = 'Mean'
STDDEV = 'StdDev'
COEFF_VARIANCE = 'CoefficientVariance'


class MeasureColor(MeasureFeatures):
    """
    Represents a feature extractor for color-based texture analysis.

    This class is a specialized _root_image feature extractor that calculates texture metrics
    based on the hue, saturation, and brightness components from an input_image _root_image. The
    extracted features are useful for texture and object-based analysis in _root_image
    processing tasks. The 'measure' method converts the extracted texture metrics into
    a DataFrame suitable for further analysis and usage.

    """
    @staticmethod
    def _operate(image: Image):
        hue_texture = MeasureColor._compute_color_metrics(image.hsv.get_foreground_hue(), labels=image.objmap[:].copy(),
                                                          )
        hue_texture = {f'{HUE}_{key}': value for key, value in hue_texture.items()}

        saturation_texture = MeasureColor._compute_color_metrics(image.hsv.get_foreground_saturation(), labels=image.objmap[:].copy(),
                                                                 )
        saturation_texture = {f'{SATURATION}_{key}': value for key, value in saturation_texture.items()}

        brightness_texture = MeasureColor._compute_color_metrics(image.hsv.get_foreground_brightness(), labels=image.objmap[:].copy(),
                                                                 )
        brightness_texture = {f'{BRIGHTNESS}_{key}': value for key, value in brightness_texture.items()}

        return pd.DataFrame(data={**hue_texture, **saturation_texture, **brightness_texture}, index=image.objects.get_labels_series())

    @staticmethod
    def _compute_color_metrics(foreground: np.ndarray, labels: np.ndarray):
        """
          Computes texture metrics from input_image _root_image data and a binary foreground mask.

          This function processes gridded _root_image objects and calculates various texture
          features using Haralick descriptors across segmented objects. The calculated
          texture metrics include statistical data and Haralick texture features, which
          are useful in descriptive and diagnostic analyses for _root_image processing applications.

          Args:
              _root_image (Image): The PhenoTypic Image object containing the _root_image data and objects information
              foreground (numpy.ndarray): A matrix array with all background pixels set
                  to 0, defining the binary mask.

          Returns:
              dict: A dictionary containing calculated measurements, including object
                  labels, statistical data (e.g., area, mean, standard deviation), and
                  multiple Haralick texture metrics (e.g., contrast, entropy).
          """

        measurements = {
            MEAN: MeasureFeatures.calculate_mean(array=foreground, labels=labels),
            STDDEV: MeasureFeatures.calculate_stddev(array=foreground, labels=labels),
            MEDIAN: MeasureFeatures.calculate_median(array=foreground, labels=labels),
            COEFF_VARIANCE: MeasureFeatures.calculate_coeff_variation(array=foreground, labels=labels),
        }
        return measurements
