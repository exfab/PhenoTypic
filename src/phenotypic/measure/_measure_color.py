from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd

from phenotypic.abstract import MeasureFeatures, MeasurementInfo

class COLOR(MeasurementInfo):
    @property
    def CATEGORY(self):
        return 'Color'

    DOMWL_MEAN = ('DomWLMean', 'The mean dominant wavelength of the object')
    DOMWL_MEDIAN = ('DomWLMedian', 'The median dominant wavelength of the object')
    DOMWL_STDDEV = ('DomWLStdDev', 'The standard deviation of the dominant wavelength of the object')
    DOMWL_COEFF_VARIANCE = ('DomWLCoefficientVariance', 'The coefficient of variation of the dominant wavelength of the object')
    DOMWL_MINIMUM = ('DomWLMinimum', 'The minimum dominant wavelength of the object')
    DOMWL_MAXIMUM = ('DomWLMaximum', 'The maximum dominant wavelength of the object')
    DOMWL_Q1 = ('DomWLLowerQuartile', 'The lower quartile (Q1) dominant wavelength of the object')
    DOMWL_Q3 = ('DomWLUpperQuartile', 'The upper quartile (Q3) dominant wavelength of the object')

    PURITY_MEAN = ('PurityMean', 'The mean excitation purity of the object')
    PURITY_MEDIAN = ('PurityMedian', 'The median excitation purity of the object')
    PURITY_STDDEV = ('PurityStdDev', 'The standard deviation of the excitation purity of the object')
    PURITY_COEFF_VARIANCE = ('PurityCoefficientVariance', 'The coefficient of variation of the excitation purity of the object')
    PURITY_MINIMUM = ('PurityMinimum', 'The minimum excitation purity of the object')
    PURITY_MAXIMUM = ('PurityMaximum', 'The maximum excitation purity of the object')
    PURITY_Q1 = ('PurityLowerQuartile', 'The lower quartile (Q1) excitation purity of the object')
    PURITY_Q3 = ('PurityUpperQuartile', 'The upper quartile (Q3) excitation purity of the object')

    HUE_MEAN = ('HueMean', 'The mean hue of the object')
    HUE_MEDIAN = ('HueMedian', 'The median hue of the object')
    HUE_STDDEV = ('HueStdDev', 'The standard deviation of the hue of the object')
    HUE_COEFF_VARIANCE = ('HueCoefficientVariance', 'The coefficient of variation of the hue of the object')
    HUE_MINIMUM = ('HueMinimum', 'The minimum hue of the object')
    HUE_MAXIMUM = ('HueMaximum', 'The maximum hue of the object')
    HUE_Q1 = ('HueLowerQuartile', 'The lower quartile (Q1) hue of the object')
    HUE_Q3 = ('HueUpperQuartile', 'The upper quartile (Q3) hue of the object')

    SATURATION_MEAN = ('SaturationMean', 'The mean saturation of the object')
    SATURATION_MEDIAN = ('SaturationMedian', 'The median saturation of the object')
    SATURATION_STDDEV = ('SaturationStdDev', 'The standard deviation of the saturation of the object')
    SATURATION_COEFF_VARIANCE = ('SaturationCoefficientVariance', 'The coefficient of variation of the saturation of the object')
    SATURATION_MINIMUM = ('SaturationMinimum', 'The minimum saturation of the object')
    SATURATION_MAXIMUM = ('SaturationMaximum', 'The maximum saturation of the object')
    SATURATION_Q1 = ('SaturationLowerQuartile', 'The lower quartile (Q1) saturation of the object')
    SATURATION_Q3 = ('SaturationUpperQuartile', 'The upper quartile (Q3) saturation of the object')

    BRIGHTNESS_MEAN = ('BrightnessMean', 'The mean brightness of the object')
    BRIGHTNESS_MEDIAN = ('BrightnessMedian', 'The median brightness of the object')
    BRIGHTNESS_STDDEV = ('BrightnessStdDev', 'The standard deviation of the brightness of the object')
    BRIGHTNESS_COEFF_VARIANCE = ('BrightnessCoefficientVariance', 'The coefficient of variation of the brightness of the object')
    BRIGHTNESS_MINIMUM = ('BrightnessMinimum', 'The minimum brightness of the object')
    BRIGHTNESS_MAXIMUM = ('BrightnessMaximum', 'The maximum brightness of the object')
    BRIGHTNESS_Q1 = ('BrightnessLowerQuartile', 'The lower quartile (Q1) brightness of the object')
    BRIGHTNESS_Q3 = ('BrightnessUpperQuartile', 'The upper quartile (Q3) brightness of the object')

class MeasureColor(MeasureFeatures):
    """
    Represents a feature extractor for color-based texture analysis.

    This class is a specialized image feature extractor that calculates texture metrics
    based on the hue, saturation, and brightness components from an input_image image. The
    extracted features are useful for texture and object-based analysis in image
    processing tasks. The 'measure' method converts the extracted texture metrics into
    a DataFrame suitable for further analysis and usage.

    """
    def _operate(self, image: Image):

        hsb_foreground = image.hsb.foreground()
        hue_texture = MeasureColor._compute_color_metrics(foreground=hsb_foreground[...,0], labels=image.objmap[:].copy(),
                                                          )
        hue_texture = {f'{HUE}_{key}': value for key, value in hue_texture.items()}

        saturation_texture = MeasureColor._compute_color_metrics(image.hsb.get_foreground_saturation(), labels=image.objmap[:].copy(),
                                                                 )
        saturation_texture = {f'{SATURATION}_{key}': value for key, value in saturation_texture.items()}

        brightness_texture = MeasureColor._compute_color_metrics(image.hsb.get_foreground_brightness(), labels=image.objmap[:].copy(),
                                                                 )
        brightness_texture = {f'{BRIGHTNESS}_{key}': value for key, value in brightness_texture.items()}

        return pd.DataFrame(data={**hue_texture, **saturation_texture, **brightness_texture}, index=image.objects.labels2series())

    @staticmethod
    def _compute_color_metrics(foreground: np.ndarray, labels: np.ndarray):
        """
          Computes texture metrics from input_image image data and a binary foreground mask.

          This function processes gridded image objects and calculates various texture
          features using Haralick descriptors across segmented objects. The calculated
          texture metrics include statistical data and Haralick texture features, which
          are useful in descriptive and diagnostic analyses for image processing applications.

          Args:
              image (Image): The PhenoTypic Image object containing the image data and objects information
              foreground (numpy.ndarray): A matrix array with all background pixels set
                  to 0, defining the binary mask.

          Returns:
              dict: A dictionary containing calculated measurements, including object
                  labels, statistical data (e.g., area, mean, standard deviation), and
                  multiple Haralick texture metrics (e.g., contrast, entropy).
          """

        measurements = {
            MEAN: MeasureFeatures._calculate_mean(array=foreground, labels=labels),
            MEDIAN: MeasureFeatures._calculate_median(array=foreground, labels=labels),
            STDDEV: MeasureFeatures._calculate_stddev(array=foreground, labels=labels),
            COEFF_VARIANCE: MeasureFeatures._calculate_coeff_variation(array=foreground, labels=labels),
        }
        return measurements
