from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd
import colour
import logging

from phenotypic.abc_ import MeasureFeatures, MeasurementInfo
from phenotypic.tools.constants_ import OBJECT

logger = logging.getLogger(__name__)


class ColorXYZ(MeasurementInfo):
    @classmethod
    def category(cls):
        return 'ColorXYZ'

    X_MINIMUM = ('CieXMin', 'The minimum X value of the object in CIE XYZ color space')
    X_Q1 = ('CieXQ1', 'The lower quartile (Q1) X value of the object in CIE XYZ color space')
    X_MEAN = ('CieXMean', 'The mean X value of the object in CIE XYZ color space')
    X_MEDIAN = ('CieXMedian', 'The median X value of the object in CIE XYZ color space')
    X_Q3 = ('CieXQ3', 'The upper quartile (Q3) X value of the object in CIE XYZ color space')
    X_MAXIMUM = ('CieXMax', 'The maximum X value of the object in CIE XYZ color space')
    X_STDDEV = ('CieXStdDev', 'The standard deviation of the X value of the object in CIE XYZ color space')
    X_COEFF_VARIANCE = ('CieXCoeffVar',
                        'The coefficient of variation of the X value of the object in CIE XYZ color space')

    @classmethod
    def cieX_headers(cls):
        return [str(cls.X_MINIMUM), str(cls.X_Q1), str(cls.X_MEAN), str(cls.X_MEDIAN), str(cls.X_Q3),
                str(cls.X_MAXIMUM), str(cls.X_STDDEV), str(cls.X_COEFF_VARIANCE)]

    Y_MINIMUM = ('CieYMin', 'The minimum Y value of the object in CIE XYZ color space')
    Y_Q1 = ('CieYQ1', 'The lower quartile (Q1) Y value of the object in CIE XYZ color space')
    Y_MEAN = ('CieYMean', 'The mean Y value of the object in CIE XYZ color space')
    Y_MEDIAN = ('CieYMedian', 'The median Y value of the object in CIE XYZ color space')
    Y_Q3 = ('CieYQ3', 'The upper quartile (Q3) Y value of the object in CIE XYZ color space')
    Y_MAXIMUM = ('CieYMax', 'The maximum Y value of the object in CIE XYZ color space')
    Y_STDDEV = ('CieYStdDev', 'The standard deviation of the Y value of the object in CIE XYZ color space')
    Y_COEFF_VARIANCE = ('CieYCoeffVar',
                        'The coefficient of variation of the Y value of the object in CIE XYZ color space')

    @classmethod
    def cieY_headers(cls):
        return [str(cls.Y_MINIMUM), str(cls.Y_Q1), str(cls.Y_MEAN), str(cls.Y_MEDIAN), str(cls.Y_Q3),
                str(cls.Y_MAXIMUM), str(cls.Y_STDDEV), str(cls.Y_COEFF_VARIANCE)]

    Z_MINIMUM = ('CieZMin', 'The minimum Z value of the object in CIE XYZ color space')
    Z_Q1 = ('CieZQ1', 'The lower quartile (Q1) Z value of the object in CIE XYZ color space')
    Z_MEAN = ('CieZMean', 'The mean Z value of the object in CIE XYZ color space')
    Z_MEDIAN = ('CieZMedian', 'The median Z value of the object in CIE XYZ color space')
    Z_Q3 = ('CieZQ3', 'The upper quartile (Q3) Z value of the object in CIE XYZ color space')
    Z_MAXIMUM = ('CieZMax', 'The maximum Z value of the object in CIE XYZ color space')
    Z_STDDEV = ('CieZStdDev', 'The standard deviation of the Z value of the object in CIE XYZ color space')
    Z_COEFF_VARIANCE = ('CieZCoeffVar',
                        'The coefficient of variation of the Z value of the object in CIE XYZ color space')

    @classmethod
    def cieZ_headers(cls):
        return [str(cls.Z_MINIMUM), str(cls.Z_Q1), str(cls.Z_MEAN), str(cls.Z_MEDIAN), str(cls.Z_Q3),
                str(cls.Z_MAXIMUM), str(cls.Z_STDDEV), str(cls.Z_COEFF_VARIANCE)]


class Colorxy(MeasurementInfo):
    @classmethod
    def category(cls):
        return 'Colorxy'

    x_MINIMUM = ('xMin', 'The minimum chromaticity x coordinate of the object')
    x_Q1 = ('xQ1', 'The lower quartile (Q1) chromaticity x coordinate of the object')
    x_MEAN = ('xMean', 'The mean chromaticity x coordinate of the object')
    x_MEDIAN = ('xMedian', 'The median chromaticity x coordinate of the object')
    x_Q3 = ('xQ3', 'The upper quartile (Q3) chromaticity x coordinate of the object')
    x_MAXIMUM = ('xMax', 'The maximum chromaticity x coordinate of the object')
    x_STDDEV = ('xStdDev', 'The standard deviation of the chromaticity x coordinate of the object')
    x_COEFF_VARIANCE = ('xCoeffVar', 'The coefficient of variation of the chromaticity x coordinate of the object')

    @classmethod
    def x_headers(cls):
        return [str(cls.x_MINIMUM), str(cls.x_Q1), str(cls.x_MEAN), str(cls.x_MEDIAN), str(cls.x_Q3),
                str(cls.x_MAXIMUM), str(cls.x_STDDEV), str(cls.x_COEFF_VARIANCE)]

    y_MINIMUM = ('yMin', 'The minimum chromaticity y coordinate of the object')
    y_Q1 = ('yQ1', 'The lower quartile (Q1) chromaticity y coordinate of the object')
    y_MEAN = ('yMean', 'The mean chromaticity y coordinate of the object')
    y_MEDIAN = ('yMedian', 'The median chromaticity y coordinate of the object')
    y_Q3 = ('yQ3', 'The upper quartile (Q3) chromaticity y coordinate of the object')
    y_MAXIMUM = ('yMax', 'The maximum chromaticity y coordinate of the object')
    y_STDDEV = ('yStdDev', 'The standard deviation of the chromaticity y coordinate of the object')
    y_COEFF_VARIANCE = ('yCoeffVar', 'The coefficient of variation of the chromaticity y coordinate of the object')

    @classmethod
    def y_headers(cls):
        return [str(cls.y_MINIMUM), str(cls.y_Q1), str(cls.y_MEAN), str(cls.y_MEDIAN), str(cls.y_Q3),
                str(cls.y_MAXIMUM), str(cls.y_STDDEV), str(cls.y_COEFF_VARIANCE)]


class ColorLab(MeasurementInfo):
    @classmethod
    def category(cls):
        return 'ColorLab'

    L_STAR_MINIMUM = ('L*Min', 'The minimum L* value of the object')
    L_STAR_Q1 = ('L*Q1', 'The lower quartile (Q1) L* value of the object')
    L_STAR_MEAN = ('L*Mean', 'The mean L* value of the object')
    L_STAR_MEDIAN = ('L*Median', 'The median L* value of the object')
    L_STAR_Q3 = ('L*Q3', 'The upper quartile (Q3) L* value of the object')
    L_STAR_MAXIMUM = ('L*Max', 'The maximum L* value of the object')
    L_STAR_STDDEV = ('L*StdDev', 'The standard deviation of the L* value of the object')
    L_STAR_COEFF_VARIANCE = ('L*CoeffVar', 'The coefficient of variation of the L* value of the object')

    @classmethod
    def l_star_headers(cls):
        return [str(cls.L_STAR_MINIMUM), str(cls.L_STAR_Q1), str(cls.L_STAR_MEAN), str(cls.L_STAR_MEDIAN),
                str(cls.L_STAR_Q3), str(cls.L_STAR_MAXIMUM), str(cls.L_STAR_STDDEV), str(cls.L_STAR_COEFF_VARIANCE)]

    A_STAR_MINIMUM = ('a*Min', 'The minimum a* value of the object')
    A_STAR_Q1 = ('a*Q1', 'The lower quartile (Q1) a* value of the object')
    A_STAR_MEAN = ('a*Mean', 'The mean a* value of the object')
    A_STAR_MEDIAN = ('a*Median', 'The median a* value of the object')
    A_STAR_Q3 = ('a*Q3', 'The upper quartile (Q3) a* value of the object')
    A_STAR_MAXIMUM = ('a*Max', 'The maximum a* value of the object')
    A_STAR_STDDEV = ('a*StdDev', 'The standard deviation of the a* value of the object')
    A_STAR_COEFF_VARIANCE = ('a*CoeffVar', 'The coefficient of variation of the a* value of the object')

    @classmethod
    def a_star_headers(cls):
        return [str(cls.A_STAR_MINIMUM), str(cls.A_STAR_Q1), str(cls.A_STAR_MEAN), str(cls.A_STAR_MEDIAN),
                str(cls.A_STAR_Q3), str(cls.A_STAR_MAXIMUM), str(cls.A_STAR_STDDEV), str(cls.A_STAR_COEFF_VARIANCE)]

    B_STAR_MINIMUM = ('b*Min', 'The minimum b* value of the object')
    B_STAR_Q1 = ('b*Q1', 'The lower quartile (Q1) b* value of the object')
    B_STAR_MEAN = ('b*Mean', 'The mean b* value of the object')
    B_STAR_MEDIAN = ('b*Median', 'The median b* value of the object')
    B_STAR_Q3 = ('b*Q3', 'The upper quartile (Q3) b* value of the object')
    B_STAR_MAXIMUM = ('b*Max', 'The maximum b* value of the object')
    B_STAR_STDDEV = ('b*StdDev', 'The standard deviation of the b* value of the object')
    B_STAR_COEFF_VARIANCE = ('b*CoeffVar', 'The coefficient of variation of the b* value of the object')

    @classmethod
    def b_star_headers(cls):
        return [str(cls.B_STAR_MINIMUM), str(cls.B_STAR_Q1), str(cls.B_STAR_MEAN), str(cls.B_STAR_MEDIAN),
                str(cls.B_STAR_Q3), str(cls.B_STAR_MAXIMUM), str(cls.B_STAR_STDDEV), str(cls.B_STAR_COEFF_VARIANCE)]

    CHROMA_EST_MEAN = (
        'ChromaEstimatedMean',
        r'The mean chroma estimation of the object calculated using :math:`\\sqrt{a*_{mean}^2 + b*_{mean})^2}`'
    )
    CHROMA_EST_MEDIAN = (
        'ChromaEstimatedMedian',
        r'The median chroma estimation of the object using :math:`\\sqrt{a*_{median}^2 + b*_{median})^2}`'
    )


class ColorHSV(MeasurementInfo):
    @classmethod
    def category(cls):
        return 'ColorHSV'

    HUE_MINIMUM = ('HueMin', 'The minimum hue of the object')
    HUE_Q1 = ('HueQ1', 'The lower quartile (Q1) hue of the object')
    HUE_MEAN = ('HueMean', 'The mean hue of the object')
    HUE_MEDIAN = ('HueMedian', 'The median hue of the object')
    HUE_Q3 = ('HueQ3', 'The upper quartile (Q3) hue of the object')
    HUE_MAXIMUM = ('HueMax', 'The maximum hue of the object')
    HUE_STDDEV = ('HueStdDev', 'The standard deviation of the hue of the object')
    HUE_COEFF_VARIANCE = ('HueCoeffVar', 'The coefficient of variation of the hue of the object')

    @classmethod
    def hue_headers(cls):
        return [str(cls.HUE_MINIMUM), str(cls.HUE_Q1), str(cls.HUE_MEAN), str(cls.HUE_MEDIAN), str(cls.HUE_Q3),
                str(cls.HUE_MAXIMUM), str(cls.HUE_STDDEV), str(cls.HUE_COEFF_VARIANCE)]

    SATURATION_MINIMUM = ('SaturationMin', 'The minimum saturation of the object')
    SATURATION_Q1 = ('SaturationQ1', 'The lower quartile (Q1) saturation of the object')
    SATURATION_MEAN = ('SaturationMean', 'The mean saturation of the object')
    SATURATION_MEDIAN = ('SaturationMedian', 'The median saturation of the object')
    SATURATION_Q3 = ('SaturationQ3', 'The upper quartile (Q3) saturation of the object')
    SATURATION_MAXIMUM = ('SaturationMax', 'The maximum saturation of the object')
    SATURATION_STDDEV = ('SaturationStdDev', 'The standard deviation of the saturation of the object')
    SATURATION_COEFF_VARIANCE = ('SaturationCoeffVar', 'The coefficient of variation of the saturation of the object')

    @classmethod
    def saturation_headers(cls):
        return [str(cls.SATURATION_MINIMUM), str(cls.SATURATION_Q1), str(cls.SATURATION_MEAN),
                str(cls.SATURATION_MEDIAN), str(cls.SATURATION_Q3), str(cls.SATURATION_MAXIMUM),
                str(cls.SATURATION_STDDEV), str(cls.SATURATION_COEFF_VARIANCE)]

    BRIGHTNESS_MINIMUM = ('BrightnessMin', 'The minimum brightness of the object')
    BRIGHTNESS_Q1 = ('BrightnessQ1', 'The lower quartile (Q1) brightness of the object')
    BRIGHTNESS_MEAN = ('BrightnessMean', 'The mean brightness of the object')
    BRIGHTNESS_MEDIAN = ('BrightnessMedian', 'The median brightness of the object')
    BRIGHTNESS_Q3 = ('BrightnessQ3', 'The upper quartile (Q3) brightness of the object')
    BRIGHTNESS_MAXIMUM = ('BrightnessMax', 'The maximum brightness of the object')
    BRIGHTNESS_STDDEV = ('BrightnessStdDev', 'The standard deviation of the brightness of the object')
    BRIGHTNESS_COEFF_VARIANCE = ('BrightnessCoeffVar', 'The coefficient of variation of the brightness of the object')

    @classmethod
    def brightness_headers(cls):
        return [str(cls.BRIGHTNESS_MINIMUM), str(cls.BRIGHTNESS_Q1), str(cls.BRIGHTNESS_MEAN),
                str(cls.BRIGHTNESS_MEDIAN), str(cls.BRIGHTNESS_Q3), str(cls.BRIGHTNESS_MAXIMUM),
                str(cls.BRIGHTNESS_STDDEV), str(cls.BRIGHTNESS_COEFF_VARIANCE)]


class MeasureColor(MeasureFeatures):
    """
    Performs color-based feature extraction and calculations on segmented image objects.

    This class extends the MeasureFeatures class to provide additional methods for analyzing
    the color properties of segmented objects within images. It computes various color-related
    metrics, including those based on the HSB (hue, saturation, brightness) color model. These calculations
    can be used in image analysis workflows to extract detailed information about the color
    characteristics of objects.

    """

    def __init__(self, white_chroma_max: float = 4.0, chroma_min: float = 8.0, include_XYZ: bool = False):
        self.white_chroma_max = white_chroma_max
        self.chroma_min = chroma_min
        self.include_XYZ = include_XYZ

    def _operate(self, image: Image):
        data = {}
        if self.include_XYZ:
            cieXYZ_foreground = image.color.XYZ.foreground()
            X_meas = MeasureColor._compute_color_metrics(foreground=cieXYZ_foreground[..., 0], labels=image.objmap[:])
            X_meas = {key: value for key, value in zip(ColorXYZ.cieX_headers(), X_meas)}

            Y_meas = MeasureColor._compute_color_metrics(foreground=cieXYZ_foreground[..., 1], labels=image.objmap[:])
            Y_meas = {key: value for key, value in zip(ColorXYZ.cieY_headers(), Y_meas)}

            Z_meas = MeasureColor._compute_color_metrics(foreground=cieXYZ_foreground[..., 2], labels=image.objmap[:])
            Z_meas = {key: value for key, value in zip(ColorXYZ.cieZ_headers(), Z_meas)}

            del cieXYZ_foreground
            data = {**data, **X_meas, **Y_meas, **Z_meas}

        xy_foreground = image.color.xy.foreground()
        x_meas = MeasureColor._compute_color_metrics(foreground=xy_foreground[..., 0], labels=image.objmap[:])
        x_meas = {key: value for key, value in zip(Colorxy.x_headers(), x_meas)}

        y_meas = MeasureColor._compute_color_metrics(foreground=xy_foreground[..., 1], labels=image.objmap[:])
        y_meas = {key: value for key, value in zip(Colorxy.y_headers(), y_meas)}

        del xy_foreground
        data = {**data, **x_meas, **y_meas}

        Lab_foreground = image.color.Lab.foreground()
        lstar_meas = MeasureColor._compute_color_metrics(foreground=Lab_foreground[..., 0], labels=image.objmap[:])
        lstar_meas = {key: value for key, value in zip(ColorLab.l_star_headers(), lstar_meas)}

        astar_meas = MeasureColor._compute_color_metrics(foreground=Lab_foreground[..., 1], labels=image.objmap[:])
        astar_meas = {key: value for key, value in zip(ColorLab.a_star_headers(), astar_meas)}

        bstar_meas = MeasureColor._compute_color_metrics(foreground=Lab_foreground[..., 2], labels=image.objmap[:])
        bstar_meas = {key: value for key, value in zip(ColorLab.b_star_headers(), bstar_meas)}

        del Lab_foreground
        data = {**data, **lstar_meas, **astar_meas, **bstar_meas}

        # HSB Measurements
        hsb_foreground = image.color.hsv.foreground()
        logger.info("Computing color metrics for hue array")
        hue_meas = MeasureColor._compute_color_metrics(foreground=hsb_foreground[..., 0], labels=image.objmap[:],
                                                       )
        hue_meas = {key: value for key, value in zip(ColorHSV.hue_headers(), hue_meas)}

        logger.info("Computing color metrics for saturation array")
        saturation_meas = MeasureColor._compute_color_metrics(foreground=hsb_foreground[..., 1], labels=image.objmap[:],
                                                              )
        saturation_meas = {key: value for key, value in zip(ColorHSV.saturation_headers(), saturation_meas)}

        logger.info("Computing color metrics for brightness array")
        brightness_meas = MeasureColor._compute_color_metrics(foreground=hsb_foreground[..., 2], labels=image.objmap[:],
                                                              )
        brightness_meas = {key: value for key, value in zip(ColorHSV.brightness_headers(), brightness_meas)}

        del hsb_foreground
        data = {**data, **hue_meas, **saturation_meas, **brightness_meas}

        meas = pd.DataFrame(data=data)

        meas.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        meas.loc[:, str(ColorLab.CHROMA_EST_MEAN)] = np.sqrt(
                (meas.loc[:, str(ColorLab.A_STAR_MEAN)] ** 2) + meas.loc[:, str(ColorLab.B_STAR_MEAN)] ** 2)
        meas.loc[:, str(ColorLab.CHROMA_EST_MEDIAN)] = np.sqrt(
                (meas.loc[:, str(ColorLab.A_STAR_MEDIAN)] ** 2) + meas.loc[:, str(ColorLab.B_STAR_MEDIAN)] ** 2)

        return meas

    @staticmethod
    def _compute_color_metrics(foreground: np.ndarray, labels: np.ndarray):
        """
          Computes texture metrics from arr image data and a binary foreground mask.

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

        measurements = [
            MeasureFeatures._calculate_minimum(array=foreground, labels=labels),
            MeasureFeatures._calculate_q1(array=foreground, labels=labels),
            MeasureFeatures._calculate_mean(array=foreground, labels=labels),
            MeasureFeatures._calculate_median(array=foreground, labels=labels),
            MeasureFeatures._calculate_q3(array=foreground, labels=labels),
            MeasureFeatures._calculate_max(array=foreground, labels=labels),
            MeasureFeatures._calculate_stddev(array=foreground, labels=labels),
            MeasureFeatures._calculate_coeff_variation(array=foreground, labels=labels),
        ]
        return measurements


MeasureColor.__doc__ = ColorHSV.append_rst_to_doc(
        ColorLab.append_rst_to_doc(
                Colorxy.append_rst_to_doc(
                        ColorXYZ.append_rst_to_doc(MeasureColor)
                )
        )
)
