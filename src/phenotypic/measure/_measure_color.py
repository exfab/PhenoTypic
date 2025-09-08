from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd
import colour
import logging

from phenotypic.abstract import MeasureFeatures, MeasurementInfo

logger = logging.getLogger(__name__)


# TODO: Dominant wavelength, purity, and luminance calculation commented out until algorithm optimization
class COLOR(MeasurementInfo):
    @property
    def CATEGORY(self):
        return 'Color'

    # DOMWL_MINIMUM = ('DomWLMinimum', 'The minimum dominant wavelength of the object')
    # DOMWL_Q1 = ('DomWLLowerQuartile', 'The lower quartile (Q1) dominant wavelength of the object')
    # DOMWL_MEAN = ('DomWLMean', 'The mean dominant wavelength of the object')
    # DOMWL_MEDIAN = ('DomWLMedian', 'The median dominant wavelength of the object')
    # DOMWL_Q3 = ('DomWLUpperQuartile', 'The upper quartile (Q3) dominant wavelength of the object')
    # DOMWL_MAXIMUM = ('DomWLMaximum', 'The maximum dominant wavelength of the object')
    # DOMWL_STDDEV = ('DomWLStdDev', 'The standard deviation of the dominant wavelength of the object')
    # DOMWL_COEFF_VARIANCE = ('DomWLCoefficientVariance', 'The coefficient of variation of the dominant wavelength of the object')
    #
    # @classmethod
    # def domwl_labels(cls):
    #     return [str(cls.DOMWL_MINIMUM), str(cls.DOMWL_Q1), str(cls.DOMWL_MEAN), str(cls.DOMWL_MEDIAN), str(cls.DOMWL_Q3),
    #             str(cls.DOMWL_MAXIMUM), str(cls.DOMWL_STDDEV), str(cls.DOMWL_COEFF_VARIANCE), ]
    #
    # PURITY_MINIMUM = ('PurityMinimum', 'The minimum excitation purity of the object')
    # PURITY_Q1 = ('PurityLowerQuartile', 'The lower quartile (Q1) excitation purity of the object')
    # PURITY_MEAN = ('PurityMean', 'The mean excitation purity of the object')
    # PURITY_MEDIAN = ('PurityMedian', 'The median excitation purity of the object')
    # PURITY_Q3 = ('PurityUpperQuartile', 'The upper quartile (Q3) excitation purity of the object')
    # PURITY_MAXIMUM = ('PurityMaximum', 'The maximum excitation purity of the object')
    # PURITY_STDDEV = ('PurityStdDev', 'The standard deviation of the excitation purity of the object')
    # PURITY_COEFF_VARIANCE = ('PurityCoefficientVariance', 'The coefficient of variation of the excitation purity of the object')
    #
    # @classmethod
    # def purity_labels(cls):
    #     return [str(cls.PURITY_MINIMUM), str(cls.PURITY_Q1), str(cls.PURITY_MEAN), str(cls.PURITY_MEDIAN), str(cls.PURITY_Q3),
    #             str(cls.PURITY_MAXIMUM), str(cls.PURITY_STDDEV), str(cls.PURITY_COEFF_VARIANCE), ]
    #
    # LUMINANCE_MINIMUM = ('LuminanceMinimum', 'The minimum luminance of the object')
    # LUMINANCE_Q1 = ('LuminanceLowerQuartile', 'The lower quartile (Q1) luminance of the object')
    # LUMINANCE_MEAN = ('LuminanceMean', 'The mean luminance of the object')
    # LUMINANCE_MEDIAN = ('LuminanceMedian', 'The median luminance of the object')
    # LUMINANCE_Q3 = ('LuminanceUpperQuartile', 'The upper quartile (Q3) luminance of the object')
    # LUMINANCE_MAXIMUM = ('LuminanceMaximum', 'The maximum luminance of the object')
    # LUMINANCE_STDDEV = ('LuminanceStdDev', 'The standard deviation of the luminance of the object')
    # LUMINANCE_COEFF_VARIANCE = ('LuminanceCoefficientVariance', 'The coefficient of variation of the luminance of the object')
    #
    # @classmethod
    # def luminance_labels(cls):
    #     return [str(cls.LUMINANCE_MINIMUM), str(cls.LUMINANCE_Q1), str(cls.LUMINANCE_MEAN), str(cls.LUMINANCE_MEDIAN),
    #             str(cls.LUMINANCE_Q3), str(cls.LUMINANCE_MAXIMUM), str(cls.LUMINANCE_STDDEV), str(cls.LUMINANCE_COEFF_VARIANCE), ]

    L_STAR_MINIMUM = ('L*', 'The minimum L* value of the object')
    L_STAR_Q1 = ('L*LowerQuartile', 'The lower quartile (Q1) L* value of the object')
    L_STAR_MEAN = ('L*Mean', 'The mean L* value of the object')
    L_STAR_MEDIAN = ('L*Median', 'The median L* value of the object')
    L_STAR_Q3 = ('L*UpperQuartile', 'The upper quartile (Q3) L* value of the object')
    L_STAR_MAXIMUM = ('L*Maximum', 'The maximum L* value of the object')
    L_STAR_STDDEV = ('L*StdDev', 'The standard deviation of the L* value of the object')
    L_STAR_COEFF_VARIANCE = ('L*CoefficientVariance', 'The coefficient of variation of the L* value of the object')

    @classmethod
    def l_star_headers(cls):
        return [str(cls.L_STAR_MINIMUM), str(cls.L_STAR_Q1), str(cls.L_STAR_MEAN), str(cls.L_STAR_MEDIAN),
                str(cls.L_STAR_Q3), str(cls.L_STAR_MAXIMUM), str(cls.L_STAR_STDDEV), str(cls.L_STAR_COEFF_VARIANCE), ]

    A_STAR_MINIMUM = ('a*', 'The minimum a* value of the object')
    A_STAR_Q1 = ('a*LowerQuartile', 'The lower quartile (Q1) a* value of the object')
    A_STAR_MEAN = ('a*Mean', 'The mean a* value of the object')
    A_STAR_MEDIAN = ('a*Median', 'The median a* value of the object')
    A_STAR_Q3 = ('a*UpperQuartile', 'The upper quartile (Q3) a* value of the object')
    A_STAR_MAXIMUM = ('a*Maximum', 'The maximum a* value of the object')
    A_STAR_STDDEV = ('a*StdDev', 'The standard deviation of the a* value of the object')
    A_STAR_COEFF_VARIANCE = ('a*CoefficientVariance', 'The coefficient of variation of the a* value of the object')

    @classmethod
    def a_star_headers(cls):
        return [str(cls.A_STAR_MINIMUM), str(cls.A_STAR_Q1), str(cls.A_STAR_MEAN), str(cls.A_STAR_MEDIAN),
                str(cls.A_STAR_Q3), str(cls.A_STAR_MAXIMUM), str(cls.A_STAR_STDDEV), str(cls.A_STAR_COEFF_VARIANCE), ]

    B_STAR_MINIMUM = ('b*', 'The minimum b* value of the object')
    B_STAR_Q1 = ('b*LowerQuartile', 'The lower quartile (Q1) b* value of the object')
    B_STAR_MEAN = ('b*Mean', 'The mean b* value of the object')
    B_STAR_MEDIAN = ('b*Median', 'The median b* value of the object')
    B_STAR_Q3 = ('b*UpperQuartile', 'The upper quartile (Q3) b* value of the object')
    B_STAR_MAXIMUM = ('b*Maximum', 'The maximum b* value of the object')
    B_STAR_STDDEV = ('b*StdDev', 'The standard deviation of the b* value of the object')
    B_STAR_COEFF_VARIANCE = ('b*CoefficientVariance', 'The coefficient of variation of the b* value of the object')

    @classmethod
    def b_star_headers(cls):
        return [str(cls.B_STAR_MINIMUM), str(cls.B_STAR_Q1), str(cls.B_STAR_MEAN), str(cls.B_STAR_MEDIAN),
                str(cls.B_STAR_Q3), str(cls.B_STAR_MAXIMUM), str(cls.B_STAR_STDDEV), str(cls.B_STAR_COEFF_VARIANCE), ]

    CHROMA_EST_MEAN = ('ChromaEstimatedMean', 'The mean chroma estimation of the object')
    CHROMA_EST_MEDIAN = ('ChromaEstimatedMedian', 'The median chroma estimation of the object')

    HUE_MINIMUM = ('HueMinimum', 'The minimum hue of the object')
    HUE_Q1 = ('HueLowerQuartile', 'The lower quartile (Q1) hue of the object')
    HUE_MEAN = ('HueMean', 'The mean hue of the object')
    HUE_MEDIAN = ('HueMedian', 'The median hue of the object')
    HUE_Q3 = ('HueUpperQuartile', 'The upper quartile (Q3) hue of the object')
    HUE_MAXIMUM = ('HueMaximum', 'The maximum hue of the object')
    HUE_STDDEV = ('HueStdDev', 'The standard deviation of the hue of the object')
    HUE_COEFF_VARIANCE = ('HueCoefficientVariance', 'The coefficient of variation of the hue of the object')

    @classmethod
    def hue_headers(cls):
        return [str(cls.HUE_MINIMUM), str(cls.HUE_Q1), str(cls.HUE_MEAN), str(cls.HUE_MEDIAN), str(cls.HUE_Q3), str(cls.HUE_MAXIMUM),
                str(cls.HUE_STDDEV), str(cls.HUE_COEFF_VARIANCE), ]

    SATURATION_MINIMUM = ('SaturationMinimum', 'The minimum saturation of the object')
    SATURATION_Q1 = ('SaturationLowerQuartile', 'The lower quartile (Q1) saturation of the object')
    SATURATION_MEAN = ('SaturationMean', 'The mean saturation of the object')
    SATURATION_MEDIAN = ('SaturationMedian', 'The median saturation of the object')
    SATURATION_Q3 = ('SaturationUpperQuartile', 'The upper quartile (Q3) saturation of the object')
    SATURATION_MAXIMUM = ('SaturationMaximum', 'The maximum saturation of the object')
    SATURATION_STDDEV = ('SaturationStdDev', 'The standard deviation of the saturation of the object')
    SATURATION_COEFF_VARIANCE = ('SaturationCoefficientVariance', 'The coefficient of variation of the saturation of the object')

    @classmethod
    def saturation_headers(cls):
        return [str(cls.SATURATION_MINIMUM), str(cls.SATURATION_Q1), str(cls.SATURATION_MEAN), str(cls.SATURATION_MEDIAN),
                str(cls.SATURATION_Q3), str(cls.SATURATION_MAXIMUM), str(cls.SATURATION_STDDEV), str(cls.SATURATION_COEFF_VARIANCE), ]

    BRIGHTNESS_MINIMUM = ('BrightnessMinimum', 'The minimum brightness of the object')
    BRIGHTNESS_Q1 = ('BrightnessLowerQuartile', 'The lower quartile (Q1) brightness of the object')
    BRIGHTNESS_MEAN = ('BrightnessMean', 'The mean brightness of the object')
    BRIGHTNESS_MEDIAN = ('BrightnessMedian', 'The median brightness of the object')
    BRIGHTNESS_Q3 = ('BrightnessUpperQuartile', 'The upper quartile (Q3) brightness of the object')
    BRIGHTNESS_MAXIMUM = ('BrightnessMaximum', 'The maximum brightness of the object')
    BRIGHTNESS_STDDEV = ('BrightnessStdDev', 'The standard deviation of the brightness of the object')
    BRIGHTNESS_COEFF_VARIANCE = ('BrightnessCoefficientVariance', 'The coefficient of variation of the brightness of the object')

    @classmethod
    def brightness_headers(cls):
        return [str(cls.BRIGHTNESS_MINIMUM), str(cls.BRIGHTNESS_Q1), str(cls.BRIGHTNESS_MEAN), str(cls.BRIGHTNESS_MEDIAN),
                str(cls.BRIGHTNESS_Q3), str(cls.BRIGHTNESS_MAXIMUM), str(cls.BRIGHTNESS_STDDEV), str(cls.BRIGHTNESS_COEFF_VARIANCE), ]


class MeasureColor(MeasureFeatures):
    """
    Performs color-based feature extraction and calculations on segmented image objects.

    This class extends the MeasureFeatures class to provide additional methods for analyzing
    the color properties of segmented objects within images. It computes various color-related
    metrics, including those based on the HSB (hue, saturation, brightness) color model. These calculations
    can be used in image analysis workflows to extract detailed information about the color
    characteristics of objects.

    """

    def __init__(self, white_chroma_max: float = 4.0, chroma_min: float = 8.0):
        self.white_chroma_max = white_chroma_max
        self.chroma_min = chroma_min

    def _operate(self, image: Image):
        # # XYZ and xy measurements
        # # Use the D65 illumination for consistency
        # xy = image.xy()
        # dom_wl, purity = self._dominant_wavelength_xy(xy=xy, white=image.illuminant, observer=image.observer)
        # logger.info("Computing color metrics for dominant wavelength array")
        # dom_wl_meas = MeasureColor._compute_color_metrics(foreground=dom_wl, labels=image.objmap[:])
        # dom_wl_meas = {key: value for key, value in zip(COLOR.domwl_labels(), dom_wl_meas)}
        #
        # logger.info("Computing color metrics for purity array")
        # purity_meas = MeasureColor._compute_color_metrics(foreground=purity, labels=image.objmap[:])
        # purity_meas = {key: value for key, value in zip(COLOR.purity_labels(), purity_meas)}
        #
        # XYZ = image.xyzD65()
        # Y = XYZ[..., 1]
        # logger.info("Computing color metrics for luminance array")
        # luminance_meas = MeasureColor._compute_color_metrics(foreground=Y, labels=image.objmap[:])
        # luminance_meas = {key: value for key, value in zip(COLOR.luminance_labels(), luminance_meas)}

        Lab_foreground = image.to_Lab()
        Lab_foreground = image.objmask._create_foreground(array=Lab_foreground, bg_label=0)

        lstar_meas = MeasureColor._compute_color_metrics(foreground=Lab_foreground[..., 0], labels=image.objmap[:])
        lstar_meas = {key: value for key, value in zip(COLOR.l_star_headers(), lstar_meas)}

        astar_meas = MeasureColor._compute_color_metrics(foreground=Lab_foreground[..., 1], labels=image.objmap[:])
        astar_meas = {key: value for key, value in zip(COLOR.a_star_headers(), astar_meas)}

        bstar_meas = MeasureColor._compute_color_metrics(foreground=Lab_foreground[..., 2], labels=image.objmap[:])
        bstar_meas = {key: value for key, value in zip(COLOR.b_star_headers(), bstar_meas)}

        del Lab_foreground

        # HSB Measurements
        hsb_foreground = image.hsv.foreground()
        logger.info("Computing color metrics for hue array")
        hue_meas = MeasureColor._compute_color_metrics(foreground=hsb_foreground[..., 0], labels=image.objmap[:],
                                                       )
        hue_meas = {key: value for key, value in zip(COLOR.hue_headers(), hue_meas)}

        logger.info("Computing color metrics for saturation array")
        saturation_meas = MeasureColor._compute_color_metrics(foreground=hsb_foreground[..., 1], labels=image.objmap[:],
                                                              )
        saturation_meas = {key: value for key, value in zip(COLOR.saturation_headers(), saturation_meas)}

        logger.info("Computing color metrics for brightness array")
        brightness_meas = MeasureColor._compute_color_metrics(foreground=hsb_foreground[..., 2], labels=image.objmap[:],
                                                              )
        brightness_meas = {key: value for key, value in zip(COLOR.brightness_headers(), brightness_meas)}

        del hsb_foreground

        meas = pd.DataFrame(data={
            # **dom_wl_meas, **purity_meas, **luminance_meas,
            **lstar_meas, **astar_meas, **bstar_meas,
            **hue_meas, **saturation_meas, **brightness_meas},
            index=image.objects.labels2series())
        meas.loc[:, str(COLOR.CHROMA_EST_MEAN)] = np.sqrt((meas.loc[:, str(COLOR.A_STAR_MEAN)] ** 2) + meas.loc[:, str(COLOR.B_STAR_MEAN)] ** 2)
        meas.loc[:, str(COLOR.CHROMA_EST_MEDIAN)] = np.sqrt((meas.loc[:, str(COLOR.A_STAR_MEDIAN)] ** 2) + meas.loc[:, str(COLOR.B_STAR_MEDIAN)] ** 2)

        return meas

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

    @staticmethod
    def _dominant_wavelength_xy(xy: np.ndarray,
                                labels: np.ndarray | None = None,
                                white: str = "D65",
                                observer: str = "CIE 1931 2 Degree Standard Observer"):
        """
        Computes the dominant wavelength and purity for given chromaticity coordinates.

        This static method calculates the dominant wavelength and excitation purity for a
        set of chromaticity coordinates (`xy`) relative to a specified white point and
        observer. The computation is vectorized, allowing for efficient processing of
        multiple chromaticity values simultaneously. Near-white numerics are handled,
        ensuring robust results for inputs very close to the white point.

        Args:
            xy (numpy.ndarray): A NumPy array of shape (..., 2) where the last dimension
                contains the chromaticity coordinates (x, y).
            white (str, optional): A string specifying the white point to use. Default
                is "D65".
            observer (str, optional): A string specifying the standard observer to use.
                Default is "CIE 1931 2 Degree Standard Observer".

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Dominant wavelengths (in nanometers) as a NumPy array
                    with the same shape as the input chromaticity coordinates without
                    the last dimension.
                - numpy.ndarray: Excitation purity as a NumPy array with the same shape
                    as the chromaticity coordinates without the last dimension.
        """
        wp = colour.CCS_ILLUMINANTS[observer][white]  # achromatic xy
        orig_shape = xy.shape[:-1]
        if labels:
            xy = xy.copy()
            xy[labels == 0] = np.nan

        xy_flat = xy.reshape(-1, 2)

        # Get the indexes of all the coords where there is an object
        xy_flat_obj_mask = ~np.isnan(xy_flat).any(axis=1)
        xy_not_nan = xy_flat[xy_flat_obj_mask, :]

        wl_nm_obj, _, _ = colour.dominant_wavelength(xy=xy_not_nan, xy_n=wp)  # vectorized
        final_wl_nm = np.zeros(shape=xy_flat.shape[0])
        final_wl_nm[xy_flat_obj_mask] = wl_nm_obj

        purity_obj = colour.excitation_purity(xy=xy_not_nan, xy_n=wp)
        final_purity = np.zeros(shape=xy_flat.shape[0])
        final_purity[xy_flat_obj_mask] = purity_obj

        return final_wl_nm.reshape(orig_shape), final_purity.reshape(orig_shape)


MeasureColor.__doc__ = COLOR.append_rst_to_doc(MeasureColor)
