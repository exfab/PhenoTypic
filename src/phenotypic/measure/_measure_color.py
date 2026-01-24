from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
import pandas as pd
import logging

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools_.constants_ import OBJECT
from ..tools_.measurement_info_ import ColorXYZ, Colorxy, ColorLab, ColorHSV

logger = logging.getLogger(__name__)


class MeasureColor(MeasureFeatures):
    """Measure color characteristics of colonies across multiple perceptual color spaces.

    This class extracts quantitative color statistics from segmented colonies using CIE XYZ,
    chromaticity (xy), CIE Lab (perceptually uniform), and HSV (hue-saturation-value) color spaces.
    For each color space, it computes intensity-independent statistical features (min, Q1, mean, median,
    Q3, max, standard deviation, coefficient of variation) per colony, plus chroma estimates in Lab space.

    **Intuition:** Colony color provides phenotypic information about pigmentation, sporulation,
    metabolic products, and stress responses. Measuring color in multiple spaces captures different
    aspects: XYZ and xy are standardized for illuminant-independent comparisons, Lab is perceptually
    uniform (equal Euclidean distances reflect equal perceived color differences), and HSV separates
    hue (pigment type) from saturation and brightness. Colony-level color variation (e.g., std dev)
    indicates uneven growth, zonation, or heterogeneous populations.

    **Use cases (agar plates):**
    - Distinguish pigmented colonies (e.g., red/yellow carotenoid-producing bacteria, dark melanin)
      from colorless ones; stratify phenotypes by pigmentation profile.
    - Detect sectoring and growth heterogeneity via high color variance within single colonies.
    - Use chromaticity (xy) or hue to identify mixed cultures or secondary growth on a plate.
    - Enable image-based selection of colonies with specific pigmentation traits (e.g., high-chroma red vs pale).
    - Assess whether color measurements cluster by genotype or growth condition for cross-plate comparisons.

    **Caveats:**
    - Color measurements are highly sensitive to illumination, camera white balance, and exposure settings;
      normalize and calibrate your imaging setup before comparing colors across plates or experiments.
    - Lab and HSV assume RGB input is correctly gamma-corrected and linearized; use image.gray or
      image.enh_gray if raw RGB is uncalibrated.
    - High saturation and brightness variance within a colony can indicate shadow regions, uneven
      lighting, or non-uniform mycelial depth; interpret texture variance alongside color variance.
    - Chroma estimates use simplified arithmetic; for critical applications, use reference color charts
      or spectrophotometry to validate color classifications.
    - XYZ inclusion is optional and slow; enable only if standardized color space analysis is essential.

    Args:
        white_chroma_max (float, optional): Chroma threshold below which a colony is classified as
            "white" (achromatic). Used to filter Lab chroma calculations. Defaults to 4.0.
        chroma_min (float, optional): Minimum chroma value to retain in analysis; colonies below this
            are sometimes treated as colorless. Defaults to 8.0.
        include_XYZ (bool, optional): Whether to compute CIE XYZ measurements (slower, less common).
            Defaults to False.

    Returns:
        pd.DataFrame: Object-level color statistics with columns organized by color space:
            - ColorXYZ: X, Y, Z tristimulus values (if include_XYZ=True).
            - Colorxy: Chromaticity coordinates x, y (perceptual color without brightness).
            - ColorLab: L* (lightness), a* (green-red), b* (blue-yellow), and chroma estimates.
            - ColorHSV: Hue (angle, color identity), Saturation (intensity of color), Brightness (luminosity).
            For each channel: Min, Q1, Mean, Median, Q3, Max, StdDev, CoeffVar.

    Examples:
        .. dropdown:: Measure colony color to detect pigmented mutants

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import OtsuDetector
                from phenotypic.measure import MeasureColor

                # Load image of colonies (may include pigmented and non-pigmented strains)
                image = Image.imread("mixed_pigment_plate.jpg")
                detector = OtsuDetector()
                image = detector.operate(image)

                # Measure color
                measurer = MeasureColor(include_XYZ=False)
                colors = measurer.operate(image)

                # Identify pigmented colonies by hue and saturation
                pigmented = colors[colors['ColorHSV_SaturationMean'] > 15]
                print(f"Found {len(pigmented)} pigmented colonies")

        .. dropdown:: Use Lab color space for perceptually uniform analysis

            .. code-block:: python

                # Measure using Lab space (perceptually uniform)
                measurer = MeasureColor()
                colors = measurer.operate(image)

                # Chroma estimates reflect perceived "colorfulness"
                bright_red = colors[
                    (colors['ColorLab_L*Mean'] > 50) &
                    (colors['ColorLab_ChromaEstimatedMean'] > 20)
                ]
                print(f"Bright red colonies: {len(bright_red)}")
    """

    _measurement_info_classes = [ColorXYZ, Colorxy, ColorLab, ColorHSV]

    def __init__(
            self,
            white_chroma_max: float = 4.0,
            chroma_min: float = 8.0,
            include_XYZ: bool = False,
    ):
        self.white_chroma_max = white_chroma_max
        self.chroma_min = chroma_min
        self.include_XYZ = include_XYZ

    def _operate(self, image: Image):
        data = {}
        if self.include_XYZ:
            cieXYZ_foreground = image.color.XYZ.foreground()
            X_meas = MeasureColor._compute_color_metrics(
                    foreground=cieXYZ_foreground[..., 0], objmap=image.objmap[:]
            )
            X_meas = {key: value for key, value in zip(ColorXYZ.cieX_headers(), X_meas)}

            Y_meas = MeasureColor._compute_color_metrics(
                    foreground=cieXYZ_foreground[..., 1], objmap=image.objmap[:]
            )
            Y_meas = {key: value for key, value in zip(ColorXYZ.cieY_headers(), Y_meas)}

            Z_meas = MeasureColor._compute_color_metrics(
                    foreground=cieXYZ_foreground[..., 2], objmap=image.objmap[:]
            )
            Z_meas = {key: value for key, value in zip(ColorXYZ.cieZ_headers(), Z_meas)}

            del cieXYZ_foreground
            data = {**data, **X_meas, **Y_meas, **Z_meas}

        xy_foreground = image.color.xy.foreground()
        x_meas = MeasureColor._compute_color_metrics(
                foreground=xy_foreground[..., 0], objmap=image.objmap[:]
        )
        x_meas = {key: value for key, value in zip(Colorxy.x_headers(), x_meas)}

        y_meas = MeasureColor._compute_color_metrics(
                foreground=xy_foreground[..., 1], objmap=image.objmap[:]
        )
        y_meas = {key: value for key, value in zip(Colorxy.y_headers(), y_meas)}

        del xy_foreground
        data = {**data, **x_meas, **y_meas}

        Lab_foreground = image.color.Lab.foreground()
        lstar_meas = MeasureColor._compute_color_metrics(
                foreground=Lab_foreground[..., 0], objmap=image.objmap[:]
        )
        lstar_meas = {
            key: value for key, value in zip(ColorLab.l_star_headers(), lstar_meas)
        }

        astar_meas = MeasureColor._compute_color_metrics(
                foreground=Lab_foreground[..., 1], objmap=image.objmap[:]
        )
        astar_meas = {
            key: value for key, value in zip(ColorLab.a_star_headers(), astar_meas)
        }

        bstar_meas = MeasureColor._compute_color_metrics(
                foreground=Lab_foreground[..., 2], objmap=image.objmap[:]
        )
        bstar_meas = {
            key: value for key, value in zip(ColorLab.b_star_headers(), bstar_meas)
        }

        del Lab_foreground
        data = {**data, **lstar_meas, **astar_meas, **bstar_meas}

        # HSB Measurements
        hsb_foreground = image.color.hsv.foreground()
        logger.info("Computing color metrics for hue array")
        hue_meas = MeasureColor._compute_color_metrics(
                foreground=hsb_foreground[..., 0],
                objmap=image.objmap[:],
        )
        hue_meas = {key: value for key, value in zip(ColorHSV.hue_headers(), hue_meas)}

        logger.info("Computing color metrics for saturation array")
        saturation_meas = MeasureColor._compute_color_metrics(
                foreground=hsb_foreground[..., 1],
                objmap=image.objmap[:],
        )
        saturation_meas = {
            key: value
            for key, value in zip(ColorHSV.saturation_headers(), saturation_meas)
        }

        logger.info("Computing color metrics for brightness array")
        brightness_meas = MeasureColor._compute_color_metrics(
                foreground=hsb_foreground[..., 2],
                objmap=image.objmap[:],
        )
        brightness_meas = {
            key: value
            for key, value in zip(ColorHSV.brightness_headers(), brightness_meas)
        }

        del hsb_foreground
        data = {**data, **hue_meas, **saturation_meas, **brightness_meas}

        meas = pd.DataFrame(data=data)

        meas.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        meas.loc[:, str(ColorLab.CHROMA_EST_MEAN)] = np.sqrt(
                (meas.loc[:, str(ColorLab.A_STAR_MEAN)] ** 2)
                + meas.loc[:, str(ColorLab.B_STAR_MEAN)] ** 2
        )
        meas.loc[:, str(ColorLab.CHROMA_EST_MEDIAN)] = np.sqrt(
                (meas.loc[:, str(ColorLab.A_STAR_MEDIAN)] ** 2)
                + meas.loc[:, str(ColorLab.B_STAR_MEDIAN)] ** 2
        )

        return meas

    @staticmethod
    def _compute_color_metrics(foreground: np.ndarray, objmap: np.ndarray):
        """
        Computes texture metrics from arr image data and a binary foreground mask.

        This function processes gridded image objects and calculates various texture
        features using Haralick descriptors across segmented objects. The calculated
        texture metrics include statistical data and Haralick texture features, which
        are useful in descriptive and diagnostic analyses for image processing applications.

        Args:
            foreground (numpy.ndarray): A matrix array with all background pixels set
                to 0, defining the binary mask.
            objmap (numpy.ndarray): Array of labels of the same shape as the foreground array.

        Returns:
            dict: A dictionary containing calculated measurements, including object
                labels, statistical data (e.g., area, mean, standard deviation), and
                multiple Haralick texture metrics (e.g., contrast, entropy).
        """

        measurements = [
            MeasureFeatures._calculate_minimum(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_q1(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_mean(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_median(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_q3(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_maximum(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_stddev(array=foreground, objmap=objmap),
            MeasureFeatures._calculate_coeff_variation(array=foreground, objmap=objmap),
        ]
        return measurements


MeasureColor.__doc__ = ColorHSV.append_rst_to_doc(
        ColorLab.append_rst_to_doc(
                Colorxy.append_rst_to_doc(ColorXYZ.append_rst_to_doc(MeasureColor))
        )
)
