from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import pandas as pd
import logging

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools_.constants_ import OBJECT
from ..tools_.measurement_info_ import ColorXYZ, Colorxy, ColorLab, ColorHSV

logger = logging.getLogger(__name__)


class MeasureColor(MeasureFeatures):
    """Measure colony color statistics across multiple perceptual color spaces.

    Extract per-colony color features from CIE XYZ, chromaticity (xy),
    CIE Lab (perceptually uniform), and HSV color spaces. For each
    channel the standard statistical suite is computed (min, Q1, mean,
    median, Q3, max, std dev, coefficient of variation), plus Lab chroma
    estimates.

    Best For:
        - Distinguishing pigmented colonies (carotenoid, melanin) from
          colorless ones to stratify phenotypes by pigmentation profile.
        - Detecting sectoring and growth heterogeneity via high
          within-colony color variance.
        - Cross-plate comparison of colony pigmentation using
          perceptually uniform Lab distances.

    Consider Also:
        - :class:`MeasureIntensity` for grayscale-only brightness and
          variability statistics.
        - :class:`MeasureColorComposition` for proportion-based color
          classification of colony pixels.
        - :class:`MeasureTexture` for surface-roughness features that
          complement color metrics.

    Args:
        white_chroma_max: Lab chroma threshold below which a colony is
            classified as achromatic (white). Default: ``4.0``.
        chroma_min: Minimum chroma value retained in analysis; colonies
            below this are treated as colorless. Default: ``8.0``.
        include_XYZ: Compute CIE XYZ tristimulus statistics (slower).
            Default: ``False``.

    Returns:
        pd.DataFrame: Object-level color statistics with column groups:

            - ColorXYZ (X, Y, Z) -- only when ``include_XYZ=True``.
            - Colorxy (x, y chromaticity).
            - ColorLab (L*, a*, b*, ChromaEstimated).
            - ColorHSV (Hue, Saturation, Brightness).
            - Each channel has Min, Q1, Mean, Median, Q3, Max, StdDev,
              CoeffVar sub-columns.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting color metrics in a biological context.
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
