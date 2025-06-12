from __future__ import annotations
from typing import TYPE_CHECKING, Callable

from mahotas.features.texture import haralick_features

if TYPE_CHECKING: from phenotypic import Image

import warnings
import mahotas as mh
import numpy as np
import pandas as pd
from skimage.util import img_as_ubyte

from phenotypic.abstract import MeasureFeatures
from phenotypic.util.constants_ import OBJECT

CATEGORY_TEXTURE = 'Texture'
ANGULAR_SECOND_MOMENT_0 = 'AngularSecondMoment-deg(0)'
ANGULAR_SECOND_MOMENT_45 = 'AngularSecondMoment-deg(45)'
ANGULAR_SECOND_MOMENT_90 = 'AngularSecondMoment-deg(90)'
ANGULAR_SECOND_MOMENT_135 = 'AngularSecondMoment-deg(135)'

CONTRAST_0 = 'Contrast-deg(0)'
CONTRAST_45 = 'Contrast-deg(45)'
CONTRAST_90 = 'Contrast-deg(90)'
CONTRAST_135 = 'Contrast-deg(135)'

CORRELATION_0 = 'Correlation-deg(0)'
CORRELATION_45 = 'Correlation-deg(45)'
CORRELATION_90 = 'Correlation-deg(90)'
CORRELATION_135 = 'Correlation-deg(135)'

VARIANCE_0 = 'HaralickVariance-deg(0)'
VARIANCE_45 = 'HaralickVariance-deg(45)'
VARIANCE_90 = 'HaralickVariance-deg(90)'
VARIANCE_135 = 'HaralickVariance-deg(135)'

INVERSE_DIFFERENCE_MOMENT_0 = 'InverseDifferenceMoment-deg(0)'
INVERSE_DIFFERENCE_MOMENT_45 = 'InverseDifferenceMoment-deg(45)'
INVERSE_DIFFERENCE_MOMENT_90 = 'InverseDifferenceMoment-deg(90)'
INVERSE_DIFFERENCE_MOMENT_135 = 'InverseDifferenceMoment-deg(135)'

SUM_AVERAGE_0 = 'SumAverage-deg(0)'
SUM_AVERAGE_45 = 'SumAverage-deg(45)'
SUM_AVERAGE_90 = 'SumAverage-deg(90)'
SUM_AVERAGE_135 = 'SumAverage-deg(135)'

SUM_VARIANCE_0 = 'SumVariance-deg(0)'
SUM_VARIANCE_45 = 'SumVariance-deg(45)'
SUM_VARIANCE_90 = 'SumVariance-deg(90)'
SUM_VARIANCE_135 = 'SumVariance-deg(135)'

SUM_ENTROPY_0 = 'SumEntropy-deg(0)'
SUM_ENTROPY_45 = 'SumEntropy-deg(45)'
SUM_ENTROPY_90 = 'SumEntropy-deg(90)'
SUM_ENTROPY_135 = 'SumEntropy-deg(135)'

ENTROPY_0 = 'Entropy-deg(0)'
ENTROPY_45 = 'Entropy-deg(45)'
ENTROPY_90 = 'Entropy-deg(90)'
ENTROPY_135 = 'Entropy-deg(135)'

DIFFERENCE_VARIANCE_0 = 'DifferenceVariance-deg(0)'
DIFFERENCE_VARIANCE_45 = 'DifferenceVariance-deg(45)'
DIFFERENCE_VARIANCE_90 = 'DifferenceVariance-deg(90)'
DIFFERENCE_VARIANCE_135 = 'DifferenceVariance-deg(135)'

DIFFERENCE_ENTROPY_0 = 'DifferenceEntropy-deg(0)'
DIFFERENCE_ENTROPY_45 = 'DifferenceEntropy-deg(45)'
DIFFERENCE_ENTROPY_90 = 'DifferenceEntropy-deg(90)'
DIFFERENCE_ENTROPY_135 = 'DifferenceEntropy-deg(135)'

IMC1_0 = 'InformationCorrelation(1)-deg(0)'
IMC1_45 = 'InformationCorrelation(1)-deg(45)'
IMC1_90 = 'InformationCorrelation(1)-deg(90)'
IMC1_135 = 'InformationCorrelation(1)-deg(135)'

IMC2_0 = 'InformationCorrelation(2)-deg(0)'
IMC2_45 = 'InformationCorrelation(2)-deg(45)'
IMC2_90 = 'InformationCorrelation(2)-deg(90)'
IMC2_135 = 'InformationCorrelation(2)-deg(135)'




class MeasureTexture(MeasureFeatures):
    """
    Represents a measurement of texture features extracted from _parent_image objects.

    This class is designed to calculate texture measurements derived from Haralick features,
    tailored for segmented objects in an _parent_image. These features include statistical properties
    that describe textural qualities, such as uniformity or variability, across different
    directional orientations. The class leverages statistical methods and _parent_image processing
    to extract meaningful characteristics applicable in _parent_image analysis tasks.

    Attributes:
        scale (int): The scale parameter used in the computation of texture features. It is
            often used to define the spatial relationship between pixels.

    References:
        [1] https://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.haralick
    """

    def __init__(self, scale: int = 5):
        """Initialize the MeasureTexture instance with a specified scale parameter.

        Args:
            scale (int, optional): The distance parameter used in calculating Haralick features.
                Defaults to 5.
        """
        self.scale = scale

    def _operate(self, image: Image):
        """Performs texture measurements on the image objects.

        This method extracts texture features from the foreground objects in the image using
        Haralick texture features. It processes the image's foreground array and returns
        the measurements as a DataFrame.

        Args:
            image (Image): The image containing objects to measure.

        Returns:
            pd.DataFrame: A DataFrame containing texture measurements for each object in the image.
                The rows are indexed by object labels, and columns represent different texture features.
        """
        texture_measurements = self._compute_matrix_texture(image=image,
                                                       foreground_array=image.matrix.get_foreground(),
                                                       foreground_name='intensity',
                                                       scale=self.scale,
                                                       )
        texture_measurements = {f'{CATEGORY_TEXTURE}_{key}': value for key, value in texture_measurements.items()}
        return pd.DataFrame(texture_measurements, index=image.objects.get_labels_series())

    @staticmethod
    def _compute_matrix_texture(image: Image, foreground_array: np.ndarray, foreground_name: str, scale: int = 5) -> dict:
        """
        Computes texture feature measurements using Haralick features for objects in a given _parent_image. The method
        calculates various statistical texture features such as Angular Second Moment, Contrast, Correlation,
        Variance, Inverse Difference Moment, among others, for different directional orientations. These
        features are computed for each segmented object within the foreground array using the specified
        scale parameter.

        Args:
            image (Image): The _parent_image containing objects and their associated properties, including
                labels and slices used for extracting foreground objects.
            foreground_array (np.ndarray): The 2D numpy array representing the foreground objects,
                where pixel values indicate the object intensity.
            foreground_name (str): The name of the foreground for labeling the resulting features.
            scale (int, optional): The distance parameter used in calculating Haralick features.
                Defaults to 5.

        Returns:
            dict: A dictionary mapping computed texture feature names (e.g.,
                "angular_second_moment", "contrast") to their corresponding values
                for each object in the foreground array.

        Raises:
            KeyboardInterrupt: If the computation process is interrupted manually.
            Warning: If an error occurs during the computation of Haralick features for specific objects, a
                warning is issued with details of the error, and NaN values are assigned for the corresponding
                measurements.
        """
        parameter_suffix = f'-{foreground_name}-scale({scale})'
        measurements = {
            ANGULAR_SECOND_MOMENT_0 + parameter_suffix: [],
            ANGULAR_SECOND_MOMENT_45 + parameter_suffix: [],
            ANGULAR_SECOND_MOMENT_90 + parameter_suffix: [],
            ANGULAR_SECOND_MOMENT_135 + parameter_suffix: [],
            CONTRAST_0 + parameter_suffix: [],
            CONTRAST_45 + parameter_suffix: [],
            CONTRAST_90 + parameter_suffix: [],
            CONTRAST_135 + parameter_suffix: [],
            CORRELATION_0 + parameter_suffix: [],
            CORRELATION_45 + parameter_suffix: [],
            CORRELATION_90 + parameter_suffix: [],
            CORRELATION_135 + parameter_suffix: [],
            VARIANCE_0 + parameter_suffix: [],
            VARIANCE_45 + parameter_suffix: [],
            VARIANCE_90 + parameter_suffix: [],
            VARIANCE_135 + parameter_suffix: [],
            INVERSE_DIFFERENCE_MOMENT_0 + parameter_suffix: [],
            INVERSE_DIFFERENCE_MOMENT_45 + parameter_suffix: [],
            INVERSE_DIFFERENCE_MOMENT_90 + parameter_suffix: [],
            INVERSE_DIFFERENCE_MOMENT_135 + parameter_suffix: [],
            SUM_AVERAGE_0 + parameter_suffix: [],
            SUM_AVERAGE_45 + parameter_suffix: [],
            SUM_AVERAGE_90 + parameter_suffix: [],
            SUM_AVERAGE_135 + parameter_suffix: [],
            SUM_VARIANCE_0 + parameter_suffix: [],
            SUM_VARIANCE_45 + parameter_suffix: [],
            SUM_VARIANCE_90 + parameter_suffix: [],
            SUM_VARIANCE_135 + parameter_suffix: [],
            SUM_ENTROPY_0 + parameter_suffix: [],
            SUM_ENTROPY_45 + parameter_suffix: [],
            SUM_ENTROPY_90 + parameter_suffix: [],
            SUM_ENTROPY_135 + parameter_suffix: [],
            ENTROPY_0 + parameter_suffix: [],
            ENTROPY_45 + parameter_suffix: [],
            ENTROPY_90 + parameter_suffix: [],
            ENTROPY_135 + parameter_suffix: [],
            DIFFERENCE_VARIANCE_0 + parameter_suffix: [],
            DIFFERENCE_VARIANCE_45 + parameter_suffix: [],
            DIFFERENCE_VARIANCE_90 + parameter_suffix: [],
            DIFFERENCE_VARIANCE_135 + parameter_suffix: [],
            DIFFERENCE_ENTROPY_0 + parameter_suffix: [],
            DIFFERENCE_ENTROPY_45 + parameter_suffix: [],
            DIFFERENCE_ENTROPY_90 + parameter_suffix: [],
            DIFFERENCE_ENTROPY_135 + parameter_suffix: [],
            IMC1_0 + parameter_suffix: [],
            IMC1_45 + parameter_suffix: [],
            IMC1_90 + parameter_suffix: [],
            IMC1_135 + parameter_suffix: [],
            IMC2_0 + parameter_suffix: [],
            IMC2_45 + parameter_suffix: [],
            IMC2_90 + parameter_suffix: [],
            IMC2_135 + parameter_suffix: [],
        }

        props = image.objects.props
        objmap = image.objmap[:]
        for idx, label in enumerate(image.objects.labels):
            slices = props[idx].slice
            obj_extracted = foreground_array[slices].copy()

            # In case there's more than one object in the crop
            obj_extracted[objmap[slices] != label] = 0

            try:
                if obj_extracted.sum() == 0:
                    return np.full((4, 13), np.nan, dtype=np.float64)
                else:
                    # Pad object with zero if its dimensions are smaller than the scale

                    haralick_features = mh.features.haralick(img_as_ubyte(obj_extracted),
                                                             distance=scale,
                                                             ignore_zeros=True,
                                                             return_mean=False,
                                                             )
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # 4 for each direction, 13 for each texture feature
                warnings.warn(f'Error in computing Haralick features for object {label}: {e}')
                haralick_features = np.full((4, 13), np.nan, dtype=np.float64)

            # Angular Second Moment - feature index 0
            measurements[ANGULAR_SECOND_MOMENT_0 + parameter_suffix].append(haralick_features[0, 0])
            measurements[ANGULAR_SECOND_MOMENT_45 + parameter_suffix].append(haralick_features[1, 0])
            measurements[ANGULAR_SECOND_MOMENT_90 + parameter_suffix].append(haralick_features[2, 0])
            measurements[ANGULAR_SECOND_MOMENT_135 + parameter_suffix].append(haralick_features[3, 0])

            # Contrast - feature index 1
            measurements[CONTRAST_0 + parameter_suffix].append(haralick_features[0, 1])
            measurements[CONTRAST_45 + parameter_suffix].append(haralick_features[1, 1])
            measurements[CONTRAST_90 + parameter_suffix].append(haralick_features[2, 1])
            measurements[CONTRAST_135 + parameter_suffix].append(haralick_features[3, 1])

            # Correlation - feature index 2
            measurements[CORRELATION_0 + parameter_suffix].append(haralick_features[0, 2])
            measurements[CORRELATION_45 + parameter_suffix].append(haralick_features[1, 2])
            measurements[CORRELATION_90 + parameter_suffix].append(haralick_features[2, 2])
            measurements[CORRELATION_135 + parameter_suffix].append(haralick_features[3, 2])

            # Variance - feature index 3
            measurements[VARIANCE_0 + parameter_suffix].append(haralick_features[0, 3])
            measurements[VARIANCE_45 + parameter_suffix].append(haralick_features[1, 3])
            measurements[VARIANCE_90 + parameter_suffix].append(haralick_features[2, 3])
            measurements[VARIANCE_135 + parameter_suffix].append(haralick_features[3, 3])

            # Inverse Difference Moment - feature index 4
            measurements[INVERSE_DIFFERENCE_MOMENT_0 + parameter_suffix].append(haralick_features[0, 4])
            measurements[INVERSE_DIFFERENCE_MOMENT_45 + parameter_suffix].append(haralick_features[1, 4])
            measurements[INVERSE_DIFFERENCE_MOMENT_90 + parameter_suffix].append(haralick_features[2, 4])
            measurements[INVERSE_DIFFERENCE_MOMENT_135 + parameter_suffix].append(haralick_features[3, 4])

            # Sum Average - feature index 5
            measurements[SUM_AVERAGE_0 + parameter_suffix].append(haralick_features[0, 5])
            measurements[SUM_AVERAGE_45 + parameter_suffix].append(haralick_features[1, 5])
            measurements[SUM_AVERAGE_90 + parameter_suffix].append(haralick_features[2, 5])
            measurements[SUM_AVERAGE_135 + parameter_suffix].append(haralick_features[3, 5])

            # Sum Variance - feature index 6
            measurements[SUM_VARIANCE_0 + parameter_suffix].append(haralick_features[0, 6])
            measurements[SUM_VARIANCE_45 + parameter_suffix].append(haralick_features[1, 6])
            measurements[SUM_VARIANCE_90 + parameter_suffix].append(haralick_features[2, 6])
            measurements[SUM_VARIANCE_135 + parameter_suffix].append(haralick_features[3, 6])

            # Sum Entropy - feature index 7
            measurements[SUM_ENTROPY_0 + parameter_suffix].append(haralick_features[0, 7])
            measurements[SUM_ENTROPY_45 + parameter_suffix].append(haralick_features[1, 7])
            measurements[SUM_ENTROPY_90 + parameter_suffix].append(haralick_features[2, 7])
            measurements[SUM_ENTROPY_135 + parameter_suffix].append(haralick_features[3, 7])

            # Entropy - feature index 8
            measurements[ENTROPY_0 + parameter_suffix].append(haralick_features[0, 8])
            measurements[ENTROPY_45 + parameter_suffix].append(haralick_features[1, 8])
            measurements[ENTROPY_90 + parameter_suffix].append(haralick_features[2, 8])
            measurements[ENTROPY_135 + parameter_suffix].append(haralick_features[3, 8])

            # Difference Variance - feature index 9
            measurements[DIFFERENCE_VARIANCE_0 + parameter_suffix].append(haralick_features[0, 9])
            measurements[DIFFERENCE_VARIANCE_45 + parameter_suffix].append(haralick_features[1, 9])
            measurements[DIFFERENCE_VARIANCE_90 + parameter_suffix].append(haralick_features[2, 9])
            measurements[DIFFERENCE_VARIANCE_135 + parameter_suffix].append(haralick_features[3, 9])

            # Difference Entropy - feature index 10
            measurements[DIFFERENCE_ENTROPY_0 + parameter_suffix].append(haralick_features[0, 10])
            measurements[DIFFERENCE_ENTROPY_45 + parameter_suffix].append(haralick_features[1, 10])
            measurements[DIFFERENCE_ENTROPY_90 + parameter_suffix].append(haralick_features[2, 10])
            measurements[DIFFERENCE_ENTROPY_135 + parameter_suffix].append(haralick_features[3, 10])

            # Information Measure of Correlation 1 - feature index 11
            measurements[IMC1_0 + parameter_suffix].append(haralick_features[0, 11])
            measurements[IMC1_45 + parameter_suffix].append(haralick_features[1, 11])
            measurements[IMC1_90 + parameter_suffix].append(haralick_features[2, 11])
            measurements[IMC1_135 + parameter_suffix].append(haralick_features[3, 11])

            # Information Measure of Correlation 2 - feature index 12
            measurements[IMC2_0 + parameter_suffix].append(haralick_features[0, 12])
            measurements[IMC2_45 + parameter_suffix].append(haralick_features[1, 12])
            measurements[IMC2_90 + parameter_suffix].append(haralick_features[2, 12])
            measurements[IMC2_135 + parameter_suffix].append(haralick_features[3, 12])

        return measurements

    @staticmethod
    def calculate_haralick(object_matrix, scale):
        """Calculates Haralick texture features for a given object matrix.

        This method computes Haralick texture features using the mahotas library. It handles
        empty matrices by returning NaN values and properly processes non-empty matrices
        to extract texture features.

        Args:
            object_matrix: The input matrix representing an object for which to calculate
                texture features.
            scale: The distance parameter used in calculating Haralick features.

        Returns:
            np.ndarray: A flattened array containing all Haralick features for the object.
                If the input matrix is empty (sum is 0), returns an array of NaN values.

        Notes:
            The returned array contains features for all four directions (0°, 45°, 90°, 135°)
            and all 13 Haralick features, flattened into a 1D array.
        """
        if object_matrix.sum() == 0:
            haralick_features = np.full((4, 13), np.nan, dtype=np.float64)
        else:
            # Pad object with zero if its dimensions are smaller than the scale

            haralick_features = mh.features.haralick(img_as_ubyte(object_matrix),
                                                     distance=scale,
                                                     ignore_zeros=True,
                                                     return_mean=False,
                                                     )
        return haralick_features.T.ravel()
