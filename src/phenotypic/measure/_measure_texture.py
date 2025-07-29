from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING: from phenotypic import Image

import warnings
from enum import Enum
from mahotas.features.texture import haralick_features

from phenotypic.abstract import MeasurementInfo


class TEXTURE(MeasurementInfo):
    @property
    def CATEGORY(self) -> str:
        return 'Texture'

    ANGULAR_SECOND_MOMENT = (
        'AngularSecondMoment',
        'Sum of squared co-occurrence probabilities (uniformity); high for smooth regions where neighbors match, low for varied textures.'
    )
    CONTRAST = (
        'Contrast',
        'Weighted sum of squared intensity differences; high for edgy patterns (large jumps), low for smooth gradients.'
    )
    CORRELATION = (
        'Correlation',
        'Linear dependency of gray levels; high if one pixel predicts its neighbor well, low if random.'
    )
    VARIANCE = (
        'HaralickVariance',
        'Spread of co-occurrence values; high for busy textures, low for uniform patches.'
    )
    INVERSE_DIFFERENCE_MOMENT = (
        'InverseDifferenceMoment',
        'Inverse of contrast weighted for similarity; high for smooth regions, low at edges.'
    )
    SUM_AVERAGE = (
        'SumAverage',
        'Mean of pixel-pair intensity sums; tracks overall brightness.'
    )
    SUM_VARIANCE = (
        'SumVariance',
        'Variance of pixel-pair sums; high if sums vary widely, low if uniform.'
    )
    SUM_ENTROPY = (
        'SumEntropy',
        'Entropy of pixel-pair sum distribution; low if predictable, high if chaotic.'
    )
    ENTROPY = (
        'Entropy',
        'Entropy of co-occurrence matrix; low for uniform patches, high for random variation.'
    )
    DIFFERENCE_VARIANCE = (
        'DifferenceVariance',
        'Variance of pixel-pair differences; high if difference sizes vary, low if uniform.'
    )
    DIFFERENCE_ENTROPY = (
        'DifferenceEntropy',
        'Entropy of pixel-pair difference distribution; measures unpredictability of differences.'
    )
    IMC1 = (
        'InformationCorrelation1',
        'Mutual-information measure between pixel pairs; how much one pixel reduces uncertainty of its neighbor.'
    )
    IMC2 = (
        'InformationCorrelation2',
        'Normalized mutual-information; strength of dependency relative to maximum possible.'
    )

    @classmethod
    def get_headers(cls, scale: int, matrix_name) -> list[str]:
        """Return full texture labels with angles in order 0, 45, 90, 135 for each feature."""
        angles = ['0', '45', '90', '135']
        labels: list[str] = []
        for member in cls.iter_labels():
            base = f"{str(member)}"
            for angle in angles:
                labels.append(f"{base}{matrix_name}-deg({angle})-scale({scale}))")
        return labels


import mahotas as mh
import numpy as np
import pandas as pd
from skimage.util import img_as_ubyte

from phenotypic.abstract import MeasureFeatures
from phenotypic.util.constants_ import OBJECT


class MeasureTexture(MeasureFeatures):
    """
    Represents a measurement of texture features extracted from image objects.

    This class is designed to calculate texture measurements derived from Haralick features,
    tailored for segmented objects in an image. These features include statistical properties
    that describe textural qualities, such as uniformity or variability, across different
    directional orientations. The class leverages statistical methods and image processing
    to extract meaningful characteristics applicable in image analysis tasks.

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

    def _operate(self, image: Image) -> pd.DataFrame:
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
        return self._compute_haralick(image=image,
                                      foreground_array=image.matrix.get_foreground(),
                                      foreground_name='Intensity',
                                      scale=self.scale,
                                      )

    @staticmethod
    def _compute_haralick(image: Image, foreground_array: np.ndarray, foreground_name: str, scale: int = 5) -> pd.DataFrame:
        """
        Computes texture feature measurements using Haralick features for objects in a given image. The method
        calculates various statistical texture features such as Angular Second Moment, Contrast, Correlation,
        Variance, Inverse Difference Moment, among others, for different directional orientations. These
        features are computed for each segmented object within the foreground array using the specified
        scale parameter.

        Args:
            image (Image): The image containing objects and their associated properties, including
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
        props = image.objects.props
        objmap = image.objmap[:]
        measurement_names = TEXTURE.get_headers(scale, foreground_name)
        measurements = np.empty(shape=(image.num_objects, len(measurement_names),), dtype=np.float64)
        for idx, label in enumerate(image.objects.labels):
            slices = props[idx].slice
            obj_extracted = foreground_array[slices].copy()

            # In case there's more than one object in the crop
            obj_extracted[objmap[slices] != label] = 0

            try:
                if obj_extracted.sum() == 0:
                    haralick_features = np.full((4, 13), np.nan, dtype=np.float64)
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

            measurements[idx, :] = haralick_features.T.ravel()

        return pd.DataFrame(measurements, index=image.objects.labels2series(), columns=measurement_names)

MeasureTexture.__doc__ = TEXTURE.append_rst_to_doc(MeasureTexture)