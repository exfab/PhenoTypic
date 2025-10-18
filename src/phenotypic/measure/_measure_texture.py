from __future__ import annotations

from typing import List, Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import warnings
import mahotas as mh
import numpy as np
import pandas as pd
from skimage import exposure
from skimage.util import img_as_ubyte

from phenotypic.abstract import MeasureFeatures
from phenotypic.tools.constants_ import OBJECT
from phenotypic.abstract import MeasurementInfo


class TEXTURE(MeasurementInfo):
    """Second-order texture features derived from the gray-level co-occurrence matrix (GLCM).

    All features assume normalized GLCMs computed at one or more pixel offsets and averaged
    across directions unless otherwise noted. Values depend on quantization, window size,
    and scale; interpret ranges comparatively within the same imaging setup.
    """

    @classmethod
    def category(cls) -> str:
        return "Texture"

    ANGULAR_SECOND_MOMENT = (
        "AngularSecondMoment",
        """Angular second moment (energy / uniformity). Measures the degree of local homogeneity
        (Σ p(i,j)²). High values → uniform texture (e.g., smooth, yeast-like colonies with consistent
        mycelial density). Low values → heterogeneous surfaces (e.g., sectored, wrinkled, or mixed
        sporulation zones). Reflects colony surface regularity rather than brightness."""
    )

    CONTRAST = (
        "Contrast",
        """Contrast (local intensity variation; Σ (i–j)² p(i,j)). High values indicate strong gray-level
        differences (e.g., sharply defined rings, radial sectors, raised or folded regions). Low values
        indicate gradual tonal changes or uniformly pigmented colonies. Quantifies visual roughness
        and zonation amplitude."""
    )

    CORRELATION = (
        "Correlation",
        """Linear gray-level correlation between neighboring pixels. Positive, high values suggest
        structured spatial dependence (e.g., oriented radial hyphae or concentric patterns); near-zero
        values indicate uncorrelated, disordered growth (e.g., diffuse cottony mycelium). Sensitive to
        illumination gradients and directional GLCM computation."""
    )

    VARIANCE = (
        "HaralickVariance",
        """GLCM variance (Σ (i–μ)² p(i,j)). Captures spread of co-occurring gray-level pairs, distinct
        from raw intensity variance. High values → complex, multi-zone textures with variable
        hyphal/spore densities. Low values → consistent gray-level relationships and simpler colony
        surfaces."""
    )

    INVERSE_DIFFERENCE_MOMENT = (
        "InverseDifferenceMoment",
        """Homogeneity (Σ p(i,j) / (1 + (i–j)²)). High values → smooth, locally uniform textures
        (e.g., glabrous colonies, uniform aerial mycelium). Low values → abrupt gray-level changes
        (e.g., granular sporulation, wrinkled surfaces). Typically inversely correlated with Contrast."""
    )

    SUM_AVERAGE = (
        "SumAverage",
        """Mean of gray-level sums (Σ k·p_{x+y}(k)). Reflects the average intensity combination of
        neighboring pixels. In fungal colonies, can loosely parallel mean colony brightness when
        illumination and exposure are controlled, but remains a second-order rather than first-order
        intensity metric."""
    )

    SUM_VARIANCE = (
        "SumVariance",
        """Variance of gray-level sum distribution. High values → heterogeneous brightness zones
        (e.g., alternating dense/sparse or pigmented/non-pigmented regions). Low values → uniform
        tone across the colony. Often correlated with Contrast; use comparatively within one setup."""
    )

    SUM_ENTROPY = (
        "SumEntropy",
        """Entropy of the gray-level sum distribution. High values → diverse brightness combinations
        and irregular zonation. Low values → repetitive or periodic brightness patterns (e.g., evenly
        spaced rings). Indicates spatial unpredictability of summed intensities."""
    )

    ENTROPY = (
        "Entropy",
        """Global GLCM entropy (–Σ p(i,j)·log p(i,j)). Measures total texture disorder and information
        content. High values → complex, irregular colony surfaces (powdery, fuzzy, or sectored growth).
        Low values → simple, smooth, predictable patterns (glabrous or uniform colonies). Sensitive to
        gray-level quantization and image dynamic range."""
    )

    DIFFERENCE_VARIANCE = (
        "DiffVariance",
        """Variance of gray-level difference distribution. High values → mixture of smooth and textured
        regions (e.g., smooth margins with wrinkled centers). Low values → consistent edge content.
        Highlights heterogeneity in edge magnitude across the colony."""
    )

    DIFFERENCE_ENTROPY = (
        "DiffEntropy",
        """Entropy of gray-level difference distribution. High values → irregular, unpredictable
        intensity transitions (e.g., random sporulation or uneven mycelial networks). Low values →
        regular periodic transitions (e.g., concentric zonation). Reflects randomness of local contrast
        rather than its magnitude."""
    )

    IMC1 = (
        "InfoCorrelation1",
        """Information measure of correlation 1. Compares joint vs marginal entropies to quantify
        mutual dependence between gray levels. Positive values → structured, predictable textures
        (e.g., organized radial growth); near-zero → independence between adjacent regions.
        Direction of sign varies with implementation."""
    )

    IMC2 = (
        "InfoCorrelation2",
        """Information measure of correlation 2 (√[1 – exp(–2 (H_xy2–H_xy))]). Always ≥ 0.
        Values approaching 1 → strong spatial dependence and organized architecture (e.g., symmetric
        rings, radial structure). Values near 0 → random, independent patterns. Captures nonlinear
        organization missed by linear correlation."""
    )

    @classmethod
    def get_headers(cls, scale: int, matrix_name) -> list[str]:
        """Return full texture labels with angles in order 0, 45, 90, 135 for each feature."""
        angles = [0, 45, 90, 135]
        labels: list[str] = []
        for member in cls.get_labels():
            for angle in angles:
                labels.append(f"{cls.category()}{matrix_name}_{member}-deg{angle:03d}-scale{scale:02d}")
        return labels


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

    def __init__(self, scale: int | List[int] = 5,
                 quant_lvl: Literal[8, 16, 32, 64] = 32,
                 enhance: bool = False,
                 warn: bool = False):
        """
        Initializes an object with specific configurations for scale, quantization level,
        enhancement, and warning behaviors. This constructor ensures that the 'scale'
        parameter is always stored as a list.

        Args:
            scale (int | List[int]): A single integer or a list of integers representing
                the scale configuration. If a single integer is provided, it will be
                converted into a list containing that integer.
            quant_lvl (Literal[8, 16, 32, 64]): The quantization level. A higher level adds
                more computational complexity but increases detail captured.
                Acceptable values are either 8, 16, 32, or 64.
            enhance (bool): A flag indicating whether to enhance the image before measuring texture. This can
                increase the amount of detail captured but can bias the measurements in cases where the relative
                variance between pixel intensities of an object are small
            warn (bool): A flag indicating whether warnings should be issued.
        """
        if not hasattr(scale, "__getitem__"):
            scale = [scale]

        self.scale = scale
        self.quant_lvl = quant_lvl
        self.enhance = enhance
        self.warn = warn

    def _operate(self, image: Image) -> pd.DataFrame:
        """Performs texture measurements on the image objects.

        This method extracts texture features from the foreground objects in the image using
        Haralick texture features. It processes the image's foreground array and returns
        the measurements as a DataFrame.

        Args:
            image (Image): The image containing objects to measure.

        Returns:
            pd.DataFrame: A DataFrame containing texture measurements for each object in the image.
                The nrows are indexed by object labels, and columns represent different texture features.
        """
        meas = self._compute_haralick(image=image,
                                      foreground_array=image.matrix.foreground(),
                                      foreground_name='Gray',
                                      scale=self.scale[0],
                                      enhance=self.enhance,
                                      warn=self.warn
                                      )
        if len(self.scale) > 1:
            for scale in self.scale[1:]:
                meas.merge(self._compute_haralick(
                        image=image,
                        foreground_array=image.matrix.foreground(),
                        foreground_name='Gray',
                        scale=scale,
                        quant_lvl=self.quant_lvl,
                        enhance=self.enhance,
                        warn=self.warn
                ), on=OBJECT.LABEL, how='outer')
        return meas

    @staticmethod
    def _compute_haralick(image: Image, foreground_array: np.ndarray, foreground_name: str,
                          scale: int,
                          quant_lvl: int,
                          enhance: bool = False,
                          warn: bool = False,
                          ) -> pd.DataFrame:
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
        if foreground_array.min() < 0 or foreground_array.max() > 1: raise ValueError(
                "Foreground array must be normalized between 0 and 1")

        props = image.objects.props
        objmap = image.objmap[:]
        measurement_names = TEXTURE.get_headers(scale, foreground_name)
        measurements = np.empty(shape=(image.num_objects, len(measurement_names),), dtype=np.float64)
        for idx, label in enumerate(image.objects.labels):
            slices = props[idx].slice
            obj_fg = foreground_array[slices].copy()

            # In case there's more than one object in the crop
            obj_fg[objmap[slices] != label] = 0

            try:
                if obj_fg.sum() == 0:  # In case an empty array is given
                    texture_statistics = np.full((4, 13), np.nan, dtype=np.float64)
                else:
                    # Pad object with zero if its dimensions are smaller than the scale

                    if enhance:
                        # contrast stretch to normalized range
                        # this can improve texture detail, but can
                        # add bias when the variance of the original range is small
                        obj_fg = exposure.rescale_intensity(obj_fg, in_range='image', out_range=(0.0, 1.0))

                    texture_statistics = mh.features.haralick(
                            MeasureTexture._quantize_arr(arr=obj_fg, quant_lvl=quant_lvl),
                            distance=scale,
                            ignore_zeros=True,
                            return_mean=False,
                    )
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # 4 for each direction, 13 for each texture feature
                if warn: warnings.warn(f'Error in computing Haralick features for object {label}: {e}')
                texture_statistics = np.full((4, 13), np.nan, dtype=np.float64)

            measurements[idx, :] = texture_statistics.T.ravel()

        measurements = pd.DataFrame(measurements, columns=measurement_names)
        measurements.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return measurements

    @staticmethod
    def _quantize_arr(arr: np.ndarray, quant_lvl) -> np.ndarray:
        """quantizes a normalized array to specific levels"""
        if arr.min() < 0 or arr.max() > 1: raise ValueError('Array is not normalized')

        quant_arr = np.floor(arr*quant_lvl)

        # handle edge case where a value was 1.0
        quant_arr = np.clip(quant_arr, a_min=0, a_max=quant_lvl - 1)
        return quant_arr.astype(np.uint8)


MeasureTexture.__doc__ = TEXTURE.append_rst_to_doc(MeasureTexture)
