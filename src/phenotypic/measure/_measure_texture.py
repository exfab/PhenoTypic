from __future__ import annotations

import functools
from typing import List, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning, module='mahotas')

import mahotas as mh
import numpy as np
import pandas as pd
from skimage import exposure

from phenotypic.abc_ import MeasureFeatures
from phenotypic.tools_.constants_ import OBJECT
from ..tools_.measurement_info_ import TEXTURE


class MeasureTexture(MeasureFeatures):
    """Measure colony surface texture using Haralick features from gray-level co-occurrence matrices.

    This class extracts second-order texture features (Haralick features) from colony grayscale images,
    quantifying surface roughness, regularity, and directional patterns. Features are computed at specified
    pixel offsets (scales) and across four directional angles (0°, 45°, 90°, 135°), then averaged.

    **Intuition:** Colony texture reflects mycelial structure, growth patterns, and physiological state.
    Smooth, glabrous colonies have low texture contrast and entropy. Wrinkled, powdery, or sporulated
    colonies exhibit high contrast and energy. Radial growth patterns show angular correlation; random
    growth shows low correlation. Texture metrics capture morphological complexity beyond area and
    perimeter, enabling fine-grained phenotypic discrimination.

    **Use cases (agar plates):**
    - Distinguish wild-type smooth colonies from rough/wrinkled mutants (e.g., Bacillus subtilis biofilm
      morphologies, Pseudomonas aeruginosa rough variants).
    - Detect sporulation and powdery growth via high contrast and entropy.
    - Assess mycelial organization in fungi: organized radial hyphae (high correlation) vs diffuse cottony
      growth (low correlation).
    - Identify growth stress or nutrient depletion via texture changes within the same strain over time.
    - Enable multi-feature clustering combining size, shape, color, and texture for robust phenotyping.

    **Caveats:**
    - Haralick features depend on image quantization level (quant_lvl); lower levels (8) reduce texture
      nuance but are faster; higher levels (64) capture detail but are sensitive to noise.
    - Scale parameter affects the neighborhood size; small scales (1-2 px) capture fine texture (mycelial
      threads), large scales (5-10 px) capture coarse patterns (overall wrinkles). No single scale is
      universal; use multiple scales and average or compare within-plate.
    - Texture metrics are sensitive to uneven illumination and shadow; preprocess with illumination
      correction or histogram equalization if images show strong gradients.
    - Enhancement (rescale_intensity) improves texture detail but can inflate contrast in low-variance
      regions (e.g., uniform smooth colonies); use judiciously and validate with manual inspection.
    - Computation is slow for large colonies and high quantization levels; optimize scale and quant_lvl
      for your specific assay.

    Attributes:
        scale (list[int]): Distance parameter(s) for Haralick co-occurrence matrix, typically 1–10 pixels.
            Larger values capture coarse texture; smaller values capture fine detail.
        quant_lvl (Literal[8, 16, 32, 64]): Gray-level quantization (number of bins). Lower values
            (8, 16) reduce dimensionality and computation time; higher values (32, 64) preserve texture
            nuance but increase noise sensitivity.
        enhance (bool): Whether to rescale intensity within each colony to [0,1] before Haralick
            computation. Improves contrast in low-variance regions but can bias comparisons. Defaults to False.
        warn (bool): Whether to issue warnings if Haralick computation fails for specific objects.
            Failures typically occur with very small colonies or empty regions. Defaults to False.

    Returns:
        pd.DataFrame: Object-level texture measurements with columns:
            - Label: Unique object identifier.
            - Texture measurements by scale and direction: AngularSecondMoment-deg000-scale##,
              Contrast-deg045-scale##, ..., Correlation-avg-scale##, etc.
            - 13 Haralick features × 4 angles = 52 directional columns per scale.
            - Final 13 columns: averages across angles for each feature at the given scale.

    References:
        [1] https://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.haralick
        [2] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural Features for Image Classification,"
            IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, Nov. 1973,
            doi: 10.1109/TSMC.1973.4309314.

    Examples:
        Measure texture to distinguish morphotypes:

        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.measure import MeasureTexture
        >>> # Load plate with smooth and wrinkled colonies
        >>> image = Image.imread("morphotype_plate.jpg")  # doctest: +SKIP
        >>> detector = OtsuDetector()
        >>> image = detector.operate(image)  # doctest: +SKIP
        >>> # Measure texture at a single scale with default quantization
        >>> measurer = MeasureTexture(scale=3, quant_lvl=32, enhance=False)
        >>> texture = measurer.operate(image)  # doctest: +SKIP
        >>> # High contrast and energy indicate wrinkled/rough morphology
        >>> wrinkled = texture[
        ...     texture['TextureGray_Contrast-avg-scale03'] > texture['TextureGray_Contrast-avg-scale03'].quantile(0.75)
        ... ]  # doctest: +SKIP
        >>> print(f"Wrinkled colonies: {len(wrinkled)}")  # doctest: +SKIP

        Multi-scale texture analysis for fine/coarse features:

        >>> # Use multiple scales to capture fine and coarse texture
        >>> measurer = MeasureTexture(scale=[1, 3, 5], quant_lvl=32, enhance=True, warn=False)
        >>> texture = measurer.operate(image)  # doctest: +SKIP
        >>> # Compare entropy across scales to assess texture organization
        >>> # Fine texture (scale 1): high entropy -> many small features
        >>> # Coarse texture (scale 5): low entropy -> organized large structures
        >>> for scale in [1, 3, 5]:  # doctest: +SKIP
        ...     col = f'TextureGray_Entropy-avg-scale0{scale}'
        ...     if col in texture.columns:
        ...         avg_entropy = texture[col].mean()
        ...         print(f"Scale {scale}px: avg entropy = {avg_entropy:.2f}")
    """

    _measurement_info_class = TEXTURE

    def __init__(
            self,
            scale: int | List[int] = 5,
            quant_lvl: Literal[8, 16, 32, 64] = 32,
            enhance: bool = False,
            warn: bool = False,
    ):
        """
        Initializes an object with specific configurations for scale, quantization level,
        enhance, and warning behaviors. This constructor ensures that the 'scale'
        parameter is always stored as a list.

        Args:
            scale (int | List[int]): A single integer or a list of integers representing
                the scale configuration. If a single integer is provided, it will be
                converted into a list containing that integer.
            quant_lvl (Literal[8, 16, 32, 64]): The quantization level. A higher level adds
                more computational complexity but captures more discrete texture changes. A higher value is
                not always more meaningful. Think of this like sensitivity to texture. Acceptable values are either 8, 16, 32, or 64.
            enhance (bool): A flag indicating whether to enhance the image before measuring texture. This can
                increase the amount of detail captured but can bias the measurements in cases where the relative
                variance between pixel intensities of an object is small.
            warn (bool): A flag indicating whether warnings should be issued.
        """
        if not hasattr(scale, "__getitem__"):  # coerce iterable input
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
        compute_haralick = functools.partial(
                self._compute_haralick,
                image=image,
                foreground_array=image.gray.foreground(),
                foreground_name="Gray",
                quant_lvl=self.quant_lvl,
                enhance=self.enhance,
                warn=self.warn,
        )

        meas = compute_haralick(scale=self.scale[0])
        if len(self.scale) > 1:
            for scale in self.scale[1:]:
                meas.merge(compute_haralick(scale=scale), on=OBJECT.LABEL, how="outer")
        return meas

    @staticmethod
    def _compute_haralick(
            image: Image,
            foreground_array: np.ndarray,
            foreground_name: str,
            scale: int,
            quant_lvl: int,
            enhance: bool,
            warn: bool,
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
        if foreground_array.min() < 0 or foreground_array.max() > 1:
            raise ValueError("Foreground array must be normalized between 0 and 1")

        props = image.objects.props
        objmap = image.objmap[:]
        measurement_names = TEXTURE.get_headers(scale, foreground_name)
        deg_measurement_names = measurement_names[
            :-13
        ]  # there are 13 haralick features so we separate the avgs out
        avg_measurement_names = measurement_names[-13:]
        deg_meas = np.empty(
                shape=(
                    image.num_objects,
                    len(deg_measurement_names),
                ),
                dtype=np.float64,
        )
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
                        obj_fg = exposure.rescale_intensity(
                                obj_fg, in_range="image", out_range=(0.0, 1.0)
                        )

                    texture_statistics = mh.features.haralick(
                            MeasureTexture._quantize_arr(arr=obj_fg,
                                                         quant_lvl=quant_lvl),
                            distance=scale,
                            ignore_zeros=True,
                            return_mean=False,
                    )
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # 4 for each direction, 13 for each texture feature
                if warn:
                    warnings.warn(
                            f"Error in computing Haralick features for object {label}: {e}"
                    )
                texture_statistics = np.full((4, 13), np.nan, dtype=np.float64)

            deg_meas[idx, :] = texture_statistics.T.ravel()

        avg_meas = np.empty(
                shape=(
                    image.num_objects,
                    len(avg_measurement_names),
                ),
                dtype=np.float64,
        )

        # step through each feature and avg across degrees
        for avg_col_idx, deg_start_idx in enumerate(range(0, deg_meas.shape[1], 4)):
            avg_meas[:, avg_col_idx] = np.average(
                    deg_meas[:, deg_start_idx: deg_start_idx + 4], axis=1
            )

        meas = pd.DataFrame(np.hstack([deg_meas, avg_meas]), columns=measurement_names)

        meas.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
        return meas

    @staticmethod
    def _quantize_arr(arr: np.ndarray, quant_lvl) -> np.ndarray:
        """quantizes a normalized array to specific levels"""
        if arr.min() < 0 or arr.max() > 1:
            raise ValueError("Array is not normalized")

        quant_arr = np.floor(arr * quant_lvl)

        # handle edge case where a value was 1.0
        quant_arr = np.clip(quant_arr, a_min=0, a_max=quant_lvl - 1)
        return quant_arr.astype(np.uint8)


MeasureTexture.__doc__ = TEXTURE.append_rst_to_doc(MeasureTexture)
