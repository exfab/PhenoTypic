from __future__ import annotations

import functools
import warnings
from typing import ClassVar, List, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import field_validator
from skimage import exposure

from phenotypic.abc_ import MeasureFeatures
from phenotypic.schema import OBJECT
from phenotypic.schema import TEXTURE

if TYPE_CHECKING:
    from phenotypic._core._image import Image

# Suppress mahotas' module-load SyntaxWarning before importing it.
warnings.filterwarnings('ignore', category=SyntaxWarning, module='mahotas')

import mahotas as mh  # noqa: E402 - must follow the filterwarnings call above


class MeasureTexture(MeasureFeatures):
    """Measure colony surface texture using Haralick features from gray-level co-occurrence matrices.

    Compute 13 second-order Haralick texture features per colony at one
    or more pixel-offset scales, across four directional angles (0, 45,
    90, 135 degrees), plus direction-averaged values. These features
    quantify surface roughness, regularity, and directional patterns
    that distinguish colony morphotypes.

    Args:
        scale: Pixel offset(s) for the co-occurrence matrix. A single
            integer or list of integers. Small values (1--2) capture fine
            texture; large values (5--10) capture coarse patterns.
            Default: ``5``.
        quant_lvl: Number of gray-level bins for quantization. Accepted
            values: ``8``, ``16``, ``32``, ``64``. Lower values are
            faster; higher values preserve texture nuance but are more
            noise-sensitive. Default: ``32``.
        enhance: Rescale each colony's intensity to [0, 1] before
            computing Haralick features. Improves contrast in
            low-variance regions but can bias cross-colony comparisons.
            Default: ``False``.
        warn: Emit warnings when Haralick computation fails for
            individual colonies (typically very small objects).
            Default: ``False``.

    Returns:
        pd.DataFrame: Object-level texture measurements with columns:

            - Label: unique object identifier.
            - 13 Haralick features x 4 angles = 52 directional columns
              per scale (e.g., ``Contrast-deg000-scale05``).
            - 13 direction-averaged columns per scale (e.g.,
              ``Contrast-avg-scale05``).

    References:
        [1] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural
        features for image classification," *IEEE Trans. Syst., Man,
        Cybern.*, vol. SMC-3, no. 6, pp. 610--621, Nov. 1973.

    Best For:
        - Distinguishing smooth wild-type colonies from rough, wrinkled,
          or sporulated mutants.
        - Assessing mycelial organization in filamentous fungi (radial
          vs cottony growth).
        - Multi-feature phenotypic clustering when combined with size,
          shape, and color measurements.

    Consider Also:
        - :class:`MeasureShape` for geometric morphology metrics
          (circularity, Feret diameters) that complement texture.
        - :class:`MeasureIntensity` for brightness statistics without
          spatial co-occurrence information.
        - :class:`MeasureColor` for pigmentation-based phenotyping.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of measuring and exporting colony data.
        :doc:`/explanation/measurement_metrics_biological_meaning` for
        interpreting texture metrics in a biological context.
    """

    _measurement_infoclass: ClassVar[type] = TEXTURE

    scale: List[int] = [5]
    quant_lvl: Literal[8, 16, 32, 64] = 32
    enhance: bool = False
    warn: bool = False

    @field_validator("scale", mode="before")
    @classmethod
    def _coerce_scale_to_list(cls, scale: int | List[int]) -> List[int]:
        """Normalize a bare integer ``scale`` to a single-element list.

        The legacy constructor stored ``scale`` as a list regardless of
        whether the caller passed an ``int`` or a ``list[int]`` (the
        ``_operate`` body indexes ``self.scale[0]`` / ``self.scale[1:]``).
        This validator reproduces that coercion so both call styles keep
        working while the declared field type stays an honest list.
        """
        if not hasattr(scale, "__getitem__"):  # coerce iterable input
            return [scale]
        return scale

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
