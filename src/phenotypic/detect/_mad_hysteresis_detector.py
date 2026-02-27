from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class MadHysteresisDetector(ThresholdDetector):
    """MAD-based hysteresis detector for noise-aware colony segmentation.

    MadHysteresisDetector estimates the background noise floor using the Median
    Absolute Deviation (MAD), then applies hysteresis thresholding with
    MAD-derived thresholds. Unlike histogram-based detectors (Otsu, Li, etc.)
    that partition intensity space, this estimates the noise structure of a
    response map — ideal for filter outputs (CED, Hessian, LoG) where the noise
    floor is known to be approximately Gaussian.

    The noise standard deviation is estimated as ``sigma_noise = 1.4826 * MAD``,
    where ``MAD = median(|data - median(data)|)``. The scale factor 1.4826
    makes the estimator consistent with the standard deviation for normally
    distributed data. Thresholds are then set as multiples of sigma_noise.

    Args:
        k_high: High threshold multiplier. The high threshold is set to
            ``k_high * sigma_noise``. Pixels above this seed connected regions.
            Higher values are more conservative. Default 5.0.

        k_low: Low threshold multiplier. The low threshold is set to
            ``k_low * sigma_noise``. Pixels above this are included if connected
            to high-threshold seeds. Must be less than k_high. Default 2.5.

        min_size: Minimum object size in pixels. Connected components smaller
            than this are removed. Default 20.

        connectivity: Connectivity for connected-component analysis. 1 for
            4-connected, 2 for 8-connected. Default 2.

        ignore_zeros: If True (default), exclude zero-intensity pixels from MAD
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges
            via clear_border(). Eliminates partial colonies at boundaries.

    Attributes:
        k_high: High threshold multiplier.
        k_low: Low threshold multiplier.
        min_size: Minimum connected component size in pixels.
        connectivity: Connectivity for labeling (1 or 2).
        ignore_zeros: Whether to exclude zero pixels from MAD computation.
        ignore_borders: Whether to remove edge-touching colonies.

    Returns:
        Image: Input image with objmask set to binary mask. True pixels represent
        detected colonies, False = background.

    Raises:
        ValueError: If k_low >= k_high.

    **Use cases**

    - **Filter response maps:** CED, Hessian, LoG, or Frangi outputs where the
      noise structure is approximately Gaussian and histogram-based methods
      produce unstable thresholds.
    - **Low-contrast colonies:** When colony signal is faint relative to
      background texture. MAD is robust to outliers and provides stable noise
      estimation.
    - **Standardized pipelines:** Multiplier-based thresholds generalize across
      plates with similar imaging conditions but varying colony density.

    **Limitations**

    - Assumes approximately Gaussian noise. Non-Gaussian noise (e.g., salt-and-
      pepper) may cause over- or under-estimation of sigma_noise.
    - Two multiplier parameters to tune. Start with defaults (k_high=5, k_low=2.5)
      and adjust based on false positive/negative rates.
    - Not suitable for raw intensity images with bimodal histograms — use Otsu or
      HysteresisDetector instead.

    **Parameter effects on colony detection**

    - **k_high (float):** Controls seed strictness. Higher → fewer seeds, fewer
      false positives, may miss faint colonies. Lower → more seeds, more noise.
    - **k_low (float):** Controls expansion sensitivity. Lower → more aggressive
      expansion from seeds, captures faint colony edges. Higher → tighter masks.
    - **min_size (int):** Removes small noise components. Increase for noisy
      images.
    - **connectivity (int):** 2 (8-connected) captures diagonal connections.
      Use 1 (4-connected) for stricter separation.

    Examples:
        Basic detection on a filter response map::

            from phenotypic import Image
            from phenotypic.detect import MadHysteresisDetector

            plate = Image.imread("agar_plate.jpg")
            detector = MadHysteresisDetector(k_high=5.0, k_low=2.5)
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels")

        Pipeline with CED preprocessing::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import MadHysteresisDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                MadHysteresisDetector(k_high=4.0, k_low=2.0, min_size=50)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(
        self,
        k_high: float = 5.0,
        k_low: float = 2.5,
        min_size: int = 20,
        connectivity: int = 2,
        ignore_zeros: bool = True,
        ignore_borders: bool = True,
    ):
        self.k_high = k_high
        self.k_low = k_low
        self.min_size = min_size
        self.connectivity = connectivity
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Apply MAD-based hysteresis thresholding to detect colonies.

        Estimates background noise via the Median Absolute Deviation, computes
        high and low thresholds as multiples of the estimated noise standard
        deviation, then applies hysteresis thresholding followed by small-object
        removal and optional border clearing.

        Args:
            image: The input image object. Must have ``detect_mat`` attribute
                (detection matrix, typically a filter response map).

        Returns:
            Image: The input image with ``objmask`` attribute set to the binary
            mask (True = detected colony pixel, False = background).

        Raises:
            ValueError: If k_low >= k_high.
        """
        if self.k_low >= self.k_high:
            raise ValueError(
                f"k_low ({self.k_low}) must be less than k_high ({self.k_high})"
            )

        response = np.clip(image.detect_mat[:].astype(np.float64), 0, None)

        # Select data for MAD computation
        if self.ignore_zeros:
            data = response[response != 0]
        else:
            data = response.ravel()

        # Handle empty data
        if data.size == 0:
            image.objmask = np.zeros(response.shape, dtype=bool)
            return image

        # Compute MAD-based noise estimate
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        sigma_noise = 1.4826 * mad

        # Handle uniform image (zero noise)
        if sigma_noise == 0.0:
            image.objmask = np.zeros(response.shape, dtype=bool)
            return image

        # Compute thresholds
        t_high = self.k_high * sigma_noise
        t_low = self.k_low * sigma_noise

        # Hysteresis thresholding
        mask = apply_hysteresis_threshold(response, t_low, t_high)
        mask = mask.astype(bool)

        # Remove small objects
        mask = remove_small_objects(mask, self.min_size, self.connectivity)

        # Optionally clear borders
        if self.ignore_borders:
            mask = clear_border(mask)

        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
MadHysteresisDetector.apply.__doc__ = MadHysteresisDetector._operate.__doc__
