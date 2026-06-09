from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import Field

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector
from ..tools_.typing_ import TuneSpec


class MadHysteresisDetector(ThresholdDetector):
    """Detect colonies by MAD-based noise estimation and hysteresis thresholding.

    Estimate the background noise floor using the Median Absolute Deviation
    (MAD), then apply hysteresis thresholding with thresholds set as
    multiples of the estimated noise standard deviation. Designed for
    filter response maps (CED, Hessian, LoG, Frangi) where the noise
    structure is approximately Gaussian and histogram-based methods produce
    unstable thresholds. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * Filter response maps (CED, Hessian, LoG, Frangi) where the
          noise floor is approximately Gaussian.
        * Low-contrast colonies where signal is faint relative to
          background texture and MAD provides a stable noise estimate.
        * Standardised pipelines where multiplier-based thresholds
          generalise across plates with varying colony density.

    Consider Also:
        * :class:`HysteresisDetector` when thresholds should be derived
          from the intensity histogram rather than noise statistics.
        * :class:`OtsuDetector` when the image is a raw intensity plate
          with a bimodal histogram.
        * :class:`ChanVeseDetector` when colonies have diffuse edges and
          region-based segmentation is more appropriate.

    Args:
        k_high: High-threshold multiplier setting the seed threshold at
            ``k_high * sigma_noise`` standard deviations above the noise
            floor. There is no universal best value -- it is tuned to the
            scene. Raising *k_high* accepts only the most prominent
            colonies as seeds (more conservative); lowering it seeds dimmer
            structures. Typical range: 3.0--8.0. Default: 5.0. On noisy or
            strongly textured plates raise toward 6--8 so agar granularity
            is not seeded; on faint enhancer outputs (e.g. Frangi on weak
            hyphae) lower toward 3--4 so genuine signal seeds at all.

        k_low: Low-threshold multiplier setting the growth threshold at
            ``k_low * sigma_noise``; pixels above this join a seed if
            connected to it. Must be strictly less than *k_high*. The ratio
            *k_high* / *k_low* trades selectivity against coverage: a wider
            gap grows seeds further into faint margins. Typical range:
            1.5--4.0. Default: 2.5. Lower toward 1.5--2.0 to recover the
            faint edges of filamentous or low-contrast colonies; raise it
            on clean high-contrast maps.

        min_size: Minimum colony area in pixels; connected components
            smaller than this are removed as noise, debris, or condensation.
            Raising it silences small artefacts at the cost of missing
            genuine micro-colonies; lowering it recovers faint colonies but
            raises false positives. Scale with resolution -- smaller for
            dense 384/1536 formats, larger for high-dpi scans. Default: 20.

        connectivity: Pixel connectivity for labelling connected
            components. ``1`` (4-connected) keeps diagonally touching
            objects separate; ``2`` (8-connected) bridges diagonal gaps in
            thin or branching structures into fewer, larger regions.
            Default: 2.

        ignore_zeros: Exclude zero-intensity pixels from MAD computation.
            Enable for plates with black borders or masked regions.
            Default: False.

        ignore_borders: Remove colonies touching image edges via
            ``clear_border()``. Recommended for grid-based colony counting
            to eliminate partial colonies at plate boundaries. Default:
            True.

    Returns:
        Image: Input image with ``objmask`` set to binary mask and
        ``objmap`` set to labeled connected components.

    Raises:
        ValueError: If ``k_low`` >= ``k_high``.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    # Search windows are kept non-overlapping (k_low high < k_high low) so the
    # tuner — which samples each field independently — can never produce a
    # k_low >= k_high pair, which ``_operate`` rejects. The windows still sit
    # inside each field's documented typical range (k_low 1.5--4.0, k_high
    # 3.0--8.0); the fact-check PR may revisit the exact split.
    k_high: Annotated[float, TuneSpec(4.0, 8.0)] = 5.0
    k_low: Annotated[float, TuneSpec(1.5, 3.5)] = 2.5
    min_size: Annotated[int, TuneSpec(10, 500, log=True)] = 20
    connectivity: Annotated[int, TuneSpec(categories=[1, 2])] = Field(2, ge=1, le=2)
    ignore_zeros: bool = False
    ignore_borders: bool = True

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
        mask = remove_small_objects(mask, min_size=self.min_size,
                                    connectivity=self.connectivity)

        # Optionally clear borders
        if self.ignore_borders:
            mask = clear_border(mask)

        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
MadHysteresisDetector.apply.__doc__ = MadHysteresisDetector._operate.__doc__
