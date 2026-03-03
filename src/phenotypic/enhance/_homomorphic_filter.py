from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ..abc_ import ImageEnhancer


class HomomorphicFilter(ImageEnhancer):
    """Homomorphic filtering for illumination correction on agar plate images.

    Separates illumination (low-frequency) and reflectance (high-frequency)
    components in the log domain, applies differential gains to suppress
    illumination gradients while boosting colony detail, then returns to the
    linear domain. This is especially useful when plates suffer from vignetting,
    uneven scanner lighting, or shadow gradients that confound global
    thresholding.

    Args:
        sigma (float): Gaussian sigma for the illumination/reflectance
            frequency cutoff. Controls the spatial scale of the illumination
            field estimate. Should be large enough that the low-pass captures
            only the illumination gradient, not individual colonies. Default
            is 200.0 pixels, suitable for high-resolution scans; scale
            proportionally with image resolution. For a 512×768 reference
            image, sigma ~40–80 is typical; for 2000×3000, sigma ~150–300.
        gamma_low (float): Gain applied to low-frequency (illumination)
            component. Values < 1.0 suppress illumination variation; 1.0
            leaves it unchanged. Default is 0.5. Does NOT scale with image
            resolution.
        gamma_high (float): Gain applied to high-frequency (reflectance)
            component. Values > 1.0 enhance colony contrast and surface detail;
            1.0 leaves it unchanged. Default is 1.5. Does NOT scale with image
            resolution.
        eps (float): Small constant added before logarithm to avoid log(0).
            Default is 1e-6. Rarely needs adjustment; prevents numerical
            underflow with near-zero pixels.

    Returns:
        Image: Modified image with illumination-corrected detect_mat,
        clipped to [0.0, 1.0].

    Raises:
        ValueError: If sigma <= 0.

    Intuition:

    Homomorphic filtering models observed plate images as a product of
    illumination (slow spatial variation due to lighting geometry, vignetting,
    scanner artifacts) and reflectance (colony pigmentation and surface
    texture). By working in the logarithmic domain, this multiplicative model
    becomes additive (log(L*R) = log(L) + log(R)), enabling frequency-domain
    separation via Gaussian blur. Differential gains then compress illumination
    while amplifying colony-scale reflectance, improving downstream thresholding
    and colony detection.

    Use Cases:

    1. **DSLR plate scans with vignetting:** Cells at the plate periphery appear
       dimmer due to lens falloff. Homomorphic filtering flattens this radial
       gradient while preserving colony boundaries, making global thresholding
       more robust across the plate.

    2. **Uneven scanner illumination (stripe artifacts):** Flatbed scanners
       sometimes exhibit horizontal/vertical brightness bands. Homomorphic
       filtering suppresses these low-frequency artifacts while keeping colony
       signal intact.

    3. **Shadow variation from uneven agar preparation:** Thick or thin regions
       of agar show variable background brightness. Correcting the illumination
       field with gamma_low < 1.0 helps standardize detection across the plate.

    4. **High-density colony plates (touching or nearly touching):** After
       illumination correction, Otsu or triangle thresholding becomes more
       consistent, improving segmentation accuracy in densely seeded plates.

    5. **Faint or pigmented colonies:** Enhancing high-frequency reflectance
       (gamma_high > 1.0) makes subtle colony boundaries more visible,
       especially on translucent agar or with diffuse colony pigmentation.

    Parameter Effects:

    **sigma (Gaussian blur radius for illumination estimation):**
    - Typical range: 40–300 pixels (resolution-dependent)
    - Resolution scaling: Use formula `sigma_adjusted = sigma_default × (image_width / 512)`
    - Example scaled values (reference: 512×768 → sigma=80):
        * 512×768: sigma ≈ 80 (reference)
        * 640×960: sigma ≈ 100 (1.25×)
        * 1024×1536: sigma ≈ 160 (2×)
        * 2000×3000: sigma ≈ 310 (3.9×)
    - **Small sigma (< 2× median colony diameter):** Illumination estimate
      includes colony signal; contrast between colonies is reduced, making
      detection harder. Also, fine illumination gradients may not be fully
      captured.
    - **Medium sigma (3–5× colony diameter):** Balances capturing broad
      illumination gradients while isolating colony reflectance. Typical
      working range.
    - **Large sigma (> 10× colony diameter):** Illumination field becomes very
      smooth; risk of missing local shadow variation. Computation time increases
      quadratically with kernel size (ksize ≈ 6*sigma + 1).
    - **Colony morphology context:**
        * Round colonies (yeast-like, ~50–100 px diameter at 512×768): sigma
          ~150–250 at reference resolution works well.
        * Irregular colonies (bacteria, ~20–60 px): May require smaller sigma
          (~80–150) to avoid smearing fine edges.
        * Filamentous colonies (fungi, ~100–300 px): Larger sigma (~200–400)
          needed to separate illumination from colony extent.

    **gamma_low (illumination suppression gain):**
    - Typical range: 0.3–0.8 (does NOT scale with resolution)
    - **< 0.5:** Aggressive illumination suppression; risk of removing legitimate
      shading within colonies or creating halo artifacts at edges.
    - **0.5 (default):** Moderate compression; good baseline for most plates.
    - **0.7–0.9:** Mild suppression; retains more subtle lighting information.
    - **1.0:** No change to illumination; mainly used with gamma_high > 1 to
      only boost reflectance.
    - **Effect on colony morphology:** Independent of colony shape; affects
      overall plate brightness uniformity, not colony-specific contrast.

    **gamma_high (reflectance enhancement gain):**
    - Typical range: 1.0–2.5 (does NOT scale with resolution)
    - **1.0:** No enhancement; reflectance unchanged.
    - **1.3–1.5 (default range):** Moderate enhancement; increases colony
      visibility and edge sharpness without amplifying noise excessively.
    - **1.7–2.0:** Strong enhancement; highlights colony surface texture and
      fine boundaries, but amplifies sensor noise and agar grain texture.
    - **> 2.0:** Risk of excessive noise amplification and artificial edge
      artifacts.
    - **Effect on colony morphology:** Uniform enhancement regardless of colony
      shape; affects all colonies proportionally.

    **eps (numerical safety constant):**
    - Typical range: 1e-8 to 1e-4 (rarely adjusted)
    - Purpose: Prevents log(0) and near-zero underflow when array contains
      zeros or very small values.
    - Default 1e-6 is safe for 8-bit and float32 images.

    Caveats and Limitations:

    - **Multichannel input:** If detect_mat is multichannel (e.g., 3-channel
      LAB), cv2.GaussianBlur applies blur per-channel independently. This may
      not produce a physically correct illumination decomposition (which assumes
      illumination is isotropic across color channels). Consider converting to
      grayscale first for best results.

    - **Kernel size growth:** Large sigma values generate huge blur kernels
      (side length = 6*sigma + 1). Kernel size ~1200 pixels at sigma=200 incurs
      significant computation time and memory overhead. For high-resolution
      images (> 3000 px wide), consider using smaller sigma or downsampling
      before filtering.

    - **Edge effects:** Gaussian blur at image borders may produce artifacts
      (cv2.GaussianBlur uses BORDER_REFLECT_101 by default: mirror-reflection
      excluding the border pixel). Illumination field estimation near edges is
      less reliable; avoid placing critical colonies very close to plate edges.

    - **Non-adaptive:** Fixed sigma and gamma parameters apply uniformly across
      the plate. Plates with spatially varying colony morphology (e.g., dense
      center, sparse edges) may need parameter tuning or region-specific
      processing.

    - **Noise amplification at gamma_high > 1.5:** Enhanced high frequencies
      amplify both legitimate colony edges and sensor noise. If input is
      inherently noisy (high-ISO DSLR, poor lighting), consider pre-filtering
      with GaussianBlur before homomorphic filtering.

    - **Interaction with downstream thresholding:** Homomorphic filtering is
      most effective before global thresholding (Otsu, triangle). Applying it
      after thresholding (on binary masks) has no effect.

    Mathematical Background:

    The homomorphic filter assumes the observed image I(x, y) as a product:

        I(x, y) = L(x, y) * R(x, y)

    where L is slowly varying illumination and R is high-frequency reflectance.
    Taking the natural logarithm converts the product to a sum:

        log(I) = log(L) + log(R)

    A Gaussian low-pass filter (kernel size ≈ 6*sigma + 1) separates the
    components:

        low_pass = GaussianBlur(log(I))  approx log(L)
        high_pass = log(I) - low_pass     approx log(R)

    Differential gains are applied:

        filtered_log = gamma_low * low_pass + gamma_high * high_pass

    Finally, exponentiating recovers the corrected image:

        result = exp(filtered_log) clipped to [0, 1]

    See Gonzalez & Woods, "Digital Image Processing," Chapter 4 (Frequency
    Domain Enhancement) for classical homomorphic filtering theory. In the 3rd
    edition (2008) homomorphic filtering appears as a standalone section in
    Chapter 4; in the 4th edition (2018) it appears within Section 4.9,
    pp. 339–341.
    """

    def __init__(
        self,
        sigma: float = 200.0,
        gamma_low: float = 0.5,
        gamma_high: float = 1.5,
        eps: float = 1e-6,
    ):
        """
        Parameters:
            sigma: Gaussian sigma for the illumination/reflectance cutoff.
                Larger values capture broader illumination gradients.  Start
                with a value several times the largest colony diameter.
            gamma_low: Gain for low frequencies (illumination).  < 1
                suppresses illumination variation; 1.0 leaves it unchanged.
            gamma_high: Gain for high frequencies (reflectance).  > 1
                enhances colony detail; 1.0 leaves it unchanged.
            eps: Offset to avoid ``log(0)``.  Rarely needs adjustment.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        self.eps = eps

    @staticmethod
    def _filter(
        array: np.ndarray,
        sigma: float = 200.0,
        gamma_low: float = 0.5,
        gamma_high: float = 1.5,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Apply homomorphic filtering to a single grayscale array.

        Separates illumination and reflectance components in the log domain,
        applies differential gains, and returns corrected image in linear domain.

        Args:
            array (np.ndarray): Grayscale image, shape (H, W), dtype float32,
                range [0.0, 1.0]. Supports both single-channel (H, W) and
                multichannel (H, W, C) arrays; multichannel is processed
                per-channel independently by cv2.GaussianBlur.
            sigma (float): Gaussian sigma for the low-pass filter (illumination
                estimation). Larger values suppress broader illumination
                gradients. Kernel size is int(6*sigma + 1), forced to be odd.
                Typical range: 20–300 pixels depending on image resolution and
                illumination artifact scale. Default is 200.0.
            gamma_low (float): Gain for low-frequency (illumination) component.
                Values < 1.0 suppress illumination variation; 1.0 is no change.
                Default is 0.5.
            gamma_high (float): Gain for high-frequency (reflectance) component.
                Values > 1.0 enhance colony detail; 1.0 is no change. Default is
                1.5.
            eps (float): Small constant added before logarithm to prevent log(0).
                Default is 1e-6. Rarely needs adjustment; use smaller values
                only if working with normalized or very low-intensity arrays.

        Returns:
            np.ndarray: Corrected image, shape matching input, dtype float32,
            range [0.0, 1.0]. Clipped to ensure valid range.

        Raises:
            ValueError: If sigma <= 0 (caught in __init__, not here).

        Processing Pipeline:

        1. Convert to float32 and compute log domain: log_image = log(array + eps)
        2. Estimate illumination field via Gaussian blur:
           - Kernel size: int(6*sigma + 1), forced odd
           - low_pass = cv2.GaussianBlur(log_image, (ksize, ksize), sigma, sigma)
        3. Compute reflectance residual: high_pass = log_image - low_pass
        4. Apply differential gains: filtered_log = gamma_low * low_pass + gamma_high * high_pass
        5. Exponentiate and clip: result = clip(exp(filtered_log) - eps, 0.0, 1.0)

        Notes:

        - This is a static method, so it can be called independently for testing
          or external use.
        - Multichannel arrays (H, W, C) are processed per-channel by
          cv2.GaussianBlur, which may not produce physically correct illumination
          decomposition. For color-aware processing, consider converting to
          single-channel (e.g., grayscale or LAB L-channel) before filtering.
        - Very large sigma values increase computation time and memory usage
          (kernel side = 6*sigma + 1 can exceed 1200 pixels). For high-resolution
          images, consider downsampling or reducing sigma.
        """
        log_image = np.log(array.astype(np.float32) + eps)

        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        low_pass = cv2.GaussianBlur(
            log_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
        )
        high_pass = log_image - low_pass

        filtered_log = gamma_low * low_pass + gamma_high * high_pass

        result = np.exp(filtered_log) - eps
        return np.clip(result, 0.0, 1.0)

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = self._filter(
            array=image.detect_mat[:],
            sigma=self.sigma,
            gamma_low=self.gamma_low,
            gamma_high=self.gamma_high,
            eps=self.eps,
        )
        return image
