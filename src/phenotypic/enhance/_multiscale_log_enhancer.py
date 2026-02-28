from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.ndimage import gaussian_laplace

from ..abc_ import ImageEnhancer

_SQRT2 = np.sqrt(2.0)


class MultiscaleLoGEnhancer(ImageEnhancer):
    """Multi-scale Laplacian of Gaussian blob enhancement for colony detection.

    Applies scale-normalised Laplacian of Gaussian (LoG) filtering across a
    geometric series of Gaussian sigmas and returns the maximum response at each
    pixel. Bright blob-like structures (colonies, inocula, droplets) on a darker
    background produce strong peaks regardless of their individual sizes, making
    this a robust preprocessing step before thresholding or GMM-based segmentation.

    Args:
        min_radius (float): Smallest target blob radius in pixels. Default: 3.0.
        max_radius (float): Largest target blob radius in pixels. Default: 12.0.
        num_scales (int): Number of logarithmically spaced scales. Default: 12.

    Returns:
        Image: Input image with detect_mat modified to contain scale-normalised
            LoG responses.

    Raises:
        ValueError: If min_radius <= 0, min_radius >= max_radius, or num_scales < 1.

    Intuition:
        The Laplacian of Gaussian (LoG) is a classic blob detector from computer
        vision. It responds strongly to circular/roughly-spherical bright regions
        on darker backgrounds—exactly matching colony morphology on agar plates.
        Unlike edge detectors (Sobel, Canny), which amplify boundaries, LoG detects
        the blob itself, producing a single peak near the colony center and suppressing
        gradual illumination changes. By applying LoG at multiple scales and taking
        the maximum response, you get size-invariant detection: small inocula and
        large mature colonies both produce strong peaks. This eliminates the need to
        know colony size in advance and works well for mixed growth stages on a single
        plate.

    Use Cases:
        1. **Sparse inoculation spots on bare agar:** When inocula are small
           (5–30 pixels), faint, and nearly invisible against the agar background,
           multiscale LoG makes them stand out. Useful for detecting initial
           inoculation on high-resolution scans before visible growth.

        2. **Mixed-size colonies on mature plates:** A single plate may have small
           emerging colonies (20–40 pixels) and large mature ones (100–200 pixels).
           Multiscale LoG detects both in one pass without parameter tuning per size.

        3. **Preprocessing before GMM segmentation:** LoG sharpens blob boundaries
           and suppresses uneven illumination, making Gaussian mixture models more
           stable and reducing false merges in clustered regions.

        4. **Low-contrast or shadowed regions:** On plates with vignetting or shadow
           variation, LoG emphasizes the blob structure itself rather than absolute
           intensity, improving contrast in poorly lit corners.

        5. **Droplet or water-spot detection:** When screening for contamination or
           condensation artifacts, LoG detects circular droplets as strongly as
           colonies, enabling their removal before analysis.

    Parameter Effects:
        **min_radius (float, pixel-based; scales with resolution):**

        - Typical range: 1.0–5.0 pixels at reference resolution (512×768)
        - Interpretation: Controls the smallest blob radius that produces strong response
        - Resolution scaling formula: `adjusted_min_radius = default_min_radius × (image_width / 512)`
        - Concrete examples at common resolutions:
          * 512×768: min_radius = 3.0 (default)
          * 640×960: min_radius ≈ 3.75 (3.0 × 1.25)
          * 1024×1536: min_radius ≈ 6.0 (3.0 × 2.0)
          * 2000×3000: min_radius ≈ 11.7 (3.0 × 3.9)
        - Effect on colony processing:
          * Small min_radius (1–2 pixels): Detects tiny inocula and noise; sensitive
            to speckles and agar texture. Produces high-magnitude peaks for small
            inocula but amplifies background noise.
          * Medium min_radius (3–5 pixels): Balanced detection of small-to-medium
            inocula (yeast spots, early bacterial colonies). Suppresses most noise
            while preserving initial growth stages.
          * Large min_radius (>5 pixels): Ignores small inocula entirely; optimized
            for detecting established colonies. Cleaner detection but misses early
            growth stages.
        - Colony morphology context:
          * Round yeast colonies: LoG response is isotropic; works well across
            all sizes.
          * Irregular bacteria (filamentous edges): LoG still responds to the blob
            center but may produce secondary peaks at irregular edges.
          * Filamentous fungi (elongated growth): LoG works best on the main blob;
            thread-like extensions produce weaker responses unless radius is tuned
            to match thread width.

        **max_radius (float, pixel-based; scales with resolution):**

        - Typical range: 8.0–50.0 pixels at reference resolution (512×768)
        - Interpretation: Controls the largest blob radius that produces strong response
        - Resolution scaling formula: `adjusted_max_radius = default_max_radius × (image_width / 512)`
        - Concrete examples at common resolutions:
          * 512×768: max_radius = 12.0 (default)
          * 640×960: max_radius ≈ 15.0 (12.0 × 1.25)
          * 1024×1536: max_radius ≈ 24.0 (12.0 × 2.0)
          * 2000×3000: max_radius ≈ 46.8 (12.0 × 3.9)
        - Effect on colony processing:
          * Small max_radius (6–10 pixels): Ignores large mature colonies; detects
            only small-to-medium inocula. LoG response drops rapidly for colonies
            larger than max_radius, creating edge artifacts.
          * Medium max_radius (10–25 pixels): Detects small-to-large colonies;
            suitable for plates with 5–150 pixel colonies. Balanced response across
            growth stages.
          * Large max_radius (>25 pixels): Includes very large or confluent regions.
            Computation increases; kernel size grows with sigma, slowing LoG
            evaluation. May merge nearby colonies into single peaks.
        - Colony morphology context:
          * Round yeast: LoG symmetry works well; increasing max_radius slightly
            improves sensitivity to large diploid strains or clumped growth.
          * Irregular bacteria: Increasing max_radius may cause multiple peaks around
            a single irregular colony if it's elongated.
          * Filamentous fungi: Large max_radius captures main growth mass but may
            blur detection of individual hyphae.

        **num_scales (int, dimensionless; does NOT scale with resolution):**

        - Typical range: 4–20 (default: 12)
        - Interpretation: Number of logarithmically spaced sigma values between
          min_radius/sqrt(2) and max_radius/sqrt(2)
        - Why this parameter does NOT scale with resolution: num_scales controls the
          *density* of scales in log-space, not absolute pixel sizes. More scales
          improve size discrimination but don't depend on image pixels—they depend
          on your task (e.g., "detect colonies from 3 to 12 pixel radii in 12 steps").
        - Examples (independent of resolution):
          * num_scales = 4: Coarse size coverage (fast); suitable for narrow size
            range (e.g., standardized inoculum size). ~3× speed boost vs. 12 scales.
          * num_scales = 8: Balanced; detects typical 2–3 octave range (e.g., 2–8
            pixel radii) without excessive overhead.
          * num_scales = 12 (default): Fine-grained size discrimination; covers
            ~4 octave range well. Moderate computational cost (~12 LoG evaluations).
          * num_scales = 20: Very fine discrimination; 2–3× slower. Use only if
            mixed-size colonies must all be detected with equal sensitivity.
        - Colony morphology context:
          * Round yeast: Fewer scales (6–8) are sufficient; size variation is modest.
          * Irregular bacteria with variable sizes: More scales (12–16) help resolve
            varying morphology.
          * Filamentous fungi with elongated growth: num_scales primarily helps
            capture blob-like central regions; thread-like structures still need
            ridge filters (Frangi, Meijering).

    Caveats and Limitations:
        1. **Intensity-dependent magnitude:** The absolute magnitude of LoG output
           depends on image contrast, blob sharpness, and local curvature. A very
           sharp, bright colony produces higher peaks than a faint, diffuse one.
           Normalize or apply percentile-based thresholding downstream if your
           detector expects fixed intensity ranges (e.g., Otsu assuming bimodal
           distribution).

        2. **Edge blur and kernel size:** Very large max_radius values increase sigma
           and thus kernel size (proportional to sigma). This slows computation
           significantly on large images (e.g., 4000×6000). Test on a small region
           first; kernel extends ~3–4 sigma in each direction.

        3. **Isotropic blob assumption:** LoG is rotationally symmetric and works best
           on roughly circular structures. Elongated or filamentous colonies (fungi,
           bacterial filaments) produce weaker, multi-peaked responses. For rod-shaped
           or thread-like structures, use ridge filters (Frangi, Sato, Meijering).

        4. **Merging of nearby blobs:** If colonies are touching or closely spaced
           and your radius range is large, multiscale LoG may produce a single merged
           peak at the center of the cluster. Use ObjectRefiner or watershed
           post-processing to separate merged detections.

        5. **Parameter sensitivity:** Results are sensitive to min_radius and
           max_radius choice. If your colony size distribution is bimodal (small
           inocula + large mature), a single radius range may under-respond to one
           mode. Consider two passes or adaptive approaches.

        6. **Interaction with downstream operations:** Place LoG *before* thresholding,
           not after. Thresholding compress dynamic range; applying LoG to a binary
           mask is meaningless.

    Mathematical Background (Brief):
        The Laplacian of Gaussian is defined as:
        LoG(x, y, σ) = -1/(π·σ⁴) · (1 - (x² + y²)/(2σ²)) · exp(-(x² + y²)/(2σ²))

        At scale σ, the LoG response peaks when σ matches the blob radius
        (approximately, σ ≈ radius/√2). The response magnitude is proportional to
        σ² times the raw LoG, giving scale normalization: responses at different
        scales become comparable. Taking the max across scales gives a single,
        size-invariant peak per blob. See Lindeberg (1998) for scale-space theory
        and blob detection foundations.

    Examples:
        Enhancing colony inocula on an agar plate:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import MultiscaleLoGEnhancer
        >>> image = load_synth_yeast_plate()
        >>> enhancer = MultiscaleLoGEnhancer(min_radius=3.0, max_radius=12.0)
        >>> enhanced = enhancer.apply(image)
        >>> enhanced.detect_mat.shape == image.detect_mat.shape
        True

        Inside a pipeline with subsequent detection:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import MultiscaleLoGEnhancer
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> pipeline = ImagePipeline([
        ...     MultiscaleLoGEnhancer(min_radius=2.0, max_radius=8.0, num_scales=8),
        ...     OtsuDetector(),
        ... ])
        >>> image = load_synth_yeast_plate()
        >>> result = pipeline.apply(image)
        >>> result.objmask is not None
        True

    Note on image resolution:
        PhenoTypic automatically handles images at any resolution. However, the
        radius parameters (min_radius, max_radius) scale with resolution; see
        "Parameter Effects" above. For images significantly different from 512×768
        (e.g., 2000×3000 from a DSLR), adjust radius parameters using the scaling
        formula: adjusted = default × (image_width / 512). The num_scales parameter
        does NOT scale and typically does not need adjustment across resolutions.
    """

    def __init__(
        self,
        min_radius: float = 3.0,
        max_radius: float = 12.0,
        num_scales: int = 12,
    ):
        """Initialize MultiscaleLoGEnhancer with radius range and scale density.

        Args:
            min_radius (float): Smallest target blob radius in pixels. The
                corresponding Gaussian sigma is ``min_radius / sqrt(2)``. Blobs
                smaller than this radius produce weaker LoG responses. Typical
                range: 1.0–5.0 pixels at 512×768 resolution. Default: 3.0.
            max_radius (float): Largest target blob radius in pixels. The
                corresponding Gaussian sigma is ``max_radius / sqrt(2)``. Blobs
                larger than this radius also produce weaker responses. Typical
                range: 8.0–50.0 pixels at 512×768 resolution. Default: 12.0.
            num_scales (int): Number of logarithmically spaced sigma values
                between ``min_radius / sqrt(2)`` and ``max_radius / sqrt(2)``.
                Controls size-discrimination resolution. Larger values give finer
                blob size resolution but increase computation (one LoG evaluation
                per scale). Typical range: 4–20. Default: 12.

        Raises:
            ValueError: If min_radius <= 0, min_radius >= max_radius, or
                num_scales < 1.
        """
        if min_radius <= 0:
            raise ValueError(f"min_radius must be positive, got {min_radius}")
        if min_radius >= max_radius:
            raise ValueError(
                f"min_radius ({min_radius}) must be less than "
                f"max_radius ({max_radius})"
            )
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")

        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.num_scales = int(num_scales)

    @staticmethod
    def _enhance(
        array: np.ndarray,
        min_radius: float = 3.0,
        max_radius: float = 12.0,
        num_scales: int = 12,
    ) -> np.ndarray:
        """Multi-scale Laplacian of Gaussian blob enhancement (core kernel).

        Applies scale-normalised Laplacian of Gaussian (LoG) filtering across a
        geometric series of Gaussian sigmas and returns the maximum response at each
        pixel. This is the core computation called by _operate() and available for
        direct use on standalone arrays.

        Args:
            array (numpy.ndarray): 2-D grayscale array, shape (height, width).
                Typically normalized to [0, 1] or [0, 255]. dtype should be float.
            min_radius (float): Smallest target blob radius in pixels. The
                corresponding Gaussian sigma is ``min_radius / sqrt(2)``. Blobs
                smaller than this produce weaker responses. Default: 3.0.
            max_radius (float): Largest target blob radius in pixels. The
                corresponding Gaussian sigma is ``max_radius / sqrt(2)``. Blobs
                larger than this also produce weaker responses. Default: 12.0.
            num_scales (int): Number of logarithmically spaced sigma values
                between ``min_radius / sqrt(2)`` and ``max_radius / sqrt(2)``.
                More scales provide finer size resolution at cost of speed.
                Default: 12.

        Returns:
            numpy.ndarray: Scale-normalised LoG response, same shape and dtype as
                *array*. Each pixel contains the maximum absolute LoG response across
                all scales, multiplied by sigma squared for scale normalization. All
                values are non-negative (≥ 0.0). Output range depends on input
                contrast and is typically [0, max_response] where max_response varies
                with image content.

        Raises:
            ValueError: If min_radius <= 0, min_radius >= max_radius, or
                num_scales < 1.

        Notes:
            At each scale σ, the Laplacian of Gaussian produces a response
            proportional to local curvature. The response is scaled by σ² so that
            blobs of different sizes produce comparable peak magnitudes. A max
            projection across scales selects the strongest response at each pixel,
            yielding size-invariant blob detection: a 3-pixel and a 12-pixel blob
            both produce strong peaks if they fall within [min_radius, max_radius].

            This function is the low-level kernel; for image processing via the
            PhenoTypic pipeline, use the MultiscaleLoGEnhancer class instead.

        Examples:
            Direct kernel use on a random array:

            >>> import numpy as np
            >>> from phenotypic.enhance._multiscale_log_enhancer import (
            ...     MultiscaleLoGEnhancer,
            ... )
            >>> rng = np.random.default_rng(0)
            >>> arr = rng.random((64, 64))
            >>> out = MultiscaleLoGEnhancer._enhance(arr)
            >>> out.shape
            (64, 64)
            >>> out.min() >= 0.0
            True

            LoG enhancement on a synthetic image with known blob:

            >>> import numpy as np
            >>> from phenotypic.enhance._multiscale_log_enhancer import (
            ...     MultiscaleLoGEnhancer,
            ... )
            >>> # Create a simple blob (Gaussian)
            >>> y, x = np.ogrid[:100, :100]
            >>> blob = np.exp(-((x - 50)**2 + (y - 50)**2) / 100.0)
            >>> out = MultiscaleLoGEnhancer._enhance(blob, min_radius=3, max_radius=12)
            >>> peak_pos = np.unravel_index(out.argmax(), out.shape)
            >>> abs(peak_pos[0] - 50) <= 2 and abs(peak_pos[1] - 50) <= 2
            True
        """
        if min_radius <= 0:
            raise ValueError(f"min_radius must be positive, got {min_radius}")
        if min_radius >= max_radius:
            raise ValueError(
                f"min_radius ({min_radius}) must be less than max_radius ({max_radius})"
            )
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")

        min_sigma = min_radius / _SQRT2
        max_sigma = max_radius / _SQRT2
        sigmas = np.geomspace(min_sigma, max_sigma, num_scales)

        enhanced = np.zeros_like(array)
        for sigma in sigmas:
            log_response = gaussian_laplace(array, sigma=sigma)
            scale_norm = sigma ** 2 * np.abs(log_response)
            np.maximum(enhanced, scale_norm, out=enhanced)

        return enhanced

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = self._enhance(
            array=image.detect_mat[:],
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            num_scales=self.num_scales,
        )
        return image
