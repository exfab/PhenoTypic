from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import structure_tensor_eigenvalues

from ..abc_ import ImageEnhancer


class CoherenceEnhancingDiffusion(ImageEnhancer):
    """
    Coherence-enhancing diffusion for filamentous structure enhancement on agar plates.

    Coherence-enhancing diffusion (CED) is an anisotropic diffusion technique that smooths
    images preferentially along coherent structures (lines, ridges, edges) while preserving
    boundaries perpendicular to those structures. Unlike isotropic smoothing (e.g., Gaussian
    blur), CED analyzes local orientation via the structure tensor and applies directional
    diffusion that follows elongated features. On fungal colony plates, CED enhances
    visibility of filamentous hyphae, streak inoculations, and elongated colony morphologies
    while suppressing background noise and texture.

    Use cases (agar plates):
    - Enhance filamentous fungal hyphae (Aspergillus, Penicillium, molds) for better
      segmentation of branching structures
    - Improve visibility of streak inoculation patterns where colonies grow along lines
    - Smooth colony interiors while preserving sharp colony-agar boundaries
    - Preprocess before ridge detection (Frangi, Sato, Meijering) to reduce noise
      without losing tubular structures
    - Enhance faint elongated features in low-contrast or noisy scans

    Tuning and effects:
    - num_iterations: Number of diffusion steps. More iterations produce stronger
      smoothing along coherent directions. Small values (5-10) provide subtle enhancement
      with fast execution. Medium values (15-30) balance enhancement and speed for
      typical use. Large values (50-100) yield heavy smoothing, useful for very noisy
      images but may over-smooth fine details. Computational cost scales linearly.
    - dt: Time step for each diffusion iteration. Must satisfy the 2D forward-Euler
      stability bound dt <= 1/8 (0.125). Smaller values (0.05-0.1) are more stable but
      require more iterations for the same effect. Larger values (0.1-0.125) converge
      faster but approach the stability limit. Recommended: 0.1 for most cases.
    - sigma: Noise/derivative scale for structure tensor computation (Gaussian derivative
      σ). Controls the scale at which image gradients are computed. Small values
      (0.5-1.5) detect fine structures but are more sensitive to noise. Medium values
      (1.5-3.0) provide robust gradient estimation for typical colony features.
      Large values (3-5) detect coarse structures but may miss fine hyphae. Match
      to the width of the structures you want to enhance.
    - rho: Integration scale for structure tensor smoothing. Controls the neighborhood
      over which gradient orientation is averaged. Must be >= sigma. When None (default),
      equals sigma (single-scale mode). Larger rho values produce smoother orientation
      fields, which is useful when structures span many pixels or when gradient estimates
      are noisy. Typical values: rho = 2*sigma to 3*sigma. Literature defaults:
      DIPlib sigma=1, rho=3; ITK sigma=0.5, rho=2.
    - alpha: Minimum diffusivity parameter (0 < alpha < 1). Prevents complete smoothing
      in uniform regions by ensuring some isotropic diffusion everywhere. Small values
      (0.001-0.01) maximize anisotropy, strongly favoring directional smoothing. Larger
      values (0.01-0.1) add more isotropic smoothing, useful for noisy images where
      orientation estimates are unreliable.
    - C: Contrast percentile for the diffusivity function (0 < C <= 100). The actual
      contrast threshold is computed as the Cth percentile of the coherence histogram
      (lambda1 - lambda2)^2 from the original image, making the parameter adaptive to
      image content. Higher values (e.g. 99) set a high threshold so only the most
      coherent structures get strong anisotropic diffusion. Lower values (e.g. 50)
      enhance weaker structures too. Recommended:
      99 for typical colony plates, 50-70 for faint structures,
      95-99 for noisy images where only the strongest coherence should drive anisotropy.

    Caveats:
    - Computational cost: CED is iterative and computes structure tensors per iteration.
      For large images or many iterations, processing can be slow. Consider downsampling
      for initial parameter tuning, then apply to full resolution.
    - Numerical stability: dt values above the stability bound (0.125) are rejected.
      If output looks noisy or has ringing, reduce dt further.
    - Structure scale: The sigma parameter must match the scale of features to enhance.
      Hyphae width of ~3 pixels works well with sigma=1.5; adjust proportionally.
    - Not for isotropic features: CED enhances elongated structures. For round colonies
      without directional features, use isotropic denoising (BilateralDenoise, NonLocalMeans).
    - Boundary effects: Edge pixels may show artifacts from gradient computation.
      Consider cropping edges after processing if artifacts appear.

    References:
        Weickert, J. (1999). Coherence-enhancing diffusion filtering.
        International Journal of Computer Vision, 31(2/3), 111-127.
        https://doi.org/10.1023/A:1008009714131

    Attributes:
        num_iterations (int): Number of diffusion iterations. More iterations produce
            stronger directional smoothing.
        dt (float): Time step for diffusion. Must be small for stability (typically 0.1).
        sigma (float): Noise/derivative scale for Gaussian gradient computation.
        rho (float | None): Integration scale for structure tensor smoothing. None means
            single-scale mode (rho = sigma).
        alpha (float): Minimum diffusivity. Prevents complete smoothing in uniform regions.
        C (float): Contrast percentile (0, 100] for adaptive coherence threshold.

    Examples:
        Enhancing filamentous fungal hyphae before ridge detection:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import CoherenceEnhancingDiffusion
        >>> # Load plate image (works with any plate, but especially useful for
        >>> # filamentous fungi like Aspergillus or molds)
        >>> image = load_synth_yeast_plate()
        >>> # Apply CED with default parameters suitable for typical hyphae
        >>> ced = CoherenceEnhancingDiffusion(num_iter=20, sigma=1.5)
        >>> enhanced = ced.apply(image)
        >>> # Detection matrix now has smoother hyphae with preserved boundaries
        >>> assert enhanced.detect_mat.shape == image.detect_mat.shape

        Pipeline with CED and ridge detection for filamentous structure analysis:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import CoherenceEnhancingDiffusion, FrangiVesselness
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> # Build pipeline for filamentous structure detection
        >>> pipeline = ImagePipeline([
        ...     # Step 1: Enhance coherent structures (hyphae, streaks)
        ...     CoherenceEnhancingDiffusion(num_iter=15, sigma=2.0, rho=5.0, dt=0.1),
        ...     # Step 2: Detect tubular/ridge-like structures
        ...     FrangiVesselness(sigmas=range(1, 4), black_ridges=False),
        ...     # Step 3: Threshold to binary mask
        ...     OtsuDetector(),
        ... ])
        >>> image = load_synth_yeast_plate()
        >>> result = pipeline.apply(image)
        >>> # Result contains detected filamentous structures
        >>> assert result.objmask is not None

        Heavy smoothing for very noisy images with prominent streaks:

        >>> from phenotypic.enhance import CoherenceEnhancingDiffusion
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> # For very noisy scans with clear streak patterns, use more iterations
        >>> heavy_ced = CoherenceEnhancingDiffusion(
        ...     num_iter=50,   # More iterations for stronger effect
        ...     dt=0.08,             # Smaller step for stability
        ...     sigma=2.5,           # Larger scale for coarse structures
        ...     alpha=0.01,          # Slightly more isotropic component
        ... )
        >>> image = load_synth_yeast_plate()
        >>> enhanced = heavy_ced.apply(image)
        >>> # Heavy smoothing along streak directions, noise suppressed
        >>> assert enhanced.detect_mat.shape == image.detect_mat.shape
    """

    def __init__(
            self,
            num_iter: int = 20,
            sigma: float = 1.5,
            rho: float | None = None,
            dt: float = 0.1,
            *,
            alpha: float = 0.001,
            C: float = 99.0,
    ):
        """
        Parameters:
            num_iter (int): Number of diffusion iterations. Controls the total
                amount of smoothing applied. Small values (5-10) give subtle enhancement;
                medium values (15-30) are typical; large values (50-100) provide heavy
                smoothing. Computational cost scales linearly with iterations.
                Recommended: 20 for balanced enhancement.
            sigma (float): Noise/derivative scale (Gaussian derivative σ). Controls the
                scale at which image gradients are computed for orientation estimation.
                Match to the width of structures you want to enhance: ~1.5 for fine
                hyphae (~3px wide), ~3.0 for coarser structures. Recommended: 1.5.
            rho (float | None): Integration scale for structure tensor smoothing. Controls
                the neighborhood over which gradient products are averaged. Must be
                >= sigma. When None (default), equals sigma (single-scale mode). Larger
                values produce smoother orientation fields. Typical: 2-3x sigma.
            dt (float): Time step for each diffusion iteration. Must satisfy the
                2D forward-Euler stability bound of 1/8 (0.125). Smaller values
                require more iterations for equivalent smoothing. Recommended:
                0.1 for stable, efficient diffusion.
            alpha (float): Minimum diffusivity parameter (0 < alpha < 1). Ensures some
                diffusion even in uniform regions, preventing numerical issues. Small
                values (0.001) maximize anisotropy; larger values (0.01-0.1) add more
                isotropic smoothing. Recommended: 0.001 for strong directional bias.
            C (float): Contrast percentile for the diffusivity function
                (0 < C <= 100). The Cth percentile of the coherence histogram
                (lambda1 - lambda2)^2 from the original image is used as the
                contrast threshold, adapting to image content. Higher values
                restrict anisotropy to the most coherent structures.
                Default: 99.
        """
        if num_iter < 1:
            raise ValueError("num_iter must be >= 1")

        if dt <= 0:
            raise ValueError("dt must be > 0")

        if dt > 0.125:
            raise ValueError(
                    "dt > 0.125 exceeds the 2D forward-Euler stability bound (1/8); "
                    "use smaller values"
            )

        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        if rho is not None:
            if rho <= 0:
                raise ValueError("rho must be > 0")
            if rho < sigma:
                raise ValueError(
                        f"rho ({rho}) must be >= sigma ({sigma}); the integration "
                        "scale cannot be smaller than the noise scale"
                )

        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")

        if not (0 < C <= 100):
            raise ValueError("C must be in (0, 100]")

        self.num_iterations = int(num_iter)
        self.dt = float(dt)
        self.sigma = float(sigma)
        self.rho = float(rho) if rho is not None else None
        self.alpha = float(alpha)
        self.C = float(C)

    @staticmethod
    def _central_diff(arr: np.ndarray, axis: int) -> np.ndarray:
        """First derivative via acc=2 central stencil (matches FinDiff(axis, 1.0, 1)).

        Interior points use second-order central differences ``[-0.5, 0, 0.5]``.
        Boundary points use second-order one-sided stencils identical to
        ``findiff.FinDiff`` with default ``acc=2``.

        Args:
            arr: Input array (2D or higher).
            axis: Axis along which to differentiate.

        Returns:
            Array of same shape with first derivative along *axis*.
        """
        out = np.empty_like(arr)
        n = arr.shape[axis]
        s = [slice(None)] * arr.ndim

        def sl(start: int | None, stop: int | None) -> tuple:
            s[axis] = slice(start, stop)
            return tuple(s)

        def ix(i: int) -> tuple:
            s[axis] = i
            return tuple(s)

        # Interior: second-order central [-0.5, 0, 0.5]
        out[sl(1, n - 1)] = 0.5 * (arr[sl(2, n)] - arr[sl(0, n - 2)])
        # Forward boundary (acc=2): [-1.5, 2.0, -0.5]
        out[ix(0)] = -1.5 * arr[ix(0)] + 2.0 * arr[ix(1)] - 0.5 * arr[ix(2)]
        # Backward boundary (acc=2): [0.5, -2.0, 1.5]
        out[ix(n - 1)] = (
            0.5 * arr[ix(n - 3)] - 2.0 * arr[ix(n - 2)] + 1.5 * arr[ix(n - 1)]
        )
        return out

    def _operate(self, image: Image) -> Image:
        """Apply coherence-enhancing diffusion to enhance filamentous structures."""
        # Work with float64 for numerical stability
        img = image.detect_mat[:].astype(np.float64)

        # Resolve integration scale (rho defaults to sigma for single-scale mode)
        rho = self.rho if self.rho is not None else self.sigma

        # Compute contrast threshold from the original image's coherence
        # histogram (Cth percentile), so it adapts to image content
        u_r0 = gaussian_filter(img, sigma=self.sigma, order=[1, 0])
        u_c0 = gaussian_filter(img, sigma=self.sigma, order=[0, 1])
        S_rr0 = gaussian_filter(u_r0 * u_r0, sigma=rho)
        S_rc0 = gaussian_filter(u_r0 * u_c0, sigma=rho)
        S_cc0 = gaussian_filter(u_c0 * u_c0, sigma=rho)
        l1_0, l2_0 = structure_tensor_eigenvalues(
                [S_rr0, S_rc0, S_cc0],
        )
        contrast_threshold = np.percentile(
                (l1_0 - l2_0) ** 2, self.C,
        )

        for _ in range(self.num_iterations):
            # Two-scale structure tensor (Weickert IJCV 1999)
            # Gaussian derivatives at noise scale sigma
            u_r = gaussian_filter(img, sigma=self.sigma, order=[1, 0])
            u_c = gaussian_filter(img, sigma=self.sigma, order=[0, 1])

            # Outer product, integrated at scale rho
            S_rr = gaussian_filter(u_r * u_r, sigma=rho)
            S_rc = gaussian_filter(u_r * u_c, sigma=rho)
            S_cc = gaussian_filter(u_c * u_c, sigma=rho)

            lambda1, lambda2 = structure_tensor_eigenvalues(
                    [S_rr, S_rc, S_cc],
            )

            # Coherence measure (unnormalized, per Weickert IJCV 1999)
            coherence = (lambda1 - lambda2) ** 2

            # Diffusion coefficients based on coherence
            # c1: diffusion perpendicular to structure (small, preserves edges)
            # c2: diffusion along structure (large where coherent)
            c1 = self.alpha
            c2 = (self.alpha
                  + (1 - self.alpha)
                  * np.exp(-contrast_threshold / (coherence + 1e-10)
                           ))

            # Local orientation from structure tensor
            theta = 0.5 * np.arctan2(2 * S_rc, S_rr - S_cc)

            # Diffusion tensor components in (row, col) coordinates
            # Cache trig to avoid redundant transcendental calls
            c = np.cos(theta)
            s = np.sin(theta)
            cos2 = c * c
            sin2 = s * s
            cossin = c * s

            D_rr = c1 * cos2 + c2 * sin2
            D_rc = (c1 - c2) * cossin
            D_cc = c1 * sin2 + c2 * cos2

            # Compute gradients using acc=2 central finite differences
            gx = self._central_diff(img, 1)  # du/dcol
            gy = self._central_diff(img, 0)  # du/drow

            # Flux: pair D_cc with du/dcol, D_rr with du/drow
            Fx = D_cc * gx + D_rc * gy
            Fy = D_rc * gx + D_rr * gy

            # Flux divergence
            div = self._central_diff(Fx, 1) + self._central_diff(Fy, 0)

            # Update image with diffusion step
            img = img + self.dt * div

        # Store result back to detection matrix, clipping to valid range
        image.detect_mat[:] = (np.clip(img, 0.0, 1.0)
                               .astype(image.detect_mat.dtype))
        return image
