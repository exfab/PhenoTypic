from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from scipy.ndimage import gaussian_filter
from skimage.feature import structure_tensor_eigenvalues
from typing_extensions import Self

from ..abc_ import ImageEnhancer


class CoherenceEnhancingDiffusion(ImageEnhancer):
    """Enhance filamentous structures via anisotropic coherence-enhancing diffusion.

    Smooths ``detect_mat`` preferentially along coherent structures (lines,
    ridges, edges) while preserving boundaries perpendicular to them. Uses the
    structure tensor to estimate local orientation and applies directional
    diffusion that follows elongated features such as fungal hyphae, streak
    inoculations, and branching colony morphologies.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        num_iter: Number of diffusion iterations. Typical range: 5--100.
            Small values (5--10) give subtle enhancement; medium values
            (15--30) are typical; large values (50--100) provide heavy
            smoothing. Default: 20.
        sigma: Noise/derivative scale for Gaussian gradient computation.
            Match to the width of structures to enhance. Typical range:
            0.5--5.0. Default: 1.5.
        rho: Integration scale for structure tensor smoothing. Must be
            >= ``sigma``. ``None`` (default) uses ``sigma`` (single-scale
            mode). Typical values: 2--3x ``sigma``.
        dt: Time step per iteration. Must satisfy the 2D forward-Euler
            stability bound (<=0.125). Typical range: 0.05--0.125.
            Default: 0.1.
        alpha: Minimum diffusivity (0 < alpha < 1). Small values (0.001)
            maximize anisotropy; larger values (0.01--0.1) add isotropic
            smoothing. Default: 0.001.
        C: Contrast percentile (0 < C <= 100) for the adaptive coherence
            threshold. Higher values restrict anisotropy to the most
            coherent structures. Default: 99.

    Returns:
        Image: Input image with ``detect_mat`` smoothed along coherent
        structures. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Filamentous fungal hyphae (Aspergillus, Penicillium, molds) where
          branching structures need enhancement.
        - Streak inoculation patterns where colonies grow along lines.
        - Preprocessing before ridge detection (Frangi, Sato, Meijering) to
          reduce noise without losing tubular structures.
        - Faint elongated features in low-contrast or noisy scans.

    Consider Also:
        - :class:`BilateralDenoise` for isotropic edge-preserving denoising
          of round colonies without directional features.
        - :class:`SatoRidgeFilter` for direct ridge detection without a
          diffusion preprocessing step.
        - :class:`MeijeringRidgeFilter` for detecting very fine neurite-like
          filaments.

    References:
        [1] J. Weickert, "Coherence-enhancing diffusion filtering," *Int.
        J. Comput. Vis.*, vol. 31, no. 2/3, pp. 111--127, Apr. 1999.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        anisotropic diffusion and structure tensor analysis.
    """

    # ``num_iter`` is the public constructor keyword; the algorithm body
    # (``_operate``) reads it as ``self.num_iterations``. The field keeps
    # the algorithm-facing name and accepts the public ``num_iter`` kwarg
    # via a validation alias, so neither the call sites nor ``_operate``
    # change. ``populate_by_name`` lets the field also be set/dumped under
    # its own name (used by JSON round-trips of migrated pipelines).
    model_config = ConfigDict(populate_by_name=True)

    num_iterations: int = Field(default=20, validation_alias="num_iter")
    sigma: float = 1.5
    rho: float | None = None
    dt: float = 0.1
    alpha: float = 0.001
    C: float = 99.0

    @model_validator(mode="after")
    def _check_diffusion_params(self) -> Self:
        """Reproduce the pre-migration ``__init__`` parameter guards.

        Several checks (notably ``rho >= sigma``) span more than one
        field, so all guards live in a single model validator rather than
        per-field validators.

        Raises:
            ValueError: If any diffusion parameter is outside its valid
                range (see the class ``Args:`` block).
        """
        if self.num_iterations < 1:
            raise ValueError("num_iter must be >= 1")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.dt > 0.125:
            raise ValueError(
                "dt > 0.125 exceeds the 2D forward-Euler stability bound (1/8); "
                "use smaller values"
            )
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.rho is not None:
            if self.rho <= 0:
                raise ValueError("rho must be > 0")
            if self.rho < self.sigma:
                raise ValueError(
                    f"rho ({self.rho}) must be >= sigma ({self.sigma}); the "
                    "integration scale cannot be smaller than the noise scale"
                )
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        if not (0 < self.C <= 100):
            raise ValueError("C must be in (0, 100]")
        return self

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
