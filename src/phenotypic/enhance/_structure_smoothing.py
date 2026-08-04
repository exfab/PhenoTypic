from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import ConfigDict, model_validator
from scipy.ndimage import gaussian_filter
from skimage.feature import structure_tensor_eigenvalues
from typing import Annotated
from typing_extensions import Self

from ..abc_ import Smoothing
from ..sdk_.typing_ import TuneSpec


class StructureSmoothing(Smoothing):
    """Enhance filamentous structures in ``detect_mat`` via coherence-enhancing anisotropic diffusion.

    Iteratively smooths the image preferentially along locally coherent
    orientations while suppressing diffusion across boundaries. A two-scale
    structure tensor (noise scale ``sigma``, integration scale ``rho``)
    estimates local orientation at each step, and the diffusion tensor
    is oriented to follow elongated features such as fungal hyphae, streak
    inoculations, and branching colony morphologies.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Filamentous fungal hyphae (Aspergillus, Penicillium, molds) where
          branching mycelial networks need reinforcement before ridge detection.
        - Streak inoculation patterns where colonies grow along lines.
        - Preprocessing before :class:`FocusEdgeSato`, :class:`FocusEdgeFrangi`,
          or :class:`FocusEdgeMeijering` to reduce noise without erasing
          tubular structures.
        - Faint elongated features in low-contrast or noisy plate scans.

    Consider Also:
        - :class:`LocalEdgeDenoise` for isotropic edge-preserving denoising
          of round colonies where directional enhancement is not needed.
        - :class:`FocusEdgeSato` for direct ridge detection on images that
          are already clean enough to skip a diffusion preprocessing step.
        - :class:`BlurGauss` when isotropic smoothing is sufficient and
          directional selectivity is not required.

    Args:
        num_iter: Number of diffusion iterations. Each iteration advances
            the PDE one time step of size ``dt``; the total diffusion extent
            is proportional to ``num_iter * dt``. Typical range: 5--100;
            values 15--30 are the practical working range for most plate-scan
            images. Default: 20.
        sigma: Noise scale for Gaussian derivative computation in pixels.
            Sets the spatial frequency band at which local orientation is
            estimated. Match to the half-width of structures to enhance:
            fine hyphae (< 3 px) need sigma 0.5--1.0; thick ridges or
            streak bands tolerate 2--4. Typical range: 0.5--5.0.
            Default: 1.5.
        rho: Integration scale for structure tensor smoothing in pixels.
            Averaging the outer-product tensor over ``rho`` produces a
            smoother orientation field that follows gently curving
            structures. Must be >= ``sigma``. ``None`` (default) sets
            ``rho = sigma`` (single-scale mode). Typical values: 2--3x
            ``sigma``; use values close to ``sigma`` for tight-turning
            branching mycelium and larger values for long straight streaks.
        dt: Time step per diffusion iteration. Must satisfy the 2D
            forward-Euler stability bound (dt <= 0.125). The extent of
            smoothing is governed by the total diffusion time
            T = ``num_iter * dt``, so prefer adjusting ``num_iter`` rather
            than ``dt`` when tuning effect magnitude. Typical range:
            0.05--0.125. Default: 0.1.
        alpha: Minimum diffusivity coefficient (0 < alpha < 1). Prevents
            the perpendicular-to-structure diffusivity from collapsing to
            zero and maximises anisotropy at small values (0.001); larger
            values (0.01--0.1) add isotropic regularisation. Default: 0.001.
        C: Percentile of the initial coherence histogram used as the
            adaptive contrast threshold (0 < C <= 100). High values (95--99)
            restrict anisotropic diffusion to only the most coherent pixels;
            lower values (70--90) spread it to weaker elongated structures.
            Default: 99.

    Returns:
        Image: Input image with ``detect_mat`` smoothed along coherent
        structures. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``num_iter`` < 1, ``dt`` is not in (0, 0.125],
            ``sigma`` <= 0, ``rho`` < ``sigma``, ``alpha`` not in (0, 1),
            or ``C`` not in (0, 100].

    References:
        [1] J. Weickert, "Coherence-enhancing diffusion filtering," *Int.
        J. Comput. Vis.*, vol. 31, no. 2/3, pp. 111--127, Apr. 1999.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
        :doc:`/tutorials/notebooks/10_detecting_filamentous_fungi` for
        pipelines that use this enhancer before ridge detection.
        :doc:`/explanation/what_enhancement_does` for background on
        anisotropic diffusion and the structure tensor.
    """

    model_config = ConfigDict(populate_by_name=True)

    num_iter: Annotated[int, TuneSpec(15, 30)] = 20
    sigma: Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 1.5
    # rho left tunable=False: it must satisfy the cross-field constraint
    # rho >= sigma (enforced in _check_diffusion_params); independent
    # sampling against a co-tuned sigma would generate invalid combos.
    rho: Annotated[float | None, TuneSpec(tunable=False)] = None
    dt: Annotated[float, TuneSpec(0.05, 0.125)] = 0.1
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
        if self.num_iter < 1:
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

        for _ in range(self.num_iter):
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
