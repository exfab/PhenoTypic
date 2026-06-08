"""Phase congruency enhancement for contrast-invariant edge detection.

Implementation follows Kovesi's phasecong3 algorithm using oriented log-Gabor wavelets.
Algorithm details from ImagePhaseCongruency.jl (Julia reference implementation).

References:
    Peter Kovesi's ImagePhaseCongruency.jl: https://github.com/peterkovesi/ImagePhaseCongruency.jl
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, List, Literal

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from pydantic import Field

from ..abc_ import FocusEdge
from ..tools_.typing_ import TuneSpec

if TYPE_CHECKING:
    from phenotypic._core._image import Image


@dataclass
class _PhaseCong3Result:
    """Internal container for phasecong3 results.

    Attributes:
        M: Maximum moment of phase congruency covariance (edge strength).
        m: Minimum moment of phase congruency covariance (corner strength).
        orientation: Feature orientation in radians [-pi/2, pi/2]. 0 corresponds
            to a vertical edge, pi/2 is horizontal. Positive is anticlockwise.
        feature_type: Local weighted mean phase angle. pi/2 corresponds to a
            bright line, 0 to a step edge, -pi/2 to a dark line.
        T: Calculated noise threshold.
        pc_sum: Mean phase congruency across all orientations (normalized).
    """

    M: np.ndarray
    m: np.ndarray
    orientation: np.ndarray
    feature_type: np.ndarray
    T: float
    pc_sum: np.ndarray


class FocusEdgePhase(FocusEdge):
    """Enhance colony edges in ``detect_mat`` using contrast-invariant phase congruency.

    Detects features where log-Gabor Fourier components are maximally in
    phase, producing an edge response that depends on phase agreement rather
    than amplitude. The result is invariant to local illumination level and
    scanner vignetting, making faint or translucent colony boundaries visible
    even where intensity-gradient methods fail. For algorithm details see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Colony boundaries that vary in opacity or contrast across the plate
          due to pigmentation differences, agar depth variation, or colony age.
        - Plates with scanner vignetting or uneven illumination where
          gradient-based filters produce inconsistent edge strength.
        - Faint, translucent colonies on bright agar where the amplitude
          signal is weak but phase coherence is preserved.
        - Filamentous fungi plates where edges span a wide range of
          orientations and ``n_orient=8`` captures all hyphal angles.

    Consider Also:
        - :class:`FocusEdgeFrangi` for elongated hyphae when vesselness
          selectivity for ridge shape is more important than illumination
          invariance.
        - :class:`FocusEdgeHessian` for multi-scale ridge and edge detection
          with explicit blob-sensitivity control.
        - :class:`SharpenEdgeGauss` for edge sharpening that preserves the
          original intensity profile on uniformly illuminated plates.

    Args:
        n_scale: Number of log-Gabor octave scales. More scales integrate
            phase evidence over a wider spatial frequency range, giving a
            smoother but spatially broader response. Typical range: 3--6.
            Default: 4.
        n_orient: Number of oriented filter lobes. 6 gives 30-degree angular
            spacing (suitable for circular yeast colony edges); 8 gives
            22.5-degree spacing and better sensitivity to hyphae at arbitrary
            angles. Typical range: 4--8. Default: 6.
        min_wavelength: Center wavelength (pixels) of the finest log-Gabor
            scale. Should be matched to the narrowest expected colony edge
            width; must be >= 2 (Nyquist limit enforced by validator). Smaller
            values detect finer high-frequency features; larger values focus on
            broader edges. Default: 3.0.
        mult: Ratio between successive scale wavelengths. Together with
            ``sigma_onf`` it determines inter-scale spectral coverage. ``2.1``
            is the upstream default. For even coverage of the spectrum Kovesi
            recommends pairing ``mult`` and ``sigma_onf`` together; e.g.
            ``sigma_onf=0.55`` / ``mult=3`` gives roughly 2-octave bandwidth
            and ``sigma_onf=0.75`` / ``mult=1.6`` gives roughly 1-octave
            bandwidth. Re-tune both whenever either changes. Must be > 1.
            Default: 2.1.
        sigma_onf: Log-Gabor bandwidth ratio (standard deviation of the
            Gaussian transfer function divided by the filter center frequency).
            Smaller values give wider bandwidth (more octaves per scale,
            broadband, suited for plates with a wide range of colony sizes);
            larger values give narrower, more frequency-selective bandwidth.
            For even spectral coverage pair with ``mult`` per the upstream
            table (``0.55`` with ``mult=3``, ``0.75`` with ``mult=1.6``).
            Valid range: 0.1--1.0. Default: 0.55.
        k: Noise threshold multiplier in units of the estimated Rayleigh
            noise standard deviation. Higher values (5--20) suppress more
            noise at the cost of missing faint colony edges; lower values
            (1--3) maximise edge recall on clean images. Value 0 disables
            noise thresholding entirely. Default: 2.0.
        cutoff: Frequency spread penalty threshold. Phase congruency values
            are penalised via a sigmoid when the multi-scale amplitude spread
            falls below this fraction, discouraging single-scale responses.
            Valid range: (0, 1) exclusive. Default: 0.5.
        g: Sigmoid sharpness controlling the transition from penalised to
            unpenalised frequency spread. Higher values create a near-binary
            gate; lower values create a gradual blend. Must be > 0.
            Default: 10.0.
        noise_method: Noise threshold estimation strategy. ``-1.0`` (default)
            estimates from the median of the smallest-scale filter amplitude
            (robust, recommended for heterogeneous plate populations). ``-2.0``
            uses the Rayleigh histogram mode (more robust on images with
            strong background gradients). Any value >= 0 bypasses estimation
            and uses that value as a fixed threshold, enabling fully
            deterministic pipelines. Default: -1.0.
        output: Which phase congruency quantity to store in ``detect_mat``.
            ``'pc_sum'`` (default) is the mean phase congruency across all
            orientations, normalised to [0, 1]; best general-purpose edge map
            for downstream thresholding. ``'M'`` is the maximum eigenvalue of
            the phase congruency covariance tensor (edge strength along
            continuous curves). ``'m'`` is the minimum eigenvalue (corner and
            junction strength). Accepted values: ``'pc_sum'``, ``'M'``,
            ``'m'``. Default: ``'pc_sum'``.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the phase
        congruency map, clipped to [0, 1]. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``n_scale`` < 1, ``n_orient`` < 1,
            ``min_wavelength`` < 2, ``mult`` <= 1, ``sigma_onf`` outside
            [0.1, 1.0], ``k`` < 0, ``cutoff`` outside (0, 1), or ``g`` <= 0.

    References:
        [1] P. Morrone and R. A. Owens, "Feature detection from local
        energy," *Pattern Recognit. Lett.*, vol. 6, no. 5, pp. 303--313,
        Dec. 1987.

        [2] M. C. Morrone and D. C. Burr, "Feature detection in human
        vision: A phase-dependent energy model," *Proc. R. Soc. London,
        Ser. B*, vol. 235, no. 1280, pp. 221--245, Dec. 1988.

        [3] P. Kovesi, "Phase congruency: A low-level image invariant,"
        *Psychol. Res.*, vol. 64, no. 2, pp. 136--148, Aug. 2000.

        [4] D. J. Field, "Relations between the statistics of natural images
        and the response properties of cortical cells," *J. Opt. Soc. Am.
        A*, vol. 4, no. 12, pp. 2379--2394, Dec. 1987.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of contrast-invariant enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        phase congruency and the Local Energy Model.
    """

    n_scale: Annotated[int, TuneSpec(3, 6)] = Field(4, ge=1)
    n_orient: Annotated[int, TuneSpec(4, 8)] = Field(6, ge=1)
    min_wavelength: Annotated[float, TuneSpec(2.0, 10.0)] = Field(3.0, ge=2.0)
    mult: Annotated[float, TuneSpec(1.5, 3.0)] = Field(2.1, gt=1.0)
    sigma_onf: Annotated[float, TuneSpec(0.1, 1.0)] = Field(0.55, ge=0.1, le=1.0)
    # Lower bound 0.5 (not 0.0): k=0 disables noise thresholding, a degenerate
    # search anchor that the optimizer should never spend trials on.
    k: Annotated[float, TuneSpec(0.5, 20.0)] = Field(2.0, ge=0.0)
    cutoff: Annotated[float, TuneSpec(0.3, 0.7)] = Field(0.5, gt=0.0, lt=1.0)
    g: Annotated[float, TuneSpec(2.0, 20.0)] = Field(10.0, gt=0.0)
    noise_method: Annotated[float, TuneSpec(tunable=False)] = -1.0
    output: Literal["M", "m", "pc_sum"] = "pc_sum"

    def _operate(self, image: Image) -> Image:
        """Apply phase congruency enhancement to the detection matrix channel."""
        result = self._phasecong3(image.detect_mat[:])

        # Select output based on configuration
        output_map = {"M": result.M, "m": result.m, "pc_sum": result.pc_sum}
        selected = output_map[self.output]

        # Ensure output is in [0, 1] range for detect_mat compatibility.
        # detect_mat enforces float32 on assignment, so no explicit cast is needed.
        image.detect_mat[:] = np.clip(selected, 0.0, 1.0)
        return image

    def _phasecong3(self, img: np.ndarray) -> _PhaseCong3Result:
        """Compute phase congruency via log-Gabor filters.

        Implementation follows Kovesi's phasecong3 algorithm with corrections
        from the Julia reference implementation (ImagePhaseCongruency.jl).

        Args:
            img: 2D grayscale image as numpy array.

        Returns:
            _PhaseCong3Result containing M, m, orientation, feature_type, T, pc_sum.
        """
        img = np.asarray(img, dtype=np.float64)
        rows, cols = img.shape
        epsilon = 1e-5  # Julia uses 1e-5

        # Construct filter grids (quadrant-shifted, DC at corners)
        radius, sintheta, costheta, freq = self._construct_filter_grids(rows, cols)

        # Construct radial component of log-Gabor filters
        log_gabor_list = self._construct_log_gabor_filters(radius)

        # Construct angular components using cosine filter (Julia reference)
        angular_spread = self._compute_angular_spread(sintheta, costheta)

        # Get FFT of image
        image_fft = fft2(img)

        # Initialize accumulators
        cov_x2 = np.zeros((rows, cols), dtype=np.float64)
        cov_y2 = np.zeros((rows, cols), dtype=np.float64)
        cov_xy = np.zeros((rows, cols), dtype=np.float64)
        energy_v = np.zeros((rows, cols, 3), dtype=np.float64)
        pc_sum = np.zeros((rows, cols), dtype=np.float64)

        # Per-scale real/imag storage, reused each orientation (replaces
        # n_scale x n_orient complex128 EO list — saves ~3.8 GB at 3000x4000)
        eo_real = np.empty((self.n_scale, rows, cols), dtype=np.float64)
        eo_imag = np.empty((self.n_scale, rows, cols), dtype=np.float64)

        # Noise threshold estimation
        T: float = 0.0

        # Process each orientation
        for o in range(self.n_orient):
            angle = o * np.pi / self.n_orient

            # Accumulators for this orientation
            sum_even = np.zeros((rows, cols), dtype=np.float64)
            sum_odd = np.zeros((rows, cols), dtype=np.float64)
            sum_amplitude = np.zeros((rows, cols), dtype=np.float64)
            max_amplitude = np.zeros((rows, cols), dtype=np.float64)

            # Initialize tau for this orientation (matches Julia logic)
            tau: float = 0.0

            for s in range(self.n_scale):
                # Combined filter: log-Gabor radial * angular spread
                filter_combined = log_gabor_list[s] * angular_spread[o]

                # Apply filter in frequency domain
                filtered_fft = image_fft * filter_combined

                # Transform back to spatial domain — store real/imag separately
                # to avoid retaining complex128 arrays across orientations
                eo_complex = ifft2(filtered_fft)
                amplitude = np.abs(eo_complex)
                eo_real[s] = eo_complex.real
                eo_imag[s] = eo_complex.imag

                # Accumulate responses
                sum_even += eo_real[s]
                sum_odd += eo_imag[s]
                sum_amplitude += amplitude
                max_amplitude = np.maximum(max_amplitude, amplitude)

                # Noise estimation from smallest scale (s=0), per orientation
                if s == 0 and self.noise_method < 0:
                    if abs(self.noise_method + 1) < epsilon:
                        # Median-based estimation
                        tau = float(np.median(amplitude) / np.sqrt(np.log(4)))
                    elif abs(self.noise_method + 2) < epsilon:
                        # Mode-based Rayleigh estimation
                        tau = self._rayleigh_mode(amplitude)

            # Compute noise threshold T for this orientation
            if self.noise_method >= 0:
                T = self.noise_method
            else:
                # Total tau across scales (geometric series)
                if tau > 0:
                    total_tau = tau * (1 - (1 / self.mult) ** self.n_scale) / (
                            1 - 1 / self.mult
                    )
                    # Expected noise energy from Rayleigh distribution
                    mean_energy = total_tau * np.sqrt(np.pi / 2)
                    sigma_energy = total_tau * np.sqrt((4 - np.pi) / 2)
                    T = mean_energy + self.k * sigma_energy
                else:
                    T = 0.0

            # Compute unit-normalized mean direction (Julia reference: XEnergy normalization)
            # MeanE and MeanO form a unit vector pointing in mean phase direction
            x_energy = np.sqrt(sum_even ** 2 + sum_odd ** 2) + epsilon
            mean_even = sum_even / x_energy
            mean_odd = sum_odd / x_energy

            # Compute energy with cross-term subtraction (Julia reference)
            # Sequential loop avoids large temporaries from vectorized sum
            energy = np.zeros((rows, cols), dtype=np.float64)
            for s in range(self.n_scale):
                even = eo_real[s]
                odd = eo_imag[s]
                energy += (
                    even * mean_even + odd * mean_odd
                    - np.abs(even * mean_odd - odd * mean_even)
                )

            # Accumulate energy vectors for orientation/feature_type (Julia reference)
            energy_v[:, :, 0] += sum_even
            energy_v[:, :, 1] += np.cos(angle) * sum_odd
            energy_v[:, :, 2] += np.sin(angle) * sum_odd

            # Frequency spread weighting (Julia reference)
            # Width measures how spread out the frequency responses are
            width = (sum_amplitude / (max_amplitude + epsilon) - 1) / (self.n_scale - 1)
            weight = 1.0 / (1.0 + np.exp((self.cutoff - width) * self.g))

            # Phase congruency for this orientation (local variable, not stored
            # in a 3D array — saves ~0.58 GB at 3000x4000)
            pc_o = weight * np.maximum(energy - T, 0) / (sum_amplitude + epsilon)

            # Accumulate covariance tensor components
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            pc_sum += pc_o

            # Square pc_o for covariance (matches Julia PCo^2)
            pc_sq = pc_o ** 2
            cov_x2 += pc_sq * cos_angle * cos_angle
            cov_y2 += pc_sq * sin_angle * sin_angle
            cov_xy += pc_sq * cos_angle * sin_angle

        # Normalize covariance (Julia reference)
        cov_x2 /= self.n_orient / 2
        cov_y2 /= self.n_orient / 2
        cov_xy *= 4.0 / self.n_orient

        # Eigenvalue analysis of covariance tensor
        denom = np.sqrt(cov_xy ** 2 + (cov_x2 - cov_y2) ** 2) + epsilon

        # Maximum and minimum moments
        M = (cov_x2 + cov_y2 + denom) / 2
        m = (cov_x2 + cov_y2 - denom) / 2

        # Ensure non-negative
        M = np.maximum(M, 0)
        m = np.maximum(m, 0)

        # Orientation (Julia reference: atan(-EnergyV[:,:,3]./EnergyV[:,:,2]))
        # Julia uses single-argument atan which gives [-pi/2, pi/2] range.
        # We use arctan for consistency, with safe division handling.
        with np.errstate(divide="ignore", invalid="ignore"):
            orientation = np.arctan(-energy_v[:, :, 2] / energy_v[:, :, 1])
        # Handle NaN/Inf from division by zero (vertical edges)
        orientation = np.nan_to_num(orientation, nan=0.0, posinf=np.pi / 2,
                                    neginf=-np.pi / 2)

        # Feature type (Julia reference)
        odd_v = np.sqrt(energy_v[:, :, 1] ** 2 + energy_v[:, :, 2] ** 2)
        feature_type = np.arctan2(energy_v[:, :, 0], odd_v)

        # Note: pc_sum normalization by n_orient is a Python-specific addition.
        # Julia's phasecong3 doesn't return pc_sum. Normalizing keeps values
        # in [0,1] range regardless of n_orient, making it suitable for detect_mat.
        return _PhaseCong3Result(
                M=M,
                m=m,
                orientation=orientation,
                feature_type=feature_type,
                T=T,
                pc_sum=pc_sum / self.n_orient,
        )

    def _construct_filter_grids(
            self, rows: int, cols: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct frequency domain grids for filter construction.

        Grids are quadrant-shifted so DC component is at [0, 0].
        Follows Julia filtergrids() implementation for odd/even handling.

        Args:
            rows: Number of rows in image.
            cols: Number of columns in image.

        Returns:
            Tuple of (radius, sintheta, costheta, freq) where:
            - radius: Radial frequency normalized [0, 0.5] with DC=1 to avoid div/0
            - sintheta: fx/freq grid for angular filter (Julia gridangles)
            - costheta: fy/freq grid for angular filter (Julia gridangles)
            - freq: Original radial frequency with DC=0
        """
        # Frequency coordinates - Julia handles odd/even differently
        if cols % 2 == 1:  # odd
            fx_range = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / cols
        else:  # even
            fx_range = np.arange(-cols / 2, cols / 2) / cols

        if rows % 2 == 1:  # odd
            fy_range = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / rows
        else:  # even
            fy_range = np.arange(-rows / 2, rows / 2) / rows

        # Quadrant shift so DC is at [0,0]
        fx_range = ifftshift(fx_range)
        fy_range = ifftshift(fy_range)

        fx, fy = np.meshgrid(fx_range, fy_range)

        # Radial frequency
        freq = np.sqrt(fx ** 2 + fy ** 2)

        # For log-Gabor, need radius with DC=1 to avoid log(0)
        radius = freq.copy()
        radius[0, 0] = 1.0

        # Compute sintheta and costheta for angular filters (Julia gridangles)
        # Temporarily set freq DC to 1 to avoid divide by zero
        freq_safe = freq.copy()
        freq_safe[0, 0] = 1.0
        sintheta = fx / freq_safe
        costheta = fy / freq_safe

        # Restore DC values
        sintheta[0, 0] = 0.0
        costheta[0, 0] = 0.0

        return radius, sintheta, costheta, freq

    def _construct_log_gabor_filters(self, radius: np.ndarray) -> List[np.ndarray]:
        """Construct log-Gabor filters for each scale.

        Log-Gabor filters have Gaussian transfer functions on a logarithmic
        frequency scale, providing constant shape ratio across scales.

        Args:
            radius: Radial frequency grid.

        Returns:
            List of n_scale log-Gabor filter arrays.
        """
        log_gabor_list = []

        # Lowpass filter depends only on radius, not scale — compute once
        lowpass = 1.0 / (1.0 + (radius / 0.45) ** 30)

        for s in range(self.n_scale):
            wavelength = self.min_wavelength * (self.mult ** s)
            f0 = 1.0 / wavelength  # Center frequency

            # Log-Gabor transfer function
            with np.errstate(divide="ignore", invalid="ignore"):
                log_rad_over_f0 = np.log(radius / f0)

            log_gabor = np.exp(
                    -(log_rad_over_f0 ** 2) / (2 * np.log(self.sigma_onf) ** 2)
            )

            # Zero out DC component
            log_gabor[0, 0] = 0

            log_gabor_list.append(log_gabor * lowpass)

        return log_gabor_list

    def _compute_angular_spread(
            self, sintheta: np.ndarray, costheta: np.ndarray
    ) -> List[np.ndarray]:
        """Compute angular spreading functions using cosine filter (Julia reference).

        Uses cosineangularfilter from Julia ImagePhaseCongruency.jl which computes
        angular distance via atan2 of sin/cos differences for proper wrap-around.

        Args:
            sintheta: fx/freq grid from _construct_filter_grids.
            costheta: fy/freq grid from _construct_filter_grids.

        Returns:
            List of n_orient angular spread arrays.
        """
        angular_spread_list = []

        # Wavelength for cosine window function (Julia reference: 4*pi/norient)
        wavelen = 4.0 * np.pi / self.n_orient

        for o in range(self.n_orient):
            angle = o * np.pi / self.n_orient
            sinangl = np.sin(angle)
            cosangl = np.cos(angle)

            # Angular distance using sin/cos difference (Julia cosineangularfilter)
            # This handles wrap-around correctly via atan2
            ds = sintheta * cosangl - costheta * sinangl  # Difference in sine
            dc = costheta * cosangl + sintheta * sinangl  # Difference in cosine
            dtheta = np.abs(np.arctan2(ds, dc))  # Absolute angular distance

            # Scale theta for cosine window and clamp to pi
            dtheta = np.minimum(dtheta * 2.0 * np.pi / wavelen, np.pi)

            # Cosine window: (cos(dtheta) + 1) / 2 gives values in [0, 1]
            spread = (np.cos(dtheta) + 1.0) / 2.0

            angular_spread_list.append(spread)

        return angular_spread_list

    def _rayleigh_mode(self, amplitude: np.ndarray) -> float:
        """Estimate Rayleigh distribution parameter from amplitude data.

        For filter responses to Gaussian noise, amplitudes follow a Rayleigh
        distribution. The mode of a Rayleigh distribution equals sigma.

        Args:
            amplitude: Array of amplitude values.

        Returns:
            Estimated Rayleigh sigma parameter.
        """
        # Flatten and remove zeros
        amp_flat = amplitude.flatten()
        amp_flat = amp_flat[amp_flat > 0]

        if len(amp_flat) == 0:
            return 0.0

        # Histogram-based mode estimation
        # Match Julia: uses 50 bins
        n_bins = 50
        hist, bin_edges = np.histogram(amp_flat, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find mode (peak of histogram)
        mode_idx = np.argmax(hist)
        mode_value = bin_centers[mode_idx]

        # For Rayleigh distribution, mode = sigma
        return float(mode_value)
