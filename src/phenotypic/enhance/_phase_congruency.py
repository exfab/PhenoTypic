"""Phase congruency enhancement for contrast-invariant edge detection.

Implementation follows Kovesi's phasecong3 algorithm using oriented log-Gabor wavelets.
Algorithm details from ImagePhaseCongruency.jl (Julia reference implementation).

References:
    Peter Kovesi's ImagePhaseCongruency.jl: https://github.com/peterkovesi/ImagePhaseCongruency.jl
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from ..abc_ import ImageEnhancer

if TYPE_CHECKING:
    from phenotypic import Image


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


class PhaseCongruencyEnhancer(ImageEnhancer):
    """Phase congruency enhancement for contrast-invariant colony edge detection.

    Phase congruency is a dimensionless measure of local feature significance based
    on the Local Energy Model. Features are detected where Fourier components are
    maximally in phase, regardless of their amplitude. This makes phase congruency
    invariant to image contrast and illumination changes - ideal for colony plates
    with uneven lighting or varying colony opacity.

    Use cases (agar plates):
    - Detecting colony boundaries independent of colony color/opacity
    - Processing images with uneven illumination or scanner vignetting
    - Enhancing faint colony edges that gradient-based methods miss
    - Preprocessing before adaptive thresholding for robust segmentation
    - Analyzing translucent or low-contrast colonies on agar

    Tuning and effects:
    - n_scale: Number of wavelet scales. More scales capture wider range of feature
      sizes. 3-4 for fine features, 5-6 for broader range. Higher values increase
      computation time.
    - n_orient: Number of filter orientations. 6 gives 30 degree spacing with good
      angular coverage. 4 is faster but may miss diagonal edges.
    - min_wavelength: Wavelength of smallest scale filter in pixels. Should match
      minimum expected colony edge width. 3.0 works for most colony imaging.
    - mult: Scaling factor between successive wavelengths. Controls spectral overlap.
      2.1 provides good coverage; smaller values give finer frequency resolution.
    - sigma_onf: Log-Gabor bandwidth parameter. 0.55 gives ~2 octave bandwidth with
      good frequency localization. 0.75 gives ~1 octave (narrower).
    - k: Number of noise standard deviations for threshold. Higher values (5-20)
      increase noise rejection but may miss faint colony edges.
    - noise_method: -1 for median-based estimation (robust), -2 for mode-based
      (Rayleigh distribution), or fixed value >= 0 for known noise levels.
    - output: Which result to use for enhancement. "pc_sum" (default) gives scalar
      phase congruency, "M" gives edge strength, "m" gives corner strength.

    Caveats:
    - Computationally intensive: FFT-based processing scales as O(N log N) per
      scale-orientation pair. For large images, consider downsampling first.
    - Memory usage: Stores filter responses for all scale-orientation combinations.
      For very large images, reduce n_scale or n_orient.
    - Output range: Phase congruency values are typically in [0, 1] but may exceed
      1 in high-contrast regions. Values are clipped to [0, 1] for detect_mat.
    - Not suitable for texture analysis: Designed for edge/line detection, not
      for characterizing surface texture or colony interior patterns.

    Attributes:
        n_scale: Number of wavelet scales (default 4).
        n_orient: Number of filter orientations (default 6).
        min_wavelength: Smallest scale wavelength in pixels (default 3.0).
        mult: Scaling factor between successive filters (default 2.1).
        sigma_onf: Log-Gabor bandwidth parameter (default 0.55).
        k: Noise threshold multiplier (default 2.0).
        cutoff: Frequency spread penalty threshold (default 0.5).
        g: Sigmoid sharpness for frequency spread weighting (default 10.0).
        noise_method: Noise estimation method (default -1 for median).
        output: Which result to store in detect_mat (default "pc_sum").

    Examples:
        Basic phase congruency enhancement:

        >>> from phenotypic.enhance import PhaseCongruencyEnhancer
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> enhancer = PhaseCongruencyEnhancer()
        >>> enhanced = enhancer.apply(image)
        >>> # Detection matrix now contains phase congruency map
        >>> enhanced.detect_mat[:].min() >= 0
        True

        Edge-focused enhancement for segmentation pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import PhaseCongruencyEnhancer, GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.0),  # Light denoising first
        ...     PhaseCongruencyEnhancer(output="M", k=3.0),  # Edge map
        ...     OtsuDetector()
        ... ])
        >>> result = pipeline.apply(image)

        High noise tolerance for grainy images:

        >>> from phenotypic.enhance import PhaseCongruencyEnhancer
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> # Increase k for aggressive noise rejection
        >>> enhancer = PhaseCongruencyEnhancer(k=10.0, noise_method=-2)
        >>> enhanced = enhancer.apply(image)
    """

    def __init__(
            self,
            n_scale: int = 4,
            n_orient: int = 6,
            min_wavelength: float = 3.0,
            mult: float = 2.1,
            sigma_onf: float = 0.55,
            k: float = 2.0,
            cutoff: float = 0.5,
            g: float = 10.0,
            noise_method: float = -1,
            output: Literal["M", "m", "pc_sum"] = "pc_sum",
    ):
        """Initialize phase congruency enhancer.

        Args:
            n_scale: Number of wavelet scales. Range [3, 6] typical.
            n_orient: Number of filter orientations. 6 gives 30 degree spacing.
            min_wavelength: Wavelength of smallest scale filter in pixels.
                Should match minimum expected feature width (default 3.0).
            mult: Scaling factor between successive filter wavelengths.
                Controls spectral overlap between scales (default 2.1).
            sigma_onf: Ratio of Gaussian standard deviation to filter center
                frequency. Controls filter bandwidth. 0.55 gives ~2 octave
                bandwidth; 0.75 gives ~1 octave (default 0.55).
            k: Number of noise standard deviations for threshold. Higher values
                increase noise rejection (default 2.0, range [2, 20]).
            cutoff: Frequency spread measure below which PC values are penalized
                (default 0.5).
            g: Sharpness of sigmoid transition for frequency spread weighting
                (default 10.0).
            noise_method: Method for noise statistics estimation. -1 uses median
                of smallest scale responses (default), -2 uses mode (Rayleigh),
                values >= 0 are used as fixed noise threshold.
            output: Which result to store in detect_mat. "pc_sum" for scalar phase
                congruency (default), "M" for edge strength, "m" for corners.
        """
        super().__init__()

        # Validate parameters
        if n_scale < 1:
            raise ValueError(f"n_scale must be >= 1, got {n_scale}")
        if n_orient < 1:
            raise ValueError(f"n_orient must be >= 1, got {n_orient}")
        if min_wavelength < 2:
            raise ValueError(f"min_wavelength must be >= 2, got {min_wavelength}")
        if mult <= 1:
            raise ValueError(f"mult must be > 1, got {mult}")
        if not 0.1 <= sigma_onf <= 1.0:
            raise ValueError(f"sigma_onf must be in [0.1, 1.0], got {sigma_onf}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if not 0 < cutoff < 1:
            raise ValueError(f"cutoff must be in (0, 1), got {cutoff}")
        if g <= 0:
            raise ValueError(f"g must be > 0, got {g}")
        if output not in ("M", "m", "pc_sum"):
            raise ValueError(f"output must be 'M', 'm', or 'pc_sum', got {output!r}")

        self.n_scale = n_scale
        self.n_orient = n_orient
        self.min_wavelength = float(min_wavelength)
        self.mult = float(mult)
        self.sigma_onf = float(sigma_onf)
        self.k = float(k)
        self.cutoff = float(cutoff)
        self.g = float(g)
        self.noise_method = float(noise_method)
        self.output = output

    def _operate(self, image: Image) -> Image:
        """Apply phase congruency enhancement to the detection matrix channel."""
        result = self._phasecong3(image.detect_mat[:])

        # Select output based on configuration
        output_map = {"M": result.M, "m": result.m, "pc_sum": result.pc_sum}
        selected = output_map[self.output]

        # Ensure output is in [0, 1] range for detect_mat compatibility
        image.detect_mat[:] = np.clip(selected, 0.0, 1.0).astype(np.float64)
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
        pc = np.zeros((self.n_orient, rows, cols), dtype=np.float64)
        cov_x2 = np.zeros((rows, cols), dtype=np.float64)
        cov_y2 = np.zeros((rows, cols), dtype=np.float64)
        cov_xy = np.zeros((rows, cols), dtype=np.float64)
        energy_v = np.zeros((rows, cols, 3), dtype=np.float64)
        pc_sum = np.zeros((rows, cols), dtype=np.float64)

        # Storage for filter responses
        EO: List[List[np.ndarray]] = [
            [np.array([]) for _ in range(self.n_orient)] for _ in range(self.n_scale)
        ]

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

                # Transform back to spatial domain
                EO[s][o] = ifft2(filtered_fft)

                # Extract even (real) and odd (imaginary) symmetric responses
                amplitude = np.abs(EO[s][o])

                # Accumulate responses
                sum_even += np.real(EO[s][o])
                sum_odd += np.imag(EO[s][o])
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
            # Energy = sum over scales of: E*MeanE + O*MeanO - |E*MeanO - O*MeanE|
            energy = np.zeros((rows, cols), dtype=np.float64)
            for s in range(self.n_scale):
                E = np.real(EO[s][o])
                O = np.imag(EO[s][o])
                energy += (
                        E * mean_even + O * mean_odd - np.abs(
                    E * mean_odd - O * mean_even)
                )

            # Accumulate energy vectors for orientation/feature_type (Julia reference)
            energy_v[:, :, 0] += sum_even
            energy_v[:, :, 1] += np.cos(angle) * sum_odd
            energy_v[:, :, 2] += np.sin(angle) * sum_odd

            # Frequency spread weighting (Julia reference)
            # Width measures how spread out the frequency responses are
            width = (sum_amplitude / (max_amplitude + epsilon) - 1) / (self.n_scale - 1)
            weight = 1.0 / (1.0 + np.exp((self.cutoff - width) * self.g))

            # Phase congruency for this orientation
            pc[o] = weight * np.maximum(energy - T, 0) / (sum_amplitude + epsilon)

            # Accumulate covariance tensor components
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            pc_sum += pc[o]

            # Square pc[o] for covariance (matches Julia PCo^2)
            pc_sq = pc[o] ** 2
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

            # Apply lowpass filter to remove high frequency aliasing
            lowpass_cutoff = 0.45
            lowpass_order = 15
            lowpass = 1.0 / (
                    1.0 + (radius / lowpass_cutoff) ** (2 * lowpass_order)
            )

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
