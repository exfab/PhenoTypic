"""Image quality metrics for colony detection analysis.

This module provides comprehensive image quality assessment tools for
evaluating detection matrices before colony detection. Metrics include
noise estimation, contrast analysis, structural coherence, and
background uniformity assessment.

The :class:`ImageMetricsCalculator` computes metrics used by both the
matplotlib-based :class:`DiagnosticsPlotter` and the Panel-based
:class:`DiagnosticsDashboard`.

Examples:
    >>> from phenotypic.data import load_synth_yeast_plate
    >>> from phenotypic.util.image_metrics import ImageMetricsCalculator
    >>> image = load_synth_yeast_plate()
    >>> calc = ImageMetricsCalculator(image.detect_mat[:])
    >>> noise = calc.compute_noise_metrics()
    >>> print(f"SNR: {noise['snr']:.2f}")
    SNR: ...
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np
from scipy import fft as scipy_fft
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import frangi, hessian, meijering

# Global thresholds for metric interpretation
THRESHOLDS = {
    "snr": {"critical": 3.0, "marginal": 7.0},
    "rms_contrast": {"critical": 0.02, "marginal": 0.05},
    "coherence": {"critical": 0.15, "marginal": 0.30},
    "nonuniformity": {"critical": 0.50, "marginal": 0.20},
}


class NoiseMetrics(TypedDict):
    """Noise-related metrics from detection matrix analysis."""

    snr: float
    sigma_mad: float
    correlation_length: float


class ContrastMetrics(TypedDict):
    """Contrast-related metrics from detection matrix analysis."""

    rms_contrast: float
    michelson: float
    dynamic_range: float
    p1: float
    p99: float


class StructureMetrics(TypedDict):
    """Structure tensor and ridge detection metrics."""

    mean_coherence: float
    optimal_scale: float
    peak_response: float
    ridge_responses: list[float]
    scales: list[float]
    ridge_method: str
    coherence_map: np.ndarray | None  # Non-serializable


class BackgroundMetrics(TypedDict):
    """Background uniformity metrics."""

    nonuniformity_ratio: float
    mean_gradient: float
    background_estimate: np.ndarray | None  # Non-serializable


class QualityScores(TypedDict):
    """Normalized 0-1 quality scores for radar chart visualization."""

    SNR: float
    Contrast: float
    Coherence: float
    Uniformity: float
    Sharpness: float


class ImageMetricsCalculator:
    """Computes image quality metrics from detection matrices.

    This class extracts noise, contrast, structure, and background
    metrics used to assess image quality for colony detection.
    It provides the computational core shared by both matplotlib-based
    and Panel-based diagnostics visualizations.

    Args:
        detect_mat: 2D grayscale detection matrix (typically ``image.detect_mat[:]``).

    Attributes:
        bit_depth: Image bit depth (8 or 16) inferred from max intensity.

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.util.image_metrics import ImageMetricsCalculator
        >>> image = load_synth_yeast_plate()
        >>> calc = ImageMetricsCalculator(image.detect_mat[:])
        >>> noise = calc.compute_noise_metrics()
        >>> print(f"SNR: {noise['snr']:.2f}")
        SNR: ...
        >>> contrast = calc.compute_contrast_metrics()
        >>> print(f"RMS contrast: {contrast['rms_contrast']:.3f}")
        RMS contrast: ...
    """

    def __init__(self, detect_mat: np.ndarray) -> None:
        self._detect_mat = detect_mat.astype(np.float64)
        self._max_intensity = 255.0 if detect_mat.max() <= 255 else 65535.0

    @property
    def bit_depth(self) -> int:
        """Image bit depth (8 or 16) inferred from max intensity."""
        return 8 if self._max_intensity == 255.0 else 16

    @property
    def max_intensity(self) -> float:
        """Maximum intensity value based on image bit depth."""
        return self._max_intensity

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _compute_autocorrelation(self, img: np.ndarray) -> np.ndarray:
        """Compute 2D autocorrelation using FFT.

        Args:
            img: Input image array.

        Returns:
            2D autocorrelation array (cropped to central region).
        """
        # Pad to avoid circular correlation artifacts
        padded = np.pad(img - np.mean(img), ((0, img.shape[0]), (0, img.shape[1])))

        # FFT-based autocorrelation
        f = scipy_fft.fft2(padded)
        power = np.abs(f) ** 2
        autocorr = np.real(scipy_fft.ifft2(power))

        # Shift zero-lag to center and crop
        autocorr = scipy_fft.fftshift(autocorr)
        h, w = img.shape
        ch, cw = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        crop_size = min(50, h // 4, w // 4)
        autocorr_cropped = autocorr[
            ch - crop_size : ch + crop_size, cw - crop_size : cw + crop_size
        ]

        return autocorr_cropped

    def compute_psd(self, img: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Compute radially averaged power spectral density.

        Args:
            img: Input image array. If None, uses the stored detection matrix.

        Returns:
            Tuple of (frequencies, radial_psd).
        """
        if img is None:
            img = self._detect_mat

        # Compute 2D FFT
        f = scipy_fft.fft2(img - np.mean(img))
        f_shifted = scipy_fft.fftshift(f)
        psd_2d = np.abs(f_shifted) ** 2

        # Radial averaging
        h, w = psd_2d.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        # Bin by radius
        max_r = min(cx, cy)
        radial_sum = np.bincount(r.ravel(), weights=psd_2d.ravel())
        radial_count = np.bincount(r.ravel())
        radial_psd = radial_sum[: max_r + 1] / (radial_count[: max_r + 1] + 1e-10)

        # Frequency axis (normalized)
        freqs = np.arange(len(radial_psd)) / len(radial_psd)

        return freqs, radial_psd

    def compute_local_contrast(
        self, img: np.ndarray | None = None, window_size: int = 15
    ) -> np.ndarray:
        """Compute local Weber contrast map.

        Args:
            img: Input image array. If None, uses the stored detection matrix.
            window_size: Size of local window.

        Returns:
            Local contrast map.
        """
        if img is None:
            img = self._detect_mat

        img_float = img.astype(np.float64)

        # Local mean
        local_mean = uniform_filter(img_float, size=window_size)

        # Local contrast (Weber-like)
        contrast = np.abs(img_float - local_mean) / (local_mean + 1e-10)

        return contrast

    def compute_local_variance(
        self, img: np.ndarray | None = None, window_size: int = 15
    ) -> np.ndarray:
        """Compute local variance map.

        Args:
            img: Input image array. If None, uses the stored detection matrix.
            window_size: Size of local window.

        Returns:
            Local variance map.
        """
        if img is None:
            img = self._detect_mat

        img_float = img.astype(np.float64)

        # Local mean and mean of squares
        local_mean = uniform_filter(img_float, size=window_size)
        local_mean_sq = uniform_filter(img_float**2, size=window_size)

        # Variance = E[X^2] - E[X]^2
        variance = np.maximum(local_mean_sq - local_mean**2, 0)

        return variance

    # =========================================================================
    # METRIC COMPUTATION METHODS
    # =========================================================================

    def compute_noise_metrics(self) -> NoiseMetrics:
        """Compute noise-related metrics (parameter-free).

        Returns:
            Dictionary with SNR, sigma_mad, and correlation_length.
        """
        img = self._detect_mat

        # Estimate noise using high-pass residual (more robust than Laplacian MAD)
        # For synthetic images with large uniform regions, Laplacian MAD can be 0
        smoothed = gaussian_filter(img, sigma=2.0)
        residual = img - smoothed

        # Use robust MAD estimator, falling back to std if MAD is too small
        mad_val = np.median(np.abs(residual - np.median(residual)))
        sigma_mad = mad_val / 0.6745 if mad_val > 1e-10 else np.std(residual)

        # Ensure minimum noise estimate (prevents division by near-zero)
        sigma_mad = max(sigma_mad, 1e-6)

        # Signal estimation (mean of image)
        signal_mean = np.mean(img)

        # SNR calculation
        snr = signal_mean / sigma_mad

        # Correlation length from autocorrelation
        # Compute half-width at half-maximum
        autocorr = self._compute_autocorrelation(img)
        center_h = autocorr.shape[0] // 2

        # Extract horizontal profile through center
        profile = autocorr[center_h, :]
        profile_norm = profile / (profile.max() + 1e-10)

        # Find half-width at half-maximum
        half_max_indices = np.where(profile_norm >= 0.5)[0]
        if len(half_max_indices) > 1:
            correlation_length = (half_max_indices[-1] - half_max_indices[0]) / 2.0
        else:
            correlation_length = 1.0

        return NoiseMetrics(
            snr=float(snr),
            sigma_mad=float(sigma_mad),
            correlation_length=float(correlation_length),
        )

    def compute_contrast_metrics(self) -> ContrastMetrics:
        """Compute contrast-related metrics (parameter-free).

        Returns:
            Dictionary with rms_contrast, michelson, dynamic_range, p1, and p99.
        """
        img = self._detect_mat

        # RMS contrast (std / mean) - bit-depth agnostic
        mean_val = np.mean(img)
        std_val = np.std(img)
        rms_contrast = std_val / (mean_val + 1e-10)

        # Michelson contrast using percentiles (more robust than min/max)
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        michelson = (p99 - p1) / (p99 + p1 + 1e-10)

        # Dynamic range (normalized by max possible intensity)
        min_val = img.min()
        max_val = img.max()
        dynamic_range = (max_val - min_val) / self._max_intensity

        return ContrastMetrics(
            rms_contrast=float(rms_contrast),
            michelson=float(michelson),
            dynamic_range=float(dynamic_range),
            p1=float(p1),
            p99=float(p99),
        )

    def compute_structure_metrics(
        self,
        sigma: float = 1.5,
        scales: list[float] | None = None,
        ridge_method: Literal["meijering", "frangi", "hessian"] = "meijering",
    ) -> StructureMetrics:
        """Compute structure tensor and ridge detection metrics.

        Args:
            sigma: Sigma for structure tensor computation.
            scales: List of scales for multiscale ridge detection. Defaults to
                [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0].
            ridge_method: Method for ridge detection: "meijering" (default),
                "frangi", or "hessian".

        Returns:
            Dictionary with mean_coherence, optimal_scale, peak_response,
            ridge_responses, scales, ridge_method, and coherence_map.
        """
        if scales is None:
            scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

        img = self._detect_mat / self._max_intensity

        # Compute structure tensor and coherence
        A_elems = structure_tensor(img, sigma=sigma)
        l1, l2 = structure_tensor_eigenvalues(A_elems)

        # Coherence: (l1 - l2) / (l1 + l2 + eps)
        coherence = (l1 - l2) / (l1 + l2 + 1e-10)
        mean_coherence = float(np.mean(coherence))

        # Multiscale ridge detection
        ridge_responses = []
        for scale in scales:
            if ridge_method == "meijering":
                response = meijering(img, sigmas=[scale], black_ridges=False)
            elif ridge_method == "frangi":
                response = frangi(img, sigmas=[scale], black_ridges=False)
            else:  # hessian
                response = hessian(img, sigmas=[scale], black_ridges=False)

            ridge_responses.append(float(np.mean(response)))

        # Find optimal scale
        if ridge_responses:
            optimal_idx = np.argmax(ridge_responses)
            optimal_scale = scales[optimal_idx]
            peak_response = ridge_responses[optimal_idx]
        else:
            optimal_scale = scales[0] if scales else 1.0
            peak_response = 0.0

        return StructureMetrics(
            mean_coherence=mean_coherence,
            optimal_scale=float(optimal_scale),
            peak_response=float(peak_response),
            ridge_responses=ridge_responses,
            scales=scales,
            ridge_method=ridge_method,
            coherence_map=coherence,
        )

    def compute_background_metrics(self, sigma: float = 50.0) -> BackgroundMetrics:
        """Compute background uniformity metrics.

        Args:
            sigma: Sigma for background estimation (Gaussian smoothing).

        Returns:
            Dictionary with nonuniformity_ratio, mean_gradient, and
            background_estimate.
        """
        img = self._detect_mat

        # Background estimate via large-sigma Gaussian smoothing
        background = gaussian_filter(img, sigma=sigma)

        # Nonuniformity ratio (std / mean of background)
        bg_mean = np.mean(background)
        bg_std = np.std(background)
        nonuniformity_ratio = bg_std / (bg_mean + 1e-10)

        # Mean gradient magnitude of background
        grad_x = np.gradient(background, axis=1)
        grad_y = np.gradient(background, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = float(np.mean(grad_mag))

        return BackgroundMetrics(
            nonuniformity_ratio=float(nonuniformity_ratio),
            mean_gradient=mean_gradient,
            background_estimate=background,
        )

    def compute_quality_scores(
        self,
        noise: NoiseMetrics,
        contrast: ContrastMetrics,
        structure: StructureMetrics,
        background: BackgroundMetrics,
    ) -> QualityScores:
        """Compute normalized 0-1 quality scores for radar chart.

        Args:
            noise: Noise metrics from compute_noise_metrics().
            contrast: Contrast metrics from compute_contrast_metrics().
            structure: Structure metrics from compute_structure_metrics().
            background: Background metrics from compute_background_metrics().

        Returns:
            Dictionary of quality scores (SNR, Contrast, Coherence, Uniformity,
            Sharpness), all normalized to [0, 1].
        """
        # SNR score (saturates at 15)
        snr_score = min(noise["snr"] / 15.0, 1.0)

        # Contrast score (RMS contrast, saturates at 0.15)
        contrast_score = min(contrast["rms_contrast"] / 0.15, 1.0)

        # Coherence score (already 0-1)
        coherence_score = min(structure["mean_coherence"], 1.0)

        # Uniformity score (inverse of nonuniformity)
        uniformity_score = max(1.0 - background["nonuniformity_ratio"] * 2, 0.0)

        # Sharpness score (based on peak ridge response as proxy)
        sharpness_score = min(structure["peak_response"] * 5, 1.0)

        return QualityScores(
            SNR=snr_score,
            Contrast=contrast_score,
            Coherence=coherence_score,
            Uniformity=uniformity_score,
            Sharpness=sharpness_score,
        )

    # =========================================================================
    # INTERPRETATION AND RECOMMENDATIONS
    # =========================================================================

    @staticmethod
    def generate_interpretation(
        section: Literal["noise", "contrast", "structure", "background"],
        metrics: dict[str, Any],
    ) -> str:
        """Generate human-readable interpretation text.

        Args:
            section: Section name ("noise", "contrast", "structure", "background").
            metrics: Computed metrics for the section.

        Returns:
            Interpretation text string.
        """
        if section == "noise":
            snr = metrics["snr"]
            if snr < THRESHOLDS["snr"]["critical"]:
                quality = "critically low"
                action = "Strong denoising required (BilateralDenoise, MedianFilter)."
            elif snr < THRESHOLDS["snr"]["marginal"]:
                quality = "marginal"
                action = "Light denoising recommended (GaussianBlur sigma=0.5-1.0)."
            else:
                quality = "adequate"
                action = "No denoising needed."
            return f"SNR ({snr:.1f}) is {quality} for reliable detection. {action}"

        elif section == "contrast":
            rms = metrics["rms_contrast"]
            if rms < THRESHOLDS["rms_contrast"]["critical"]:
                quality = "critically low"
                action = "Strong contrast enhancement required (CLAHE, HistogramEqualization)."
            elif rms < THRESHOLDS["rms_contrast"]["marginal"]:
                quality = "marginal"
                action = "Contrast enhancement recommended (CLAHE clip_limit=2.0)."
            else:
                quality = "adequate"
                action = "Contrast is sufficient for detection."
            return f"RMS contrast ({rms:.3f}) is {quality}. {action}"

        elif section == "structure":
            coherence = metrics["mean_coherence"]
            if coherence < THRESHOLDS["coherence"]["critical"]:
                quality = "low"
                desc = "weak directional structure"
            elif coherence < THRESHOLDS["coherence"]["marginal"]:
                quality = "moderate"
                desc = "some linear features present"
            else:
                quality = "high"
                desc = "strong linear/edge structure"
            opt_scale = metrics["optimal_scale"]
            return (
                f"Mean coherence ({coherence:.3f}) indicates {desc}. "
                f"Optimal ridge scale: sigma={opt_scale:.1f} px."
            )

        elif section == "background":
            nonunif = metrics["nonuniformity_ratio"]
            if nonunif > THRESHOLDS["nonuniformity"]["critical"]:
                quality = "severe"
                action = "Background correction critical (RollingBallRemoveBG, FlatFieldCorrection)."
            elif nonunif > THRESHOLDS["nonuniformity"]["marginal"]:
                quality = "moderate"
                action = "Background correction recommended."
            else:
                quality = "acceptably low"
                action = "Background is sufficiently uniform."
            return f"Background non-uniformity ({nonunif:.1%}) is {quality}. {action}"

        return ""

    @staticmethod
    def generate_recommendations(
        noise: NoiseMetrics,
        contrast: ContrastMetrics,
        structure: StructureMetrics,
        background: BackgroundMetrics,
    ) -> list[str]:
        """Generate actionable preprocessing recommendations.

        Args:
            noise: Noise metrics from compute_noise_metrics().
            contrast: Contrast metrics from compute_contrast_metrics().
            structure: Structure metrics from compute_structure_metrics().
            background: Background metrics from compute_background_metrics().

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Noise recommendations
        snr = noise["snr"]
        corr_len = noise["correlation_length"]
        if snr < THRESHOLDS["snr"]["critical"]:
            recommendations.append(
                "Apply strong denoising: BilateralDenoise(sigma_spatial=3, sigma_intensity=0.1)"
            )
        elif snr < THRESHOLDS["snr"]["marginal"]:
            recommendations.append("Apply light denoising: GaussianBlur(sigma=0.5-1.0)")

        if corr_len > 5:
            recommendations.append(
                f"Structured noise detected (xi={corr_len:.1f}px). "
                "Consider MedianFilter or morphological opening."
            )

        # Contrast recommendations
        rms = contrast["rms_contrast"]
        dynamic = contrast["dynamic_range"]
        if rms < THRESHOLDS["rms_contrast"]["critical"]:
            recommendations.append(
                "Apply contrast enhancement: CLAHE(clip_limit=3.0) or HistogramEqualization"
            )
        elif rms < THRESHOLDS["rms_contrast"]["marginal"]:
            recommendations.append("Consider CLAHE(clip_limit=2.0) for improved contrast")

        if dynamic < 0.3:
            recommendations.append(
                "Low dynamic range detected. Consider ContrastStretching or exposure adjustment."
            )

        # Structure recommendations
        opt_scale = structure["optimal_scale"]
        recommendations.append(
            f"Use sigma_range=[{opt_scale*0.7:.1f}, {opt_scale*1.5:.1f}] for multiscale structure detection"
        )

        # Background recommendations
        nonunif = background["nonuniformity_ratio"]
        if nonunif > THRESHOLDS["nonuniformity"]["critical"]:
            recommendations.append(
                "Apply background correction: RollingBallRemoveBG(radius=50) or FlatFieldCorrection"
            )
        elif nonunif > THRESHOLDS["nonuniformity"]["marginal"]:
            recommendations.append(
                "Consider GaussianSubtract(sigma=50) for background uniformity"
            )

        return recommendations

    # =========================================================================
    # CONVENIENCE METHOD
    # =========================================================================

    def compute_all(
        self,
        structure_sigma: float = 1.5,
        ridge_scales: list[float] | None = None,
        ridge_method: Literal["meijering", "frangi", "hessian"] = "meijering",
        background_sigma: float = 50.0,
        include_non_serializable: bool = False,
    ) -> dict[str, Any]:
        """Compute all metrics and return comprehensive dict.

        Args:
            structure_sigma: Sigma for structure tensor computation.
            ridge_scales: Scales for ridge detection.
            ridge_method: Ridge detection algorithm.
            background_sigma: Sigma for background estimation.
            include_non_serializable: If False, removes numpy arrays from output.

        Returns:
            Dict with keys: bit_depth, noise, contrast, structure, background,
            quality_scores, interpretations, recommendations.
        """
        # Compute all metrics
        noise = self.compute_noise_metrics()
        contrast = self.compute_contrast_metrics()
        structure = self.compute_structure_metrics(
            sigma=structure_sigma,
            scales=ridge_scales,
            ridge_method=ridge_method,
        )
        background = self.compute_background_metrics(sigma=background_sigma)

        # Compute quality scores
        quality_scores = self.compute_quality_scores(noise, contrast, structure, background)

        # Generate interpretations
        interpretations = {
            "noise": self.generate_interpretation("noise", noise),
            "contrast": self.generate_interpretation("contrast", contrast),
            "structure": self.generate_interpretation("structure", structure),
            "background": self.generate_interpretation("background", background),
        }

        # Generate recommendations
        recommendations = self.generate_recommendations(
            noise, contrast, structure, background
        )

        # Clean non-serializable items if requested
        if include_non_serializable:
            structure_out = dict(structure)
            background_out = dict(background)
        else:
            structure_out = {k: v for k, v in structure.items() if k != "coherence_map"}
            background_out = {
                k: v for k, v in background.items() if k != "background_estimate"
            }

        return {
            "bit_depth": self.bit_depth,
            "noise": dict(noise),
            "contrast": dict(contrast),
            "structure": structure_out,
            "background": background_out,
            "quality_scores": dict(quality_scores),
            "interpretations": interpretations,
            "recommendations": recommendations,
        }


__all__ = [
    "ImageMetricsCalculator",
    "THRESHOLDS",
    "NoiseMetrics",
    "ContrastMetrics",
    "StructureMetrics",
    "BackgroundMetrics",
    "QualityScores",
]
