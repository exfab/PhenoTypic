"""Dataclasses for image diagnostics panel descriptions and metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelDescription:
    """Description of a diagnostic panel for interpretation guidance.

    Each diagnostic panel in the figure has an associated description that helps
    users interpret what they're seeing and understand how it relates to their
    preprocessing pipeline.

    Attributes:
        label: Single letter panel identifier (e.g., "A", "B", "C").
        title: Short title for the panel (e.g., "Intensity Histogram").
        what_it_shows: Brief explanation of what the panel visualizes.
        how_to_read: Guidance on interpreting axes, colormaps, or patterns.
        good_values: What indicates good image quality (shown in green).
        poor_values: What indicates problems requiring attention (shown in red).
        pipeline_link: Suggested PhenoTypic operations to address issues.
    """

    label: str
    title: str
    what_it_shows: str
    how_to_read: str
    good_values: str
    poor_values: str
    pipeline_link: str


# Pre-defined panel descriptions for each diagnostic panel

PANEL_A_HISTOGRAM = PanelDescription(
    label="A",
    title="Intensity Histogram",
    what_it_shows="Distribution of pixel intensities in enhanced grayscale",
    how_to_read="X-axis: intensity (0-255), Y-axis: frequency. Blue=data, red=Gaussian fit",
    good_values="Bell-shaped, centered, uses full dynamic range",
    poor_values="Bimodal, clipped at edges, very narrow spread",
    pipeline_link="CLAHE, ContrastStretching, GammaCorrection",
)

PANEL_B_AUTOCORR = PanelDescription(
    label="B",
    title="Noise Autocorrelation",
    what_it_shows="Spatial correlation structure of noise (residual high-freq)",
    how_to_read="Center=max correlation. Width indicates noise correlation length",
    good_values="Tight central peak (uncorrelated noise, easy to filter)",
    poor_values="Wide/elongated peak (structured noise, harder to remove)",
    pipeline_link="GaussianBlur, BilateralDenoise, MedianFilter",
)

PANEL_C_PSD = PanelDescription(
    label="C",
    title="Power Spectral Density",
    what_it_shows="Frequency content on log-log scale (radially averaged)",
    how_to_read="X: spatial frequency, Y: power. Slope indicates noise type",
    good_values="Smooth rolloff, signal above noise floor at low freq",
    poor_values="Flat spectrum (white noise dominant), peaks (periodic artifacts)",
    pipeline_link="GaussianBlur (high-freq), RollingBallRemoveBG (low-freq)",
)

PANEL_D_ORIGINAL = PanelDescription(
    label="D",
    title="Enhanced Grayscale",
    what_it_shows="Current state of enh_gray used for detection",
    how_to_read="Grayscale rendering of preprocessed image",
    good_values="Clear colony/background separation, uniform background",
    poor_values="Low contrast, uneven illumination, visible noise",
    pipeline_link="(Reference panel - shows preprocessing result)",
)

PANEL_E_CONTRAST = PanelDescription(
    label="E",
    title="Local Contrast Map",
    what_it_shows="Weber contrast computed in local windows",
    how_to_read="Brighter=higher local contrast. Reveals edge strength",
    good_values="High values at colony boundaries, low in background",
    poor_values="Uniformly low (flat image) or noisy (texture dominates)",
    pipeline_link="CLAHE, UnsharpMask, LocalContrastEnhance",
)

PANEL_F_BARS = PanelDescription(
    label="F",
    title="Contrast Metrics",
    what_it_shows="Quantified contrast measures with quality thresholds",
    how_to_read="Bar height shows metric value. Colors: green=good, yellow=marginal, red=poor",
    good_values="All bars in green zone (RMS>0.05, Michelson>0.3)",
    poor_values="Bars in red zone indicate insufficient contrast",
    pipeline_link="CLAHE, ContrastStretching, HistogramEqualization",
)

PANEL_G_GRADIENT = PanelDescription(
    label="G",
    title="Gradient Magnitude",
    what_it_shows="Edge strength from Sobel gradient magnitude",
    how_to_read="Brighter=stronger edges. Reveals structure boundaries",
    good_values="Sharp colony edges, minimal background texture",
    poor_values="Weak edges, noisy background, double edges",
    pipeline_link="UnsharpMask, GaussianBlur (pre-smooth), CannyDetector",
)

PANEL_H_COHERENCE = PanelDescription(
    label="H",
    title="Orientation Coherence",
    what_it_shows="Local directional consistency from structure tensor",
    how_to_read="Brighter=stronger directional pattern (edges, ridges)",
    good_values="High at colony boundaries, low in uniform regions",
    poor_values="Uniformly high (texture) or low (no structure)",
    pipeline_link="CoherenceEnhancingDiffusion, DirectionalFilter",
)

PANEL_I_RIDGE = PanelDescription(
    label="I",
    title="Multiscale Ridge Response",
    what_it_shows="Ridge detection response vs. scale (sigma)",
    how_to_read="Peak shows optimal scale for tubular structures",
    good_values="Clear peak at colony-relevant scale (1-5 px)",
    poor_values="No peak (no ridges) or peak at wrong scale",
    pipeline_link="MeijeringRidgeFilter, FrangiVesselness, SatoRidgeFilter",
)

PANEL_J_BACKGROUND = PanelDescription(
    label="J",
    title="Background Estimate",
    what_it_shows="Large-scale intensity variation (Gaussian smoothed)",
    how_to_read="Should be uniform if illumination is even",
    good_values="Flat/uniform intensity across field",
    poor_values="Gradients, vignetting, uneven illumination",
    pipeline_link="RollingBallRemoveBG, GaussianSubtract, FlatFieldCorrection",
)

PANEL_K_VARIANCE = PanelDescription(
    label="K",
    title="Local Variance Map",
    what_it_shows="Windowed variance showing local activity (log scale)",
    how_to_read="Brighter=higher variance. Reveals texture and noise",
    good_values="High in colonies, low in background",
    poor_values="High everywhere (noisy) or uniform (no features)",
    pipeline_link="BilateralDenoise, MedianFilter, VarianceFilter",
)


__all__ = [
    "PanelDescription",
    "PANEL_A_HISTOGRAM",
    "PANEL_B_AUTOCORR",
    "PANEL_C_PSD",
    "PANEL_D_ORIGINAL",
    "PANEL_E_CONTRAST",
    "PANEL_F_BARS",
    "PANEL_G_GRADIENT",
    "PANEL_H_COHERENCE",
    "PANEL_I_RIDGE",
    "PANEL_J_BACKGROUND",
    "PANEL_K_VARIANCE",
]
