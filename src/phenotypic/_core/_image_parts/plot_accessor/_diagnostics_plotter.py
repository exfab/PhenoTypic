"""Comprehensive image quality diagnostics plotter for preprocessing pipeline development."""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from scipy import fft as scipy_fft
from scipy.stats import norm
from skimage.filters import sobel

from phenotypic.abc_.plotting import Control, PhtPlot, figure
from phenotypic.util.image_metrics import ImageMetricsCalculator, THRESHOLDS
from phenotypic.sdk_.viz.figures._theme import NAVY, OKABE_ITO

from ._base_plotter import BasePlotter
from ._diagnostics_types import (
    PANEL_B_AUTOCORR,
    PANEL_C_PSD,
    PANEL_D_ORIGINAL,
    PANEL_E_CONTRAST,
    PANEL_F_BARS,
    PANEL_G_GRADIENT,
    PANEL_H_COHERENCE,
    PANEL_I_RIDGE,
    PANEL_J_BACKGROUND,
    PANEL_K_VARIANCE,
    PanelDescription,
)

# Module-level Control constants — defined ONCE and shared BY IDENTITY across
# the @figure methods that reference them, so the dashboard renders one widget
# per logical knob rather than a duplicate per figure.
STRUCTURE_SIGMA = Control(
    label="Structure σ", kind="float", default=1.5, bounds=(0.5, 5.0), step=0.1
)
RIDGE_METHOD = Control(
    label="Ridge method",
    kind="select",
    default="meijering",
    options=("meijering", "frangi", "hessian"),
)
BACKGROUND_SIGMA = Control(
    label="Background σ", kind="float", default=50.0, bounds=(10.0, 150.0), step=5.0
)

# Multiscale ridge detection scales are held FIXED (a list does not fit the
# float/select/bool/text Control kinds, so ``ridge_scales`` is intentionally
# not exposed as a control). This mirrors the matplotlib ``diagnostics()``
# default and is reused by every structure/summary figure below.
DEFAULT_RIDGE_SCALES: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

# Quality-threshold trace colors carried over verbatim from the matplotlib
# panels (these are explicit per-trace data colors, not theme styling, so they
# are kept rather than delegated to the template).
_POOR_COLOR = "#c0392b"  # red — below critical threshold
_MARGINAL_COLOR = "#f39c12"  # yellow — between critical and marginal
_GOOD_COLOR = "#27ae60"  # green — above marginal threshold
#: Border color for the white annotation callout boxes.
AXIS_ANNOTATION_BORDER = "#cccccc"


class DiagnosticsPlotter(BasePlotter, PhtPlot):
    """Generates comprehensive image quality diagnostics for preprocessing pipeline development.

    This class provides multi-panel figures analyzing noise, contrast, structure, and
    background characteristics of images. It computes quantitative metrics and provides
    data-driven recommendations for preprocessing operations.

    All metrics are computed from the detection matrix (detect_mat), which reflects
    the current state of preprocessing applied to the image.

    The metrics computation is delegated to :class:`ImageMetricsCalculator` from
    ``phenotypic.util.image_metrics``, shared by the static matplotlib figure and
    the standalone ``PlotDiagnostics`` report.

    Dual renderer:
        The static matplotlib :meth:`diagnostics` (returning
        ``(matplotlib Figure, metrics dict)``) is preserved unchanged. In
        addition, this class mixes in :class:`~phenotypic.abc_.plotting.PhtPlot`
        and declares a parallel set of interactive Plotly ``@figure`` methods
        (named ``fig_*``) that surface the uniform ``.inspect()`` / ``.report()``
        / dashboard contract. The Plotly figures mirror the matplotlib panels
        one-for-one; control-bearing figures (structure σ, ridge method,
        background σ) recompute on demand from a fresh
        :class:`ImageMetricsCalculator`.

    Examples:
        Generate full diagnostics report:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> plotter = DiagnosticsPlotter(image)
        >>> fig, metrics = plotter.diagnostics()
        >>> print(f"SNR: {metrics['noise']['snr']:.2f}")
        >>> plt.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
        >>> plt.close(fig)

        Analyze specific sections:

        >>> plotter = DiagnosticsPlotter(image)
        >>> fig, metrics = plotter.diagnostics(sections=["noise", "contrast"])
        >>> plt.close(fig)
    """

    call_name = "diagnostics"

    @property
    def _max_intensity(self) -> int:
        """Maximum intensity value based on image bit depth."""
        return 255 if self._root_image.bit_depth == 8 else 65535

    def _get_calculator(self) -> ImageMetricsCalculator:
        """Build a fresh ImageMetricsCalculator over the current detect_mat."""
        return ImageMetricsCalculator(self._root_image.detect_mat[:])

    def _figure_subject(self) -> Any:
        """Subject for the ``PhtPlot`` mixin.

        This plotter is a *helper* provider: each ``@figure`` method reads
        ``self`` directly and takes no subject parameter, so the returned
        subject is informational only (the root :class:`Image`).
        """
        return self._root_image

    def _get_histogram_panel_description(self) -> PanelDescription:
        """Generate histogram panel description with bit-depth-aware intensity range."""
        return PanelDescription(
            label="A",
            title="Intensity Histogram",
            what_it_shows="Distribution of pixel intensities in detection matrix",
            how_to_read=f"X-axis: intensity (0-{self._max_intensity}), Y-axis: frequency. Blue=data, red=Gaussian fit",
            good_values="Bell-shaped, centered, uses full dynamic range",
            poor_values="Bimodal, clipped at edges, very narrow spread",
            pipeline_link="EnhanceLocalContrast, ContrastStretching, GammaCorrection",
        )

    # ============================================================================
    # PLOTLY FIGURE METHODS (interactive dual renderer)
    #
    # Each ``fig_*`` method returns a RAW ``go.Figure`` (the @figure decorator
    # auto-applies the house "phenotypic" theme). They mirror the matplotlib
    # ``_plot_*`` panels one-for-one but never touch the matplotlib path.
    # ============================================================================

    # -- section "noise" --------------------------------------------------------

    @figure(
        title="A: Intensity Histogram",
        section="noise",
    )
    def fig_intensity_histogram(self) -> go.Figure:
        """Plotly intensity histogram with Gaussian fit (Panel A).

        Mirrors :meth:`_plot_intensity_histogram`: a filled histogram of the
        detection-matrix intensities overlaid with a fitted Gaussian, annotated
        with mean, standard deviation, and the robust MAD noise estimate.

        Returns:
            A ``go.Figure`` whose primary trace is the histogram density
            (``go.Scatter`` fill) plus the Gaussian fit overlay.
        """
        detect_mat = self._root_image.detect_mat[:]
        if min(detect_mat.shape[:2]) >= 4:
            calculator = self._get_calculator()
            sigma_mad: float | None = calculator.compute_noise_metrics()[
                "sigma_mad"
            ]
        else:
            sigma_mad = None

        bins = 256
        counts, bin_edges = np.histogram(
            detect_mat.ravel(), bins=bins, range=(0, self._max_intensity)
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        counts_norm = counts / (counts.sum() * (bin_edges[1] - bin_edges[0]))

        mean_val = float(np.mean(detect_mat))
        std_val = float(np.std(detect_mat))
        x_fit = np.linspace(0, self._max_intensity, 500)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=counts_norm,
                fill="tozeroy",
                mode="lines",
                line=dict(color=NAVY, width=1),
                name="Data",
            )
        )
        if std_val > np.finfo(float).eps:
            gaussian_fit = norm.pdf(x_fit, mean_val, std_val)
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=gaussian_fit,
                    mode="lines",
                    line=dict(color=OKABE_ITO[6], width=2, dash="dash"),
                    name="Gaussian fit",
                )
            )
            fit_note = ""
        else:
            fig.add_vline(
                x=mean_val,
                line=dict(color=OKABE_ITO[6], width=2, dash="dash"),
                annotation_text="Mean intensity",
            )
            fit_note = "<br>Gaussian fit omitted"
        fig.update_layout(
            title="A: Intensity Histogram",
            xaxis_title="Intensity",
            yaxis_title="Density",
            xaxis_range=[0, self._max_intensity],
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            text=(
                f"μ={mean_val:.1f}<br>σ={std_val:.1f}<br>MAD={sigma_mad:.2f}"
                if sigma_mad is not None
                else f"μ={mean_val:.1f}<br>σ={std_val:.1f}<br>MAD=N/A"
            ),
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=AXIS_ANNOTATION_BORDER,
            borderwidth=1,
        )
        if fit_note:
            fig.layout.annotations[-1].text = (
                f"{fig.layout.annotations[-1].text}{fit_note}"
            )
        return fig

    @staticmethod
    def _empty_plotly_figure(title: str, message: str) -> go.Figure:
        """Return an annotated empty Plotly figure for degenerate diagnostics."""
        fig = go.Figure()
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(color=NAVY, size=12),
        )
        fig.update_layout(title=title)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    @figure(
        title="B: Noise Autocorrelation",
        section="noise",
        description=PANEL_B_AUTOCORR,
    )
    def fig_noise_autocorrelation(self) -> go.Figure:
        """Plotly noise autocorrelation heatmap (Panel B).

        Mirrors :meth:`_plot_noise_autocorrelation`: the FFT-based 2D
        autocorrelation of the detection matrix, normalized to ``[0, 1]`` and
        rendered with a Hot colorscale, annotated with the correlation length
        ξ in pixels.

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Heatmap``.
        """
        detect_mat = self._root_image.detect_mat[:]
        correlation_length = self._get_calculator().compute_noise_metrics()[
            "correlation_length"
        ]

        autocorr = self._compute_autocorrelation(detect_mat.astype(np.float64))
        autocorr_norm = autocorr / (autocorr.max() + 1e-10)

        fig = go.Figure(
            go.Heatmap(
                z=autocorr_norm,
                colorscale="Hot",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Corr."),
            )
        )
        fig.update_layout(
            title="B: Noise Autocorrelation",
            xaxis_title="Lag (pixels)",
            yaxis_title="Lag (pixels)",
        )
        fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            text=f"ξ={correlation_length:.1f} px",
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=AXIS_ANNOTATION_BORDER,
            borderwidth=1,
        )
        return fig

    @figure(
        title="C: Power Spectral Density",
        section="noise",
        description=PANEL_C_PSD,
    )
    def fig_power_spectral_density(self) -> go.Figure:
        """Plotly radially-averaged power spectral density (Panel C).

        Mirrors :meth:`_plot_power_spectral_density`: the radially-averaged PSD
        on log-log axes with a fitted power-law slope overlay.

        Returns:
            A ``go.Figure`` whose primary trace is a log-log ``go.Scatter``.
        """
        detect_mat = self._root_image.detect_mat[:]
        freqs, psd = self._compute_psd(detect_mat.astype(np.float64))

        valid = (freqs > 0.01) & (psd > 0)
        freqs_valid = freqs[valid]
        psd_valid = psd[valid]

        if len(freqs_valid) == 0:
            return self._empty_plotly_figure(
                "C: Power Spectral Density",
                "No positive PSD bins to display.",
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freqs_valid,
                y=psd_valid,
                mode="lines",
                line=dict(color=NAVY, width=1.5),
                name="PSD",
            )
        )
        if len(freqs_valid) > 10:
            log_f = np.log10(freqs_valid)
            log_p = np.log10(psd_valid)
            coeffs = np.polyfit(log_f, log_p, 1)
            slope = coeffs[0]
            fit_line = 10 ** (coeffs[0] * log_f + coeffs[1])
            fig.add_trace(
                go.Scatter(
                    x=freqs_valid,
                    y=fit_line,
                    mode="lines",
                    line=dict(color=OKABE_ITO[6], width=1, dash="dash"),
                    name=f"slope={slope:.2f}",
                )
            )
        fig.update_layout(
            title="C: Power Spectral Density",
            xaxis_title="Spatial Frequency (normalized)",
            yaxis_title="Power",
            xaxis_type="log",
            yaxis_type="log",
        )
        return fig

    # -- section "contrast" -----------------------------------------------------

    @figure(
        title="D: Detection Matrix",
        section="contrast",
        description=PANEL_D_ORIGINAL,
    )
    def fig_detection_matrix(self) -> go.Figure:
        """Plotly detection-matrix grayscale image (Panel D).

        Mirrors :meth:`_plot_original_image`: a grayscale rendering of the
        current ``detect_mat`` used for detection.

        Returns:
            A ``go.Figure`` whose primary trace is a grayscale ``go.Heatmap``.
        """
        detect_mat = self._root_image.detect_mat[:]
        fig = go.Figure(
            go.Heatmap(
                z=detect_mat,
                colorscale="gray",
                zmin=0,
                zmax=self._max_intensity,
                colorbar=dict(title="Intensity"),
            )
        )
        fig.update_layout(title="D: Detection Matrix")
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            scaleanchor="x",
            constrain="domain",
        )
        return fig

    @figure(
        title="E: Local Contrast Map",
        section="contrast",
        description=PANEL_E_CONTRAST,
    )
    def fig_local_contrast_map(self) -> go.Figure:
        """Plotly local Weber-contrast map (Panel E).

        Mirrors :meth:`_plot_local_contrast_map`: the local Weber contrast,
        clipped at the 99th percentile and rendered with a Magma colorscale.

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Heatmap``.
        """
        calculator = self._get_calculator()
        contrast_map = calculator.compute_local_contrast()  # uses internal detect_mat

        fig = go.Figure(
            go.Heatmap(
                z=contrast_map,
                colorscale="Magma",
                zmin=0,
                zmax=float(np.percentile(contrast_map, 99)),
                colorbar=dict(title="Weber Contrast"),
            )
        )
        fig.update_layout(title="E: Local Contrast Map")
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            scaleanchor="x",
            constrain="domain",
        )
        return fig

    @figure(
        title="F: Contrast Metrics",
        section="contrast",
        description=PANEL_F_BARS,
    )
    def fig_contrast_metrics(self) -> go.Figure:
        """Plotly contrast-metric bars with quality color coding (Panel F).

        Mirrors :meth:`_plot_contrast_metrics_bars`: RMS contrast, Michelson
        contrast, and dynamic range as a horizontal bar chart, each colored
        red/yellow/green against critical/marginal thresholds.

        Returns:
            A ``go.Figure`` whose primary trace is a horizontal ``go.Bar``.
        """
        metrics = self._get_calculator().compute_contrast_metrics()

        metric_names = ["RMS Contrast", "Michelson", "Dynamic Range"]
        metric_values = [
            metrics["rms_contrast"],
            metrics["michelson"],
            metrics["dynamic_range"],
        ]
        thresholds = {
            "RMS Contrast": (0.02, 0.05),
            "Michelson": (0.1, 0.3),
            "Dynamic Range": (0.3, 0.6),
        }
        colors = []
        for name, val in zip(metric_names, metric_values):
            crit, marg = thresholds[name]
            if val < crit:
                colors.append(_POOR_COLOR)
            elif val < marg:
                colors.append(_MARGINAL_COLOR)
            else:
                colors.append(_GOOD_COLOR)

        fig = go.Figure(
            go.Bar(
                x=metric_values,
                y=metric_names,
                orientation="h",
                marker=dict(color=colors, line=dict(color="black", width=1)),
                text=[f"{v:.3f}" for v in metric_values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="F: Contrast Metrics",
            xaxis_title="Value",
            xaxis_range=[0, 1],
            showlegend=False,
        )
        return fig

    # -- section "structure" ----------------------------------------------------

    @figure(
        title="G: Gradient Magnitude",
        section="structure",
        description=PANEL_G_GRADIENT,
    )
    def fig_gradient_magnitude(self) -> go.Figure:
        """Plotly Sobel gradient-magnitude map (Panel G).

        Mirrors :meth:`_plot_gradient_magnitude`: the Sobel edge-strength map,
        clipped at the 99th percentile and rendered with a Viridis colorscale.

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Heatmap``.
        """
        detect_mat = self._root_image.detect_mat[:]
        img_norm = detect_mat.astype(np.float64) / self._max_intensity
        gradient = sobel(img_norm)

        fig = go.Figure(
            go.Heatmap(
                z=gradient,
                colorscale="Viridis",
                zmin=0,
                zmax=float(np.percentile(gradient, 99)),
                colorbar=dict(title="Edge Strength"),
            )
        )
        fig.update_layout(title="G: Gradient Magnitude")
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            scaleanchor="x",
            constrain="domain",
        )
        return fig

    @figure(
        title="H: Orientation Coherence",
        section="structure",
        controls={"sigma": STRUCTURE_SIGMA},
        description=PANEL_H_COHERENCE,
    )
    def fig_orientation_coherence(self, *, sigma) -> go.Figure:
        """Plotly orientation-coherence map from the structure tensor (Panel H).

        Mirrors :meth:`_plot_orientation_coherence`: the per-pixel coherence
        ``(λ1 - λ2) / (λ1 + λ2)`` rendered with a Plasma colorscale and
        annotated with the mean coherence. Recomputes from a fresh
        :class:`ImageMetricsCalculator` using the passed ``sigma``.

        Args:
            sigma: Structure-tensor smoothing σ (bound to ``STRUCTURE_SIGMA``).

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Heatmap``.
        """
        calculator = ImageMetricsCalculator(self._root_image.detect_mat[:])
        structure = calculator.compute_structure_metrics(
            sigma=sigma, scales=DEFAULT_RIDGE_SCALES
        )
        coherence_map = structure["coherence_map"]
        mean_coherence = structure["mean_coherence"]

        fig = go.Figure(
            go.Heatmap(
                z=coherence_map,
                colorscale="Plasma",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Coherence"),
            )
        )
        fig.update_layout(title="H: Orientation Coherence")
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            scaleanchor="x",
            constrain="domain",
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            text=f"Mean: {mean_coherence:.3f}",
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=AXIS_ANNOTATION_BORDER,
            borderwidth=1,
        )
        return fig

    @figure(
        title="I: Ridge Response",
        section="structure",
        controls={"sigma": STRUCTURE_SIGMA, "ridge_method": RIDGE_METHOD},
        description=PANEL_I_RIDGE,
    )
    def fig_ridge_response(self, *, sigma, ridge_method) -> go.Figure:
        """Plotly multiscale ridge-response curve (Panel I).

        Mirrors :meth:`_plot_multiscale_ridge_response`: the mean ridge
        response across the fixed scale set, marking the optimal scale.
        Recomputes from a fresh :class:`ImageMetricsCalculator` using the
        passed ``sigma`` and ``ridge_method``.

        Args:
            sigma: Structure-tensor smoothing σ (bound to ``STRUCTURE_SIGMA``).
            ridge_method: Ridge detector, one of ``"meijering"``, ``"frangi"``,
                ``"hessian"`` (bound to ``RIDGE_METHOD``).

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Scatter`` curve.
        """
        calculator = ImageMetricsCalculator(self._root_image.detect_mat[:])
        structure = calculator.compute_structure_metrics(
            sigma=sigma, scales=DEFAULT_RIDGE_SCALES, ridge_method=ridge_method
        )
        scales = structure["scales"]
        ridge_responses = structure["ridge_responses"]
        optimal_scale = structure["optimal_scale"]
        opt_idx = scales.index(optimal_scale) if optimal_scale in scales else 0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scales,
                y=ridge_responses,
                mode="lines+markers",
                line=dict(color=NAVY, width=2),
                marker=dict(size=8),
                name="Response",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[optimal_scale],
                y=[ridge_responses[opt_idx]],
                mode="markers",
                marker=dict(color=OKABE_ITO[6], size=16, symbol="star"),
                name=f"Optimal σ={optimal_scale:.1f}",
            )
        )
        fig.add_vline(
            x=optimal_scale,
            line=dict(color=OKABE_ITO[6], width=1, dash="dash"),
        )
        fig.update_layout(
            title=f"I: Ridge Response ({ridge_method})",
            xaxis_title="Scale (σ)",
            yaxis_title="Mean Response",
        )
        return fig

    # -- section "background" ---------------------------------------------------

    @figure(
        title="J: Background Estimate",
        section="background",
        controls={"bg_sigma": BACKGROUND_SIGMA},
        description=PANEL_J_BACKGROUND,
    )
    def fig_background_estimate(self, *, bg_sigma) -> go.Figure:
        """Plotly background-estimate map (Panel J).

        Mirrors :meth:`_plot_background_estimate`: the large-sigma Gaussian
        background estimate rendered grayscale, annotated with the
        nonuniformity percentage (colored by quality threshold). Recomputes
        from a fresh :class:`ImageMetricsCalculator` using the passed
        ``bg_sigma``.

        Args:
            bg_sigma: Gaussian σ for background estimation (bound to
                ``BACKGROUND_SIGMA``).

        Returns:
            A ``go.Figure`` whose primary trace is a grayscale ``go.Heatmap``.
        """
        calculator = ImageMetricsCalculator(self._root_image.detect_mat[:])
        background_metrics = calculator.compute_background_metrics(sigma=bg_sigma)
        background = background_metrics["background_estimate"]
        nonuniformity = background_metrics["nonuniformity_ratio"]

        if nonuniformity > THRESHOLDS["nonuniformity"]["critical"]:
            color = _POOR_COLOR
        elif nonuniformity > THRESHOLDS["nonuniformity"]["marginal"]:
            color = _MARGINAL_COLOR
        else:
            color = _GOOD_COLOR

        fig = go.Figure(
            go.Heatmap(
                z=background,
                colorscale="gray",
                colorbar=dict(title="Intensity"),
            )
        )
        fig.update_layout(title="J: Background Estimate")
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            scaleanchor="x",
            constrain="domain",
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            text=f"Nonunif: {nonuniformity:.1%}",
            showarrow=False,
            align="right",
            font=dict(color=color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=AXIS_ANNOTATION_BORDER,
            borderwidth=1,
        )
        return fig

    @figure(
        title="K: Local Variance (log)",
        section="background",
        description=PANEL_K_VARIANCE,
    )
    def fig_local_variance(self) -> go.Figure:
        """Plotly local-variance map on a log scale (Panel K).

        Mirrors :meth:`_plot_local_variance_map`: the windowed local variance,
        log-scaled and rendered with an Inferno colorscale.

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Heatmap``.
        """
        calculator = self._get_calculator()
        variance = calculator.compute_local_variance()  # uses internal detect_mat
        variance_log = np.log10(variance + 1)

        fig = go.Figure(
            go.Heatmap(
                z=variance_log,
                colorscale="Inferno",
                colorbar=dict(title="log₁₀(variance)"),
            )
        )
        fig.update_layout(title="K: Local Variance (log)")
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            scaleanchor="x",
            constrain="domain",
        )
        return fig

    # -- section "summary" ------------------------------------------------------

    @figure(
        title="Image Quality Summary",
        section="summary",
        controls={
            "sigma": STRUCTURE_SIGMA,
            "ridge_method": RIDGE_METHOD,
            "bg_sigma": BACKGROUND_SIGMA,
        },
        primary=True,
    )
    def fig_quality_summary(self, *, sigma, ridge_method, bg_sigma) -> go.Figure:
        """Plotly executive-summary radar chart (the ``inspect()`` figure).

        Mirrors :meth:`_plot_spider_chart`: the five normalized quality scores
        (SNR, Contrast, Coherence, Uniformity, Sharpness) as a closed
        ``go.Scatterpolar`` radar. Recomputes every metric from a fresh
        :class:`ImageMetricsCalculator` using all three controls; marked
        ``primary=True`` so :meth:`inspect` selects it.

        Args:
            sigma: Structure-tensor smoothing σ (bound to ``STRUCTURE_SIGMA``).
            ridge_method: Ridge detector (bound to ``RIDGE_METHOD``).
            bg_sigma: Background-estimation σ (bound to ``BACKGROUND_SIGMA``).

        Returns:
            A ``go.Figure`` whose primary trace is a ``go.Scatterpolar``.
        """
        calculator = ImageMetricsCalculator(self._root_image.detect_mat[:])
        noise = calculator.compute_noise_metrics()
        contrast = calculator.compute_contrast_metrics()
        structure = calculator.compute_structure_metrics(
            sigma=sigma, scales=DEFAULT_RIDGE_SCALES, ridge_method=ridge_method
        )
        background = calculator.compute_background_metrics(sigma=bg_sigma)
        quality_scores = calculator.compute_quality_scores(
            noise, contrast, structure, background
        )

        categories = list(quality_scores.keys())
        values = list(quality_scores.values())
        # Close the polygon back to the first vertex.
        categories_closed = categories + categories[:1]
        values_closed = values + values[:1]

        fig = go.Figure(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                mode="lines+markers",
                line=dict(color=NAVY, width=2),
                fillcolor="rgba(0,54,96,0.25)",
                name="Quality",
            )
        )
        fig.update_layout(
            title="Image Quality Summary",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
        )
        return fig

    # ============================================================================
    # HELPER METHODS FOR PLOTTING
    # ============================================================================

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

    def _compute_psd(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute radially averaged power spectral density.

        Args:
            img: Input image array.

        Returns:
            Tuple of (frequencies, radial_psd).
        """
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

    # ============================================================================
    # PLOTTING METHODS (one per panel)
    # ============================================================================

    def _plot_intensity_histogram(
        self, ax: plt.Axes, detect_mat: np.ndarray, sigma_mad: float
    ) -> None:
        """Plot intensity histogram with Gaussian fit (Panel A)."""
        # Use 256 bins regardless of bit depth for readability
        bins = 256
        hist_range = (0, self._max_intensity)

        counts, bin_edges = np.histogram(detect_mat.ravel(), bins=bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize for density
        counts_norm = counts / (counts.sum() * (bin_edges[1] - bin_edges[0]))

        # Plot histogram
        ax.fill_between(bin_centers, counts_norm, alpha=0.7, color="steelblue")
        ax.plot(bin_centers, counts_norm, color="navy", linewidth=1)

        # Fit and plot Gaussian
        mean_val = np.mean(detect_mat)
        std_val = np.std(detect_mat)
        x_fit = np.linspace(0, self._max_intensity, 500)
        gaussian_fit = norm.pdf(x_fit, mean_val, std_val)
        ax.plot(x_fit, gaussian_fit, "r--", linewidth=2, label="Gaussian fit")

        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
        ax.set_title("A: Intensity Histogram", fontweight="bold")
        ax.set_xlim(0, self._max_intensity)
        ax.legend(loc="upper right", fontsize=8)

        # Add stats annotation
        stats_text = f"μ={mean_val:.1f}\nσ={std_val:.1f}\nMAD={sigma_mad:.2f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _plot_noise_autocorrelation(
        self, ax: plt.Axes, detect_mat: np.ndarray, correlation_length: float
    ) -> None:
        """Plot noise autocorrelation (Panel B)."""
        autocorr = self._compute_autocorrelation(detect_mat.astype(np.float64))

        # Normalize to [0, 1]
        autocorr_norm = autocorr / (autocorr.max() + 1e-10)

        im = ax.imshow(autocorr_norm, cmap="hot", vmin=0, vmax=1)
        ax.set_title("B: Noise Autocorrelation", fontweight="bold")
        ax.set_xlabel("Lag (pixels)")
        ax.set_ylabel("Lag (pixels)")

        # Add correlation length annotation
        ax.text(
            0.95,
            0.95,
            f"ξ={correlation_length:.1f} px",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_power_spectral_density(self, ax: plt.Axes, detect_mat: np.ndarray) -> None:
        """Plot radially-averaged power spectral density (Panel C)."""
        freqs, psd = self._compute_psd(detect_mat.astype(np.float64))

        # Filter out zero frequency and very low values
        valid = (freqs > 0.01) & (psd > 0)
        freqs_valid = freqs[valid]
        psd_valid = psd[valid]

        if len(freqs_valid) > 0:
            ax.loglog(freqs_valid, psd_valid, color="navy", linewidth=1.5)

            # Fit power law to estimate slope
            if len(freqs_valid) > 10:
                log_f = np.log10(freqs_valid)
                log_p = np.log10(psd_valid)
                coeffs = np.polyfit(log_f, log_p, 1)
                slope = coeffs[0]

                # Plot fit line
                fit_line = 10 ** (coeffs[0] * log_f + coeffs[1])
                ax.loglog(
                    freqs_valid, fit_line, "r--", linewidth=1, label=f"slope={slope:.2f}"
                )
                ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("Spatial Frequency (normalized)")
        ax.set_ylabel("Power")
        ax.set_title("C: Power Spectral Density", fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

    def _plot_original_image(self, ax: plt.Axes, detect_mat: np.ndarray) -> None:
        """Plot the detection matrix image (Panel D)."""
        ax.imshow(detect_mat, cmap="gray", vmin=0, vmax=self._max_intensity)
        ax.set_title("D: Detection Matrix", fontweight="bold")
        ax.axis("off")

    def _plot_local_contrast_map(
        self, ax: plt.Axes, detect_mat: np.ndarray, calculator: ImageMetricsCalculator
    ) -> None:
        """Plot local contrast map (Panel E)."""
        contrast_map = calculator.compute_local_contrast(detect_mat)

        im = ax.imshow(contrast_map, cmap="magma", vmin=0, vmax=np.percentile(contrast_map, 99))
        ax.set_title("E: Local Contrast Map", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Weber Contrast")

    def _plot_contrast_metrics_bars(
        self, ax: plt.Axes, metrics: dict[str, float]
    ) -> None:
        """Plot contrast metrics as horizontal bars with quality thresholds (Panel F)."""
        metric_names = ["RMS Contrast", "Michelson", "Dynamic Range"]
        metric_values = [
            metrics["rms_contrast"],
            metrics["michelson"],
            metrics["dynamic_range"],
        ]

        # Thresholds for color coding
        thresholds = {
            "RMS Contrast": (0.02, 0.05),  # (critical, marginal)
            "Michelson": (0.1, 0.3),
            "Dynamic Range": (0.3, 0.6),
        }

        colors = []
        for name, val in zip(metric_names, metric_values):
            crit, marg = thresholds[name]
            if val < crit:
                colors.append("#c0392b")  # Red - poor
            elif val < marg:
                colors.append("#f39c12")  # Yellow - marginal
            else:
                colors.append("#27ae60")  # Green - good

        y_pos = np.arange(len(metric_names))
        bars = ax.barh(y_pos, metric_values, color=colors, edgecolor="black")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel("Value")
        ax.set_title("F: Contrast Metrics", fontweight="bold")
        ax.set_xlim(0, 1)

        # Add value labels
        for bar, val in zip(bars, metric_values):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

        # Add threshold lines
        for i, (name, (crit, marg)) in enumerate(thresholds.items()):
            ax.axvline(crit, color="#c0392b", linestyle="--", alpha=0.5, linewidth=1)
            ax.axvline(marg, color="#f39c12", linestyle="--", alpha=0.5, linewidth=1)

    def _plot_gradient_magnitude(self, ax: plt.Axes, detect_mat: np.ndarray) -> None:
        """Plot Sobel gradient magnitude (Panel G)."""
        img_norm = detect_mat.astype(np.float64) / self._max_intensity
        gradient = sobel(img_norm)

        im = ax.imshow(gradient, cmap="viridis", vmin=0, vmax=np.percentile(gradient, 99))
        ax.set_title("G: Gradient Magnitude", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Edge Strength")

    def _plot_orientation_coherence(
        self, ax: plt.Axes, coherence_map: np.ndarray, mean_coherence: float
    ) -> None:
        """Plot orientation coherence from structure tensor (Panel H)."""
        im = ax.imshow(coherence_map, cmap="plasma", vmin=0, vmax=1)
        ax.set_title("H: Orientation Coherence", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Coherence")

        # Add mean coherence annotation
        ax.text(
            0.95,
            0.95,
            f"Mean: {mean_coherence:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _plot_multiscale_ridge_response(
        self,
        ax: plt.Axes,
        scales: list[float],
        ridge_responses: list[float],
        optimal_scale: float,
        ridge_method: str,
    ) -> None:
        """Plot multiscale ridge detection response (Panel I)."""
        ax.plot(scales, ridge_responses, "o-", color="navy", linewidth=2, markersize=6)

        # Mark optimal scale
        opt_idx = scales.index(optimal_scale) if optimal_scale in scales else 0
        ax.axvline(optimal_scale, color="red", linestyle="--", alpha=0.7, label=f"Optimal σ={optimal_scale:.1f}")
        ax.plot(optimal_scale, ridge_responses[opt_idx], "r*", markersize=15)

        ax.set_xlabel("Scale (σ)")
        ax.set_ylabel("Mean Response")
        ax.set_title(f"I: Ridge Response ({ridge_method})", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_background_estimate(
        self, ax: plt.Axes, background: np.ndarray, nonuniformity: float
    ) -> None:
        """Plot background estimate (Panel J)."""
        im = ax.imshow(background, cmap="gray")
        ax.set_title("J: Background Estimate", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")

        # Determine quality color
        if nonuniformity > THRESHOLDS["nonuniformity"]["critical"]:
            color = "#c0392b"
        elif nonuniformity > THRESHOLDS["nonuniformity"]["marginal"]:
            color = "#f39c12"
        else:
            color = "#27ae60"

        ax.text(
            0.95,
            0.95,
            f"Nonunif: {nonuniformity:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _plot_local_variance_map(
        self, ax: plt.Axes, detect_mat: np.ndarray, calculator: ImageMetricsCalculator
    ) -> None:
        """Plot local variance map on log scale (Panel K)."""
        variance = calculator.compute_local_variance(detect_mat)

        # Log scale for better visualization
        variance_log = np.log10(variance + 1)

        im = ax.imshow(variance_log, cmap="inferno")
        ax.set_title("K: Local Variance (log)", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log₁₀(variance)")

    def _plot_spider_chart(
        self, ax: plt.Axes, quality_scores: dict[str, float]
    ) -> None:
        """Plot executive summary spider/radar chart."""
        categories = list(quality_scores.keys())
        values = list(quality_scores.values())
        n = len(categories)

        # Compute angles
        angles = [i / n * 2 * np.pi for i in range(n)]
        angles += angles[:1]  # Close the polygon
        values += values[:1]

        # Plot (polar axes have these methods but mypy doesn't know about them)
        ax.set_theta_offset(np.pi / 2)  # type: ignore[attr-defined]
        ax.set_theta_direction(-1)  # type: ignore[attr-defined]

        ax.plot(angles, values, "o-", linewidth=2, color="navy")
        ax.fill(angles, values, alpha=0.25, color="steelblue")

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Image Quality Summary", fontweight="bold", pad=20)

        # Add concentric reference circles for quality zones
        for r in [0.3, 0.7]:
            theta_circle = np.linspace(0, 2 * np.pi, 100)
            ax.plot(theta_circle, [r] * 100, "--", color="gray", alpha=0.3, linewidth=0.5)

    def _render_panel_descriptions(
        self, ax: plt.Axes, descriptions: list[PanelDescription]
    ) -> None:
        """Render panel descriptions as formatted text in a text axis."""
        ax.axis("off")

        # Colors for semantic formatting
        colors = {
            "label": "#1a237e",  # Navy for labels
            "good": "#27ae60",  # Green for good values
            "poor": "#c0392b",  # Red for poor values
            "link": "#2980b9",  # Blue for pipeline links
            "text": "#333333",  # Dark gray for body text
        }

        n_panels = len(descriptions)
        col_width = 1.0 / n_panels

        for i, desc in enumerate(descriptions):
            x_pos = (i + 0.5) * col_width
            y_base = 0.95

            # Panel label and title
            ax.text(
                x_pos,
                y_base,
                f"[{desc.label}] {desc.title}",
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color=colors["label"],
                transform=ax.transAxes,
            )

            # What it shows
            ax.text(
                x_pos,
                y_base - 0.15,
                f"Shows: {desc.what_it_shows}",
                ha="center",
                va="top",
                fontsize=7,
                color=colors["text"],
                transform=ax.transAxes,
                wrap=True,
            )

            # How to read
            ax.text(
                x_pos,
                y_base - 0.35,
                f"Read: {desc.how_to_read}",
                ha="center",
                va="top",
                fontsize=7,
                color=colors["text"],
                transform=ax.transAxes,
                wrap=True,
            )

            # Good values
            ax.text(
                x_pos,
                y_base - 0.55,
                f"Good: {desc.good_values}",
                ha="center",
                va="top",
                fontsize=7,
                color=colors["good"],
                transform=ax.transAxes,
                wrap=True,
            )

            # Poor values
            ax.text(
                x_pos,
                y_base - 0.70,
                f"Poor: {desc.poor_values}",
                ha="center",
                va="top",
                fontsize=7,
                color=colors["poor"],
                transform=ax.transAxes,
                wrap=True,
            )

            # Pipeline link
            ax.text(
                x_pos,
                y_base - 0.85,
                f"→ {desc.pipeline_link}",
                ha="center",
                va="top",
                fontsize=7,
                color=colors["link"],
                transform=ax.transAxes,
                wrap=True,
            )

    # ============================================================================
    # MAIN PUBLIC METHOD
    # ============================================================================

    def diagnostics(
        self,
        sections: Literal["all", "noise", "contrast", "structure", "background"]
        | list[str] = "all",
        figsize: tuple[float, float] | None = None,
        include_descriptions: bool = True,
        include_recommendations: bool = True,
        background_sigma: float = 50.0,
        structure_sigma: float = 1.5,
        ridge_scales: list[float] | None = None,
        ridge_method: Literal["meijering", "frangi", "hessian"] = "meijering",
    ) -> tuple[plt.Figure, dict[str, Any]]:
        """Generate comprehensive image quality diagnostics as a static matplotlib figure.

        For the interactive report, use ``PlotDiagnostics().report(image)``.

        Args:
            sections: Which sections to include. "all" for complete diagnostics, or a
                list of section names: ["noise", "contrast", "structure", "background"].
            figsize: Figure size as (width, height) in inches. If None, computed
                automatically based on number of sections.
            include_descriptions: If True, include panel description text below each
                section.
            include_recommendations: If True, include recommendations summary panel.
            background_sigma: Sigma for background estimation Gaussian smoothing.
            structure_sigma: Sigma for structure tensor computation.
            ridge_scales: Scales for multiscale ridge detection. Defaults to
                [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0].
            ridge_method: Method for ridge detection: "meijering" (default, optimized for
                neurite-like structures), "frangi" (vesselness), or "hessian" (raw eigenvalues).

        Returns:
            Tuple of (fig, metrics_dict) where:
                - fig: A matplotlib Figure with the diagnostics visualization.
                - metrics_dict: Dictionary containing:
                    - "bit_depth": Image bit depth (8 or 16)
                    - "noise": Noise metrics (snr, sigma_mad, correlation_length)
                    - "contrast": Contrast metrics (rms_contrast, michelson, dynamic_range)
                    - "structure": Structure metrics (mean_coherence, optimal_scale, etc.)
                    - "background": Background metrics (nonuniformity_ratio, mean_gradient)
                    - "quality_scores": Normalized 0-1 scores for spider chart
                    - "interpretations": Data-driven interpretation text per section
                    - "recommendations": List of actionable recommendations

        Examples:
            Static matplotlib figure:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> plotter = DiagnosticsPlotter(image)
            >>> fig, metrics = plotter.diagnostics()
            >>> print(f"SNR: {metrics['noise']['snr']:.2f}")
            >>> plt.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
            >>> plt.close(fig)

            For an interactive dashboard:

            >>> from phenotypic.plotting import PlotDiagnostics
            >>> dashboard = PlotDiagnostics().report(image)  # doctest: +SKIP
        """
        # Default ridge scales
        if ridge_scales is None:
            ridge_scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

        # Get detection matrix
        detect_mat = self._root_image.detect_mat[:]

        # Create metrics calculator
        calculator = ImageMetricsCalculator(detect_mat)

        return self._diagnostics_matplotlib(
            sections=sections,
            figsize=figsize,
            include_descriptions=include_descriptions,
            include_recommendations=include_recommendations,
            background_sigma=background_sigma,
            structure_sigma=structure_sigma,
            ridge_scales=ridge_scales,
            ridge_method=ridge_method,
            detect_mat=detect_mat,
            calculator=calculator,
        )

    def _diagnostics_matplotlib(
        self,
        *,
        sections: Literal["all", "noise", "contrast", "structure", "background"]
        | list[str],
        figsize: tuple[float, float] | None,
        include_descriptions: bool,
        include_recommendations: bool,
        background_sigma: float,
        structure_sigma: float,
        ridge_scales: list[float],
        ridge_method: Literal["meijering", "frangi", "hessian"],
        detect_mat: np.ndarray,
        calculator: ImageMetricsCalculator,
    ) -> tuple[plt.Figure, dict[str, Any]]:
        """Static matplotlib fallback for diagnostics().

        Produces the original multi-panel figure when Panel is not installed.
        """
        # Parse sections
        if sections == "all":
            section_list = ["noise", "contrast", "structure", "background"]
        elif isinstance(sections, str):
            section_list = [sections]
        else:
            section_list = list(sections)

        # Validate sections
        valid_sections = {"noise", "contrast", "structure", "background"}
        for s in section_list:
            if s not in valid_sections:
                raise ValueError(f"Invalid section: {s}. Valid: {valid_sections}")

        # Compute all metrics using ImageMetricsCalculator
        noise_metrics = calculator.compute_noise_metrics()
        contrast_metrics = calculator.compute_contrast_metrics()
        structure_metrics = calculator.compute_structure_metrics(
            sigma=structure_sigma, scales=ridge_scales, ridge_method=ridge_method
        )
        background_metrics = calculator.compute_background_metrics(sigma=background_sigma)

        # Compute quality scores
        quality_scores = calculator.compute_quality_scores(
            noise_metrics, contrast_metrics, structure_metrics, background_metrics
        )

        # Generate interpretations
        interpretations = {
            "noise": calculator.generate_interpretation("noise", noise_metrics),
            "contrast": calculator.generate_interpretation("contrast", contrast_metrics),
            "structure": calculator.generate_interpretation("structure", structure_metrics),
            "background": calculator.generate_interpretation("background", background_metrics),
        }

        # Generate recommendations
        recommendations = calculator.generate_recommendations(
            noise_metrics, contrast_metrics, structure_metrics, background_metrics
        )

        # ========================================================================
        # FIGURE LAYOUT
        # ========================================================================

        # Compute figure size if not provided
        if figsize is None:
            height = 4.0  # Executive summary
            for _ in section_list:
                height += 2.5  # Panel row
                if include_descriptions:
                    height += 1.0  # Description row
            if include_recommendations:
                height += 1.5  # Recommendations row
            figsize = (14.0, height)

        # Compute grid layout
        n_rows = 1  # Executive summary
        height_ratios = [1.5]

        for s in section_list:
            n_rows += 1  # Panel row
            if s == "background":
                height_ratios.append(1.0)
            else:
                height_ratios.append(1.0)

            if include_descriptions:
                n_rows += 1  # Description row
                height_ratios.append(0.5)

        if include_recommendations:
            n_rows += 1
            height_ratios.append(0.6)

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = GridSpec(n_rows, 3, figure=fig, height_ratios=height_ratios, hspace=0.4, wspace=0.3)

        row_idx = 0

        # ========================================================================
        # EXECUTIVE SUMMARY (Spider Chart)
        # ========================================================================
        ax_spider = fig.add_subplot(gs[row_idx, :], projection="polar")
        self._plot_spider_chart(ax_spider, quality_scores)
        row_idx += 1

        # ========================================================================
        # SECTION PANELS
        # ========================================================================

        for section in section_list:
            descriptions = []

            if section == "background":
                # Two-column layout for background section
                ax1 = fig.add_subplot(gs[row_idx, 0:2])
                ax2 = fig.add_subplot(gs[row_idx, 2])

                self._plot_background_estimate(
                    ax1,
                    background_metrics["background_estimate"],
                    background_metrics["nonuniformity_ratio"],
                )
                self._plot_local_variance_map(ax2, detect_mat, calculator)

                descriptions = [PANEL_J_BACKGROUND, PANEL_K_VARIANCE]
            else:
                # Three-column layout
                ax1 = fig.add_subplot(gs[row_idx, 0])
                ax2 = fig.add_subplot(gs[row_idx, 1])
                ax3 = fig.add_subplot(gs[row_idx, 2])

                if section == "noise":
                    # Use bit-depth aware histogram description
                    hist_desc = self._get_histogram_panel_description()
                    self._plot_intensity_histogram(ax1, detect_mat, noise_metrics["sigma_mad"])
                    self._plot_noise_autocorrelation(ax2, detect_mat, noise_metrics["correlation_length"])
                    self._plot_power_spectral_density(ax3, detect_mat)
                    descriptions = [hist_desc, PANEL_B_AUTOCORR, PANEL_C_PSD]

                elif section == "contrast":
                    self._plot_original_image(ax1, detect_mat)
                    self._plot_local_contrast_map(ax2, detect_mat, calculator)
                    self._plot_contrast_metrics_bars(ax3, contrast_metrics)
                    descriptions = [PANEL_D_ORIGINAL, PANEL_E_CONTRAST, PANEL_F_BARS]

                elif section == "structure":
                    self._plot_gradient_magnitude(ax1, detect_mat)
                    self._plot_orientation_coherence(
                        ax2,
                        structure_metrics["coherence_map"],
                        structure_metrics["mean_coherence"],
                    )
                    self._plot_multiscale_ridge_response(
                        ax3,
                        structure_metrics["scales"],
                        structure_metrics["ridge_responses"],
                        structure_metrics["optimal_scale"],
                        structure_metrics["ridge_method"],
                    )
                    descriptions = [PANEL_G_GRADIENT, PANEL_H_COHERENCE, PANEL_I_RIDGE]

            row_idx += 1

            # Description row
            if include_descriptions and descriptions:
                ax_desc = fig.add_subplot(gs[row_idx, :])
                self._render_panel_descriptions(ax_desc, descriptions)
                row_idx += 1

        # ========================================================================
        # RECOMMENDATIONS PANEL
        # ========================================================================
        if include_recommendations:
            ax_rec = fig.add_subplot(gs[row_idx, :])
            ax_rec.axis("off")

            # Title
            ax_rec.text(
                0.5,
                0.95,
                "Recommendations",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                transform=ax_rec.transAxes,
            )

            # Interpretations summary
            y_pos = 0.75
            for section in section_list:
                interp = interpretations[section]
                ax_rec.text(
                    0.02,
                    y_pos,
                    f"• {interp}",
                    ha="left",
                    va="top",
                    fontsize=8,
                    transform=ax_rec.transAxes,
                    wrap=True,
                )
                y_pos -= 0.15

            # Action items
            y_pos -= 0.05
            ax_rec.text(
                0.02,
                y_pos,
                "Suggested Actions:",
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="#2980b9",
                transform=ax_rec.transAxes,
            )
            y_pos -= 0.12

            for rec in recommendations[:5]:  # Limit to top 5
                ax_rec.text(
                    0.04,
                    y_pos,
                    f"→ {rec}",
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="#2980b9",
                    transform=ax_rec.transAxes,
                    wrap=True,
                )
                y_pos -= 0.10

        # Use subplots_adjust instead of tight_layout to avoid polar axis warnings
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # ========================================================================
        # PREPARE RETURN DICTIONARY
        # ========================================================================

        # Remove non-serializable items from metrics
        structure_metrics_clean = {
            k: v
            for k, v in structure_metrics.items()
            if k != "coherence_map"
        }
        background_metrics_clean = {
            k: v
            for k, v in background_metrics.items()
            if k != "background_estimate"
        }

        metrics_dict = {
            "bit_depth": self._root_image.bit_depth,
            "noise": dict(noise_metrics),
            "contrast": dict(contrast_metrics),
            "structure": structure_metrics_clean,
            "background": background_metrics_clean,
            "quality_scores": dict(quality_scores),
            "interpretations": interpretations,
            "recommendations": recommendations,
        }

        return fig, metrics_dict


__all__ = ["DiagnosticsPlotter"]
