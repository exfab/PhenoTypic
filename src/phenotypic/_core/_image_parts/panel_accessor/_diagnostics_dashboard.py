"""Interactive Panel dashboard for image quality diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from phenotypic.tools_.panel_ import PANEL_AVAILABLE

if PANEL_AVAILABLE:
    import param
    import panel as pn
else:
    param = None  # type: ignore[assignment]
    pn = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from phenotypic._core._image import Image


def _format_description(desc: Any) -> str:
    """Format a PanelDescription as an HTML block.

    Args:
        desc: PanelDescription dataclass instance.

    Returns:
        HTML string with styled description.
    """
    return (
        f"<div style='font-size:12px; padding:4px 8px; line-height:1.5;'>"
        f"<b style='color:#1a237e;'>[{desc.label}] {desc.title}</b><br/>"
        f"<span style='color:#333;'>Shows: {desc.what_it_shows}</span><br/>"
        f"<span style='color:#333;'>Read: {desc.how_to_read}</span><br/>"
        f"<span style='color:#27ae60;'>Good: {desc.good_values}</span><br/>"
        f"<span style='color:#c0392b;'>Poor: {desc.poor_values}</span><br/>"
        f"<span style='color:#2980b9;'>&rarr; {desc.pipeline_link}</span>"
        f"</div>"
    )


if PANEL_AVAILABLE:
    from phenotypic.tools_.register import register_dashboard
    from phenotypic.util.image_metrics import ImageMetricsCalculator

    from ..plot_accessor._diagnostics_types import (
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
    )

    @register_dashboard
    class DiagnosticsDashboard(param.Parameterized):
        """Interactive Panel dashboard for image quality diagnostics.

        Provides live-updating plots and metrics when parameters change.
        Section toggles show/hide noise, contrast, structure, and background
        analysis panels.

        The metrics computation is delegated to :class:`ImageMetricsCalculator`
        from ``phenotypic.util.image_metrics``, which is shared with the
        matplotlib-based :class:`DiagnosticsPlotter`.

        Access via ``image.panel.diagnostics()``:

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> dashboard = image.panel.diagnostics()
            >>> dashboard.panel()  # Display interactive dashboard

            Get metrics from the dashboard:

            >>> metrics = dashboard.metrics
            >>> print(f"SNR: {metrics['noise']['snr']:.2f}")
        """

        call_name = "diagnostics"

        # Interactive parameters
        structure_sigma = param.Number(
                default=1.5, bounds=(0.1, 10.0), step=0.1,
                doc="Sigma for structure tensor computation.",
        )
        ridge_method = param.Selector(
                default="meijering", objects=["meijering", "frangi", "hessian"],
                doc="Method for ridge detection.",
        )
        ridge_scales_str = param.String(
                default="0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0",
                doc="Comma-separated list of scales for multiscale ridge detection.",
        )
        background_sigma = param.Number(
                default=50.0, bounds=(5.0, 200.0), step=5.0,
                doc="Sigma for background estimation Gaussian smoothing.",
        )

        # Section toggles
        show_noise = param.Boolean(default=True, doc="Show noise analysis section.")
        show_contrast = param.Boolean(default=True,
                                      doc="Show contrast analysis section.")
        show_structure = param.Boolean(default=True,
                                       doc="Show structure analysis section.")
        show_background = param.Boolean(default=True,
                                        doc="Show background analysis section.")

        def __init__(
                self,
                root_image: Image,
                **params: Any,
        ) -> None:
            """Initialize DiagnosticsDashboard.

            Args:
                root_image: The Image instance to analyze.
                **params: Additional param.Parameterized parameters.
            """
            super().__init__(**params)

            self._root_image = root_image
            self._detect_mat = root_image.detect_mat[:]
            self._calculator = ImageMetricsCalculator(self._detect_mat)

            # Create plotter for rendering (lazy import to avoid circular)
            from ..plot_accessor._diagnostics_plotter import DiagnosticsPlotter
            self._plotter = DiagnosticsPlotter(root_image)

            # Pre-compute parameter-free metrics once
            self._noise_metrics = self._calculator.compute_noise_metrics()
            self._contrast_metrics = self._calculator.compute_contrast_metrics()

        # ------------------------------------------------------------------
        # Public method for registry-based access
        # ------------------------------------------------------------------

        def diagnostics(
            self,
            structure_sigma: float | None = None,
            ridge_method: str | None = None,
            ridge_scales: list[float] | None = None,
            background_sigma: float | None = None,
        ) -> "DiagnosticsDashboard":
            """Return the dashboard instance with optional parameter updates.

            Args:
                structure_sigma: Sigma for structure tensor computation.
                ridge_method: Method for ridge detection.
                ridge_scales: Scales for multiscale ridge detection.
                background_sigma: Sigma for background estimation.

            Returns:
                This DiagnosticsDashboard instance.
            """
            if structure_sigma is not None:
                self.structure_sigma = structure_sigma
            if ridge_method is not None:
                self.ridge_method = ridge_method
            if ridge_scales is not None:
                self.ridge_scales_str = ", ".join(str(s) for s in ridge_scales)
            if background_sigma is not None:
                self.background_sigma = background_sigma

            return self

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _parse_ridge_scales(self) -> list[float]:
            """Parse ridge_scales_str into a list of floats."""
            try:
                return [float(s.strip()) for s in self.ridge_scales_str.split(",") if
                        s.strip()]
            except ValueError:
                return [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

        def _make_panel_figure(
                self,
                plot_method: Any,
                *args: Any,
                figsize: tuple[float, float] = (5, 4),
                polar: bool = False,
        ) -> pn.pane.Matplotlib:
            """Create a Panel-wrapped matplotlib figure from a plotter method.

            Args:
                plot_method: Callable that draws on a matplotlib Axes.
                *args: Extra args forwarded to plot_method after the Axes.
                figsize: Figure size in inches.
                polar: Whether to use polar projection.

            Returns:
                A Panel Matplotlib pane.
            """
            if polar:
                fig, ax = plt.subplots(figsize=figsize,
                                       subplot_kw={"projection": "polar"})
            else:
                fig, ax = plt.subplots(figsize=figsize)
            plot_method(ax, *args)
            fig.tight_layout()
            pane = pn.pane.Matplotlib(fig, tight=True, dpi=100)
            plt.close(fig)
            return pane

        # ------------------------------------------------------------------
        # Metric accessors (recompute parameter-dependent on demand)
        # ------------------------------------------------------------------

        def _current_structure_metrics(self) -> dict[str, Any]:
            return self._calculator.compute_structure_metrics(
                    sigma=self.structure_sigma,
                    scales=self._parse_ridge_scales(),
                    ridge_method=self.ridge_method,
            )

        def _current_background_metrics(self) -> dict[str, Any]:
            return self._calculator.compute_background_metrics(
                    sigma=self.background_sigma,
            )

        @property
        def metrics(self) -> dict[str, Any]:
            """Return the current metrics dictionary.

            Noise and contrast metrics are pre-computed; structure and background
            metrics are recomputed using the current interactive parameter values.

            Returns:
                Dictionary with noise, contrast, structure, background,
                quality_scores, interpretations, and recommendations.
            """
            structure = self._current_structure_metrics()
            background = self._current_background_metrics()
            quality_scores = self._calculator.compute_quality_scores(
                    self._noise_metrics, self._contrast_metrics, structure, background,
            )
            interpretations = {
                "noise"     : self._calculator.generate_interpretation("noise",
                                                                     self._noise_metrics),
                "contrast"  : self._calculator.generate_interpretation("contrast",
                                                                     self._contrast_metrics),
                "structure" : self._calculator.generate_interpretation("structure",
                                                                     structure),
                "background": self._calculator.generate_interpretation("background",
                                                                     background),
            }
            recommendations = self._calculator.generate_recommendations(
                    self._noise_metrics, self._contrast_metrics, structure, background,
            )

            # Clean non-serializable arrays
            structure_clean = {k: v for k, v in structure.items() if
                               k != "coherence_map"}
            background_clean = {k: v for k, v in background.items() if
                                k != "background_estimate"}

            return {
                "bit_depth"      : self._calculator.bit_depth,
                "noise"          : dict(self._noise_metrics),
                "contrast"       : dict(self._contrast_metrics),
                "structure"      : structure_clean,
                "background"     : background_clean,
                "quality_scores" : dict(quality_scores),
                "interpretations": interpretations,
                "recommendations": recommendations,
            }

        # ------------------------------------------------------------------
        # Reactive section renderers
        # ------------------------------------------------------------------

        @param.depends("show_noise")
        def _noise_section(self) -> pn.Column:
            if not self.show_noise:
                return pn.Column()

            hist_desc = self._plotter._get_histogram_panel_description()

            hist = self._make_panel_figure(
                    self._plotter._plot_intensity_histogram,
                    self._detect_mat,
                    self._noise_metrics["sigma_mad"],
            )
            autocorr = self._make_panel_figure(
                    self._plotter._plot_noise_autocorrelation,
                    self._detect_mat,
                    self._noise_metrics["correlation_length"],
            )
            psd = self._make_panel_figure(
                    self._plotter._plot_power_spectral_density,
                    self._detect_mat,
            )

            interpretation = self._calculator.generate_interpretation("noise",
                                                                    self._noise_metrics)

            return pn.Card(
                    pn.Row(hist, autocorr, psd, sizing_mode="stretch_width"),
                    pn.Row(
                            pn.pane.HTML(_format_description(hist_desc)),
                            pn.pane.HTML(_format_description(PANEL_B_AUTOCORR)),
                            pn.pane.HTML(_format_description(PANEL_C_PSD)),
                            sizing_mode="stretch_width",
                    ),
                    pn.pane.Markdown(f"*{interpretation}*"),
                    title="Noise Analysis",
                    collapsed=False,
                    sizing_mode="stretch_width",
            )

        @param.depends("show_contrast")
        def _contrast_section(self) -> pn.Column:
            if not self.show_contrast:
                return pn.Column()

            original = self._make_panel_figure(
                    self._plotter._plot_original_image, self._detect_mat,
            )
            contrast_map = self._make_panel_figure(
                    self._plotter._plot_local_contrast_map, self._detect_mat, self._calculator,
            )
            bars = self._make_panel_figure(
                    self._plotter._plot_contrast_metrics_bars, self._contrast_metrics,
            )

            interpretation = self._calculator.generate_interpretation("contrast",
                                                                    self._contrast_metrics)

            return pn.Card(
                    pn.Row(original, contrast_map, bars, sizing_mode="stretch_width"),
                    pn.Row(
                            pn.pane.HTML(_format_description(PANEL_D_ORIGINAL)),
                            pn.pane.HTML(_format_description(PANEL_E_CONTRAST)),
                            pn.pane.HTML(_format_description(PANEL_F_BARS)),
                            sizing_mode="stretch_width",
                    ),
                    pn.pane.Markdown(f"*{interpretation}*"),
                    title="Contrast Analysis",
                    collapsed=False,
                    sizing_mode="stretch_width",
            )

        @param.depends("show_structure", "structure_sigma", "ridge_method",
                       "ridge_scales_str")
        def _structure_section(self) -> pn.Column:
            if not self.show_structure:
                return pn.Column()

            structure = self._current_structure_metrics()

            gradient = self._make_panel_figure(
                    self._plotter._plot_gradient_magnitude, self._detect_mat,
            )
            coherence = self._make_panel_figure(
                    self._plotter._plot_orientation_coherence,
                    structure["coherence_map"],
                    structure["mean_coherence"],
            )
            ridge = self._make_panel_figure(
                    self._plotter._plot_multiscale_ridge_response,
                    structure["scales"],
                    structure["ridge_responses"],
                    structure["optimal_scale"],
                    structure["ridge_method"],
            )

            interpretation = self._calculator.generate_interpretation("structure",
                                                                    structure)

            return pn.Card(
                    pn.Row(gradient, coherence, ridge, sizing_mode="stretch_width"),
                    pn.Row(
                            pn.pane.HTML(_format_description(PANEL_G_GRADIENT)),
                            pn.pane.HTML(_format_description(PANEL_H_COHERENCE)),
                            pn.pane.HTML(_format_description(PANEL_I_RIDGE)),
                            sizing_mode="stretch_width",
                    ),
                    pn.pane.Markdown(f"*{interpretation}*"),
                    title="Structure Analysis",
                    collapsed=False,
                    sizing_mode="stretch_width",
            )

        @param.depends("show_background", "background_sigma")
        def _background_section(self) -> pn.Column:
            if not self.show_background:
                return pn.Column()

            background = self._current_background_metrics()

            bg_plot = self._make_panel_figure(
                    self._plotter._plot_background_estimate,
                    background["background_estimate"],
                    background["nonuniformity_ratio"],
            )
            variance = self._make_panel_figure(
                    self._plotter._plot_local_variance_map, self._detect_mat, self._calculator,
            )

            interpretation = self._calculator.generate_interpretation("background",
                                                                    background)

            return pn.Card(
                    pn.Row(bg_plot, variance, sizing_mode="stretch_width"),
                    pn.Row(
                            pn.pane.HTML(_format_description(PANEL_J_BACKGROUND)),
                            pn.pane.HTML(_format_description(PANEL_K_VARIANCE)),
                            sizing_mode="stretch_width",
                    ),
                    pn.pane.Markdown(f"*{interpretation}*"),
                    title="Background Analysis",
                    collapsed=False,
                    sizing_mode="stretch_width",
            )

        @param.depends(
                "structure_sigma", "ridge_method", "ridge_scales_str",
                "background_sigma",
        )
        def _spider_section(self) -> pn.pane.Matplotlib:
            structure = self._current_structure_metrics()
            background = self._current_background_metrics()
            quality_scores = self._calculator.compute_quality_scores(
                    self._noise_metrics, self._contrast_metrics, structure, background,
            )
            return self._make_panel_figure(
                    self._plotter._plot_spider_chart, quality_scores,
                    figsize=(5, 5), polar=True,
            )

        @param.depends(
                "structure_sigma", "ridge_method", "ridge_scales_str",
                "background_sigma",
        )
        def _recommendations_section(self) -> pn.pane.Markdown:
            structure = self._current_structure_metrics()
            background = self._current_background_metrics()

            interpretations = {
                "noise"     : self._calculator.generate_interpretation("noise",
                                                                     self._noise_metrics),
                "contrast"  : self._calculator.generate_interpretation("contrast",
                                                                     self._contrast_metrics),
                "structure" : self._calculator.generate_interpretation("structure",
                                                                     structure),
                "background": self._calculator.generate_interpretation("background",
                                                                     background),
            }
            recommendations = self._calculator.generate_recommendations(
                    self._noise_metrics, self._contrast_metrics, structure, background,
            )

            lines = ["## Recommendations\n"]
            for section_name in ("noise", "contrast", "structure", "background"):
                lines.append(f"- {interpretations[section_name]}")
            lines.append("\n**Suggested Actions:**\n")
            for rec in recommendations[:5]:
                lines.append(f"- {rec}")

            return pn.pane.Markdown("\n".join(lines))

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------

        def panel(self) -> pn.Column:
            """Return the interactive Panel layout.

            Returns:
                A Panel Column containing the full interactive dashboard.
            """
            controls_params = pn.Card(
                    pn.Param(
                            self.param.structure_sigma,
                            widgets={"structure_sigma": pn.widgets.FloatSlider},
                    ),
                    pn.Param(
                            self.param.background_sigma,
                            widgets={"background_sigma": pn.widgets.FloatSlider},
                    ),
                    pn.Param(
                            self.param.ridge_method,
                            widgets={"ridge_method": pn.widgets.Select},
                    ),
                    pn.Param(
                            self.param.ridge_scales_str,
                            widgets={"ridge_scales_str": pn.widgets.TextInput},
                    ),
                    title="Parameters",
                    collapsed=False,
                    width=300,
            )

            controls_sections = pn.Card(
                    pn.Param(self.param.show_noise),
                    pn.Param(self.param.show_contrast),
                    pn.Param(self.param.show_structure),
                    pn.Param(self.param.show_background),
                    title="Sections",
                    collapsed=False,
                    width=300,
            )

            sidebar = pn.Column(controls_params, controls_sections, width=320)

            return pn.Column(
                    pn.pane.Markdown("# Image Quality Diagnostics"),
                    pn.Row(
                            sidebar,
                            pn.Column(self._spider_section,
                                      sizing_mode="stretch_width"),
                            sizing_mode="stretch_width",
                    ),
                    self._noise_section,
                    self._contrast_section,
                    self._structure_section,
                    self._background_section,
                    self._recommendations_section,
                    sizing_mode="stretch_width",
            )

__all__ = ["PANEL_AVAILABLE"]
if PANEL_AVAILABLE:
    __all__.append("DiagnosticsDashboard")
