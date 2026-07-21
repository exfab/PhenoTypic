"""Standalone replacements for the removed ``Image.plot`` diagnostics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from phenotypic.abc_.plotting import PlotImage


def _diagnostics_figure(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Copy figure metadata while changing diagnostics to a call-time subject."""
    from phenotypic._core._image_parts.plot_accessor._diagnostics_plotter import (
        DiagnosticsPlotter,
    )

    source = getattr(DiagnosticsPlotter, name).__figure_spec__

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        method.__figure_spec__ = replace(  # type: ignore[attr-defined]
            source,
            method=method,
            wants_subject=True,
            subject_param="subject",
        )
        return method

    return decorate


class PlotDiagnostics(BaseModel, PlotImage):
    """Build image-quality diagnostics from a weakly bound call-time image."""

    model_config = ConfigDict(extra="forbid")

    def _render_diagnostic(
        self, subject: Any, name: str, **controls: Any
    ) -> Any:
        """Render through a short-lived legacy helper without retaining it."""
        from phenotypic._core._image_parts.plot_accessor._diagnostics_plotter import (
            DiagnosticsPlotter,
        )

        if subject is None:
            raise TypeError("PlotDiagnostics requires an Image subject")
        return getattr(DiagnosticsPlotter(subject), name)(**controls)

    @_diagnostics_figure("fig_intensity_histogram")
    def fig_intensity_histogram(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(
            subject, "fig_intensity_histogram", **controls
        )

    @_diagnostics_figure("fig_noise_autocorrelation")
    def fig_noise_autocorrelation(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(
            subject, "fig_noise_autocorrelation", **controls
        )

    @_diagnostics_figure("fig_power_spectral_density")
    def fig_power_spectral_density(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(
            subject, "fig_power_spectral_density", **controls
        )

    @_diagnostics_figure("fig_detection_matrix")
    def fig_detection_matrix(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(subject, "fig_detection_matrix", **controls)

    @_diagnostics_figure("fig_local_contrast_map")
    def fig_local_contrast_map(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(
            subject, "fig_local_contrast_map", **controls
        )

    @_diagnostics_figure("fig_contrast_metrics")
    def fig_contrast_metrics(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(subject, "fig_contrast_metrics", **controls)

    @_diagnostics_figure("fig_gradient_magnitude")
    def fig_gradient_magnitude(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(
            subject, "fig_gradient_magnitude", **controls
        )

    @_diagnostics_figure("fig_orientation_coherence")
    def fig_orientation_coherence(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(
            subject, "fig_orientation_coherence", **controls
        )

    @_diagnostics_figure("fig_ridge_response")
    def fig_ridge_response(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(subject, "fig_ridge_response", **controls)

    @_diagnostics_figure("fig_background_estimate")
    def fig_background_estimate(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(
            subject, "fig_background_estimate", **controls
        )

    @_diagnostics_figure("fig_local_variance")
    def fig_local_variance(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(subject, "fig_local_variance", **controls)

    @_diagnostics_figure("fig_quality_summary")
    def fig_quality_summary(self, subject: Any, **controls: Any) -> Any:
        return self._render_diagnostic(subject, "fig_quality_summary", **controls)


class PlotDetectModes(BaseModel, PlotImage):
    """Compare every registered detection mode for a supplied image."""

    model_config = ConfigDict(extra="forbid")

    def inspect(
        self,
        subject: Any = None,
        *,
        for_save: bool = False,
        **overrides: Any,
    ) -> Any:
        del for_save
        from phenotypic._core._image_parts.plot_accessor._detect_modes_plotter import (
            DetectModesPlotter,
        )

        if subject is None:
            raise TypeError("PlotDetectModes.inspect requires an Image subject")
        return DetectModesPlotter(subject).inspect(**overrides)

    def report(self, subject: Any = None, **overrides: Any) -> Any:
        from phenotypic._core._image_parts.plot_accessor._detect_modes_plotter import (
            DetectModesPlotter,
        )

        if subject is None:
            raise TypeError("PlotDetectModes.report requires an Image subject")
        return DetectModesPlotter(subject).report(**overrides)


__all__ = ["PlotDetectModes", "PlotDiagnostics"]
