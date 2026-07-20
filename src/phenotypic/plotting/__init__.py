"""Public runtime plotting configuration and output infrastructure.

Exports are resolved lazily so core pipeline imports do not pull pandas, Dash,
Plotly, or Matplotlib into lightweight worker and schema-loading paths.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AnalysisInput": "._bindings",
    "MeasurementInput": "._bindings",
    "PipelineObjectRef": "._bindings",
    "PlotBinding": "._bindings",
    "FigureAdapter": "._adapter",
    "FigureLike": "._output",
    "PlotOutput": "._output",
    "PlotPage": "._output",
    "canonical_group_key": "._output",
    "publish_plot_output": "._writer",
    "safe_path_component": "._writer",
    "PlotMeasTimeSeries": "._plot_meas_time_series",
    "PlotDetectModes": "._image_plots",
    "PlotDiagnostics": "._image_plots",
    "AnalysisArtifactIntegrityError": "._analysis_artifacts",
    "AnalysisArtifactPaths": "phenotypic.sdk_._io_constants",
    "AnalysisManifest": "._analysis_artifacts",
    "AnalysisManifestEntry": "._analysis_artifacts",
    "AnalysisManifestError": "._analysis_artifacts",
    "analysis_manifest_path": "phenotypic.sdk_._io_constants",
    "build_analysis_manifest_entry": "._analysis_artifacts",
    "file_sha256": "._analysis_artifacts",
    "named_analysis_csv_path": "phenotypic.sdk_._io_constants",
    "named_analysis_parquet_path": "phenotypic.sdk_._io_constants",
    "named_analysis_paths": "phenotypic.sdk_._io_constants",
    "publish_analysis_manifest_entry": "._analysis_artifacts",
    "read_analysis_manifest": "._analysis_artifacts",
    "recover_analysis_publication": "._analysis_artifacts",
    "write_analysis_publication_journal": "._analysis_artifacts",
    "validate_analysis_id": "phenotypic.sdk_._io_constants",
    "AnalysisNotFoundError": "._analysis_registry",
    "AnalysisRegistry": "._analysis_registry",
    "AnalysisResult": "._analysis_registry",
    "PlotCoordinator": "._coordinator",
    "QcPlotSubject": "._coordinator",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - normal Python attribute path
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
