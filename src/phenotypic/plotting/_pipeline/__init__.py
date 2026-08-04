"""Private runtime integration for configured pipeline plots.

This package owns pipeline bindings, lifecycle dispatch, figure publication,
and named analysis artifact resolution. Plot authors should use contracts from
``phenotypic.abc_.plotting`` and concrete models from ``phenotypic.plotting``.
"""

from ._adapter import FigureAdapter
from ._analysis_artifacts import (
    AnalysisArtifactIntegrityError,
    AnalysisArtifactPaths,
    AnalysisManifest,
    AnalysisManifestEntry,
    AnalysisManifestError,
    analysis_manifest_path,
    build_analysis_manifest_entry,
    file_sha256,
    named_analysis_csv_path,
    named_analysis_parquet_path,
    named_analysis_paths,
    publish_analysis_manifest_entry,
    read_analysis_manifest,
    recover_analysis_publication,
    validate_analysis_id,
    write_analysis_publication_journal,
)
from ._analysis_registry import (
    AnalysisInputLike,
    AnalysisNotFoundError,
    AnalysisRegistry,
    AnalysisResult,
)
from ._bindings import (
    AnalysisInput,
    MeasurementInput,
    PipelineObjectRef,
    PlotBinding,
    PlotInput,
    deserialize_plot_bindings,
    normalize_plot_bindings,
    serialize_plot_binding,
)
from ._coordinator import PlotCoordinator, QcPlotSubject
from ._writer import (
    PlotPublicationBlocked,
    publish_plot_output,
    safe_path_component,
)

__all__ = [
    "AnalysisArtifactIntegrityError",
    "AnalysisArtifactPaths",
    "AnalysisInput",
    "AnalysisInputLike",
    "AnalysisManifest",
    "AnalysisManifestEntry",
    "AnalysisManifestError",
    "AnalysisNotFoundError",
    "AnalysisRegistry",
    "AnalysisResult",
    "FigureAdapter",
    "MeasurementInput",
    "PipelineObjectRef",
    "PlotBinding",
    "PlotCoordinator",
    "PlotInput",
    "PlotPublicationBlocked",
    "QcPlotSubject",
    "analysis_manifest_path",
    "build_analysis_manifest_entry",
    "deserialize_plot_bindings",
    "file_sha256",
    "named_analysis_csv_path",
    "named_analysis_parquet_path",
    "named_analysis_paths",
    "normalize_plot_bindings",
    "publish_analysis_manifest_entry",
    "publish_plot_output",
    "read_analysis_manifest",
    "recover_analysis_publication",
    "safe_path_component",
    "serialize_plot_binding",
    "validate_analysis_id",
    "write_analysis_publication_journal",
]
