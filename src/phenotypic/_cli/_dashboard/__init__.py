"""Dashboard generation package for the PhenoTypic CLI."""

from ._analysis_data import write_analysis_sidecar
from ._generator import generate_dashboard, regenerate_dashboard_artifacts
from ._manifest_builder import build_manifest

__all__ = [
    "generate_dashboard",
    "regenerate_dashboard_artifacts",
    "build_manifest",
    "write_analysis_sidecar",
]
