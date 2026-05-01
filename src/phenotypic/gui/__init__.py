"""PhenoTypic GUI components.

This package provides interactive interfaces for building and visualising
ImagePipelines.  Components are lazy-loaded so that optional dependencies
(Panel, napari, Dash) are only imported when actually used.

Sub-packages
------------
- ``gui.builder``        — Dash node-graph pipeline builder
- ``gui.results_viewer`` — Dash CLI-output results viewer
- ``gui.sweep``          — napari parameter-sweep viewer

Utilities
---------
- ``OperationRegistry``  — discovers operations and extracts parameter metadata
"""

from __future__ import annotations


def _check_gui_deps() -> bool:
    """Check if Panel GUI dependencies are available."""
    import importlib.util

    return all(
        importlib.util.find_spec(pkg) is not None for pkg in ["panel", "param"]
    )


GUI_AVAILABLE = _check_gui_deps()


def __getattr__(name: str):
    """Lazy import with helpful error message for missing dependencies."""
    if name == "OperationRegistry":
        from ._operation_registry import OperationRegistry

        return OperationRegistry

    # Napari sweep viewer (requires napari, no Panel needed)
    if name == "NapariSweepViewer":
        from .sweep import NapariSweepViewer

        return NapariSweepViewer
    if name == "launch_sweep_viewer":
        from .sweep import launch_sweep_viewer

        return launch_sweep_viewer

    # Dash node-graph builder (requires dash + dash-cytoscape, lazy)
    if name == "create_builder_app":
        from .builder import create_app

        return create_app
    if name == "BuilderState":
        from .builder import BuilderState

        return BuilderState

    # Dash CLI-output results viewer (Dash + OpenSeadragon, lazy)
    if name == "create_results_viewer_app":
        from .results_viewer import create_app

        return create_app
    if name == "launch_results_viewer":
        from .results_viewer import launch_results_viewer

        return launch_results_viewer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OperationRegistry",
    "GUI_AVAILABLE",
    # Napari sweep viewer
    "NapariSweepViewer",
    "launch_sweep_viewer",
    # Dash node-graph builder
    "create_builder_app",
    "BuilderState",
    # Dash CLI-output results viewer
    "create_results_viewer_app",
    "launch_results_viewer",
]
