"""PhenoTypic GUI components for interactive pipeline building.

This module provides Panel-based interactive interfaces for building and testing
ImagePipelines in Jupyter notebooks.

Requires optional dependencies:
    pip install phenotypic[gui]

or with uv:
    uv sync --extra gui

Components:
- PipelineBuilder: Main GUI for building pipelines interactively
- InstanceManager: Workspace manager for saving/loading pipelines
- OperationRegistry: Metadata registry for discovering operations

Global session management (like napari):
- get_global_manager(): Get or create global InstanceManager
- reset_global_manager(): Reset with new workspace

Examples:
    Basic usage (no pn.extension() needed in Jupyter):

    >>> from phenotypic.gui import PipelineBuilder
    >>> from phenotypic.data import load_synth_yeast_plate
    >>>
    >>> image = load_synth_yeast_plate()
    >>> builder = PipelineBuilder(image=image)  # Auto-uses global manager
    >>> builder.panel()  # Auto-initializes Panel in Jupyter

    Manual manager (advanced):

    >>> from phenotypic.gui import PipelineBuilder, InstanceManager
    >>> manager = InstanceManager(workspace="./my_pipelines")
    >>> builder = PipelineBuilder(manager=manager, image=image)
    >>> builder.panel()
"""

from __future__ import annotations


def _check_gui_deps() -> bool:
    """Check if GUI dependencies are available.

    Returns:
        True if panel and param are installed, False otherwise
    """
    import importlib.util

    return all(
        importlib.util.find_spec(pkg) is not None for pkg in ["panel", "param"]
    )


GUI_AVAILABLE = _check_gui_deps()


def __getattr__(name: str):
    """Lazy import with helpful error message for missing dependencies.

    Args:
        name: Attribute name to import

    Returns:
        Requested module/class

    Raises:
        ImportError: If GUI dependencies not installed
        AttributeError: If attribute doesn't exist
    """
    # Core GUI components (require panel/param)
    if name in ("PipelineBuilder", "InstanceManager", "OperationRegistry",
                "get_global_manager", "reset_global_manager"):
        if not GUI_AVAILABLE and name == "PipelineBuilder":
            raise ImportError(
                f"GUI component '{name}' requires optional dependencies. "
                "Install with: pip install phenotypic[gui]"
            )
        if name == "PipelineBuilder":
            from ._pipeline_builder import PipelineBuilder

            return PipelineBuilder
        elif name == "InstanceManager":
            from ._instance_manager import InstanceManager

            return InstanceManager
        elif name == "OperationRegistry":
            from ._operation_registry import OperationRegistry

            return OperationRegistry
        elif name == "get_global_manager":
            from ._global_session import get_global_manager

            return get_global_manager
        elif name == "reset_global_manager":
            from ._global_session import reset_global_manager

            return reset_global_manager

    # Napari sweep viewer (requires napari, no Panel needed)
    if name in ("NapariSweepViewer", "launch_sweep_viewer"):
        if name == "NapariSweepViewer":
            from .sweep import NapariSweepViewer

            return NapariSweepViewer
        elif name == "launch_sweep_viewer":
            from .sweep import launch_sweep_viewer

            return launch_sweep_viewer

    # Dash node-graph builder (requires dash + dash-cytoscape, lazy)
    if name in ("create_builder_app", "BuilderState"):
        if name == "create_builder_app":
            from .builder import create_app

            return create_app
        elif name == "BuilderState":
            from .builder import BuilderState

            return BuilderState

    # Dash CLI-output results viewer (Dash + OpenSeadragon, lazy)
    if name in ("create_results_viewer_app", "launch_results_viewer"):
        if name == "create_results_viewer_app":
            from .results_viewer import create_app

            return create_app
        elif name == "launch_results_viewer":
            from .results_viewer import launch_results_viewer

            return launch_results_viewer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core GUI components
    "PipelineBuilder",
    "InstanceManager",
    "OperationRegistry",
    "GUI_AVAILABLE",
    "get_global_manager",
    "reset_global_manager",
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
