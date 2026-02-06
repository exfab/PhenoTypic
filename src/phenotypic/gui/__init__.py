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

Pipeline Exploration (parameter sweeps):
- PipelineGraph: Graph for exploring pipeline variants
- SweepSpec: Parameter sweep specification
- SweepExecutor: Batch execution engine
- SweepResults: Results container with analysis methods

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

    Parameter sweep (programmatic):

    >>> from phenotypic.gui.explorer import PipelineGraph, SweepSpec, SweepExecutor
    >>> from phenotypic.enhance import GaussianBlur
    >>> from phenotypic.detect import OtsuDetector
    >>>
    >>> graph = PipelineGraph()
    >>> gauss = graph.add_operation(GaussianBlur, sigma=1.5)
    >>> otsu = graph.add_operation(OtsuDetector)
    >>> output = graph.add_output()
    >>> graph.connect(gauss, otsu).connect(otsu, output)
    >>> graph.add_sweep(gauss, SweepSpec.from_range('sigma', 1.0, 3.0, 0.5))
    >>>
    >>> executor = SweepExecutor(graph, output_dir='./results')
    >>> results = executor.run(images=['./plate.tif'])
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

    # Explorer components (programmatic API - only require networkx)
    if name in ("PipelineGraph", "SweepSpec", "SweepExecutor",
                "SweepResult", "SweepResults", "GraphNode"):
        if name == "PipelineGraph":
            from .explorer import PipelineGraph

            return PipelineGraph
        elif name == "SweepSpec":
            from .explorer import SweepSpec

            return SweepSpec
        elif name == "SweepExecutor":
            from .explorer import SweepExecutor

            return SweepExecutor
        elif name == "SweepResult":
            from .explorer import SweepResult

            return SweepResult
        elif name == "SweepResults":
            from .explorer import SweepResults

            return SweepResults
        elif name == "GraphNode":
            from .explorer import GraphNode

            return GraphNode

    # Explorer GUI components (require Panel)
    if name in ("PipelineExplorer", "PipelineNodeEditor"):
        if not GUI_AVAILABLE:
            raise ImportError(
                f"GUI component '{name}' requires optional dependencies. "
                "Install with: pip install phenotypic[gui]"
            )
        if name == "PipelineExplorer":
            from .explorer import PipelineExplorer

            return PipelineExplorer
        elif name == "PipelineNodeEditor":
            from .explorer import PipelineNodeEditor

            return PipelineNodeEditor

    # Viewer components (require Panel)
    if name in ("SweepComparisonWidget", "SweepHTMLExporter"):
        if name == "SweepHTMLExporter":
            # HTML exporter only needs Jinja2, not Panel
            from .viewer import SweepHTMLExporter

            return SweepHTMLExporter
        elif name == "SweepComparisonWidget":
            if not GUI_AVAILABLE:
                raise ImportError(
                    f"GUI component '{name}' requires optional dependencies. "
                    "Install with: pip install phenotypic[gui]"
                )
            from .viewer import SweepComparisonWidget

            return SweepComparisonWidget

    # Napari sweep viewer (requires napari, no Panel needed)
    if name in ("NapariSweepViewer", "launch_sweep_viewer"):
        if name == "NapariSweepViewer":
            from .sweep import NapariSweepViewer

            return NapariSweepViewer
        elif name == "launch_sweep_viewer":
            from .sweep import launch_sweep_viewer

            return launch_sweep_viewer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core GUI components
    "PipelineBuilder",
    "InstanceManager",
    "OperationRegistry",
    "GUI_AVAILABLE",
    "get_global_manager",
    "reset_global_manager",
    # Explorer components (programmatic API)
    "PipelineGraph",
    "SweepSpec",
    "SweepExecutor",
    "SweepResult",
    "SweepResults",
    "GraphNode",
    # Explorer GUI components
    "PipelineExplorer",
    "PipelineNodeEditor",
    # Viewer components
    "SweepComparisonWidget",
    "SweepHTMLExporter",
    # Napari sweep viewer
    "NapariSweepViewer",
    "launch_sweep_viewer",
]
