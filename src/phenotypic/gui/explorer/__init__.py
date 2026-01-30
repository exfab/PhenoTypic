"""Pipeline Variant Explorer - programmatic and GUI interfaces for parameter sweeps.

This module provides tools for exploring pipeline configurations by:
- Building exploration graphs with branching paths
- Configuring parameter sweeps on operations
- Executing all variants in parallel
- Comparing results

The programmatic API works without GUI dependencies (only requires networkx).
The GUI components require the full [gui] optional dependencies.

Programmatic Usage:
    >>> from phenotypic.gui.explorer import PipelineGraph, SweepSpec, SweepExecutor
    >>> from phenotypic.enhance import GaussianBlur
    >>> from phenotypic.detect import OtsuDetector
    >>>
    >>> # Build graph
    >>> graph = PipelineGraph()
    >>> gauss = graph.add_operation(GaussianBlur, sigma=1.5)
    >>> otsu = graph.add_operation(OtsuDetector)
    >>> output = graph.add_output()
    >>> graph.connect(gauss, otsu).connect(otsu, output)
    >>>
    >>> # Add sweep
    >>> graph.add_sweep(gauss, SweepSpec.from_range('sigma', 1.0, 3.0, 0.5))
    >>>
    >>> # Execute
    >>> executor = SweepExecutor(graph, output_dir='./results')
    >>> results = executor.run(images=['./plate.tif'])

GUI Usage:
    >>> from phenotypic.gui import PipelineExplorer
    >>> explorer = PipelineExplorer()
    >>> explorer.panel()
"""

from __future__ import annotations

from ._sweep_spec import SweepSpec
from ._pipeline_graph import PipelineGraph, GraphNode
from ._sweep_executor import SweepExecutor
from ._sweep_results import SweepResult, SweepResults

__all__ = [
    "SweepSpec",
    "PipelineGraph",
    "GraphNode",
    "SweepExecutor",
    "SweepResult",
    "SweepResults",
]
