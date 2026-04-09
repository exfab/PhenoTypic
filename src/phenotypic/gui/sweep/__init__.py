"""Napari-based viewer for browsing sweep output directories.

Launch from the command line::

    pixi run python -m phenotypic.gui.sweep [path/to/sweep_output]

Or use programmatically::

    from phenotypic.gui.sweep import NapariSweepViewer, launch_sweep_viewer
    launch_sweep_viewer(Path("./sweep_output"))
"""

from __future__ import annotations

from ._sweep_data_model import (
    IntermediateStep,
    PipelineConfig,
    ResolvedLayerSources,
    SweepHDF5File,
    SweepOutputData,
    SweepOutputScanner,
    build_layer_resolution_index,
)

__all__ = [
    "NapariSweepViewer",
    "launch_sweep_viewer",
    "SweepOutputScanner",
    "SweepOutputData",
    "PipelineConfig",
    "SweepHDF5File",
    "IntermediateStep",
    "ResolvedLayerSources",
    "build_layer_resolution_index",
    "StepSliderWidget",
    "ParameterExplorerWidget",
    "PipelineConfigBar",
]


def __getattr__(name: str):
    """Lazy-import napari-dependent classes."""
    if name == "NapariSweepViewer":
        from ._napari_sweep_viewer import NapariSweepViewer

        return NapariSweepViewer
    if name == "launch_sweep_viewer":
        from ._napari_sweep_viewer import launch_sweep_viewer

        return launch_sweep_viewer
    if name == "StepSliderWidget":
        from ._step_slider_widget import StepSliderWidget

        return StepSliderWidget
    if name == "ParameterExplorerWidget":
        from ._parameter_explorer_widget import ParameterExplorerWidget

        return ParameterExplorerWidget
    if name == "PipelineConfigBar":
        from ._pipeline_config_bar import PipelineConfigBar

        return PipelineConfigBar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
