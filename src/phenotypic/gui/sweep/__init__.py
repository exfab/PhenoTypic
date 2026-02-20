"""Napari-based viewer for browsing sweep output directories.

Launch from the command line::

    uv run python -m phenotypic.gui.sweep [path/to/sweep_output]

Or use programmatically::

    from phenotypic.gui.sweep import NapariSweepViewer, launch_sweep_viewer
    launch_sweep_viewer(Path("./sweep_output"))
"""

from __future__ import annotations

from ._sweep_data_model import (
    PipelineConfig,
    SweepHDF5File,
    SweepOutputData,
    SweepOutputScanner,
)

__all__ = [
    "NapariSweepViewer",
    "launch_sweep_viewer",
    "SweepOutputScanner",
    "SweepOutputData",
    "PipelineConfig",
    "SweepHDF5File",
]


def __getattr__(name: str):
    """Lazy-import napari-dependent classes."""
    if name == "NapariSweepViewer":
        from ._napari_sweep_viewer import NapariSweepViewer

        return NapariSweepViewer
    if name == "launch_sweep_viewer":
        from ._napari_sweep_viewer import launch_sweep_viewer

        return launch_sweep_viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
