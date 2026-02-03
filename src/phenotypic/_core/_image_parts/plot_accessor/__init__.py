"""Plot accessor classes for PhenoTypic visualization."""

from ._all_data_plotter import AllDataPlotter
from ._base_plotter import BasePlotter
from ._diagnostics_dashboard import PANEL_AVAILABLE as _DIAG_PANEL_AVAILABLE
from ._diagnostics_plotter import DiagnosticsPlotter
from ._diagnostics_types import PanelDescription
from ._morphology_plotter import MorphologyPlotter
from ._overlay_plotter import OverlayPlotter
from ._size_distribution_plotter import SizeDistributionPlotter
from ._spatial_plotter import SpatialPlotter
from ._threshold_plotter import ThresholdPlotter

if _DIAG_PANEL_AVAILABLE:
    from ._diagnostics_dashboard import DiagnosticsDashboard

__all__ = [
    "AllDataPlotter",
    "BasePlotter",
    "DiagnosticsPlotter",
    "MorphologyPlotter",
    "OverlayPlotter",
    "PanelDescription",
    "SizeDistributionPlotter",
    "SpatialPlotter",
    "ThresholdPlotter",
]

if _DIAG_PANEL_AVAILABLE:
    __all__.append("DiagnosticsDashboard")



