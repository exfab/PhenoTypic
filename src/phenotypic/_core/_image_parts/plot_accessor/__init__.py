"""Plot accessor classes for PhenoTypic visualization.

Plotters are registered via the ``@register_plotter`` decorator when imported.
Use ``available_plotters()`` to list registered plotters and ``get_plotter(name)``
to retrieve a plotter class by name.
"""

# Import all plotters to trigger @register_plotter decorators
from ._all_data_plotter import AllDataPlotter
from ._base_plotter import BasePlotter
from ._detect_modes_plotter import DetectModesPlotter
from ._diagnostics_plotter import DiagnosticsPlotter
from ._diagnostics_types import PanelDescription
from ._morphology_plotter import MorphologyPlotter
from ._size_distribution_plotter import SizeDistributionPlotter
from ._spatial_plotter import SpatialPlotter
from ._threshold_plotter import ThresholdPlotter

# Re-export registry functions for convenience
from phenotypic.sdk_.register import available_plotters, get_plotter

__all__ = [
    "AllDataPlotter",
    "BasePlotter",
    "DetectModesPlotter",
    "DiagnosticsPlotter",
    "MorphologyPlotter",
    "PanelDescription",
    "SizeDistributionPlotter",
    "SpatialPlotter",
    "ThresholdPlotter",
    "available_plotters",
    "get_plotter",
]



