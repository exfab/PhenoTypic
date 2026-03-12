"""Registry utilities for PhenoTypic components.

Provides registration decorators and lookup functions for extensible
components like plotters and dashboards.

Examples:
    Register a new plotter::

        from phenotypic.tools_.register import register_plotter

        @register_plotter
        class MyPlotter(BasePlotter):
            name = "my_plot"

            def my_plot(self, **kwargs):
                ...

    Query available plotters::

        from phenotypic.tools_.register import available_plotters, get_plotter

        print(available_plotters())
        # ('all', 'diagnostics', 'morph_progression', ...)

        plotter_cls = get_plotter("overlay")
"""

from ._analysis_plugin_registry import (
    AnalysisPluginRegistry,
    available_analysis_plugins,
    get_analysis_plugin,
    register_analysis,
)
from ._base_registry import BaseRegistry
from ._dashboard_registry import (
    DashboardRegistry,
    available_dashboards,
    get_dashboard,
    register_dashboard,
)
from ._plotter_registry import (
    PlotterRegistry,
    available_plotters,
    get_plotter,
    register_plotter,
)

__all__ = [
    "AnalysisPluginRegistry",
    "BaseRegistry",
    "DashboardRegistry",
    "PlotterRegistry",
    "available_analysis_plugins",
    "available_dashboards",
    "available_plotters",
    "get_analysis_plugin",
    "get_dashboard",
    "get_plotter",
    "register_analysis",
    "register_dashboard",
    "register_plotter",
]
