"""Panel dashboard accessor classes for interactive image exploration.

Dashboards are registered via the ``@register_dashboard`` decorator when imported.
Use ``available_dashboards()`` to list registered dashboards and ``get_dashboard(name)``
to retrieve a dashboard class by name.
"""

# Import dashboards to trigger @register_dashboard decorators
from ._base_dashboard import BaseDashboard
from ._detect_mode_dashboard import DetectModeDashboard
from phenotypic.tools_.panel_ import PANEL_AVAILABLE

if PANEL_AVAILABLE:
    from ._diagnostics_dashboard import DiagnosticsDashboard

# Re-export registry functions for convenience
from phenotypic.tools_.register import available_dashboards, get_dashboard

__all__ = [
    "BaseDashboard",
    "DetectModeDashboard",
    "PANEL_AVAILABLE",
    "available_dashboards",
    "get_dashboard",
]

if PANEL_AVAILABLE:
    __all__.append("DiagnosticsDashboard")
