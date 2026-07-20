"""Registry utilities for PhenoTypic analysis dashboard components.

Examples:
    Register a new analysis plugin::

        from phenotypic.sdk_.register import register_analysis

        @register_analysis
        class MyAnalysisPlugin(BaseAnalysisPlugin):
            call_name = "my_analysis"

    Query available analysis plugins::

        from phenotypic.sdk_.register import available_analysis_plugins

        print(available_analysis_plugins())
"""

from ._analysis_plugin_registry import (
    AnalysisPluginRegistry,
    available_analysis_plugins,
    get_analysis_plugin,
    register_analysis,
)
from ._base_registry import BaseRegistry

__all__ = [
    "AnalysisPluginRegistry",
    "BaseRegistry",
    "available_analysis_plugins",
    "get_analysis_plugin",
    "register_analysis",
]
