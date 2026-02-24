"""Reusable GUI components for PhenoTypic.

This module provides modular Panel components that can be composed
for different UIs (PipelineBuilder, SweepExecutor, etc.).
"""

from __future__ import annotations

# Components will be imported as needed (lazy imports to avoid Panel dependency)
__all__ = [
    "ParamEditor",
    "OperationCard",
    "EmbeddedPipelineEditor",
    "AddOperationMenu",
    "PreviewPanel",
]


def __getattr__(name: str):
    """Lazy import components to avoid Panel dependency at module level."""
    if name == "ParamEditor":
        from ._param_editor import ParamEditor

        return ParamEditor
    elif name == "OperationCard":
        from ._operation_card import OperationCard

        return OperationCard
    elif name == "EmbeddedPipelineEditor":
        from ._embedded_pipeline import EmbeddedPipelineEditor

        return EmbeddedPipelineEditor
    elif name == "AddOperationMenu":
        from ._add_operation_menu import AddOperationMenu

        return AddOperationMenu
    elif name == "PreviewPanel":
        from ._preview_panel import PreviewPanel

        return PreviewPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
