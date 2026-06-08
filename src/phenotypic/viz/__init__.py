"""Shared visualization layer for PhenoTypic.

Hosts the centralized Plotly theme that every figure routes through. The
public theme entry points are re-exported here for convenience.
"""
from __future__ import annotations

from .figures import PHENOTYPIC_TEMPLATE_NAME, apply_theme

__all__ = [
    "PHENOTYPIC_TEMPLATE_NAME",
    "apply_theme",
]
