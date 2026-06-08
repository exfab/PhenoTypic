"""Figure construction helpers and the centralized Plotly theme.

Re-exports the public theme surface from :mod:`._theme` so callers (and
the forthcoming ``@figure`` decorator) can import it from the package
root.
"""
from __future__ import annotations

from ._theme import (
    FONT_FAMILY,
    OKABE_ITO,
    PHENOTYPIC_TEMPLATE_NAME,
    apply_theme,
    register_phenotypic_template,
)

__all__ = [
    "PHENOTYPIC_TEMPLATE_NAME",
    "OKABE_ITO",
    "FONT_FAMILY",
    "register_phenotypic_template",
    "apply_theme",
]
