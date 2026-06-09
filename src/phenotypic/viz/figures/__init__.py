"""Figure construction helpers and the centralized Plotly theme.

Re-exports the public theme surface from :mod:`._theme` so callers (and
the forthcoming ``@figure`` decorator) can import it from the package
root.
"""
from __future__ import annotations

from ._mpl_theme import phenotypic_mpl_context, phenotypic_rc
from ._theme import (
    FAILED_FILL,
    FONT_FAMILY,
    FONT_FAMILY_MONO,
    OKABE_ITO,
    PHENOTYPIC_TEMPLATE_NAME,
    SEQUENTIAL_COLORSCALE,
    apply_theme,
    register_phenotypic_template,
)

__all__ = [
    "PHENOTYPIC_TEMPLATE_NAME",
    "OKABE_ITO",
    "SEQUENTIAL_COLORSCALE",
    "FAILED_FILL",
    "FONT_FAMILY",
    "FONT_FAMILY_MONO",
    "register_phenotypic_template",
    "apply_theme",
    "phenotypic_rc",
    "phenotypic_mpl_context",
]
