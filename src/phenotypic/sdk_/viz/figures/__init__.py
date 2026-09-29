"""Figure construction helpers, the Plotly theme and the static-figure theme.

Re-exports the public theme surface from :mod:`._theme` (Plotly, GUI charts)
and :mod:`._mpl_theme` (matplotlib, static figures for print) so callers can
import it from the package root.
"""
from __future__ import annotations

from ._mpl_theme import (
    FIGURE_WIDTHS_MM,
    MAX_FIGURE_HEIGHT_MM,
    export_figure,
    figure_size_mm,
    phenotypic_mpl_context,
    phenotypic_rc,
)
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
    "FIGURE_WIDTHS_MM",
    "MAX_FIGURE_HEIGHT_MM",
    "figure_size_mm",
    "export_figure",
]
