"""Centralized Plotly theme for PhenoTypic figures.

Registers a single ``plotly.io`` template named ``"phenotypic"`` carrying
the brand palette, Okabe-Ito data series order, and Roboto typography
defined in ``DESIGN.md`` (and mirrored by
``phenotypic.gui._design``). The ``@figure`` decorator applies this
theme to every figure via :func:`apply_theme`, so individual call sites
never re-spell hex codes, fonts, or axis styling.

Design-token sources (single source of truth is ``DESIGN.md``):

* Brand / UI palette -- ``01 -- Color Palette / Primary Colors``
  (navy ``#003660``, blue ``#1b75bc``, gold ``#febc11``).
* Data series order -- ``06 -- Data Visualization / Categorical Series
  Order`` and the matplotlib ``OKABE_ITO`` block in ``07 -- Code
  Integration`` (navy-anchored: navy, orange, sky, green, blue, purple,
  vermilion).
* Typography -- ``02 -- Typography`` (Roboto-based stack).
* Chart styling -- ``06 -- Data Visualization / Chart Styling Rules``
  (gridlines ``#e8ecf2``, axes ``#dde3ed``, muted axis labels
  ``#8892a4``, navy title).

This module imports only ``plotly`` and the stdlib. It is deliberately
free of ``dash`` / project imports so the theme stays cheap to import
from any layer (CLI, GUI, notebooks, tests).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "PHENOTYPIC_TEMPLATE_NAME",
    "NAVY",
    "BLUE",
    "GOLD",
    "WHITE",
    "BG",
    "GRID",
    "AXIS",
    "MUTED",
    "BODY",
    "OKABE_ITO",
    "FONT_FAMILY",
    "register_phenotypic_template",
    "apply_theme",
]

#: Name the template is registered under in :data:`plotly.io.templates`.
PHENOTYPIC_TEMPLATE_NAME: str = "phenotypic"

# ---------------------------------------------------------------------------
# Brand / UI palette (DESIGN.md "01 -- Color Palette / Primary Colors")
# ---------------------------------------------------------------------------

#: Brand navy -- headings, title font, primary anchor.
NAVY: str = "#003660"
#: Brand blue -- accents, interactive states.
BLUE: str = "#1b75bc"
#: Brand gold -- highlights, emphasized accents.
GOLD: str = "#febc11"

# ---------------------------------------------------------------------------
# Surface / neutral tokens (DESIGN.md "Surface & Neutral Tokens" +
# "Chart Styling Rules")
# ---------------------------------------------------------------------------

#: Card / plot surface.
WHITE: str = "#ffffff"
#: App background (paper backdrop behind the plotting area).
BG: str = "#f5f7fa"
#: Chart gridlines (``--color-rule``).
GRID: str = "#e8ecf2"
#: Axis lines (``--color-border``).
AXIS: str = "#dde3ed"
#: Secondary text -- axis labels, captions (``--color-muted``).
MUTED: str = "#8892a4"
#: Primary body text (``--color-body``).
BODY: str = "#2e3a4e"

# ---------------------------------------------------------------------------
# Okabe-Ito data palette (DESIGN.md "Categorical Series Order" + the
# matplotlib OKABE_ITO block in "07 -- Code Integration")
# ---------------------------------------------------------------------------
#
# Colorblind-safe series order is fixed and must not be reordered:
# navy (series 1, UI-harmonized), orange, sky, green, blue, purple, then
# vermilion reserved for the error/alert series. Black closes the cycle so
# overflow series past the named seven fall back to a neutral ink rather
# than wrapping to navy.

#: Categorical color cycle for data series, in DESIGN.md's fixed order.
OKABE_ITO: tuple[str, ...] = (
    "#003660",  # navy      -- series 1, UI-harmonized
    "#E69F00",  # orange    -- series 2
    "#56B4E9",  # sky blue  -- series 3
    "#009E73",  # green     -- series 4
    "#0072B2",  # blue      -- series 5
    "#CC79A7",  # purple    -- series 6
    "#D55E00",  # vermilion -- error / alert series
    "#000000",  # black     -- overflow / ink
)

# ---------------------------------------------------------------------------
# Typography (DESIGN.md "02 -- Typography")
# ---------------------------------------------------------------------------

#: Roboto-based font stack matching the active GUI font in
#: ``phenotypic.gui._design``.
FONT_FAMILY: str = (
    "'Roboto', -apple-system, BlinkMacSystemFont, "
    '"Segoe UI", "Helvetica Neue", Arial, sans-serif'
)


def register_phenotypic_template() -> None:
    """Build and register the ``"phenotypic"`` Plotly template.

    Constructs a :class:`plotly.graph_objects.layout.Template` carrying
    the brand palette, Okabe-Ito ``colorway``, Roboto typography, and the
    axis / grid / legend styling from ``DESIGN.md``, then stores it in
    :data:`plotly.io.templates` under :data:`PHENOTYPIC_TEMPLATE_NAME`.

    The function is idempotent: it rebuilds and re-assigns the template on
    every call, so importing this module (which calls it once) and any
    later explicit calls leave a single, identical entry. It is invoked
    automatically at import time.

    Example:
        >>> import plotly.io as pio
        >>> register_phenotypic_template()
        >>> "phenotypic" in pio.templates
        True
    """
    template = go.layout.Template(
        layout=dict(
            colorway=list(OKABE_ITO),
            font=dict(family=FONT_FAMILY, color=BODY),
            paper_bgcolor=BG,
            plot_bgcolor=WHITE,
            title=dict(font=dict(family=FONT_FAMILY, color=NAVY)),
            xaxis=dict(
                gridcolor=GRID,
                linecolor=AXIS,
                zerolinecolor=GRID,
                tickfont=dict(family=FONT_FAMILY, color=MUTED),
                title=dict(font=dict(family=FONT_FAMILY, color=BODY)),
            ),
            yaxis=dict(
                gridcolor=GRID,
                linecolor=AXIS,
                zerolinecolor=GRID,
                tickfont=dict(family=FONT_FAMILY, color=MUTED),
                title=dict(font=dict(family=FONT_FAMILY, color=BODY)),
            ),
            legend=dict(font=dict(family=FONT_FAMILY, color=BODY)),
        )
    )
    pio.templates[PHENOTYPIC_TEMPLATE_NAME] = template


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the PhenoTypic theme to ``fig`` in place and return it.

    Ensures the ``"phenotypic"`` template is registered, then sets the
    figure to use it *composed* with Plotly's default template
    (``"plotly+phenotypic"``). Composition preserves Plotly's built-in
    trace defaults while letting the PhenoTypic layer override layout
    styling (colorway, fonts, axes, legend).

    Traces (``fig.data``) are never read, deleted, or mutated; only
    ``fig.layout.template`` is set. The function returns the same figure
    object it was given and is idempotent -- applying it repeatedly leaves
    a single combined template and the traces untouched.

    Args:
        fig: The figure to theme.

    Returns:
        The same ``fig`` object, with its layout template set to the
        composed ``"plotly+phenotypic"`` template.

    Example:
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        >>> themed = apply_theme(fig)
        >>> themed is fig
        True
        >>> len(themed.data)
        1
    """
    if PHENOTYPIC_TEMPLATE_NAME not in pio.templates:
        register_phenotypic_template()
    fig.layout.template = f"plotly+{PHENOTYPIC_TEMPLATE_NAME}"
    return fig


# Register the template once at import time so importing this module is
# sufficient to make the "phenotypic" template available.
register_phenotypic_template()
