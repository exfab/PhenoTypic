"""Okabe-Ito colour constants that are safe to import at module scope.

``sdk_/viz/figures/_theme.py`` owns the plotly template and imports plotly at
module scope, so an operation module cannot import colours from it without
breaking the lazy-startup gate (``tests/unit/ci/test_startup_imports.py``).
This module holds the same palette with no third-party imports; the theme
re-exports :data:`OKABE_ITO` from here so the two can never drift.

The series order is fixed by DESIGN.md "Categorical Series Order".
"""

from __future__ import annotations

OKABE_ITO_NAVY: str = "#003660"
OKABE_ITO_ORANGE: str = "#E69F00"
OKABE_ITO_SKY: str = "#56B4E9"
OKABE_ITO_GREEN: str = "#009E73"
OKABE_ITO_BLUE: str = "#0072B2"
OKABE_ITO_PURPLE: str = "#CC79A7"
OKABE_ITO_VERMILION: str = "#D55E00"
OKABE_ITO_BLACK: str = "#000000"
#: Okabe-Ito yellow. Nearly invisible as a line, point or text on white, so it
#: is kept out of both cycles below and reserved for large filled areas.
OKABE_ITO_YELLOW: str = "#F0E442"

#: Categorical colour cycle for data series, in DESIGN.md's fixed order:
#: navy (series 1, UI-harmonized), orange, sky, green, blue, purple, then
#: vermilion reserved for the error/alert series, and black as overflow ink.
OKABE_ITO: tuple[str, ...] = (
    OKABE_ITO_NAVY,
    OKABE_ITO_ORANGE,
    OKABE_ITO_SKY,
    OKABE_ITO_GREEN,
    OKABE_ITO_BLUE,
    OKABE_ITO_PURPLE,
    OKABE_ITO_VERMILION,
    OKABE_ITO_BLACK,
)


#: Okabe-Ito in the order Wong (2011, Nature Methods 8:441) prints it, with
#: yellow skipped: black, orange, sky blue, bluish green, blue, vermilion,
#: reddish purple. Static matplotlib figures use this order (DESIGN.md
#: "Figures"); GUI and Plotly charts keep the navy-first :data:`OKABE_ITO`.
OKABE_ITO_PUBLISHED: tuple[str, ...] = (
    OKABE_ITO_BLACK,
    OKABE_ITO_ORANGE,
    OKABE_ITO_SKY,
    OKABE_ITO_GREEN,
    OKABE_ITO_BLUE,
    OKABE_ITO_VERMILION,
    OKABE_ITO_PURPLE,
)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Return a CSS ``rgba(...)`` string for a ``#RRGGBB`` colour.

    Args:
        hex_color: Six-digit hex colour, with or without a leading ``#``.
        alpha: Opacity in ``[0, 1]``, formatted to three decimals.

    Returns:
        ``"rgba(r, g, b, a)"`` for plotly fill and line colours.
    """
    digits = hex_color.lstrip("#")
    red = int(digits[0:2], 16)
    green = int(digits[2:4], 16)
    blue = int(digits[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"
