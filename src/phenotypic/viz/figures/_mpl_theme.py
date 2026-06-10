"""Matplotlib rcParams mirror of the PhenoTypic chart theme.

The plotly side lives in :mod:`._theme`; this module carries the matplotlib
equivalent from ``DESIGN.md`` "07 -- Code Integration" (the ``rcParams`` block):
the Okabe-Ito ``axes.prop_cycle``, navy-tinted grid / edge colors, hidden top /
right spines, and the role fonts (IBM Plex Sans body, JetBrains Mono numerics).

``matplotlib`` is imported lazily inside the functions so importing this module
(and the ``phenotypic.viz.figures`` package) stays cheap for callers that only
need the plotly theme.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ._theme import AXIS, BODY, GRID, MUTED, NAVY, OKABE_ITO, WHITE

__all__ = ["phenotypic_rc", "phenotypic_mpl_context"]


def phenotypic_rc() -> dict[str, Any]:
    """Return the PhenoTypic matplotlib ``rcParams`` dict (DESIGN.md "07").

    Suitable for ``matplotlib.rcParams.update(...)`` or
    ``matplotlib.rc_context(phenotypic_rc())``. Numeric tick labels render in
    JetBrains Mono; axis labels / titles in IBM Plex Sans. The Okabe-Ito series
    order anchors ``axes.prop_cycle`` so the first plotted series is brand navy.

    Returns:
        A fresh ``dict`` of rcParam name -> value.
    """
    from cycler import cycler

    # Drop the overflow "ink" black (index 7) so the prop_cycle is the six
    # categorical series + the vermilion error color, matching the plotly
    # ``colorway`` intent.
    prop_colors = list(OKABE_ITO[:7])
    return {
        "axes.prop_cycle": cycler(color=prop_colors),
        "figure.facecolor": "#FBFEF8",
        "axes.facecolor": WHITE,
        "axes.edgecolor": AXIS,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "Helvetica Neue", "Arial"],
        "font.monospace": ["JetBrains Mono", "DejaVu Sans Mono", "Courier New"],
        "axes.labelcolor": BODY,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "600",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        # Tick labels are numeric data -> render in the mono family.
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }


@contextmanager
def phenotypic_mpl_context() -> Iterator[None]:
    """Context manager applying :func:`phenotypic_rc` via ``rc_context``.

    Use around figure construction so the theme is scoped and never leaks into
    a caller's global ``rcParams``::

        with phenotypic_mpl_context():
            fig = analyzer.show()
            fig.savefig(buf, format="png")

    Yields:
        ``None``; the themed rcParams are active for the duration of the block.
    """
    import matplotlib as mpl

    with mpl.rc_context(phenotypic_rc()):
        yield
