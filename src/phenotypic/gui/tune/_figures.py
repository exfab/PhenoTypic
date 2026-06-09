"""Shared Plotly figure scaffolding for the ``/tune/`` co-pilot.

Every tune figure -- the Monitor objective / importance plots, the Curate
candidate / difference overlays, and the placeholder figures shown while a
render is in flight -- sits on the same transparent canvas so the surrounding
card's design-token background shows through. This module single-sources that
shared base layout (mono numeric font + transparent paper/plot backgrounds) so
the three call sites can't drift on the font token or the transparent sentinel.

The base font is the mono family because tune figures are data plots: axis
ticks, hover values, and legends are numeric data, which DESIGN.md "02" renders
in JetBrains Mono.

Like the rest of :mod:`phenotypic.gui.tune`, importing this module must never
drag ``optuna`` into ``sys.modules`` -- it imports only ``plotly`` and the
design-token font family.
"""
from __future__ import annotations

from phenotypic.gui._design import FONT_FAMILY_MONO

#: The fully-transparent Plotly paper/plot fill — NOT a brand color (so not a
#: ``_design`` token), just the sentinel that lets the card background show
#: through. Single-sourced here so the three tune figure builders stay in step.
_TRANSPARENT_FILL: str = "rgba(0,0,0,0)"


def transparent_layout(**overrides: object) -> dict[str, object]:
    """The shared tune-figure layout: mono numeric font + transparent fill.

    Returns a fresh ``dict`` (safe to splat into ``update_layout``) carrying the
    design-token mono font and the transparent paper/plot backgrounds every tune
    figure shares. Each caller layers its own ``margin`` / axes / ``dragmode`` /
    annotations on top via ``overrides`` (later keys win).

    Args:
        **overrides: Layout keys merged over the shared base (e.g. ``margin``,
            ``xaxis``, ``dragmode``); an override for a base key replaces it.

    Returns:
        A new layout ``dict`` ready to pass to
        :meth:`plotly.graph_objects.Figure.update_layout`.
    """
    base: dict[str, object] = {
        "font": {"family": FONT_FAMILY_MONO},
        "paper_bgcolor": _TRANSPARENT_FILL,
        "plot_bgcolor": _TRANSPARENT_FILL,
    }
    base.update(overrides)
    return base


__all__ = ["transparent_layout"]
