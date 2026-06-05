"""Dash adapter for rendering a ``FigureProvider`` in the web GUI — DEFERRED STUB.

# TODO(plotting-migration): wire FigureProvider dashboards into the Dash web GUI.

This module is intentionally a stub. The Panel→Plotly migration replaced the four
notebook dashboards (diagnostics, detect_modes, grid-finder, color-correction)
with **notebook-native** surfaces — ipywidgets via ``image.plot.dash.*`` and the
``*.dashboard()`` helpers — and none are mounted in the Dash hub. Web-GUI
integration was descoped.

A complete implementation of this adapter was prototyped during Phase 4 (build a
controls panel + a ``dbc.Accordion`` of per-figure ``dcc.Graph``s from
``provider.iter_figures()``, plus one Dash callback per control-bearing figure for
selective re-render, with controls deduped by identity and figures seeded on
load). It was removed when GUI integration was dropped.

When a web-GUI dashboard is wanted later — most likely **color-correction**, the
only control-bearing dashboard — restore the adapter here. References:

* the protocol: :mod:`phenotypic.abc_._figure_provider`
  (``FigureProvider.iter_figures()`` / ``figures(subject)`` →
  ``BoundFigures.render(spec, **controls)``, ``Control``, ``@figure``);
* the design: ``docs/design_outlines/migrate_plotting_panel_to_plotly_dash/design.md`` §7;
* the existing GUI Plotly seam to mirror: ``gui/analysis/_render.py`` (renders
  ``node.dash() -> go.Figure`` into ``dcc.Graph``).
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["build_figure_dashboard", "register_figure_dashboard_callbacks"]

_DEFERRED = (
    "The Dash web-GUI figure adapter is a deferred stub — diagnostics and the "
    "other dashboards are notebook-only (image.plot.dash.* / *.dashboard()). "
    "See the module docstring to restore it."
)


def build_figure_dashboard(
    provider: Any, subject: Any = None, *, id_prefix: str = "figdash"
) -> Any:
    """Deferred stub — raises :class:`NotImplementedError`. See module docstring."""
    raise NotImplementedError(_DEFERRED)


def register_figure_dashboard_callbacks(
    app: Any, *, id_prefix: str, bound_getter: Callable[[], Any]
) -> None:
    """Deferred stub — raises :class:`NotImplementedError`. See module docstring."""
    raise NotImplementedError(_DEFERRED)
