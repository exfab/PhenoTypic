"""Plot rendering with autodetection for the analysis sub-app.

Filters (``SetAnalyzer`` subclasses except ``ModelFitter``) ship with a
matplotlib :meth:`show`; we render those to PNG bytes and embed as ``<img>``.
Models expose :meth:`report` through ``PlotAnalysis`` and return a Plotly
:class:`~plotly.graph_objects.Figure`; we wrap those in ``dcc.Graph`` for
the fast path.

The autodetection is intentionally permissive — any unexpected exception
during rendering bubbles up as a small error card so the rest of the
page keeps working.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # safe in dash worker threads; must precede pyplot import

import matplotlib.pyplot as plt
from dash import dcc, html

from phenotypic.plotting import FigureAdapter
from phenotypic.plotting._output import normalize_plot_output
from phenotypic.sdk_.viz.figures import apply_theme, phenotypic_mpl_context

if TYPE_CHECKING:
    from phenotypic.analysis.abc_ import SetAnalyzer

logger = logging.getLogger(__name__)


def render_plot(node: "SetAnalyzer | Any", **plot_kwargs: Any) -> Any:
    """Render *node*'s preview plot, autodetecting plotly vs matplotlib.

    Args:
        node: A :class:`SetAnalyzer` (filter) or :class:`ModelFitter`
            (model) instance. Must have ``.analyze()`` already been called
            so internal results are populated; the caller is responsible
            for ordering.
        **plot_kwargs: Plotting parameters forwarded to whichever
            visualization method runs — ``node.report(**plot_kwargs)`` on
            the plotting-capability fast path, ``node.show(**plot_kwargs)`` on the
            matplotlib fallback. The caller is responsible for passing
            only kwargs valid for the method that will actually run
            (see :func:`._plot_controls.plotting_params`, which
            introspects the same method this function selects).

    Returns:
        A Dash component ready for layout: ``dcc.Graph`` on the plotly
        fast path, ``html.Img`` (data-URI PNG) on the matplotlib fallback,
        or an inline error card on any unexpected failure.
    """
    report = getattr(node, "report", None)
    if callable(report):
        try:
            figure = report(**plot_kwargs)
        except Exception as exc:  # noqa: BLE001 - render failures are surfaced inline
            logger.warning(
                "report() raised on %s: %s", type(node).__name__, exc
            )
            return _error_card(f"report(): {exc}")
    else:
        figure = None

    if figure is not None:
        return _render_output(figure, producer=type(node).__name__)

    try:
        # Apply the matplotlib rcParams mirror (DESIGN.md "07") for the duration
        # of figure construction + raster so filter previews carry the brand
        # palette, fonts, and spine rules.
        with phenotypic_mpl_context():
            mpl_fig = node.show(**plot_kwargs)
            if mpl_fig is None:
                # ``show()`` implementations that mutate ``plt.gca()`` without
                # returning the figure leave the active one for us to grab.
                mpl_fig = plt.gcf()

            return FigureAdapter.to_dash_component(
                mpl_fig,
                class_name="analysis-section-plot",
                image_style={"maxWidth": "100%", "height": "auto"},
                mpl_savefig_kwargs={"dpi": 110, "bbox_inches": "tight"},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("show() raised on %s: %s", type(node).__name__, exc)
        return _error_card(f"show(): {exc}")


def _render_output(value: Any, *, producer: str) -> Any:
    """Render one or more figure pages with a reusable Dash page selector.

    A single page preserves the existing direct ``Graph``/``Img`` result. A
    multi-page output becomes ``dcc.Tabs`` whose labels come from
    :class:`PlotPage`; each failed page is isolated to an inline error card.

    Args:
        value: Raw supported figure or :class:`PlotOutput`.
        producer: Human-readable producer name used in log messages.

    Returns:
        A Dash figure component, tab selector, empty-state card, or error card.
    """
    output = normalize_plot_output(value)
    if not output.pages:
        return html.Div(
            "No plot pages were produced.",
            className="analysis-plot-empty",
        )

    rendered: list[Any] = []
    for page in output.pages:
        try:
            backend = FigureAdapter.backend_name(page.figure)
            if backend == "plotly":
                apply_theme(page.figure)
            component = FigureAdapter.to_dash_component(
                page.figure,
                graph_config={"displayModeBar": False},
                class_name="analysis-section-plot",
                image_style={"maxWidth": "100%", "height": "auto"},
                mpl_savefig_kwargs={"dpi": 110, "bbox_inches": "tight"},
            )
        except Exception as exc:  # noqa: BLE001 - isolate failed siblings
            FigureAdapter.close(page.figure)
            logger.warning(
                "report() page %s rendering failed on %s: %s",
                page.key,
                producer,
                exc,
            )
            component = _error_card(f"report() page {page.key!r}: {exc}")
        rendered.append((page, component))

    if len(rendered) == 1:
        return rendered[0][1]

    return dcc.Tabs(
        [
            dcc.Tab(
                label=page.label or page.key,
                value=page.key,
                children=component,
            )
            for page, component in rendered
        ],
        value=rendered[0][0].key,
        className="analysis-plot-pages",
    )


def _error_card(message: str) -> Any:
    """Inline error card for failed renders."""
    return html.Div(
        [
            html.Strong("Preview unavailable"),
            html.Pre(message, className="analysis-error-pre"),
        ],
        className="analysis-error-card",
    )
