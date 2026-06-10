"""Plot rendering with autodetection for the analysis sub-app.

Filters (``SetAnalyzer`` subclasses except ``ModelFitter``) ship with a
matplotlib :meth:`show` but raise ``NotImplementedError`` from their
:meth:`dash` — we render those to PNG bytes and embed as ``<img>``.
Models (``ModelFitter`` subclasses) override :meth:`dash` to return a
plotly :class:`~plotly.graph_objects.Figure`; we wrap those in
``dcc.Graph`` for the fast path.

The autodetection is intentionally permissive — any unexpected exception
during rendering bubbles up as a small error card so the rest of the
page keeps working.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # safe in dash worker threads; must precede pyplot import

import matplotlib.pyplot as plt
from dash import dcc, html

from phenotypic.viz.figures import apply_theme, phenotypic_mpl_context

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
            visualization method runs — ``node.dash(**plot_kwargs)`` on
            the plotly fast path, ``node.show(**plot_kwargs)`` on the
            matplotlib fallback. The caller is responsible for passing
            only kwargs valid for the method that will actually run
            (see :func:`._plot_controls.plotting_params`, which
            introspects the same method this function selects).

    Returns:
        A Dash component ready for layout: ``dcc.Graph`` on the plotly
        fast path, ``html.Img`` (data-URI PNG) on the matplotlib fallback,
        or an inline error card on any unexpected failure.
    """
    # Try plotly fast path first.
    try:
        figure = node.dash(**plot_kwargs)
    except NotImplementedError:
        figure = None
    except Exception as exc:  # noqa: BLE001 - render failures are surfaced inline
        logger.warning("dash() raised on %s: %s", type(node).__name__, exc)
        return _error_card(f"dash(): {exc}")

    if figure is not None:
        # Stamp the shared PhenoTypic theme (Okabe-Ito colorway, mono numeric
        # axes, navy title, brand grid/axis colors) so every model figure is
        # spec-faithful without per-analyzer styling.
        apply_theme(figure)
        return dcc.Graph(
            figure=figure,
            config={"displayModeBar": False},
            className="analysis-section-plot",
        )

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

            buf = io.BytesIO()
            mpl_fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(mpl_fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("show() raised on %s: %s", type(node).__name__, exc)
        return _error_card(f"show(): {exc}")

    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return html.Img(
        src=f"data:image/png;base64,{encoded}",
        className="analysis-section-plot",
        style={"maxWidth": "100%", "height": "auto"},
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
