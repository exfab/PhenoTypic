"""Faceted detection-mode comparison plotter (notebook Plotly ``FigureProvider``).

Plotly successor to the Panel ``DetectModeDashboard``: instead of a dropdown +
thumbnail Panel layout, it renders one faceted ``go.Figure`` with a subplot per
registered detection mode so every candidate source channel for colony detection
can be eyeballed side by side.
"""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phenotypic.abc_ import FigureProvider, figure
from phenotypic.tools_._plotly_helpers import plotly_imshow
from phenotypic.tools_.register import register_plotter

from ._base_plotter import BasePlotter


@register_plotter
class DetectModesPlotter(BasePlotter, FigureProvider):
    """Compare every registered detection mode as a faceted Plotly figure.

    This is the notebook-only Plotly replacement for the Panel
    ``DetectModeDashboard``. Each registered :class:`DetectionMode` is computed
    over the root image and rendered as one grayscale subplot, plus a final
    panel for the image's current (possibly enhanced) ``detect_mat``. Laying
    every candidate source channel out at once makes it easy to pick the best
    channel for colony detection on a given plate.

    Modes that require RGB data are skipped when the image has none.

    The single ``@figure`` method :meth:`detect_modes` is control-free, so both
    ``image.plot.detect_modes()`` (the themed faceted figure) and
    ``image.plot.dash.detect_modes()`` (``.dash()`` on the provider) return the
    same ``go.Figure`` directly.

    Examples:
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> fig = image.plot.detect_modes()
        >>> # fig.show()  # interactive faceted comparison in a notebook
    """

    call_name = "detect_modes"

    def _figure_subject(self) -> Any:
        """Subject for the ``FigureProvider`` mixin.

        This plotter is a *helper* provider: the ``@figure`` method reads
        ``self`` directly and takes no subject parameter, so the returned
        subject is informational only (the root :class:`Image`).
        """
        return self._root_image

    def _compute_mode_matrices(self) -> dict[str, Any]:
        """Compute detection matrices for every registered mode, plus ``detect_mat``.

        Skips modes that require RGB data when the image has none, then appends
        the current ``detect_mat`` as a final labelled entry (mirroring the Panel
        dashboard's special "current" panel).

        Returns:
            Ordered mapping of panel label to its 2-D array.
        """
        from phenotypic._core._image_parts.detection_modes import (
            available_modes,
            get_detection_mode,
        )

        has_rgb = not self._root_image.rgb.isempty()
        matrices: dict[str, Any] = {}
        for name in available_modes():
            mode = get_detection_mode(name)
            if mode.requires_rgb and not has_rgb:
                continue
            matrices[name] = mode.compute(self._root_image)

        detect_mat_label = f"detect_mat (current: {self._root_image.detect_mode})"
        matrices[detect_mat_label] = self._root_image.detect_mat[:]
        return matrices

    @figure(
        title="Detection Mode Comparison",
        section="detect_modes",
    )
    def detect_modes(self) -> go.Figure:
        """Faceted grayscale comparison of every detection mode (one per subplot).

        Computes each registered :class:`DetectionMode` over the root image,
        plus the current ``detect_mat``, and arranges them in a roughly square
        subplot grid. Each panel is a grayscale image layer built with
        :func:`~phenotypic.tools_._plotly_helpers.plotly_imshow`; axes are hidden
        and aspect-locked so the panels read as thumbnails. The house theme is
        applied automatically by the ``@figure`` decorator.

        Returns:
            A faceted ``go.Figure`` with one subplot per mode (and a final
            ``detect_mat`` panel), each holding a grayscale image trace.
        """
        matrices = self._compute_mode_matrices()
        labels = list(matrices.keys())
        n_panels = len(labels)

        ncols = math.ceil(math.sqrt(n_panels))
        nrows = math.ceil(n_panels / ncols)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=labels,
            horizontal_spacing=0.04,
            vertical_spacing=0.08,
        )

        for idx, label in enumerate(labels):
            row = idx // ncols + 1
            col = idx % ncols + 1
            panel = plotly_imshow(matrices[label], cmap="gray")
            for trace in panel.data:
                fig.add_trace(trace, row=row, col=col)
            # Image traces read top-to-bottom; lock each panel's aspect to ITS OWN
            # x-axis (make_subplots numbers axes row-major: panel k → x{k}/y{k},
            # with x1 written as "x") and hide ticks so panels read as thumbnails.
            x_axis = "x" if idx == 0 else f"x{idx + 1}"
            fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
            fig.update_yaxes(
                autorange="reversed",
                showticklabels=False,
                showgrid=False,
                scaleanchor=x_axis,
                constrain="domain",
                row=row,
                col=col,
            )

        fig.update_layout(title="Detection Mode Comparison")
        return fig


__all__ = ["DetectModesPlotter"]
