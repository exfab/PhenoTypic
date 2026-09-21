"""Lazy backend adapter shared by CLI publication and Dash rendering."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, cast

from phenotypic.abc_.plotting import figure_backend_of


class FigureAdapter:
    """Save and render supported Plotly and Matplotlib figures."""

    @staticmethod
    def save_png(
        figure: Any,
        path: Path,
        *,
        title_prefix: str = "",
    ) -> None:
        """Save one figure as PNG and close Matplotlib figures reliably.

        Args:
            figure: Plotly or Matplotlib figure.
            path: Destination PNG path.
            title_prefix: Optional prefix applied to a temporary Plotly copy.

        Raises:
            TypeError: If the figure backend is unsupported.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if FigureAdapter._is_plotly(figure):
            from plotly.graph_objects import Figure

            target = Figure(figure)
            if title_prefix:
                current = target.layout.title.text or ""
                target.update_layout(title=f"{title_prefix}{current}")
            target.write_image(str(path), format="png")
            return

        if FigureAdapter._is_matplotlib(figure):
            try:
                figure.savefig(path, format="png")
            finally:
                from matplotlib import pyplot as plt

                plt.close(figure)
            return
        raise TypeError(
            "supported plot figures are plotly.graph_objects.Figure or "
            f"matplotlib.figure.Figure; got {type(figure).__module__}."
            f"{type(figure).__qualname__}"
        )

    @staticmethod
    def save_html(figure: Any, path: Path, *, plotlyjs_src: str) -> None:
        """Write one Plotly figure as a standalone interactive HTML page.

        No Kaleido and no Chrome are involved. ``plotlyjs_src`` is emitted
        verbatim as the script src, so the 4.8 MB bundle is referenced rather
        than embedded.

        Args:
            figure: A Plotly figure.
            path: Destination ``.html`` path.
            plotlyjs_src: Relative src, from ``plotlyjs_src_for``.

        Raises:
            TypeError: If *figure* is not a Plotly figure.
        """
        if figure_backend_of(figure) != "plotly":
            raise TypeError(
                "HTML export is Plotly-only; got "
                f"{type(figure).__module__}.{type(figure).__qualname__}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path, include_plotlyjs=plotlyjs_src)

    @staticmethod
    def to_dash_component(
        figure: Any,
        *,
        graph_config: Mapping[str, Any] | None = None,
        class_name: str | None = None,
        image_style: Mapping[str, Any] | None = None,
        mpl_savefig_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Convert a supported figure to a native Dash component.

        Matplotlib figures are rasterized and closed before this method
        returns. They are also closed when rasterization or Dash component
        construction fails.

        Args:
            figure: Plotly or Matplotlib figure.
            graph_config: Optional configuration for a Plotly ``dcc.Graph``.
            class_name: Optional CSS class applied to the Dash component.
            image_style: Optional inline style for a Matplotlib ``html.Img``.
            mpl_savefig_kwargs: Optional keyword arguments passed to
                Matplotlib's ``savefig`` in addition to ``format="png"``.

        Returns:
            A ``dcc.Graph`` for Plotly or an ``html.Img`` for Matplotlib.

        Raises:
            TypeError: If the figure backend is unsupported.
        """
        if FigureAdapter._is_plotly(figure):
            from dash import dcc

            return dcc.Graph(
                figure=figure,
                config=cast(
                    Any,
                    None if graph_config is None else dict(graph_config),
                ),
                className=class_name,
            )
        if FigureAdapter._is_matplotlib(figure):
            from matplotlib import pyplot as plt

            try:
                from dash import html

                savefig_kwargs = dict(mpl_savefig_kwargs or {})
                savefig_kwargs["format"] = "png"
                with BytesIO() as buffer:
                    figure.savefig(buffer, **savefig_kwargs)
                    encoded = base64.b64encode(buffer.getvalue()).decode(
                        "ascii"
                    )
                return html.Img(
                    src=f"data:image/png;base64,{encoded}",
                    className=class_name,
                    style=None if image_style is None else dict(image_style),
                )
            finally:
                plt.close(figure)
        raise TypeError(
            "cannot render unsupported figure type "
            f"{type(figure).__module__}.{type(figure).__qualname__}"
        )

    @staticmethod
    def close(figure: Any) -> None:
        """Close a Matplotlib figure; Plotly figures require no cleanup."""
        if FigureAdapter._is_matplotlib(figure):
            from matplotlib import pyplot as plt

            plt.close(figure)

    @staticmethod
    def backend_name(figure: Any) -> str:
        """Return the stable backend name for a supported figure.

        The published manifest spells the matplotlib backend ``"matplotlib"``
        while the ``@figure`` decorator spells it ``"mpl"``. This is the one
        place the two vocabularies meet; the wire format is not changed here.
        """
        backend = figure_backend_of(figure)
        if backend == "plotly":
            return "plotly"
        if backend == "mpl":
            return "matplotlib"
        raise TypeError(
            "unsupported figure type "
            f"{type(figure).__module__}.{type(figure).__qualname__}"
        )

    @staticmethod
    def _is_plotly(figure: Any) -> bool:
        return figure_backend_of(figure) == "plotly"

    @staticmethod
    def _is_matplotlib(figure: Any) -> bool:
        return figure_backend_of(figure) == "mpl"


__all__ = ["FigureAdapter"]
