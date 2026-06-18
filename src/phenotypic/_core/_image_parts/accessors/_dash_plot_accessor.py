"""Interactive ``.dash`` sub-accessor for ``FigureProvider`` plotters.

``image.plot.dash.<name>()`` resolves ``<name>`` against the same
``@register_plotter`` registry that powers ``image.plot.<name>()``; if the
registered plotter implements the figure protocol (``FigureProvider``), calling
it returns that plotter's interactive view — a composed ``go.Figure`` for
control-free providers, or an ipywidgets dashboard when the figures declare
``Control``s.

The ``FigureProvider`` check is duck-typed (the presence of ``iter_figures`` /
``dash``) to avoid a ``_core`` → ``abc_`` import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from phenotypic.sdk_.register import available_plotters, get_plotter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._plot_accessor import PlotAccessor

__all__ = ("DashPlotAccessor",)


def _is_figure_provider(cls: type) -> bool:
    """Duck-typed ``FigureProvider`` check (has ``iter_figures`` and ``dash``)."""
    return callable(getattr(cls, "iter_figures", None)) and callable(
        getattr(cls, "dash", None)
    )


class DashPlotAccessor:
    """Dispatches ``image.plot.dash.<name>()`` to a plotter's ``.dash()``.

    Reuses the parent :class:`PlotAccessor`'s instance cache so a plotter built
    for ``image.plot.<name>()`` and ``image.plot.dash.<name>()`` is the same
    object.
    """

    def __init__(self, plot_accessor: "PlotAccessor") -> None:
        """Bind to the parent plot accessor.

        Args:
            plot_accessor: The :class:`PlotAccessor` that owns this sub-accessor.
        """
        self._plot = plot_accessor

    def _dashable(self) -> list[str]:
        """Names of registered plotters that support ``.dash()``."""
        names = []
        for name in available_plotters():
            try:
                if _is_figure_provider(get_plotter(name)):
                    names.append(name)
            except ValueError:  # pragma: no cover - registry race, defensive
                continue
        return names

    def __getattr__(self, name: str) -> Any:
        """Resolve ``name`` to a registered FigureProvider plotter's ``.dash()``.

        Args:
            name: A registered plotter name (its ``call_name``).

        Returns:
            A callable that invokes the plotter instance's ``.dash(**kwargs)``.
            The callable forwards keyword arguments only; for operation-style
            providers pass the subject by keyword (``...(subject=image)``).

        Raises:
            AttributeError: If ``name`` is not a registered plotter or that
                plotter does not implement the figure protocol.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            plotter_cls = get_plotter(name)
        except ValueError:
            raise AttributeError(
                f"'{type(self).__name__}' has no interactive plotter {name!r}. "
                f"Dashable plotters: {', '.join(self._dashable())}"
            ) from None
        if not _is_figure_provider(plotter_cls):
            raise AttributeError(
                f"Plotter {name!r} does not support .dash() — it is not a "
                f"FigureProvider."
            )
        instance = self._plot._get_or_create(plotter_cls)

        def call(**kwargs: Any) -> Any:
            return instance.dash(**kwargs)

        call.__name__ = name
        call.__qualname__ = f"plot.dash.{name}"
        return call

    def __dir__(self) -> list[str]:
        """Include dashable plotter names for tab-completion."""
        return sorted(set(super().__dir__()) | set(self._dashable()))
