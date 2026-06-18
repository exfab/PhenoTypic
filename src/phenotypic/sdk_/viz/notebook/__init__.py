"""Notebook (ipywidgets) rendering shell for the figure protocol.

Exposes :func:`build_notebook_dashboard`, used by
:meth:`phenotypic.abc_.FigureProvider.dash` when a provider's figures declare
``Control``s. ipywidgets is imported lazily inside the builder.
"""

from ._adapter import build_notebook_dashboard

__all__ = ["build_notebook_dashboard"]
