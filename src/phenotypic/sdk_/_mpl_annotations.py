"""Runtime stand-ins for the matplotlib names used only in annotations.

Deferring matplotlib moved names like ``plt``, ``Figure`` and ``Axes`` into
``if TYPE_CHECKING:`` blocks. With ``from __future__ import annotations`` the
annotations that mention them become strings — and a string nothing can resolve
is a broken public API, not a free optimization: ``typing.get_type_hints`` on
any annotated method raises ``NameError``, and ``docs/source/conf.py`` sets
``autodoc_typehints = "both"``, so Sphinx resolves annotations at runtime. The
failure surfaces as a docs-build error, never as a test failure.

This is the same remedy stage 2 applied to ``MeasurementFrame`` for polars:
type checkers keep the precise type from the ``TYPE_CHECKING`` branch, while at
runtime the name resolves to something harmless.

``plt`` is a namespace rather than ``Any`` because the annotations spell
``plt.Figure`` and ``plt.Axes``, and ``Any.Figure`` would raise ``AttributeError``.
It deliberately carries *only* the attributes the annotations use: a function
body that reached for ``plt.subplots`` through this stand-in would fail loudly
instead of silently drawing nothing. Bodies never do — every one imports pyplot
locally, which shadows this name.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = [
    "Axes",
    "Colormap",
    "Figure",
    "GridSpec",
    "Normalize",
    "PathCollection",
    "Quiver",
    "Rectangle",
    "plt",
]

#: Stand-in for ``matplotlib.pyplot``, carrying only the annotation attributes.
plt = SimpleNamespace(Figure=Any, Axes=Any)

Axes = Any
Colormap = Any
Figure = Any
GridSpec = Any
Normalize = Any
PathCollection = Any
Quiver = Any
Rectangle = Any
