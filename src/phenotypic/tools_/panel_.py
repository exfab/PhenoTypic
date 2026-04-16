"""Shared Panel toolkit for interactive dashboards.

Centralises Panel availability checks, Jupyter environment detection,
and ``pn.extension()`` initialization so that every module using Panel
imports from one place instead of reimplementing these patterns.
"""

from __future__ import annotations

import importlib.util

__all__ = [
    "PANEL_AVAILABLE",
    "PANEL_IMPORT_ERROR",
    "PLOTLY_AVAILABLE",
    "display_or_return",
    "ensure_panel_extension",
    "in_ipython",
    "in_jupyter",
    "require_panel",
]

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

PANEL_AVAILABLE: bool = importlib.util.find_spec("panel") is not None
"""``True`` when the ``panel`` package is importable."""

PLOTLY_AVAILABLE: bool = importlib.util.find_spec("plotly") is not None
"""``True`` when the ``plotly`` package is importable."""

PANEL_IMPORT_ERROR: str = (
    "Panel is required for interactive dashboards. "
    "Install it with: pip install 'phenotypic[gui]'"
)
"""User-facing error message when Panel is missing."""

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def require_panel() -> None:
    """Raise :class:`ImportError` with an actionable message if Panel is missing.

    Raises:
        ImportError: If ``panel`` is not installed.
    """
    if not PANEL_AVAILABLE:
        raise ImportError(PANEL_IMPORT_ERROR)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def in_ipython() -> bool:
    """Detect whether code is running inside any IPython session.

    Returns:
        ``True`` when an IPython kernel or terminal is active.
    """
    try:
        get_ipython()  # type: ignore[name-defined]  # noqa: F821
        return True
    except NameError:
        return False


def in_jupyter() -> bool:
    """Detect whether code is running inside a Jupyter notebook.

    This is stricter than :func:`in_ipython` — it returns ``True`` only
    for notebook kernels (``ZMQInteractiveShell``), not plain IPython
    terminals.

    Returns:
        ``True`` when running in a Jupyter notebook kernel.
    """
    try:
        from IPython import get_ipython as _get_ipython

        shell = _get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Extension initialisation
# ---------------------------------------------------------------------------

_panel_initialized: bool = False


def ensure_panel_extension(*extensions: str, **kwargs) -> None:
    """Initialise ``pn.extension()`` once, only when inside IPython.

    Safe to call multiple times — the extension is loaded at most once
    per process.  Outside IPython the call is a silent no-op. When
    :mod:`plotly` is importable it is automatically included as an
    extension so Plotly figures render correctly in Panel layouts.

    Args:
        *extensions: Extra Panel extensions to load (e.g. ``"tabulator"``).
            ``"plotly"`` is added automatically when Plotly is installed.
        **kwargs: Forwarded to ``pn.extension()`` on the first call
            (e.g. ``inline=True``).
    """
    global _panel_initialized

    if _panel_initialized:
        return

    if not in_ipython():
        return

    if not PANEL_AVAILABLE:
        return

    import panel as pn

    ext_set = list(dict.fromkeys(extensions))
    if PLOTLY_AVAILABLE and "plotly" not in ext_set:
        ext_set.append("plotly")

    pn.extension(*ext_set, **kwargs)
    _panel_initialized = True


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------


def display_or_return(layout, *, show: bool = True):
    """Display a Panel layout in Jupyter or return it for programmatic use.

    When *show* is ``True`` and the session is a Jupyter notebook the
    layout is passed to ``IPython.display.display`` (after ensuring the
    Panel extension is loaded).  Otherwise the layout is returned as-is.

    Args:
        layout: A Panel layout object (e.g. ``pn.Column``).
        show: If ``True``, attempt to display interactively.

    Returns:
        The *layout* object (always returned, even after display).
    """
    if show and in_jupyter():
        ensure_panel_extension()
        from IPython.display import display

        display(layout)

    return layout
