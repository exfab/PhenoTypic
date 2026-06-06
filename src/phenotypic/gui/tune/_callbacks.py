"""Dash callbacks for the ``/tune/`` co-pilot.

Two concerns live here, each a thin Dash adapter around a pure, headless-
testable helper:

* **Sub-tab switching** — :func:`active_view` maps a clicked sub-tab button's
  ID to its view name (falling back to the default Monitor view for an unknown
  or absent trigger); the registered callback toggles which view container is
  visible and which button carries the active class.

The module never imports ``optuna`` at import time. (The Monitor poll callback,
added alongside, opens the live study lazily inside its own body — gated on the
``tune`` extra being importable — so this module stays in the package's
optuna-free import surface.)
"""
from __future__ import annotations

from dash import ctx

from phenotypic.gui.tune import _ids as ids

#: The default view shown when no (or an unknown) sub-tab is active.
_DEFAULT_VIEW: ids.SubTabName = "monitor"

#: Reverse map: a sub-tab button ID -> its view name. Built once from the
#: ordered sub-tab names so the helper never re-derives strings at call time.
_BUTTON_ID_TO_VIEW: dict[str, ids.SubTabName] = {
    ids.subtab_button_id(name): name for name in ids.SUBTAB_ORDER
}


def active_view(trigger_id: str | None) -> ids.SubTabName:
    """Resolve which sub-tab view a click on ``trigger_id`` should show.

    Pure routing logic, unit-testable without Dash: a known sub-tab button ID
    (``tune-subtab-<name>``) resolves to its view name; ``None``, an empty
    string, or any unrecognised ID falls back to the default Monitor view (so
    the initial render and any stray trigger land somewhere valid).

    Args:
        trigger_id: The ID of the component that fired the callback (Dash's
            ``ctx.triggered_id``), or ``None`` on the initial call.

    Returns:
        The resolved view name (one of :data:`ids.SUBTAB_ORDER`).
    """
    if not trigger_id:
        return _DEFAULT_VIEW
    return _BUTTON_ID_TO_VIEW.get(trigger_id, _DEFAULT_VIEW)


def _view_container_class(name: ids.SubTabName, active: ids.SubTabName) -> str:
    """The class string for a view container (non-active gets the hidden class)."""
    classes = ["tune-view"]
    if name != active:
        classes.append("tune-view-hidden")
    return " ".join(classes)


def _subtab_button_class(name: ids.SubTabName, active: ids.SubTabName) -> str:
    """The class string for a sub-tab button (active gets the highlight class)."""
    classes = ["tune-subtab"]
    if name == active:
        classes.append("tune-subtab-active")
    return " ".join(classes)


def register_callbacks(app) -> None:  # type: ignore[no-untyped-def]
    """Register the tune sub-app's Dash callbacks on ``app``.

    Wires the sub-tab switch: a click on any of the four sub-tab buttons
    re-resolves the active view via :func:`active_view`, then toggles each view
    container's visibility and each button's active class. The active view name
    is mirrored into :data:`ids.TUNE_ACTIVE_VIEW_STORE` so later callbacks can
    read it without re-deriving the trigger.

    Args:
        app: The :class:`dash.Dash` instance whose layout is assigned.
    """
    from dash import Input, Output

    @app.callback(
        Output(ids.TUNE_ACTIVE_VIEW_STORE, "data"),
        *[
            Output(ids.view_container_id(name), "className")
            for name in ids.SUBTAB_ORDER
        ],
        *[
            Output(ids.subtab_button_id(name), "className")
            for name in ids.SUBTAB_ORDER
        ],
        *[
            Input(ids.subtab_button_id(name), "n_clicks")
            for name in ids.SUBTAB_ORDER
        ],
        prevent_initial_call=True,
    )
    def _switch_subtab(*_n_clicks: int) -> tuple[str, ...]:
        active = active_view(ctx.triggered_id)
        container_classes = [
            _view_container_class(name, active) for name in ids.SUBTAB_ORDER
        ]
        button_classes = [
            _subtab_button_class(name, active) for name in ids.SUBTAB_ORDER
        ]
        return (active, *container_classes, *button_classes)


__all__ = ["active_view", "register_callbacks"]
