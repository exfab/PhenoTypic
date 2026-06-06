"""Tests for the tune sub-tab switch.

The switch is a thin Dash callback around a pure ``active_view(trigger_id)``
helper, so the routing logic unit-tests headless: a clicked sub-tab button's
ID maps to its view name; an unknown or ``None`` trigger falls back to the
default Monitor view.
"""
from __future__ import annotations

import pytest


def test_active_view_maps_each_subtab() -> None:
    from phenotypic.gui.tune._callbacks import active_view

    assert active_view("tune-subtab-monitor") == "monitor"
    assert active_view("tune-subtab-curate") == "curate"
    assert active_view("tune-subtab-space") == "space"
    assert active_view("tune-subtab-launch") == "launch"


@pytest.mark.parametrize("trigger", [None, "", "tune-subtab-bogus", "nonsense"])
def test_active_view_unknown_falls_back_to_monitor(trigger: str | None) -> None:
    from phenotypic.gui.tune._callbacks import active_view

    assert active_view(trigger) == "monitor"
