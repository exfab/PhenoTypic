"""Integration tests for the ``/tune/`` co-pilot factory + hub (un)mount.

Boots the tune Dash factory directly (empty state) and confirms the four
sub-tab buttons render — the tune package itself is unaffected by the
unmount and keeps working standalone. The hub composition, however, no
longer mounts it: Tune is unmounted per
docs/superpowers/plans/2026-08-26-gui-simplification-removals
(phase-4-tune-unmount.md), so ``test_hub_mounts_tune`` below asserts the
mount is *absent* from the ``DispatcherMiddleware`` mounts dict.
``tests/unit/gui/shell/test_tune_is_unmounted.py`` carries the narrower,
faster unit-level version of the same assertion.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.shell import SandboxRoot


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def test_create_app_empty_state_has_subtabs() -> None:
    from phenotypic.gui.tune import create_app

    app = create_app(root=None, url_prefix="/tune/")
    layout = str(app.layout)
    for tid in (
        "tune-subtab-monitor",
        "tune-subtab-curate",
        "tune-subtab-space",
        "tune-subtab-launch",
    ):
        assert tid in layout


def test_hub_does_not_mount_tune(sandbox: SandboxRoot) -> None:
    """Tune is unmounted: no ``/tune`` entry in the composed hub's mounts."""
    from phenotypic.gui.shell._app import compose_hub
    from phenotypic.gui.shell._sandbox import SandboxRoot as SandboxRootDirect

    assert SandboxRootDirect is SandboxRoot  # confirm the import path

    hub_app, _session = compose_hub(sandbox, start_idle_thread=False)
    mounts = hub_app.server.wsgi_app.mounts
    assert not any(m.rstrip("/") == "/tune" for m in mounts)
