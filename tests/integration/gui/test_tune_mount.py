"""Integration tests for the ``/tune/`` co-pilot mount + empty-state layout.

Boots the tune Dash factory directly (empty state) and confirms the four
sub-tab buttons render, then composes the full hub and confirms the tune
mount lands in the ``DispatcherMiddleware`` mounts dict alongside the
builder / results / run / analysis mounts.
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


def test_hub_mounts_tune(sandbox: SandboxRoot) -> None:
    from phenotypic.gui.shell._app import compose_hub
    from phenotypic.gui.shell._sandbox import SandboxRoot as SandboxRootDirect

    assert SandboxRootDirect is SandboxRoot  # confirm the import path

    hub_app, _session = compose_hub(sandbox, start_idle_thread=False)
    mounts = hub_app.server.wsgi_app.mounts
    assert any(m.rstrip("/") == "/tune" for m in mounts)
