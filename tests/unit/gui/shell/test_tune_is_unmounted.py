"""Tune is unmounted: unreachable from the UI, still importable on disk.

Both halves matter. The first two tests are the unmount; the third is the
'still importable' half of the contract, which is what distinguishes this
phase (docs/superpowers/plans/2026-08-26-gui-simplification-removals,
phase-4-tune-unmount.md) from a deletion.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from phenotypic.gui._config import MOUNT_TUNE
from phenotypic.gui.shell._layout import NAV_MODEL, _NavGroup
from phenotypic.gui.shell._sandbox import SandboxRoot


def _leaf_ids(model) -> set[str]:
    found: set[str] = set()
    for entry in model:
        if isinstance(entry, _NavGroup):
            found.update(entry.members)
        else:
            found.add(entry)
    return found


@pytest.fixture()
def built_hub_dispatcher_mounts(tmp_path: Path) -> set[str]:
    """The composed hub's ``DispatcherMiddleware`` mount keys.

    Mirrors ``tests/integration/gui/test_tune_mount.py``'s
    ``test_hub_mounts_tune`` pattern (now retired by this phase): build a
    sandbox, compose the hub via ``create_app``, and read the mount prefixes
    off the underlying ``DispatcherMiddleware``.
    """
    from phenotypic.gui.shell._app import create_app

    sandbox = SandboxRoot.from_path(tmp_path)
    app = create_app(sandbox)
    mounts = app.server.wsgi_app.mounts
    return {m.rstrip("/") for m in mounts}


def test_nav_model_has_no_tune_leaf():
    assert "shell-tab-tune" not in _leaf_ids(NAV_MODEL)


def test_dispatcher_has_no_tune_mount(built_hub_dispatcher_mounts):
    assert MOUNT_TUNE.rstrip("/") not in built_hub_dispatcher_mounts


def test_tune_package_is_still_importable():
    assert importlib.import_module("phenotypic.gui.tune") is not None
