"""Shared fixtures for the plotting package tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_chrome_probe():
    """Clear the process-global Chrome verdict around every test in this package.

    ``chrome_available`` memoises on first call, so a test that probes for real
    poisons every later test in the same process: a monkeypatch applied
    afterwards is simply inert. Measured -- patching ``pio.to_image`` to succeed
    after one real probe still returned ``False`` until ``reset_chrome_probe()``
    was called.

    There is no ``pytest-randomly`` here, so the result is a *stable* wrong
    answer rather than an intermittent one, which is harder to notice and
    easier to trust.

    This was a file-local fixture in ``test_backends.py``. It is package scope
    now because Task 5 puts ``chrome_available()`` on the publication path and
    Task 7's tests call ``preflight_plot_backends`` unpatched, so the leak would
    reach files that never mention the probe.
    """
    from phenotypic.plotting._pipeline._backends import reset_chrome_probe

    reset_chrome_probe()
    yield
    reset_chrome_probe()
