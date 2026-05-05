"""Save-preset E2E test (mutates sandbox; needs function-scoped fixtures).

This module is split out from ``test_run_console.py`` because the test
writes ``<sandbox>/.phenotypic-gui/presets/<name>.json``. Sharing a
module-scoped sandbox with read-only tests would leak state across runs,
so we override ``fake_sandbox`` and ``live_server`` here with
function-scoped variants that build a fresh sandbox per test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page

from .conftest import _build_sandbox, _start_live_server


# ---------------------------------------------------------------------------
# Function-scoped fixture overrides — isolate mutation from module-scoped
# tests in sibling files.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    """Function-scoped override: fresh sandbox per test."""
    return _build_sandbox(tmp_path)


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Function-scoped override: fresh Werkzeug boot per test."""
    yield from _start_live_server(fake_sandbox)


@pytest.fixture
def hub_url(live_server: str) -> str:
    return live_server


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_save_preset_writes_file(
    page: Page, hub_url: str, fake_sandbox: Path,
) -> None:
    """Clicking Save preset with a name writes
    ``<sandbox>/.phenotypic-gui/presets/<name>.json``.

    The form state is empty here; that's fine — the preset just round-trips
    whatever the form's `RC_STORE_FORM_STATE` currently holds (default
    ``RunConsoleState`` round-tripped to JSON).
    """
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#rc-input-preset-name")
    page.fill("#rc-input-preset-name", "smoke_preset")
    page.click("#rc-btn-save-preset")
    # The save callback returns a toast; wait for any toast text.
    page.wait_for_function(
        "() => {"
        "  const t = document.getElementById('rc-toast');"
        "  return t && (t.textContent || '').toLowerCase().includes('saved');"
        "}",
        timeout=5_000,
    )
    target = fake_sandbox / ".phenotypic-gui" / "presets" / "smoke_preset.json"
    assert target.is_file()
