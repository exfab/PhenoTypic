"""Integration test for the dashboard's postMessage upgrade door.

We can't exercise actual ``window.parent`` semantics without a browser
(that's deferred to the Phase 9 Playwright sweep). What we CAN check
without a browser:

    * The generated dashboard HTML contains the ``postShellEvent`` helper.
    * The helper is guarded by ``window.parent !== window`` so it stays
      silent when the dashboard is opened standalone (file://, double-
      click), preventing console errors.
    * Refresh / completion paths actually call the helper.
    * The wire format is structured JSON — ``source`` /
      ``kind`` / ``payload`` keys present.
"""
from __future__ import annotations

import re
from pathlib import Path

from phenotypic._cli._dashboard._generator import generate_dashboard
from phenotypic.tools_ import dashboard_html_path


def _generate_html(tmp_path: Path) -> str:
    """Generate dashboard.html under tmp_path/deliverables; return its source."""
    generate_dashboard(tmp_path, execution_mode="local")
    return dashboard_html_path(tmp_path).read_text(encoding="utf-8")


def test_dashboard_includes_postshell_event_helper(tmp_path: Path) -> None:
    html = _generate_html(tmp_path)
    assert "function postShellEvent(" in html


def test_postshell_event_guarded_by_parent_check(tmp_path: Path) -> None:
    """The standalone path must stay silent — no console spam from cross-
    origin postMessage attempts when opened from file://."""
    html = _generate_html(tmp_path)
    assert "window.parent === window" in html
    # Standalone: returns BEFORE attempting postMessage.
    assert re.search(
        r"window\.parent\s*===\s*window\s*\)\s*return",
        html,
    ) is not None


def test_refresh_callback_emits_manifest_event(tmp_path: Path) -> None:
    """The ``refresh()`` callback POSTs a 'manifest' shell event each tick."""
    html = _generate_html(tmp_path)
    assert "postShellEvent('manifest'" in html


def test_complete_callback_emits_complete_event(tmp_path: Path) -> None:
    """When ``data.is_complete`` flips, a 'complete' event is posted."""
    html = _generate_html(tmp_path)
    assert "postShellEvent('complete'" in html


def test_message_payload_carries_structured_keys(tmp_path: Path) -> None:
    """Wire format: ``source`` / ``kind`` / ``payload`` keys present."""
    html = _generate_html(tmp_path)
    assert "source: 'phenotypic-dashboard'" in html
    assert "kind: kind" in html
    assert "payload: payload" in html
