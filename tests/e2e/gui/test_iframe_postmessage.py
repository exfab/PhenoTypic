"""Iframed dashboard + ``postShellEvent`` upgrade door (E2E).

Confirms the cross-frame contract that lets the iframed
``dashboard.html`` push manifest updates back to the parent shell:

1. Click a Recent Runs row → iframe ``src`` lands on the dashboard.
2. The dashboard's ``postShellEvent('manifest', ...)`` JS fires on its
   refresh tick.
3. The parent receives a ``{source: 'phenotypic-dashboard', kind, payload}``
   payload via ``window.addEventListener('message', ...)``.

The integration test in ``tests/integration/gui/test_postmessage_listener.py``
already proves the JS exists in newly-generated dashboard HTML; this
test proves the message actually crosses the iframe boundary in a real
browser.
"""
from __future__ import annotations

from playwright.sync_api import Page


def _install_message_listener(page: Page) -> None:
    """Wire a window-scoped message listener that records phenotypic-dashboard
    events into ``window.__pheno_received`` for later inspection."""
    page.evaluate(
        "() => {"
        "  window.__pheno_received = [];"
        "  window.addEventListener('message', (e) => {"
        "    if (e?.data?.source === 'phenotypic-dashboard') {"
        "      window.__pheno_received.push({"
        "        kind: e.data.kind,"
        "        hasPayload: !!e.data.payload,"
        "      });"
        "    }"
        "  });"
        "}"
    )


def test_iframe_loads_runs_blueprint_url(page: Page, hub_url: str) -> None:
    """The iframe's ``src`` resolves under the ``/runs/`` Flask blueprint
    (NOT a Dash mount), regardless of which page hosts the iframe."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector('[id*="rc-recents-row"]', timeout=10_000)
    page.locator('[id*="rc-recents-row"]').first.click()
    page.wait_for_function(
        "() => {"
        "  const f = document.getElementById('rc-iframe');"
        "  return f && (f.src || '').includes('/runs/');"
        "}",
        timeout=5_000,
    )
    src = page.locator("#rc-iframe").get_attribute("src") or ""
    # Specifically NOT a Dash sub-app prefix: ``/runs/`` lives on the shell
    # Flask fallback, so the URL is rooted at the host with no ``/builder/``,
    # ``/results/``, or ``/run/`` segment in front.
    assert "/runs/" in src
    assert "/builder/runs/" not in src
    assert "/results/runs/" not in src
    assert "/run/runs/" not in src


def test_postshell_event_crosses_iframe_boundary(
    page: Page, hub_url: str,
) -> None:
    """The dashboard's ``postShellEvent`` reaches the parent window.

    We click into the recent run, install a parent-side message listener,
    then wait for the dashboard's refresh tick (default ~5 s). At least
    one ``manifest`` event should land.
    """
    page.goto(hub_url + "/run/")
    page.wait_for_selector('[id*="rc-recents-row"]', timeout=10_000)
    page.locator('[id*="rc-recents-row"]').first.click()
    page.wait_for_selector("#rc-iframe[src*='/runs/']", state="attached", timeout=5_000)

    _install_message_listener(page)

    # Wait up to 12 s for at least one manifest event to arrive.
    page.wait_for_function(
        "() => (window.__pheno_received || []).length > 0",
        timeout=12_000,
    )
    received = page.evaluate("() => window.__pheno_received")
    assert any(m.get("kind") == "manifest" for m in received), (
        f"no manifest events received; got: {received!r}"
    )
