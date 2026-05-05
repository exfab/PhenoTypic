"""Top-bar tab navigation + RSS readout + help modal (E2E).

These tests exercise the chrome that wraps every Dash mount. They confirm:

* Clicking a tab actually navigates (not just a label change).
* The active tab's class flips on each navigation.
* The RSS readout populates after the first ``dcc.Interval`` tick.
* The Help modal opens and contains the SSH-tunnel + cloud-deploy copy.
"""
from __future__ import annotations

from playwright.sync_api import Page, expect


def test_home_loads_with_chrome(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-top-bar")
    expect(page.locator("#shell-top-bar")).to_be_visible()
    expect(page.locator("#shell-sidebar")).to_be_visible()
    expect(page.locator("#shell-tab-home")).to_have_class(
        # Home tab carries the active class on the home page.
        # ``to_have_class`` matches a substring when the value is a string.
        "shell-tab shell-tab-active"
    )


def test_tab_navigation_active_class_tracks(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-tab-builder")

    page.click("#shell-tab-builder")
    page.wait_for_url("**/builder/")
    expect(page.locator("#shell-tab-builder")).to_have_class(
        "shell-tab shell-tab-active"
    )
    expect(page.locator("#shell-tab-home")).to_have_class("shell-tab")

    page.click("#shell-tab-viewer")
    page.wait_for_url("**/results/")
    expect(page.locator("#shell-tab-viewer")).to_have_class(
        "shell-tab shell-tab-active"
    )

    page.click("#shell-tab-run")
    page.wait_for_url("**/run/")
    expect(page.locator("#shell-tab-run")).to_have_class(
        "shell-tab shell-tab-active"
    )


def test_rss_readout_populates(page: Page, hub_url: str) -> None:
    """The RSS label updates from the placeholder once ``shell-rss-interval``
    fires. The placeholder text is ``"--"``; after the tick it becomes
    something like ``"RSS 444 MB"``."""
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-rss-label")
    # Default Interval is 5 s, but the first tick fires immediately on
    # page load. Give it 8 s to show up before failing.
    rss_locator = page.locator("#shell-rss-label")
    rss_locator.wait_for(state="visible", timeout=10_000)
    text = rss_locator.text_content() or ""
    # Match ``RSS <int> MB`` to make the assertion resilient to the actual
    # memory footprint.
    import re

    assert re.match(r"RSS \d+ MB", text), f"unexpected RSS readout: {text!r}"


def test_help_modal_opens_and_contains_copy(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-help-button")
    page.click("#shell-help-button")
    # dbc.Modal renders the title element when open.
    page.wait_for_selector("#shell-help-modal", timeout=5_000)
    text = page.locator("#shell-help-modal").text_content() or ""
    assert "SSH tunnel" in text
    assert "Cloud deployment" in text


def test_sandbox_label_renders_root(page: Page, hub_url: str, fake_sandbox) -> None:
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-root-label")
    text = page.locator("#shell-root-label").text_content() or ""
    assert str(fake_sandbox) in text
