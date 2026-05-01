"""Sidebar file browser interactions (E2E).

Confirms:

* Tree renders the expected child entries from ``fake_sandbox``.
* Capability badges show on the image directory and the CLI output.
* Hidden-files toggle flips the input checked state.
* External-symlinks toggle flips the input checked state.
* Refresh button click does not throw (chromium console error count
  stays flat across the click).
"""
from __future__ import annotations

from playwright.sync_api import Page, expect


def test_tree_renders_expected_entries(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-sidebar-tree")
    text = page.locator("#shell-sidebar-tree").text_content() or ""
    assert "plate1" in text
    assert "results" in text


def test_image_dir_carries_image_count_badge(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-sidebar-tree")
    text = page.locator("#shell-sidebar-tree").text_content() or ""
    # ``plate1/`` has one image.tif → ``img (1)`` badge per the classifier.
    assert "img (1)" in text


def test_hidden_toggle_changes_state(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/")
    sel = "#shell-sidebar-hidden-toggle input"
    page.wait_for_selector(sel)
    expect(page.locator(sel)).not_to_be_checked()
    page.click(sel)
    expect(page.locator(sel)).to_be_checked()


def test_symlink_toggle_changes_state(page: Page, hub_url: str) -> None:
    page.goto(hub_url + "/")
    sel = "#shell-sidebar-symlink-toggle input"
    page.wait_for_selector(sel)
    expect(page.locator(sel)).not_to_be_checked()
    page.click(sel)
    expect(page.locator(sel)).to_be_checked()


def test_refresh_button_does_not_throw(page: Page, hub_url: str) -> None:
    """Clicking Refresh fires the classifier-cache flush callback. A regression
    that broke the callback would surface as a JS console error on the
    page."""
    errors: list[str] = []
    page.on(
        "console",
        lambda msg: errors.append(msg.text) if msg.type == "error" else None,
    )
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-sidebar-refresh")
    page.click("#shell-sidebar-refresh")
    # Give Dash time to round-trip the callback.
    page.wait_for_timeout(1_500)
    assert errors == [], f"console errors after refresh: {errors!r}"
