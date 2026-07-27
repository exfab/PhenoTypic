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

from pathlib import Path

import pytest
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


@pytest.mark.parametrize("toggle_selector", [
    "#shell-sidebar-hidden-toggle input",
    "#shell-sidebar-symlink-toggle input",
])
def test_sidebar_toggle_changes_state(
    page: Page, hub_url: str, toggle_selector: str,
) -> None:
    """Hidden-files and symlink toggles both flip from unchecked to checked
    on click."""
    page.goto(hub_url + "/")
    page.wait_for_selector(toggle_selector)
    expect(page.locator(toggle_selector)).not_to_be_checked()
    page.click(toggle_selector)
    expect(page.locator(toggle_selector)).to_be_checked()


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


def test_refresh_updates_nested_badges_open_picker_and_source_label(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
) -> None:
    """One revision refreshes every shell-owned filesystem snapshot."""
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"results\\""][id*="shell-sidebar-entry"]'
    )
    expect(page.locator("#shell-sidebar-tree")).to_contain_text(
        "CliOutputExample",
        timeout=5_000,
    )

    page.click("#shell-settings-button")
    page.click("#shell-settings-input-folder-pick")
    page.wait_for_selector(
        "#shell-source-image-root-modal-body",
        state="visible",
        timeout=5_000,
    )

    late_source = fake_sandbox / "late-source"
    late_source.mkdir()
    (late_source / "plate.tif").write_bytes(b"")
    late_output = fake_sandbox / "results" / "LateOutput"
    (late_output / "results").mkdir(parents=True)
    deliverables = late_output / "deliverables"
    deliverables.mkdir()
    (deliverables / "master_measurements.parquet").write_bytes(b"")

    # The modal backdrop blocks pointer clicks on the sidebar. A DOM click
    # exercises the same Dash callback while leaving the picker open.
    page.evaluate(
        "() => document.getElementById('shell-sidebar-refresh').click()"
    )

    expect(page.locator("#shell-source-image-root-modal-body")).to_contain_text(
        "late-source",
        timeout=5_000,
    )
    expect(page.locator("#shell-sidebar-tree")).to_contain_text(
        "LateOutput",
        timeout=5_000,
    )
    late_output_row = page.locator(
        'button[id*="\\"path\\":\\"results/LateOutput\\""]'
        '[id*="shell-sidebar-entry"]'
    )
    expect(late_output_row.locator("xpath=..")).to_contain_text(
        "out",
        timeout=5_000,
    )

    page.click(
        '[id*="shell-source-image-root-entry"][id*="late-source"]'
    )
    page.click("#shell-source-image-root-confirm")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: late-source",
        timeout=5_000,
    )

    (late_source / "plate.tif").unlink()
    late_source.rmdir()
    page.click("#shell-sidebar-refresh")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "Previous source unavailable in this sandbox",
        timeout=5_000,
    )
