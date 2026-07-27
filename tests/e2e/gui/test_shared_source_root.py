"""Cross-page shared source-image-root flow."""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


def _open_settings(page: Page) -> None:
    if not page.locator("#shell-settings-popover").is_visible():
        page.click("#shell-settings-button")
    page.wait_for_selector("#shell-settings-popover", state="visible", timeout=5_000)


def _expect_settings_source_label(page: Page, text: str) -> None:
    _open_settings(page)
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        text,
        timeout=5_000,
    )


def _select_plate1_source(page: Page, hub_url: str) -> None:
    """Click the shared fixture's image directory in the sidebar."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )
    _expect_settings_source_label(page, "source: plate1")


def test_shared_source_persists_across_pages_and_seeds_builder(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
) -> None:
    """Run-selected source stays visible across mounted pages.

    Builder also consumes the shared source as the Load Image browse root, while
    Viewer and Analysis still load their own empty/output-root surfaces.
    """
    _select_plate1_source(page, hub_url)
    plate = fake_sandbox / "plate1"

    page.goto(hub_url + "/tune/")
    _expect_settings_source_label(page, "source: plate1")

    page.goto(hub_url + "/builder/")
    _expect_settings_source_label(page, "source: plate1")
    page.click("#btn-load-image")
    page.wait_for_function(
        "() => {"
        "  const modal = document.getElementById('modal-load-image');"
        "  return modal && getComputedStyle(modal).display !== 'none';"
        "}",
        timeout=5_000,
    )
    expect(page.locator("#modal-load-image-body")).to_contain_text(
        "image.tif",
        timeout=5_000,
    )
    _open_settings(page)
    expect(page.locator("#shell-source-image-root-label")).to_have_attribute(
        "title",
        str(plate),
    )

    page.goto(hub_url + "/results/")
    _expect_settings_source_label(page, "source: plate1")
    expect(page.locator("#results-viewer-empty-state")).to_be_visible(
        timeout=10_000
    )

    page.goto(hub_url + "/analysis/")
    _expect_settings_source_label(page, "source: plate1")
    expect(page.locator("#analysis-page")).to_be_visible(timeout=10_000)


def test_status_bar_source_picker_sets_shared_source_and_page_inputs(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
) -> None:
    """Top-bar source picker writes the same shared source store."""
    plate = fake_sandbox / "plate1"

    page.goto(hub_url + "/run/")
    _open_settings(page)
    page.click("#shell-settings-input-folder-pick")
    page.wait_for_selector("#shell-source-image-root-modal", timeout=5_000)
    page.wait_for_function(
        "() => {"
        "  const m = document.getElementById('shell-source-image-root-modal');"
        "  return m && getComputedStyle(m).display !== 'none';"
        "}",
        timeout=5_000,
    )
    expect(page.locator("#shell-source-image-root-modal-body")).to_contain_text(
        "plate1",
        timeout=5_000,
    )
    page.click(
        '[id*="shell-source-image-root-entry"][id*="plate1"]'
    )
    page.click("#shell-source-image-root-confirm")

    _expect_settings_source_label(page, "source: plate1")
    expect(page.locator("#shell-source-image-root-label")).to_have_attribute(
        "title",
        str(plate),
    )
    expect(page.locator("#rc-label-input")).to_contain_text(
        "plate1",
        timeout=5_000,
    )

    page.goto(hub_url + "/builder/")
    _expect_settings_source_label(page, "source: plate1")
    page.click("#btn-load-image")
    page.wait_for_function(
        "() => {"
        "  const modal = document.getElementById('modal-load-image');"
        "  return modal && getComputedStyle(modal).display !== 'none';"
        "}",
        timeout=5_000,
    )
    expect(page.locator("#modal-load-image-body")).to_contain_text(
        "image.tif",
        timeout=5_000,
    )

    page.goto(hub_url + "/tune/")
    _expect_settings_source_label(page, "source: plate1")


@pytest.mark.parametrize(
    "payload",
    [
        {
            "version": 1,
            "abs_path": "/previous/sandbox/plate1",
            "rel_path": "plate1",
            "label": "plate1",
            "validated": True,
        },
        {
            "version": 2,
            "kind": "image_source",
            "relative_path": "plate1",
            "absolute_path_at_selection": "/previous/sandbox/plate1",
            "sandbox_fingerprint": "different-sandbox",
            "validation": {"exists": True, "is_directory": True},
            "selected_at": "2026-07-23T00:00:00+00:00",
            "abs_path": "/previous/sandbox/plate1",
            "rel_path": "plate1",
            "label": "plate1",
            "validated": True,
        },
    ],
    ids=["v1", "v2"],
)
def test_previous_sandbox_source_is_unavailable_until_reselected(
    page: Page,
    hub_url: str,
    fake_sandbox: Path,
    payload: dict[str, object],
) -> None:
    """V1 and V2 stores never rebind the same relative name across roots."""
    payload = dict(payload)
    if payload["version"] == 2:
        # Keep the compatibility absolute path identical to the current
        # selection. The explicit click must still replace this payload based
        # on its mismatched fingerprint.
        current_path = str((fake_sandbox / "plate1").resolve())
        payload["absolute_path_at_selection"] = current_path
        payload["abs_path"] = current_path
    page.goto(hub_url + "/")
    page.wait_for_selector("#shell-settings-button", timeout=10_000)
    page.evaluate(
        """
        payload => window.dash_clientside.set_props(
            'shell-source-image-root-store',
            {data: payload}
        )
        """,
        payload,
    )

    _expect_settings_source_label(
        page,
        "Previous source unavailable in this sandbox",
    )
    page.click(
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )
    _expect_settings_source_label(page, "source: plate1")
