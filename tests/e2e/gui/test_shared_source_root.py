"""Cross-page shared source-image-root flow."""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page, expect


def _select_plate1_source(page: Page, hub_url: str) -> None:
    """Click the shared fixture's image directory in the sidebar."""
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: plate1",
        timeout=5_000,
    )


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
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: plate1",
        timeout=5_000,
    )

    page.goto(hub_url + "/builder/")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: plate1",
        timeout=5_000,
    )
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
    expect(page.locator("#shell-source-image-root-label")).to_have_attribute(
        "title",
        str(plate),
    )

    page.goto(hub_url + "/results/")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: plate1",
        timeout=5_000,
    )
    expect(page.locator("#results-viewer-empty-state")).to_be_visible(
        timeout=10_000
    )

    page.goto(hub_url + "/analysis/")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: plate1",
        timeout=5_000,
    )
    expect(page.locator("#analysis-page")).to_be_visible(timeout=10_000)
