"""Playwright E2E coverage for optimized Browse Single-view navigation."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect


@pytest.fixture()
def live_browse_single(fake_sandbox, live_server, hub_url, page):
    """Open Browse over a deterministic twelve-image source folder."""
    from PIL import Image as PILImage

    plate1 = fake_sandbox / "plate1"
    for index in range(12):
        PILImage.new(
            "RGB",
            (320, 240),
            (20 + index, 80, 120),
        ).save(plate1 / f"image{index:02d}.png")

    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click(
        'button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]'
    )
    if not page.locator("#shell-settings-popover").is_visible():
        page.click("#shell-settings-button")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: plate1",
        timeout=5_000,
    )
    page.goto(hub_url + "/browse/")
    page.wait_for_selector(".browse-filmstrip-item", timeout=15_000)
    page.wait_for_selector("#browse-osd-div canvas", timeout=15_000)
    return page


def test_jk_and_shifted_jumps_clamp_through_canonical_picker(
    live_browse_single,
) -> None:
    page = live_browse_single
    position = page.locator("#browse-position")
    expect(position).to_contain_text("1 of")
    expect(page.locator("#browse-meta-image-name")).to_have_text("image.tif")

    page.keyboard.press("k")
    expect(position).to_contain_text("2 of")
    expect(page.locator("#browse-meta-image-name")).to_have_text("image00.png")
    page.keyboard.press("j")
    expect(position).to_contain_text("1 of")
    page.keyboard.press("Shift+k")
    expect(position).to_contain_text("11 of")
    page.keyboard.press("Shift+k")
    expect(position).to_contain_text("13 of")
    page.keyboard.press("k")
    expect(position).to_contain_text("13 of")


def test_shortcut_ignored_in_editing_control(live_browse_single) -> None:
    page = live_browse_single
    position = page.locator("#browse-position")
    expect(position).to_contain_text("1 of")
    page.locator("#browse-keep-position").focus()
    page.keyboard.press("k")
    expect(position).to_contain_text("1 of")


def test_viewer_is_reused_and_filmstrip_is_bounded(live_browse_single) -> None:
    page = live_browse_single
    page.evaluate(
        "window.__browseViewerBefore = window.__phenotypicBrowse.singleViewer"
    )
    page.keyboard.press("k")
    expect(page.locator("#browse-position")).to_contain_text("2 of")
    page.wait_for_function(
        "window.__phenotypicBrowse.singleViewer === window.__browseViewerBefore"
    )
    assert page.locator(".browse-filmstrip-item").count() <= 9
    assert (
        page.locator('.browse-filmstrip-item[aria-current="true"]').count()
        == 1
    )
    page.wait_for_selector(
        '.browse-filmstrip-thumb[data-loaded="true"]',
        timeout=10_000,
    )
    page.locator(".browse-filmstrip-item").nth(2).click()
    expect(page.locator("#browse-position")).to_contain_text("3 of")


def test_arrow_key_pans_osd_without_changing_selection(
    live_browse_single,
) -> None:
    page = live_browse_single
    expect(page.locator("#browse-position")).to_contain_text("1 of")
    before = page.evaluate(
        "window.__phenotypicBrowse.singleViewer.viewport.getCenter().x"
    )
    page.locator("#browse-osd-div").click()
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "before => window.__phenotypicBrowse.singleViewer.viewport.getCenter().x > before",
        arg=before,
        timeout=5_000,
    )
    expect(page.locator("#browse-position")).to_contain_text("1 of")


def test_keep_position_restores_equal_dimension_viewport(
    live_browse_single,
) -> None:
    page = live_browse_single
    page.keyboard.press("k")
    expect(page.locator("#browse-position")).to_contain_text("2 of")
    page.locator("#browse-keep-position").check()
    page.evaluate(
        """() => {
            const viewer = window.__phenotypicBrowse.singleViewer;
            viewer.viewport.panTo(new OpenSeadragon.Point(0.58, 0.52), true);
            viewer.viewport.zoomTo(1.35, null, true);
            viewer.viewport.applyConstraints();
        }"""
    )
    before = page.evaluate(
        "window.__phenotypicBrowse.singleViewer.viewport.getZoom()"
    )
    page.locator("#browse-osd-div").click()
    page.keyboard.press("k")
    expect(page.locator("#browse-position")).to_contain_text("3 of")
    page.wait_for_function(
        "before => Math.abs(window.__phenotypicBrowse.singleViewer.viewport.getZoom() - before) < 0.05",
        arg=before,
        timeout=10_000,
    )


def test_prepare_stop_clear_controls_report_progress(
    live_browse_single,
) -> None:
    page = live_browse_single
    page.click("#browse-prepare-btn")
    expect(page.locator("#browse-preparation-status")).to_contain_text(
        re.compile(r"Preparing dataset|Prepared \d+ of \d+"),
        timeout=10_000,
    )
    stop = page.locator("#browse-stop-prepare-btn")
    if stop.is_enabled():
        stop.click()
        expect(page.locator("#browse-preparation-status")).to_contain_text(
            re.compile(r"Stopping after current image|Prepared"),
            timeout=10_000,
        )
    clear = page.locator("#browse-clear-cache-btn")
    expect(clear).to_be_enabled(timeout=20_000)
    clear.click()
    expect(page.locator("#browse-cache-usage")).not_to_be_empty()
