"""Browse synced Compare strip e2e (spec §7). Mirror once Phase 3 lands Results.

Drives the real browser against the Browse Timeline surface: shift/ctrl-click
multi-select, the "Compare selected" button (`#browse-tl-compare-btn`), the
axis row-header trigger, the shared feedback-guarded viewport, and the over-cap
notice — all owned by the shared `timeline.js` controller
(`window.__phenotypicTimeline.openCompareStrip` + the `__compareViewers` seam).

Requires PLAYWRIGHT=1 (enforced by the conftest module-skip).

Fixture design notes:

* The fixtures BIO_REPLICATE the Phase 2 ``live_browse_timeline`` seeding idiom
  (sidebar-tree-click source seeding, verified at
  ``test_shared_source_root._select_plate1_source`` /
  ``test_browse_timeline.live_browse_timeline``) rather than import it: a pytest
  fixture defined in another test module is not shared across modules and this
  group is scoped to add only this file.
* They declare FUNCTION-SCOPED overrides of ``fake_sandbox`` + ``live_server``
  (the conftest's documented ``_build_sandbox`` + ``_start_live_server``
  pattern) so each test gets a pristine sandbox — the over-cap fixture must not
  inherit another test's seeded PNGs from the module-scoped default.
* The base sandbox seeds a valid ``plate1/image.tif`` in the ``.``/``(root)``
  row. Assertions target the additional seeded PNG rows
  (``[data-row="t0"]`` …) so matrix counts stay scoped to each test's fixture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import expect

from .conftest import _build_sandbox, _start_live_server

# Module-level marker: skipped on CI via ``-m "not ci_flaky"``. OSD mount + the
# per-viewer DZI tile fetch (<=cap concurrent pyramid builds, spec §15.10) on
# GHA shared runners stochastically exceeds the ``wait_for_selector`` budget;
# locally the flow is reliable. tests/CLAUDE.md.
pytestmark = pytest.mark.ci_flaky

_VIEWPORT = {"width": 600, "height": 450}
# 3x3 matrix of REAL PNGs (9 cells) — under the cap of 12, so the small
# fixture's selection never trips the over-cap path. The always-present valid
# root-row TIFF is intentionally outside the scoped PNG selection.
_SMALL_FOLDERS = ("t0", "t1", "t2")
_SMALL_NAMES = ("plateA.png", "plateB.png", "plateC.png")
# Exactly 14 REAL PNG cells (t0..t6 x plateA, plateB) for the over-cap notice.
_LARGE_FOLDERS = ("t0", "t1", "t2", "t3", "t4", "t5", "t6")
_LARGE_NAMES = ("plateA.png", "plateB.png")


@pytest.fixture()
def fake_sandbox(tmp_path: Path) -> Path:
    """Function-scoped sandbox override (per-test isolation).

    Overrides the conftest's module-scoped ``fake_sandbox`` so the small and
    large timeline fixtures never share (and pollute) one ``plate1`` directory.
    """
    return _build_sandbox(tmp_path)


@pytest.fixture()
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Function-scoped live server bound to the per-test ``fake_sandbox``."""
    yield from _start_live_server(fake_sandbox)


@pytest.fixture()
def hub_url(live_server: str) -> str:
    """Function-scoped ``hub_url`` override (the conftest alias is module-scoped
    and would scope-mismatch against the per-test ``live_server`` above)."""
    return live_server


def _seed_and_open_timeline(page, fake_sandbox, hub_url, folders, names):
    """Seed ``folders x names`` PNGs under ``plate1``, select it via the sidebar
    tree, open Browse, switch to Timeline mode, and wait for a non-empty grid.

    Mirrors the Phase 2 ``live_browse_timeline`` seeding path exactly (sidebar
    tree click → shared source store write → Browse mount → Timeline toggle).
    """
    from PIL import Image as PILImage

    plate1 = fake_sandbox / "plate1"
    for folder in folders:
        d = plate1 / folder
        d.mkdir(parents=True, exist_ok=True)
        for name in names:
            PILImage.new("RGB", (300, 200), (40, 80, 120)).save(d / name)

    page.set_viewport_size(_VIEWPORT)

    # Select ``plate1`` as the shared source via the sidebar tree (the proven
    # idiom — mirrors test_shared_source_root._select_plate1_source).
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click('button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]')
    # Confirm the store write landed before navigating away.
    if not page.locator("#shell-settings-popover").is_visible():
        page.click("#shell-settings-button")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
            "source: plate1", timeout=5_000
    )

    # Open the Browse mount (it reads the shared source from the store).
    page.goto(hub_url + "/browse/")
    page.click("text=Timeline")
    # Validate the fixture renders a non-empty grid before any test relies on it.
    page.wait_for_selector(".timeline-cell[data-src]", timeout=10_000)
    # And that the test-scoped PNG rows rendered (not just the base root row).
    page.wait_for_selector('.timeline-cell[data-src][data-row="t0"]', timeout=10_000)
    return page


@pytest.fixture()
def live_browse_timeline(fake_sandbox, live_server, hub_url, page):
    """Browse open in Timeline mode over a seeded 3x3 real-PNG matrix (<= cap).

    Replicates the Phase 2 fixture's sidebar-tree seeding idiom. The 9 real
    populated cells are under ``TIMELINE_COMPARE_CAP`` (12), so the Compare
    strip mounts every selected viewer without the over-cap notice.
    """
    return _seed_and_open_timeline(
            page, fake_sandbox, hub_url, _SMALL_FOLDERS, _SMALL_NAMES
    )


@pytest.fixture()
def live_browse_timeline_large(fake_sandbox, live_server, hub_url, page):
    """Browse open in Timeline mode over EXACTLY 14 real-PNG cells (over cap).

    ``t0..t6 x plateA, plateB`` = 7x2 = 14 > ``TIMELINE_COMPARE_CAP`` (12). The
    exact count is load-bearing: the over-cap test selects only the seeded PNG
    cells (excluding the valid base root-row TIFF) and asserts the full notice
    string
    ``"Showing first 12 of 14 — narrow the selection"``.
    """
    return _seed_and_open_timeline(
            page, fake_sandbox, hub_url, _LARGE_FOLDERS, _LARGE_NAMES
    )


def _real_png_cells(page):
    """Populated cells in the test-scoped PNG rows.

    The base sandbox also seeds a valid ``plate1/image.tif`` in the ``.`` row.
    Tests target ``t0..`` rows so selection counts cover only the matrices
    seeded by this module.
    """
    return page.query_selector_all(
            '.timeline-cell[data-src][data-row^="t"][data-ref]'
    )


def test_multiselect_then_compare_mounts_exactly_n_viewers(
        live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector('.timeline-cell[data-src][data-row="t0"]')
    cells = _real_png_cells(page)
    assert len(cells) >= 3
    # Shift-click 3 distinct populated real-PNG tiles → 3 selected.
    for cell in cells[:3]:
        cell.click(modifiers=["Shift"])
    assert len(page.query_selector_all(".timeline-cell--selected")) == 3
    page.click("#browse-tl-compare-btn")
    # Exactly 3 OSD viewers mount, each with a canvas.
    page.wait_for_selector(
            "#timeline-compare-modal .timeline-compare-osd canvas", timeout=15_000
    )
    osd_cells = page.query_selector_all(
            "#timeline-compare-modal .timeline-compare-cell"
    )
    assert len(osd_cells) == 3
    canvases = page.query_selector_all(
            "#timeline-compare-modal .timeline-compare-osd canvas"
    )
    assert len(canvases) >= 3  # OSD draws ≥1 canvas per viewer
    # Teardown: close via the × button and assert no viewer cell remains.
    page.click("#timeline-compare-close")
    page.wait_for_function(
            "() => document.querySelectorAll('.timeline-compare-cell').length === 0"
    )
    assert page.query_selector("#timeline-compare-modal") is None


def test_pan_zoom_one_viewer_propagates_to_peers(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector('.timeline-cell[data-src][data-row="t0"]')
    cells = _real_png_cells(page)
    for cell in cells[:2]:
        cell.click(modifiers=["Shift"])
    page.click("#browse-tl-compare-btn")
    page.wait_for_selector(
            "#timeline-compare-modal .timeline-compare-osd canvas", timeout=15_000
    )
    # Drive viewer[0]'s viewport via the OSD API and poll viewer[1]'s zoom +
    # center to confirm the shared viewport propagated. window.__phenotypicTimeline
    # .__compareViewers is the COMMITTED test seam assigned in the controller.
    page.evaluate(
            "() => { const vs = window.__phenotypicTimeline.__compareViewers; "
            "vs[0].viewport.zoomTo(2.0); }"
    )
    page.wait_for_function(
            "() => { const vs = window.__phenotypicTimeline.__compareViewers; "
            "return Math.abs(vs[1].viewport.getZoom(true) "
            "- vs[0].viewport.getZoom(true)) < 0.05; }",
            timeout=10_000,
    )
    # Center follows too (within tolerance): pan viewer[0] and poll viewer[1].
    page.evaluate(
            "() => { const vs = window.__phenotypicTimeline.__compareViewers; "
            "const OSD = window.OpenSeadragon; "
            "vs[0].viewport.panTo(new OSD.Point(0.4, 0.3)); }"
    )
    page.wait_for_function(
            "() => { const vs = window.__phenotypicTimeline.__compareViewers; "
            "const a = vs[0].viewport.getCenter(true); "
            "const b = vs[1].viewport.getCenter(true); "
            "return Math.abs(a.x - b.x) < 0.05 && Math.abs(a.y - b.y) < 0.05; }",
            timeout=10_000,
    )


def test_row_header_click_opens_strip_for_that_row(live_browse_timeline) -> None:
    page = live_browse_timeline
    # Click the test-scoped ``t0`` row header, excluding the base root row.
    sel = '.timeline-axis-label--y[data-row="t0"]'
    page.wait_for_selector(sel)
    page.click(sel)
    page.wait_for_selector(
            "#timeline-compare-modal .timeline-compare-osd canvas", timeout=15_000
    )
    osd_cells = page.query_selector_all(
            "#timeline-compare-modal .timeline-compare-cell"
    )
    # The seeded matrix has 3 time columns per row → 3 viewers for that row.
    assert len(osd_cells) == 3


def test_over_cap_selection_shows_notice(live_browse_timeline_large) -> None:
    page = live_browse_timeline_large  # exactly 14 real-PNG cells (see fixture)
    page.wait_for_selector('.timeline-cell[data-src][data-row="t0"]')
    # Select via JS by toggling the class on every REAL populated cell directly
    # (exclude the valid base root row), NOT by physical clicks: in the no-scroll
    # centered window, off-window cells are positioned via CSS transform and are
    # not reliably hit-testable, so 14 physical shift-clicks would be flaky. The
    # selection source of truth is the .timeline-cell--selected class, so
    # setting it is equivalent to clicking.
    total = page.evaluate(
            "() => { const cs = document.querySelectorAll("
            "'.timeline-cell[data-src][data-row^=\"t\"][data-ref]'); "
            "cs.forEach(c => c.classList.add('timeline-cell--selected')); "
            "return cs.length; }"
    )
    assert total == 14
    page.click("#browse-tl-compare-btn")
    page.wait_for_selector(
            "#timeline-compare-modal .timeline-compare-notice", timeout=15_000
    )
    notice = page.text_content("#timeline-compare-modal .timeline-compare-notice")
    # EXACT full string — guards the em-dash "—" coupling between the JS mirror
    # and the Python compare_selection_plan(...).notice. 14 selected, cap 12 →
    # "Showing first 12 of 14 — narrow the selection".
    assert notice == "Showing first 12 of 14 — narrow the selection"
    # Cap honored: exactly 12 viewer cells despite 14 selected.
    assert (
            len(page.query_selector_all(
                "#timeline-compare-modal .timeline-compare-cell"))
            == 12
    )


def test_shift_click_does_not_open_popout(live_browse_timeline) -> None:
    # A modified click toggles selection only; the deep-zoom pop-out (Phase 2)
    # must NOT open on shift-click.
    page = live_browse_timeline
    page.wait_for_selector('.timeline-cell[data-src][data-row="t0"]')
    cell = _real_png_cells(page)[0]
    cell.click(modifiers=["Shift"])
    # Selection toggled on…
    assert "timeline-cell--selected" in (cell.get_attribute("class") or "")
    # …but neither the compare strip nor the single-image pop-out opened.
    assert page.query_selector("#timeline-compare-modal") is None
    popout_open = page.evaluate(
            "() => { const d = document.getElementById('browse-tl-popout-modal');"
            " const m = d && d.closest('.modal');"
            " return !!(m && m.classList.contains('show')); }"
    )
    assert popout_open is False
