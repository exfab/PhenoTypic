"""Playwright e2e: Browse Timeline focus-and-navigate controller (spec §16).

Requires PLAYWRIGHT=1 (enforced by the conftest module-skip).

The fixture seeds the matrix UNDER the shared ``plate1`` source root and
selects it via the proven sidebar-tree-click idiom
(``test_shared_source_root._select_plate1_source``); it then switches to
Timeline mode and validates a non-empty grid renders before any test relies
on it. A deliberately SMALL viewport (so the ``data-focus-margin`` ring
provably extends OFF-SCREEN at the default (0,0) focus) makes the
bounded-window + off-screen-pre-mount assertions meaningful.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import expect

from phenotypic.gui.shell._metadata_context import metadata_payload_from_path
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import source_payload_from_path

# Tight DOM-poll budget on a fresh Werkzeug server: stochastically slow on GHA.
pytestmark = pytest.mark.ci_flaky

# Small enough that the centered focus window + the 2-cell focus margin extends
# beyond the visible rectangle at the default (0,0) corner focus, so an
# off-screen-but-mounted cell provably exists; large enough that the 6x6 matrix
# still has FAR cells the bounded window never mounts.
_VIEWPORT = {"width": 600, "height": 450}
_FOLDERS = ("t0", "t1", "t2", "t3", "t4", "t5")
_NAMES = (
    "plateA.png",
    "plateB.png",
    "plateC.png",
    "plateD.png",
    "plateE.png",
    "plateF.png",
)


@pytest.fixture()
def live_browse_timeline(fake_sandbox, live_server, hub_url, page):
    """Browse open in Timeline mode over a seeded 6x6 source matrix.

    Seeds ``>=3`` sub-folders x ``>=3`` PNGs UNDER the shared fixture's
    ``plate1`` source root (here 6x6 so far cells stay unmounted), selects
    ``plate1`` via the sidebar tree (the proven seeding idiom — sets the shared
    source-image-root store + label through the real UI path), opens Browse,
    switches to Timeline mode, and confirms a non-empty grid renders.
    """
    from PIL import Image as PILImage

    plate1 = fake_sandbox / "plate1"
    for folder in _FOLDERS:
        d = plate1 / folder
        d.mkdir(parents=True, exist_ok=True)
        for name in _NAMES:
            PILImage.new("RGB", (300, 200), (40, 80, 120)).save(d / name)

    # A small viewport makes the margin ring provably extend off-screen at the
    # corner focus (see module docstring).
    page.set_viewport_size(_VIEWPORT)

    # Select ``plate1`` as the shared source via the sidebar tree (the proven
    # idiom — mirrors test_shared_source_root._select_plate1_source). This sets
    # the source-image-root store + label through the real UI path, so the
    # Browse dataset callback sees a validated payload.
    page.goto(hub_url + "/run/")
    page.wait_for_selector("#shell-sidebar-tree", timeout=10_000)
    page.click('button[id*="\\"path\\":\\"plate1\\""][id*="shell-sidebar-entry"]')
    # Confirm the store write landed before navigating away — the sidebar click
    # fires a Dash callback that persists the source-image-root store; opening
    # Browse before it settles would race the localStorage write.
    if not page.locator("#shell-settings-popover").is_visible():
        page.click("#shell-settings-button")
    expect(page.locator("#shell-source-image-root-label")).to_contain_text(
        "source: plate1", timeout=5_000
    )

    # Open the Browse mount (it reads the shared source from the store).
    page.goto(hub_url + "/browse/")
    # Switch to Timeline mode.
    page.click("text=Timeline")
    # Validate the fixture renders a non-empty grid before any test relies on it.
    page.wait_for_selector(".timeline-cell[data-src]", timeout=10_000)
    return page


def test_focus_starts_on_first_populated_cell(live_browse_timeline) -> None:
    page = live_browse_timeline  # fixture: server up, Browse open, Timeline mode on
    # Cells exist immediately (placeholders carry data-src + grid coordinates)…
    page.wait_for_selector(".timeline-cell[data-src]")
    # …and exactly one cell is focused — the first populated cell (top-left of
    # the ordered matrix: smallest row-index, then col-index).
    page.wait_for_selector(".timeline-cell--focused")
    focused = page.eval_on_selector_all(".timeline-cell--focused", "els => els.length")
    assert focused == 1
    # That focused cell is the smallest-coordinate populated cell.
    coord = page.eval_on_selector(
        ".timeline-cell--focused",
        "el => el.getAttribute('data-row-index') + ',' + el.getAttribute('data-col-index')",
    )
    assert coord == "0,0"  # seeded fixture has a populated (0,0) cell


def test_arrow_right_moves_focus_and_mounts_new_neighborhood(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    # Focus the viewport, then press ArrowRight → focus advances one column.
    # Click the focusable viewport wrapper (tabIndex=0) so its scoped keydown
    # listener receives the event (the inner #browse-tl-grid is not focusable).
    page.click(".browse-tl-viewport")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-col-index') === '1'"
    )
    # The newly-near neighborhood mounts <img> (focus + margin ring).
    page.wait_for_function(
        "document.querySelectorAll('#browse-tl-grid .timeline-cell img').length > 0"
    )


def test_edge_button_right_moves_focus(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    page.click("#browse-tl-nav-right")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-col-index') === '1'"
    )


def test_far_cell_is_not_mounted_window_is_bounded(live_browse_timeline) -> None:
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    # Give the focus window a beat to mount before counting.
    page.wait_for_function(
        "document.querySelectorAll('#browse-tl-grid .timeline-cell img').length > 0"
    )
    total = page.eval_on_selector_all(".timeline-cell[data-src]", "els => els.length")
    mounted = page.eval_on_selector_all(
        "#browse-tl-grid .timeline-cell img", "els => els.length"
    )
    # Bounded window: only the focused neighborhood + margin ring mounts, NEVER
    # every cell. (Seed enough cells that total > the visible window + margin.)
    assert 0 < mounted < total


def test_offscreen_margin_ring_is_pre_mounted(live_browse_timeline) -> None:
    # User-required smooth-UX behaviour (spec §16.3): a cell JUST OUTSIDE the
    # visible window but WITHIN data-focus-margin must already carry an <img>
    # (pre-mounted), so a quick step into it is instant.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    page.wait_for_function(
        "document.querySelectorAll('#browse-tl-grid .timeline-cell img').length > 0"
    )
    # A cell within the focus margin but off-screen is mounted; assert at least
    # one mounted <img> sits outside the viewport's visible rectangle.
    off_screen_mounted = page.evaluate(
        """() => {
            const vp = document.querySelector('.browse-tl-viewport').getBoundingClientRect();
            const imgs = document.querySelectorAll('#browse-tl-grid .timeline-cell img');
            let n = 0;
            imgs.forEach((img) => {
                const r = img.getBoundingClientRect();
                const visible = r.right > vp.left && r.left < vp.right
                    && r.bottom > vp.top && r.top < vp.bottom;
                if (!visible) { n += 1; }
            });
            return n;
        }"""
    )
    assert off_screen_mounted >= 1  # the margin ring pre-mounted at least one


# --- Task 9: single-image deep-zoom pop-out --------------------------------
def test_hover_reveals_popout_button(live_browse_timeline) -> None:
    # The ⤢ button is HIDDEN by default and revealed ONLY on tile hover (CSS
    # :hover gate, B1 timeline.css / spec §16.4). This genuinely exercises the
    # gate: assert visibility === 'hidden' BEFORE hover, then 'visible' after.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell[data-src]")
    cell = page.query_selector(".timeline-cell[data-src]")
    btn = cell.query_selector(".timeline-cell-popout")
    assert btn is not None
    # Before hover: the CSS gate keeps the button hidden.
    before = page.evaluate(
        "(el) => getComputedStyle(el).visibility", btn
    )
    assert before == "hidden", f"expected ⤢ hidden before hover, got {before!r}"
    # Hover the cell → the :hover rule reveals the button.
    cell.hover()
    page.wait_for_function(
        "() => { const b = document.querySelector("
        "'.timeline-cell[data-src]:hover .timeline-cell-popout'); "
        "return b && getComputedStyle(b).visibility === 'visible' "
        "&& getComputedStyle(b).display !== 'none'; }"
    )


def test_focused_cell_has_visible_highlight(live_browse_timeline) -> None:
    # The single focused cell must be VISIBLY highlighted (spec §16.1) — B1's
    # timeline.css gives .timeline-cell--focused a blue outline + ring shadow.
    # Assert at least one of outline / box-shadow is a non-'none' value so the
    # focus indicator can never silently regress to invisible.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    styles = page.eval_on_selector(
        ".timeline-cell--focused",
        "el => { const s = getComputedStyle(el);"
        " return {outline: s.outlineStyle, outlineWidth: s.outlineWidth,"
        " boxShadow: s.boxShadow}; }",
    )
    has_outline = (
        styles["outline"] not in ("none", "")
        and styles["outlineWidth"] not in ("0px", "")
    )
    has_box_shadow = styles["boxShadow"] not in ("none", "")
    assert has_outline or has_box_shadow, (
        f"focused cell has no visible highlight: {styles!r}"
    )


def test_popout_opens_deep_zoom_on_hover_click(live_browse_timeline) -> None:
    page = live_browse_timeline
    # Target a SINGLE real seeded PNG (t0/plateA.png at col 1) — the shared
    # fixture's (0,0) ``image.tif`` is an empty stub whose DZI 500s, so the
    # deep-zoom would have no canvas to mount. The explicit col-index keeps the
    # selector unambiguous (one cell, not the whole t0 row).
    sel = '.timeline-cell[data-src][data-row="t0"][data-col-index="1"]'
    page.wait_for_selector(sel)
    cell = page.query_selector(sel)
    cell.hover()  # reveal the ⤢ before clicking it
    page.click(f"{sel} .timeline-cell-popout")
    # dbc.Modal puts the component id on the inner .modal-dialog; the .show
    # open-state class lands on the OUTER .modal wrapper (its closest ancestor).
    page.wait_for_function(
        "() => { const d = document.getElementById('browse-tl-popout-modal');"
        " const m = d && d.closest('.modal');"
        " return m && m.classList.contains('show'); }",
        timeout=10000,
    )
    # OSD mounts its canvas inside the modal body.
    page.wait_for_function(
        "document.querySelector('#browse-tl-popout-osd canvas') !== null"
    )


def test_enter_opens_popout_for_focused_cell(live_browse_timeline) -> None:
    # The keyboard path (spec §16.4): Enter on the focused cell opens the same
    # pop-out modal + OSD canvas. The controller (Task 7) wires this via the
    # shared #browse-tl-popout-input bridge.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".browse-tl-viewport")  # focus the viewport (not a text input)
    # Move focus off the (0,0) empty-image.tif stub onto a real seeded PNG
    # (row t0, col plateA.png) so the deep-zoom has a decodable source.
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "() => { const f = document.querySelector('.timeline-cell--focused');"
        " return f && f.getAttribute('data-src') && f.getAttribute('data-row') === 't0'"
        " && f.getAttribute('data-col-index') === '1'; }"
    )
    page.keyboard.press("Enter")
    # dbc.Modal id is on the .modal-dialog; .show lands on the outer .modal.
    page.wait_for_function(
        "() => { const d = document.getElementById('browse-tl-popout-modal');"
        " const m = d && d.closest('.modal');"
        " return m && m.classList.contains('show'); }",
        timeout=10000,
    )
    page.wait_for_function(
        "document.querySelector('#browse-tl-popout-osd canvas') !== null"
    )


# --- M4: <body> MutationObserver re-attach (Phase 3 depends on it) ----------
def test_reattach_observer_rebinds_on_wholesale_grid_swap(live_browse_timeline) -> None:
    # Spec §15.7 / §16.8: the controller's <body> MutationObserver re-attaches
    # when Dash REPLACES the grid container wholesale (a fresh node lacking
    # data-tl-attached). Browse's own clientside callbacks normally mask this
    # path (they call attach() explicitly), but Phase 3's dbc.Tabs re-mount
    # will exercise it, so guard it now: swap the #browse-tl-grid node for a
    # brand-new one with fresh cells and assert focus works on the NEW node.
    page = live_browse_timeline
    page.wait_for_selector(".timeline-cell--focused")
    # Swap the grid node wholesale: build a fresh .timeline-grid-container with
    # the same id but a NEW 2x2 cell set, no data-tl-attached flag, and replace
    # the live one inside the viewport. The <body> observer must fire attach()
    # on the fresh node (it sees a .timeline-grid-container without tlAttached).
    page.evaluate(
        """() => {
            const old = document.getElementById('browse-tl-grid');
            const parent = old.parentNode;
            const fresh = document.createElement('div');
            fresh.id = 'browse-tl-grid';
            fresh.className = 'timeline-grid-container';
            fresh.setAttribute('data-focus-margin', '2');
            fresh.setAttribute('data-mount-cap', '400');
            fresh.setAttribute('data-warm-concurrency', '2');
            // Two populated cells so firstPopulatedCell() + focus has a target.
            for (let c = 0; c < 2; c++) {
                const cell = document.createElement('div');
                cell.className = 'timeline-cell';
                cell.style.width = '120px';
                cell.style.height = '120px';
                cell.style.position = 'relative';
                cell.setAttribute('data-row-index', '0');
                cell.setAttribute('data-col-index', String(c));
                cell.setAttribute('data-src', 'about:blank#swap' + c);
                cell.setAttribute('data-ref', 'swap' + c);
                cell.setAttribute('data-row', 'swapRow');
                cell.setAttribute('data-col', String(c));
                fresh.appendChild(cell);
            }
            old.remove();              // drop the attached node
            parent.appendChild(fresh); // insert the fresh, un-attached node
        }"""
    )
    # The observer re-attaches → a cell on the FRESH node becomes focused and
    # the controller marks the new node data-tl-attached="1".
    page.wait_for_function(
        "() => { const g = document.getElementById('browse-tl-grid');"
        " return g && g.dataset.tlAttached === '1'"
        " && g.querySelector('.timeline-cell--focused') !== null; }",
        timeout=10000,
    )
    # Exactly one focused cell on the new node (no double-binding / stale state).
    focused = page.eval_on_selector_all(
        "#browse-tl-grid .timeline-cell--focused", "els => els.length"
    )
    assert focused == 1


# --- M5: same-cell pop-out re-open ------------------------------------------
def test_same_cell_popout_reopens_after_close(live_browse_timeline) -> None:
    # Closing the dbc modal leaves the server is_open stale; clicking the SAME
    # tile must reopen it. browse.js adds a monotonic sequence to each
    # revision-bound event-store write, so the Dash input changes every time.
    page = live_browse_timeline
    sel = '.timeline-cell[data-src][data-row="t0"][data-col-index="1"]'
    page.wait_for_selector(sel)

    def _open_via_cell() -> None:
        cell = page.query_selector(sel)
        cell.hover()  # reveal the ⤢
        page.click(f"{sel} .timeline-cell-popout")
        page.wait_for_function(
            "() => { const d = document.getElementById('browse-tl-popout-modal');"
            " const m = d && d.closest('.modal');"
            " return m && m.classList.contains('show'); }",
            timeout=10000,
        )

    # 1) Open the pop-out on the cell.
    _open_via_cell()
    # 2) Close the modal via its header close button; wait until it is gone.
    page.click("#browse-tl-popout-modal .btn-close")
    page.wait_for_function(
        "() => { const d = document.getElementById('browse-tl-popout-modal');"
        " const m = d && d.closest('.modal');"
        " return !m || !m.classList.contains('show'); }",
        timeout=10000,
    )
    # 3) Click the SAME cell again. The sequence makes the event-store value
    #    change, so the server callback re-fires and the modal reopens.
    _open_via_cell()


def _wait_for_authorized_grid(page) -> None:
    """Wait until the server has acknowledged the browser-applied revision."""
    page.wait_for_function(
        """() => {
            const grid = document.getElementById('browse-tl-grid');
            if (!grid) { return false; }
            const revision = grid.getAttribute('data-grid-revision');
            return !!revision
                && grid.getAttribute('data-authorized-revision') === revision
                && !!grid.getAttribute('data-revision-generation')
                && !!grid.getAttribute('data-session-id');
        }""",
        timeout=10_000,
    )


def test_delayed_popout_approval_cannot_reopen_retired_revision(
    live_browse_timeline,
) -> None:
    """A revision-A response arriving after B is ignored by the live gate."""
    page = live_browse_timeline
    _wait_for_authorized_grid(page)
    retired = page.evaluate(
        """() => {
            const grid = document.getElementById('browse-tl-grid');
            const cell = grid.querySelector('.timeline-cell[data-ref]');
            return {
                session_id: grid.getAttribute('data-session-id'),
                generation: Number(
                    grid.getAttribute('data-revision-generation')),
                revision: grid.getAttribute('data-grid-revision'),
                sequence: 919,
                token: cell.getAttribute('data-ref'),
                label: 'retired/source.png',
            };
        }"""
    )

    page.click("#browse-tl-tile-size-plus")
    page.wait_for_function(
        """
        previous => {
            const grid = document.getElementById('browse-tl-grid');
            return grid && grid.getAttribute('data-grid-revision') !== previous;
        }
        """,
        arg=retired["revision"],
    )
    _wait_for_authorized_grid(page)

    # Simulate a delayed server response and its request payload arriving after
    # revision B. Neither may publish modal/store/title outputs for A.
    page.evaluate(
        """retired => {
            window.dash_clientside.set_props(
                'browse-tl-popout-event', {data: retired});
            window.dash_clientside.set_props(
                'browse-tl-popout-approved', {data: retired});
        }""",
        retired,
    )
    page.wait_for_timeout(300)
    modal_open = page.evaluate(
        """() => {
            const dialog = document.getElementById('browse-tl-popout-modal');
            const modal = dialog && dialog.closest('.modal');
            return !!(modal && modal.classList.contains('show'));
        }"""
    )
    assert modal_open is False


def test_compare_await_is_cancelled_by_grid_revision(
    live_browse_timeline,
) -> None:
    """An OSD-ready continuation cannot mount after its grid is retired."""
    page = live_browse_timeline
    _wait_for_authorized_grid(page)
    cells = page.query_selector_all(".timeline-cell[data-src][data-ref]")
    assert len(cells) >= 2
    for cell in cells[:2]:
        cell.click(modifiers=["Shift"])

    old_revision = page.locator("#browse-tl-grid").get_attribute(
        "data-grid-revision"
    )
    page.evaluate(
        """() => {
            window.__browseResolveOsd = null;
            window.__phenotypicTimeline.osdReady = new Promise(resolve => {
                window.__browseResolveOsd = resolve;
            });
        }"""
    )
    page.click("#browse-tl-compare-btn")
    page.click("#browse-tl-tile-size-plus")
    page.wait_for_function(
        """
        previous => {
            const grid = document.getElementById('browse-tl-grid');
            return grid && grid.getAttribute('data-grid-revision') !== previous;
        }
        """,
        arg=old_revision,
    )
    _wait_for_authorized_grid(page)
    page.evaluate("() => window.__browseResolveOsd()")
    page.wait_for_timeout(300)

    assert page.query_selector("#timeline-compare-modal") is None


def test_source_revision_clears_stale_timeline_and_rebinds_actions(
    live_browse_timeline,
    fake_sandbox: Path,
) -> None:
    """Reproduce F-004–F-006 without a page navigation.

    Author stale pattern/axis state and open Compare on ``plate1``, then publish
    a new shared source revision for ``plate2``. The reset must retire every old
    value in one response, and delegated popout/Compare actions must work on the
    newly rendered grid without exposing encoded refs.
    """
    from PIL import Image as PILImage

    page = live_browse_timeline
    plate2 = fake_sandbox / "plate2"
    for folder in ("new-t0", "new-t1"):
        directory = plate2 / folder
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("newA.png", "newB.png"):
            PILImage.new("RGB", (300, 200), (120, 60, 30)).save(
                directory / filename
            )

    source_payload = source_payload_from_path(
        SandboxRoot.from_path(fake_sandbox),
        plate2,
        source="manual",
    )
    assert source_payload is not None
    metadata_path = fake_sandbox / "plate2.csv"
    metadata_path.write_text(
        "Metadata_ImageFileName,Group,Time\n"
        "newA.png,control,0\n"
        "newB.png,stress,1\n",
        encoding="utf-8",
    )
    metadata_payload = metadata_payload_from_path(
        SandboxRoot.from_path(fake_sandbox),
        metadata_path,
    )
    assert metadata_payload is not None

    # Make old-source authoring visibly stale and open a Compare overlay whose
    # selection must be retired with that source revision.
    page.evaluate(
        """() => {
            window.dash_clientside.set_props(
                'browse-tl-row-source', {value: 'pattern'});
            window.dash_clientside.set_props(
                'browse-tl-time-source', {value: 'folder'});
            window.dash_clientside.set_props(
                'browse-tl-pattern-input', {value: '(?P<plate>.+)'});
            window.dash_clientside.set_props(
                'browse-tl-pattern-advanced', {value: ['advanced']});
        }"""
    )
    page.wait_for_function(
        "() => document.getElementById('browse-tl-pattern-input').value"
        " === '(?P<plate>.+)'"
    )
    page.wait_for_selector(".timeline-cell[data-src][data-ref]")
    old_cells = page.query_selector_all(".timeline-cell[data-src][data-ref]")
    assert len(old_cells) >= 2
    for cell in old_cells[:2]:
        cell.click(modifiers=["Shift"])
    page.click("#browse-tl-compare-btn")
    page.wait_for_selector("#timeline-compare-modal")

    # Change the shared source in place, exactly where the old implementation
    # retained its matrix, pattern preview, selection, and event bindings.
    page.evaluate(
        """
        payload => window.dash_clientside.set_props(
            'shell-source-image-root-store',
            {data: payload}
        )
        """,
        source_payload,
    )

    page.wait_for_function(
        "() => document.getElementById('browse-tl-pattern-input').value === ''"
    )
    expect(page.locator("#browse-tl-row-source")).to_contain_text("Folder")
    expect(page.locator("#browse-tl-time-source")).to_contain_text(
        "EXIF capture time"
    )
    expect(page.locator("#browse-tl-pattern-preview")).to_contain_text(
        "Enter a pattern to preview matches."
    )
    page.wait_for_selector(
        '.timeline-cell[data-src][data-row="new-t0"][data-ref]',
        timeout=10_000,
    )
    assert page.query_selector("#timeline-compare-modal") is None
    assert page.query_selector_all(".timeline-cell--selected") == []

    # A metadata revision rebuilds the grid again. The next delegated event is
    # bound to that replacement generation, not the retired source-only one.
    source_grid_revision = page.locator("#browse-tl-grid").get_attribute(
        "data-grid-revision"
    )
    page.evaluate(
        """
        payload => window.dash_clientside.set_props(
            'shell-metadata-csv-store',
            {data: payload}
        )
        """,
        metadata_payload,
    )
    page.wait_for_function(
        """
        previous => {
            const grid = document.getElementById('browse-tl-grid');
            return grid && grid.getAttribute('data-grid-revision') !== previous;
        }
        """,
        arg=source_grid_revision,
    )

    # Hover-click uses the revision-stamped Dash event after the source/grid
    # and metadata revisions. Its title is a readable sandbox-relative path.
    new_cell_selector = (
        '.timeline-cell[data-src][data-row="new-t0"][data-col-index="0"]'
    )
    new_cell = page.query_selector(new_cell_selector)
    assert new_cell is not None
    new_cell.hover()
    page.click(f"{new_cell_selector} .timeline-cell-popout")
    page.wait_for_function(
        "() => { const d = document.getElementById('browse-tl-popout-modal');"
        " const m = d && d.closest('.modal');"
        " return m && m.classList.contains('show'); }",
        timeout=10_000,
    )
    expect(page.locator("#browse-tl-popout-title")).to_contain_text(
        "plate2/new-t0/newA.png"
    )
    page.click("#browse-tl-popout-modal .btn-close")

    # Enter goes through the same delegated event store after the remount.
    page.click(".browse-tl-viewport")
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => { const d = document.getElementById('browse-tl-popout-modal');"
        " const m = d && d.closest('.modal');"
        " return m && m.classList.contains('show'); }",
        timeout=10_000,
    )
    page.click("#browse-tl-popout-modal .btn-close")

    # Compare also rebinds after the same revision and decodes internal tokens
    # before rendering visible titles.
    new_cells = page.query_selector_all(
        '.timeline-cell[data-src][data-row="new-t0"][data-ref]'
    )
    assert len(new_cells) == 2
    for cell in new_cells:
        cell.click(modifiers=["Shift"])
    page.click("#browse-tl-compare-btn")
    page.wait_for_selector("#timeline-compare-modal .timeline-compare-cell-title")
    titles = page.locator(
        "#timeline-compare-modal .timeline-compare-cell-title"
    ).all_text_contents()
    assert titles == [
        "plate2/new-t0/newA.png",
        "plate2/new-t0/newB.png",
    ]


def test_shared_refresh_rescans_selected_source_and_retires_timeline_state(
    live_browse_timeline,
    fake_sandbox: Path,
) -> None:
    """Refresh keeps source authority stable while rebuilding Browse in place."""
    from PIL import Image as PILImage

    page = live_browse_timeline
    late_dataset = fake_sandbox / "plate1" / "late-refresh"
    late_image = late_dataset / "late.png"
    source_before = page.evaluate(
        "() => window.localStorage.getItem('shell-source-image-root-store')"
    )

    page.evaluate(
        """() => {
            window.dash_clientside.set_props(
                'browse-tl-row-source', {value: 'pattern'});
            window.dash_clientside.set_props(
                'browse-tl-time-source', {value: 'folder'});
            window.dash_clientside.set_props(
                'browse-tl-pattern-input', {value: '(?P<plate>.+)'});
            window.dash_clientside.set_props(
                'browse-tl-pattern-advanced', {value: ['advanced']});
        }"""
    )
    page.wait_for_function(
        "() => document.getElementById('browse-tl-pattern-input').value"
        " === '(?P<plate>.+)'"
    )
    cells = page.query_selector_all(".timeline-cell[data-src][data-ref]")
    assert len(cells) >= 2
    for cell in cells[:2]:
        cell.click(modifiers=["Shift"])
    page.click("#browse-tl-compare-btn")
    page.wait_for_selector("#timeline-compare-modal")

    try:
        late_dataset.mkdir()
        PILImage.new("RGB", (300, 200), (25, 50, 75)).save(late_image)
        old_grid_revision = page.locator("#browse-tl-grid").get_attribute(
            "data-grid-revision"
        )

        # The open Compare strip intentionally covers the chrome. A DOM click
        # exercises the same shared Dash action while proving the refresh
        # itself retires the overlay.
        page.evaluate(
            "() => document.getElementById('shell-sidebar-refresh').click()"
        )

        page.click("#browse-dataset-picker")
        expect(page.get_by_role("option", name="late-refresh")).to_be_visible(
            timeout=10_000,
        )
        page.keyboard.press("Escape")
        page.wait_for_function(
            """
            previous => {
                const grid = document.getElementById('browse-tl-grid');
                return grid
                    && grid.getAttribute('data-grid-revision') !== previous;
            }
            """,
            arg=old_grid_revision,
        )
        page.wait_for_selector(
            '.timeline-cell[data-src][data-row="late-refresh"][data-ref]',
            timeout=10_000,
        )

        expect(page.locator("#browse-tl-row-source")).to_contain_text("Folder")
        expect(page.locator("#browse-tl-time-source")).to_contain_text(
            "EXIF capture time"
        )
        expect(page.locator("#browse-tl-pattern-preview")).to_contain_text(
            "Enter a pattern to preview matches."
        )
        assert page.input_value("#browse-tl-pattern-input") == ""
        assert page.query_selector("#timeline-compare-modal") is None
        assert page.query_selector_all(".timeline-cell--selected") == []
        assert (
            page.evaluate(
                "() => window.localStorage.getItem("
                "'shell-source-image-root-store')"
            )
            == source_before
        )
    finally:
        late_image.unlink(missing_ok=True)
        late_dataset.rmdir()
