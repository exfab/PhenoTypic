"""Executable contract for the generic drag-splitter in ``results_viewer.js``.

The splitter used to be the QC worklist's alone: every identifier was a
literal (``#qc-review-splitter``, ``#qc-review-worklist``,
``store-qc-sidebar-width``, a ``_qcSplitter`` flag), so a second surface
could not use it. It is now driven by a data-attribute contract --
``data-splitter-target`` names the pane to resize, ``data-splitter-store``
names the ``dcc.Store`` the final width persists to -- and the Python side
supplies both.

These drive the real handler in a real browser: ``mousedown`` ->
``mousemove`` -> ``mouseup`` against two independent splitters on one page.
A source-text assertion cannot tell "generalized" from "renamed", and the
Python-side clamp mirror in ``tests/unit/gui/results_viewer/
test_qc_review_layout.py`` cannot actuate a pointer at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_LIFECYCLE = (
    Path(__file__).parents[3]
    / "src/phenotypic/gui/results_viewer/_assets/results_viewer.js"
)
_SOURCE = _LIFECYCLE.read_text(encoding="utf-8")

#: The DOM shape a real viewer page has: the two containers that ARE
#: always mounted, no ``#qc-review-gallery`` (that surface is unmounted).
#: Any timer still live on this page is a session-long poll for something
#: that will never arrive.
_PRODUCTION_SHAPE = """
<div id="cards-container"></div>
<div id="colony-grid-container"></div>
"""

#: Two independent splitters, so a handle that resizes "the" pane rather
#: than *its* pane is a failure rather than a coincidence.
_TWO_SPLITTERS = _PRODUCTION_SHAPE + """
<div id="row" style="display:flex;align-items:stretch;height:240px;margin:0">
  <div id="pane-a" style="flex:0 0 auto;width:180px;background:#eee"></div>
  <div id="handle-a" data-splitter-target="pane-a" data-splitter-store="store-a"
       style="flex:0 0 auto;width:10px;background:#333;cursor:col-resize"></div>
  <div id="pane-b" style="flex:0 0 auto;width:180px;background:#ddd"></div>
  <div id="handle-b" data-splitter-target="pane-b" data-splitter-store="store-b"
       style="flex:0 0 auto;width:10px;background:#666;cursor:col-resize"></div>
  <div style="flex:1 1 auto"></div>
</div>
"""

#: Installed BEFORE the script tag: tracks which ``setInterval`` timers
#: the file leaves running, and captures ``dash_clientside.set_props``.
_INSTRUMENT = """() => {
    window.__setProps = [];
    window.dash_clientside = {
        set_props: (id, props) => { window.__setProps.push([id, props]); },
    };
    window.__liveIntervals = new Set();
    const realSet = window.setInterval.bind(window);
    const realClear = window.clearInterval.bind(window);
    window.setInterval = function (fn, ms) {
        const id = realSet(fn, ms);
        window.__liveIntervals.add(id);
        return id;
    };
    window.clearInterval = function (id) {
        window.__liveIntervals.delete(id);
        return realClear(id);
    };
}"""


def _load(page, body_html: str):
    """Set page content, instrument the window, then evaluate the module."""
    page.set_content(body_html)
    page.evaluate(_INSTRUMENT)
    page.add_script_tag(content=_SOURCE)
    return page


@pytest.fixture
def splitter_page(page):
    """A page with two mounted splitters and the module already evaluated."""
    return _load(page, _TWO_SPLITTERS)


def _width(page, element_id: str) -> float:
    """Rendered width in px of one element."""
    return page.evaluate(
        "(id) => document.getElementById(id).getBoundingClientRect().width",
        element_id,
    )


def _drag(page, handle_id: str, dx: int) -> None:
    """Drive a real pointer drag of *dx* px on one handle."""
    box = page.locator(f"#{handle_id}").bounding_box()
    assert box is not None
    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    step = 20 if dx > 0 else -20
    moved = 0
    while abs(moved) < abs(dx):
        moved += step
        if abs(moved) > abs(dx):
            moved = dx
        page.mouse.move(start_x + moved, start_y)
    page.mouse.up()


# ---------------------------------------------------------------------------
# The data-attribute contract
# ---------------------------------------------------------------------------


def test_the_module_names_no_qc_identifier(page) -> None:
    """Generalizing means the QC literals are gone, not renamed.

    Pinned as source text because it is a statement about the module's
    vocabulary, which no runtime observation can make: a splitter that
    still reads ``#qc-review-worklist`` behaves identically on the QC
    surface and is simply absent on every other one.
    """
    assert "qc-review-splitter" not in _SOURCE
    assert "qc-review-worklist" not in _SOURCE
    assert "store-qc-sidebar-width" not in _SOURCE
    assert "_qcSplitter" not in _SOURCE


def test_each_handle_resizes_only_its_own_target(splitter_page) -> None:
    """``data-splitter-target`` routes the drag; the peer pane is untouched."""
    page = splitter_page
    assert _width(page, "pane-a") == pytest.approx(180, abs=1)
    assert _width(page, "pane-b") == pytest.approx(180, abs=1)

    _drag(page, "handle-a", 120)

    assert _width(page, "pane-a") == pytest.approx(300, abs=2)
    assert _width(page, "pane-b") == pytest.approx(180, abs=1)

    _drag(page, "handle-b", -20)

    assert _width(page, "pane-a") == pytest.approx(300, abs=2)
    assert _width(page, "pane-b") == pytest.approx(160, abs=2)


def test_each_handle_persists_to_its_own_store(splitter_page) -> None:
    """``data-splitter-store`` routes the persisted width, one store each."""
    page = splitter_page
    _drag(page, "handle-a", 120)
    _drag(page, "handle-b", -20)

    calls = page.evaluate("() => window.__setProps")
    assert [store for store, _ in calls] == ["store-a", "store-b"]
    assert calls[0][1]["data"] == pytest.approx(300, abs=2)
    assert calls[1][1]["data"] == pytest.approx(160, abs=2)


def test_a_drag_past_the_bounds_clamps(splitter_page) -> None:
    """The rendered pane and the persisted width both stop at the clamp."""
    page = splitter_page
    _drag(page, "handle-a", 400)
    assert _width(page, "pane-a") == pytest.approx(380, abs=2)

    _drag(page, "handle-a", -400)
    assert _width(page, "pane-a") == pytest.approx(140, abs=2)

    widths = page.evaluate("() => window.__setProps.map((c) => c[1].data)")
    assert widths == [380, 140]


def test_clamp_sidebar_width_stays_on_the_namespace(splitter_page) -> None:
    """The exported clamp keeps its bounds and its garbage-input default."""
    page = splitter_page
    clamp = "(v) => window.__phenotypicResultsViewer.clampSidebarWidth(v)"
    assert page.evaluate(clamp, 250) == 250
    assert page.evaluate(clamp, 9999) == 380
    assert page.evaluate(clamp, -5) == 140
    assert page.evaluate(clamp, "not-a-number") == 180


# ---------------------------------------------------------------------------
# Attachment: late mounts, and no timer left behind
# ---------------------------------------------------------------------------


def test_a_handle_mounted_after_load_still_attaches(page) -> None:
    """Dash mounts tabs lazily, so the handle usually arrives late.

    Every assertion here is behavioural: that the handle attached is
    proven by the drag working, not by reading the attach flag. An
    earlier version waited on ``[data-splitter-attached='1']`` before
    dragging, which quietly turned this into a test of the flag -- it
    then "caught" a mutation that removed the flag while leaving the
    splitter working perfectly, reporting coverage the suite did not
    have.

    No wait is needed and none is honest: ``MutationObserver`` callbacks
    are delivered at the microtask checkpoint ending the task that
    mutated the DOM, so attachment has already happened by the time the
    next CDP round-trip begins the drag.
    """
    _load(page, "<div id='root' style='margin:0'></div>")

    page.evaluate(
        """() => {
            document.getElementById("root").innerHTML = `
              <div style="display:flex;align-items:stretch;height:240px">
                <div id="pane-late" style="flex:0 0 auto;width:180px"></div>
                <div id="handle-late" data-splitter-target="pane-late"
                     data-splitter-store="store-late"
                     style="flex:0 0 auto;width:10px;background:#333"></div>
              </div>`;
        }"""
    )
    _drag(page, "handle-late", 60)

    assert _width(page, "pane-late") == pytest.approx(240, abs=2)
    calls = page.evaluate("() => window.__setProps")
    assert [store for store, _ in calls] == ["store-late"]


def test_an_unmounted_surface_leaves_no_polling_timer(page) -> None:
    """A surface that never mounts must not be polled for forever.

    Two attach loops in this file cleared only once their target existed.
    The QC worklist splitter and the QC gallery are both unmounted, so
    neither loop ever cleared: two 100 ms timers ran for the life of every
    session, next to body-wide observers already doing the same work.
    Attachment is observer-driven now, so both terminate by construction.

    Asserted on the production DOM shape rather than a bare page, because
    a bare page also starves the cards-container observer -- and that one
    does terminate in a real viewer.
    """
    _load(page, _PRODUCTION_SHAPE)
    page.wait_for_timeout(400)

    assert page.evaluate("() => window.__liveIntervals.size") == 0


def test_a_mounted_splitter_leaves_no_polling_timer(splitter_page) -> None:
    """Attaching must also stop the file polling, not just start working."""
    splitter_page.wait_for_timeout(400)
    assert splitter_page.evaluate("() => window.__liveIntervals.size") == 0
