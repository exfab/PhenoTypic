"""Browser-driven E2E for the QC Review worklist drag-splitter.

Pins the clientside drag the Python unit tests cannot actuate: the unit
tests in ``tests/unit/gui/results_viewer/test_qc_review_layout.py`` cover
the clamp + the width-apply callback (the Python half), but only a real
browser can drive the ``results_viewer.js`` mousedown→move→up handler.
This test drags the ``#qc-review-splitter`` handle and asserts the
worklist width changes (clamped) and **persists across a collapse/expand
cycle** via ``STORE_QC_SIDEBAR_WIDTH``.

``ci_flaky``-gated like the other browser E2E (single-threaded Werkzeug
dev server + Dash callback-chain timing flakes on shared CI runners; the
SUT is correct — see ``tests/CLAUDE.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from PIL import Image as PILImage
from playwright.sync_api import Page

from phenotypic import ImagePipeline
from phenotypic.analysis import ReplicateAgreement
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc

from tests._output_layout import write_master, write_measurements_mirror, write_pipeline_json
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server

# Single-threaded dev server + Dash callback chain stochastically exceeds
# the wait budgets on GHA shared runners; correct locally. See test_qc_tab.py.
pytestmark = pytest.mark.ci_flaky

_OUTPUT_NAME = "CliOutputExample"
_INSTANCE_ID = "qc-SE-splitter01"
# Two images × a 3-row × 4-col grid → groups with several members so the
# Review worklist + gallery render real content.
_IMAGES = ("plate_001.tif", "plate_002.tif")
_NROWS, _NCOLS = 3, 4


def _build_master() -> pl.DataFrame:
    """Build a master frame the viewer + ReplicateAgreement can load."""
    rows: list[dict[str, object]] = []
    label = 0
    for image in _IMAGES:
        for r in range(1, _NROWS + 1):
            for c in range(1, _NCOLS + 1):
                label += 1
                rows.append(
                    {
                        "Metadata_Dataset": "ds1",
                        "Metadata_ImageFile": image,
                        "Metadata_Time": 0.0,
                        "Object_Label": label,
                        "Grid_RowNum": r,
                        "Grid_ColNum": c,
                        "Bbox_CenterRR": 50,
                        "Bbox_CenterCC": 50,
                        "Bbox_MinRR": 40,
                        "Bbox_MaxRR": 60,
                        "Bbox_MinCC": 40,
                        "Bbox_MaxCC": 60,
                        "Size_Area": float(100 + r * 10 + c),
                    }
                )
    return pl.DataFrame(rows)


def _pipeline() -> ImagePipeline:
    """A pipeline carrying one ReplicateAgreement QC entry grouped by row."""
    pipeline = ImagePipeline(name="qc-splitter-e2e")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": ["Grid_RowNum"],
                    "min_replicates": 2,
                },
                instance_id=_INSTANCE_ID,
                enabled=True,
            )
        ]
    )
    return pipeline


def _seed_review_output(sandbox: Path) -> Path:
    """Seed a CLI output dir with the master + overlays + qc/ artifact."""
    out = sandbox / "results" / _OUTPUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    master = _build_master()
    write_master(out, master)
    write_measurements_mirror(out, master)

    overlays = out / "results" / "ds1" / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _IMAGES:
        stem = Path(image).stem
        PILImage.new("RGB", (120, 120), (200, 0, 0)).save(overlays / f"{stem}.png")

    pipeline = _pipeline()
    write_pipeline_json(out, pipeline)
    # Generate the qc/ artifact the Review tab reads.
    run_qc(master.to_pandas(), pipeline, out)
    return out


@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    """Function-scoped sandbox seeded with a QC review artifact."""
    sandbox = _build_sandbox(tmp_path)
    _seed_review_output(sandbox)
    return sandbox


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Function-scoped live server over the review-seeded sandbox."""
    yield from _start_live_server(fake_sandbox)


@pytest.fixture
def hub_url(live_server: str) -> str:
    """Alias for the live-server URL."""
    return live_server


def _open_review(page: Page, hub_url: str) -> None:
    """Hand off the output root, open the QC tab, and switch to Review."""
    output_rel = f"results/{_OUTPUT_NAME}"
    page.goto(hub_url + "/")
    page.wait_for_load_state("networkidle")
    resp = page.evaluate(
        """async (path) => {
            const r = await fetch('/sandbox/api/viewer/output-root', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: path}),
            });
            return r.status;
        }""",
        output_rel,
    )
    assert resp == 200, f"viewer hand-off failed: {resp}"

    page.goto(hub_url + "/results/")
    page.wait_for_selector("#qc-cards-container", state="attached", timeout=15_000)
    page.locator("a.nav-link", has_text="QC").first.click()
    # Flip the Configure | Review toggle.
    page.wait_for_selector(
        'label[for$="qc-subview-toggle_input_review"]', timeout=10_000
    )
    page.locator('label[for$="qc-subview-toggle_input_review"]').click()
    # Wait for the worklist to populate (Review is now active).
    page.wait_for_selector(".qc-worklist-row", timeout=15_000)
    page.wait_for_selector("#qc-review-splitter", state="attached", timeout=10_000)


def _worklist_width(page: Page) -> float:
    """Current rendered width (px) of the worklist."""
    return page.evaluate(
        "() => document.querySelector('#qc-review-worklist')"
        ".getBoundingClientRect().width"
    )


def _selected_worklist_text(page: Page) -> str:
    """Return the currently highlighted Review worklist row text."""
    return page.evaluate(
        """() => {
            const rows = Array.from(document.querySelectorAll('.qc-worklist-row'));
            const selected = rows.find((row) =>
                (row.getAttribute('style') || '').includes('rgba(0, 54, 96, 0.06)')
                || (row.getAttribute('style') || '').includes('rgba(0,54,96,0.06)')
            );
            return selected ? (selected.textContent || '').trim() : '';
        }"""
    )


def test_group_icon_navigation_previous_and_next(page: Page, hub_url: str) -> None:
    """Icon-only Review group buttons move selection backward and forward."""
    _open_review(page, hub_url)
    first = _selected_worklist_text(page)
    assert first, "initial Review group was not selected"

    page.locator("#qc-review-next-btn").click()
    page.wait_for_function(
        "(before) => {"
        "  const rows = Array.from(document.querySelectorAll('.qc-worklist-row'));"
        "  const selected = rows.find((row) => "
        "    (row.getAttribute('style') || '').includes('rgba(0, 54, 96, 0.06)')"
        "    || (row.getAttribute('style') || '').includes('rgba(0,54,96,0.06)'));"
        "  return selected && (selected.textContent || '').trim() !== before;"
        "}",
        arg=first,
        timeout=10_000,
    )
    second = _selected_worklist_text(page)
    assert second and second != first

    page.locator("#qc-review-prev-btn").click()
    page.wait_for_function(
        "(before) => {"
        "  const rows = Array.from(document.querySelectorAll('.qc-worklist-row'));"
        "  const selected = rows.find((row) => "
        "    (row.getAttribute('style') || '').includes('rgba(0, 54, 96, 0.06)')"
        "    || (row.getAttribute('style') || '').includes('rgba(0,54,96,0.06)'));"
        "  return selected && (selected.textContent || '').trim() !== before;"
        "}",
        arg=second,
        timeout=10_000,
    )
    assert _selected_worklist_text(page) != second


def test_splitter_drag_resizes_and_persists_across_collapse(
    page: Page, hub_url: str
) -> None:
    """Dragging the splitter widens the worklist; the width survives collapse.

    Drives the real ``results_viewer.js`` drag (mousedown on the handle →
    mousemove → mouseup), then collapses + expands via the chevron and
    asserts the dragged width is restored (proving STORE_QC_SIDEBAR_WIDTH
    persisted it — not reset to the 180px default).
    """
    _open_review(page, hub_url)

    width_before = _worklist_width(page)
    assert 175 <= width_before <= 185, f"default width not ~180px: {width_before}"

    # Drag the splitter handle right by +120px via real pointer events.
    handle = page.locator("#qc-review-splitter")
    box = handle.bounding_box()
    assert box is not None
    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + 60
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    # Several incremental moves so the JS mousemove handler tracks the drag.
    for dx in range(20, 121, 20):
        page.mouse.move(start_x + dx, start_y)
    page.mouse.up()
    page.wait_for_timeout(600)  # let set_props + the width callback resolve

    width_after = _worklist_width(page)
    assert width_after > width_before + 40, (
        f"drag did not widen the worklist: {width_before} -> {width_after}"
    )
    # Still within the clamp ceiling.
    assert width_after <= 381, f"width exceeded max clamp: {width_after}"
    dragged = width_after

    # Collapse via the chevron, then expand — the dragged width must return.
    toggle = page.locator("#qc-review-sidebar-toggle")
    toggle.click()
    page.wait_for_function(
        "() => getComputedStyle("
        "document.querySelector('#qc-review-worklist')).display === 'none'",
        timeout=10_000,
    )
    toggle.click()
    page.wait_for_function(
        "() => getComputedStyle("
        "document.querySelector('#qc-review-worklist')).display !== 'none'",
        timeout=10_000,
    )
    page.wait_for_timeout(400)

    width_restored = _worklist_width(page)
    assert abs(width_restored - dragged) <= 2, (
        f"dragged width not preserved across collapse: "
        f"{dragged} -> {width_restored}"
    )


def test_qc_tile_click_selects_via_shared_store(page: Page, hub_url: str) -> None:
    """A QC-gallery tile checkbox click selects the tile (M1: selection parity).

    Drives the full in-browser chain the unit/integration tests cannot:
    the JS shift-click bridge (now attached to ``#qc-review-gallery``) emits
    a delta → the QC consumer folds it into the SHARED
    ``store-colony-selection`` → the clientside styler (which reads that
    store and sweeps ``#qc-review-gallery``) toggles ``.is-selected`` on the
    clicked tile. Observing the ``.is-selected`` class is a stronger
    end-to-end proof than reading the store directly: it only lights up if
    every link in the chain is wired.

    This replaces the hand-injected-State integration test as the FEATURES
    "QC selection parity" reference.
    """
    _open_review(page, hub_url)

    # The first worklist group is auto-selected on Review open, so the
    # detail gallery already carries tiles. Wait for at least one tile
    # checkbox to mount.
    gallery = page.locator("#qc-review-gallery")
    checkbox = gallery.locator(".colony-cell-checkbox").first
    checkbox.wait_for(state="attached", timeout=15_000)

    # No tile is selected before the click.
    selected_before = page.evaluate(
        "() => document.querySelectorAll("
        "'#qc-review-gallery .colony-cell.is-selected').length"
    )
    assert selected_before == 0, (
        f"expected no selected QC tiles before the click, got {selected_before}"
    )

    # Click the checkbox: bridge → QC delta store → consumer → shared
    # selection store → styler → .is-selected.
    checkbox.click()

    # The styler must light exactly the clicked tile.
    page.wait_for_function(
        "() => document.querySelectorAll("
        "'#qc-review-gallery .colony-cell.is-selected').length >= 1",
        timeout=10_000,
    )
    selected_after = page.evaluate(
        "() => document.querySelectorAll("
        "'#qc-review-gallery .colony-cell.is-selected').length"
    )
    assert selected_after == 1, (
        f"QC tile click should select exactly one tile, got {selected_after}"
    )
