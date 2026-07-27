"""Browser-driven E2E for the filter-sidebar Offcanvas toggle + filtering.

Pins the toggle open/close sequence and the badge/match-count assertions that
Python unit tests cannot actuate: a real browser is required to drive the
Dash callback chain that opens/closes ``dbc.Offcanvas`` and updates the
``#filter-toggle-badge`` and ``#filter-match-count`` elements.

``ci_flaky``-gated: single-threaded Werkzeug dev server + Dash callback-chain
timing stochastically exceeds wait budgets on GHA shared runners; the SUT is
correct — see ``tests/CLAUDE.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from PIL import Image as PILImage
from playwright.sync_api import Page

from playwright.sync_api import expect

from phenotypic.schema import CULTURE_METADATA, EXPERIMENT_METADATA, METADATA
from tests.e2e.gui.conftest import (
    _build_sandbox,
    _start_live_server,
    bind_results_output,
    publish_coherent_terminal_evidence,
)

# Module-level marker: skipped on CI via ``-m "not ci_flaky"`` in the
# gui-e2e workflow. Locally these tests pass reliably; on GHA ubuntu-latest
# shared runners the Dash callback chain (open-offcanvas → add-row →
# column-dropdown select → values-dropdown select) stochastically exceeds
# the wait budgets.
pytestmark = pytest.mark.ci_flaky

_OUTPUT_NAME = "FilterOffcanvasOutput"
# Two datasets with distinct values in the canonical dataset column.
_DS1_IMAGES = ("plate_001.tif", "plate_002.tif")
_DS2_IMAGES = ("plate_003.tif",)
_NROWS, _NCOLS = 2, 3
_DATASET_COLUMN = str(EXPERIMENT_METADATA.DATASET)
_TIME_COLUMN = str(CULTURE_METADATA.TIME)


# ---------------------------------------------------------------------------
# Sandbox seeding
# ---------------------------------------------------------------------------


def _build_master() -> pl.DataFrame:
    """Build a master frame with two distinct canonical dataset values.

    ``ds1`` has two image files; ``ds2`` has one. This lets the filter
    test narrow from 3 to 2 image files when ds1 is selected.
    """
    rows: list[dict[str, object]] = []
    label = 0
    for dataset, images in [("ds1", _DS1_IMAGES), ("ds2", _DS2_IMAGES)]:
        for image in images:
            for r in range(1, _NROWS + 1):
                for c in range(1, _NCOLS + 1):
                    label += 1
                    rows.append(
                        {
                            _DATASET_COLUMN: dataset,
                            str(METADATA.IMAGE_NAME): image,
                            _TIME_COLUMN: 0.0,
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


def _seed_viewer_output(sandbox: Path) -> Path:
    """Seed a CLI output dir with master + measurements + overlay PNGs.

    Both the clean master archive and the post-applied mirror now live under
    ``<out>/deliverables/`` (the deliverables/ cutover); ``OutputRoot.discover``
    requires the master there. Resolve the paths via the ``phenotypic.sdk_``
    helpers rather than hand-joining names.
    """
    from phenotypic.sdk_ import (
        master_measurements_parquet_path,
        measurements_parquet_path,
    )

    out = sandbox / "results" / _OUTPUT_NAME
    out.mkdir(parents=True, exist_ok=True)

    master = _build_master()
    master_path = master_measurements_parquet_path(out)
    mirror_path = measurements_parquet_path(out)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master.write_parquet(master_path)
    master.write_parquet(mirror_path)

    # Overlay PNGs — seed ds1 overlays so the viewer can resolve them.
    (out / "results" / "ds1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = out / "deliverables" / "overlays" / "ds1"
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _DS1_IMAGES:
        stem = Path(image).stem
        PILImage.new("RGB", (120, 120), (200, 0, 0)).save(overlays / f"{stem}.png")

    publish_coherent_terminal_evidence(out, total_images=len(_DS1_IMAGES))
    return out


# ---------------------------------------------------------------------------
# Fixtures (function-scoped — each test needs a fresh server with its sandbox)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    """Function-scoped sandbox seeded with a filterable viewer output."""
    sandbox = _build_sandbox(tmp_path)
    _seed_viewer_output(sandbox)
    return sandbox


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Function-scoped live server over the filter-seeded sandbox."""
    yield from _start_live_server(fake_sandbox)


@pytest.fixture
def hub_url(live_server: str) -> str:
    """Alias for the live-server URL."""
    return live_server


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


def _open_viewer(page: Page, hub_url: str) -> None:
    """Hand off the output root to the viewer and navigate to /results/."""
    output_rel = f"results/{_OUTPUT_NAME}"
    bind_results_output(page, hub_url, output_rel)
    page.wait_for_selector("#btn-filters-toggle", state="attached", timeout=15_000)


def _offcanvas_is_visible(page: Page) -> bool:
    """Return True if the filter offcanvas is currently visible in the DOM."""
    return page.evaluate(
        """() => {
            const el = document.querySelector('#filter-offcanvas');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none'
                && style.visibility !== 'hidden'
                && rect.width > 0;
        }"""
    )


def _open_filter_row_dropdown(page: Page, row_index: int, dropdown_index: int) -> None:
    """Open the Nth dcc.Dropdown inside a filter row via keyboard (Enter).

    Dash 4 ``dcc.Dropdown`` renders via Radix UI; the most reliable opener
    is keyboard ``Enter`` after focusing the trigger element. The column
    dropdown is index 0 inside the filter row; the values dropdown is index 1.

    Args:
        page: Playwright page.
        row_index: 0-based index of the filter row (``nth(.filter-row)``).
        dropdown_index: 0-based index of the ``dcc.Dropdown`` inside the row.
    """
    row = page.locator(".filter-row").nth(row_index)
    # dcc.Dropdown renders a div with role="combobox" (the trigger button).
    # Each dropdown in the row is a separate .dash-dropdown container.
    dropdown_trigger = row.locator(".dash-dropdown").nth(dropdown_index)
    dropdown_trigger.scroll_into_view_if_needed()
    dropdown_trigger.focus()
    page.keyboard.press("Enter")
    # Wait for the listbox to appear.
    page.wait_for_selector(
        '[role="listbox"] [role="option"]', state="attached", timeout=6_000
    )


def _pick_listbox_option(page: Page, label_text: str) -> None:
    """Click the first listbox option whose text matches ``label_text``."""
    page.locator('[role="listbox"] [role="option"]', has_text=label_text).first.click()


def _open_offcanvas(page: Page) -> None:
    """Click the sticky tab-row Filters button and wait for the panel to open.

    Mirrors Step 3 of :func:`test_filter_offcanvas_toggle_and_filtering`: the
    Dash callback flips ``is_open`` which re-renders the ``dbc.Offcanvas`` with
    ``rect.width > 0``.
    """
    page.locator("#btn-filters-toggle").click()
    page.wait_for_function(
        "() => {"
        "  const el = document.querySelector('#filter-offcanvas');"
        "  if (!el) return false;"
        "  const rect = el.getBoundingClientRect();"
        "  return rect.width > 0;"
        "}",
        timeout=12_000,
    )
    assert _offcanvas_is_visible(page), (
        "filter offcanvas should be visible after clicking #btn-filters-toggle"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_filter_offcanvas_toggle_and_filtering(page: Page, hub_url: str) -> None:
    """Filter offcanvas boots closed; toggle opens it; filter narrows results.

    Steps exercised:

    1. Open the viewer via the hand-off pattern.
    2. Assert the offcanvas boots CLOSED (zero width / not visible).
    3. Click ``#btn-filters-toggle`` → assert the offcanvas becomes visible.
    4. Click ``#btn-add-filter-row`` → choose the canonical dataset column in the
       column dropdown → assert ``#filter-toggle-badge`` shows "1".
    5. Choose "ds1" in the values dropdown → assert ``#filter-match-count``
       contains "images match".
    6. Close the offcanvas via the backdrop → assert it is hidden again.
    """
    _open_viewer(page, hub_url)

    # Step 2 — offcanvas boots closed.
    assert not _offcanvas_is_visible(page), (
        "filter offcanvas should boot closed but appears visible"
    )

    # Step 3 — open via toggle button.
    page.locator("#btn-filters-toggle").click()
    # Wait for the offcanvas to open — Bootstrap adds the "show" class and
    # transitions the panel into view. The Dash callback flips ``is_open``
    # which re-renders the dbc.Offcanvas component with rect.width > 0.
    page.wait_for_function(
        "() => {"
        "  const el = document.querySelector('#filter-offcanvas');"
        "  if (!el) return false;"
        "  const rect = el.getBoundingClientRect();"
        "  return rect.width > 0;"
        "}",
        timeout=12_000,
    )
    assert _offcanvas_is_visible(page), (
        "filter offcanvas should be visible after clicking #btn-filters-toggle"
    )

    # Step 4 — add a filter row and pick the canonical dataset column.
    page.locator("#btn-add-filter-row").click()

    # Wait for a filter row to appear in the container.
    page.wait_for_selector(".filter-row", state="attached", timeout=10_000)
    page.wait_for_timeout(400)  # let the row fully render its child dropdowns

    # Open the column dropdown (first dcc.Dropdown in the row, index 0)
    # via keyboard — Dash 4 uses Radix UI triggers, keyboard Enter is
    # the most reliable cross-platform opener.
    _open_filter_row_dropdown(page, row_index=0, dropdown_index=0)
    page.wait_for_timeout(200)

    # Click the canonical dataset option in the open listbox.
    _pick_listbox_option(page, _DATASET_COLUMN)
    page.wait_for_timeout(800)  # let Dash callback write the spec store

    # Under the method-aware redesign a row is only "active" once it has a
    # usable payload (here: at least one selected value). Picking only the
    # column is a no-op, so the badge must STAY hidden/empty at this point.
    badge = page.locator("#filter-toggle-badge")
    assert badge.inner_text().strip() == "", (
        "#filter-toggle-badge should stay empty after selecting only a column "
        f"(no values yet); got {badge.inner_text()!r}"
    )

    # Step 5 — choose "ds1" in the values dropdown (second dcc.Dropdown for a
    # text column: column=0, method=1, values=2). Wait for its options to
    # populate after the column selection.
    page.wait_for_timeout(400)  # let values dropdown repopulate

    _open_filter_row_dropdown(page, row_index=0, dropdown_index=2)
    page.wait_for_timeout(200)

    _pick_listbox_option(page, "ds1")
    page.wait_for_timeout(800)  # let filter callback resolve image pairs

    # Now the row is active — the badge shows "1".
    badge.wait_for(state="visible", timeout=8_000)
    badge_text = badge.inner_text()
    assert badge_text.strip() == "1", (
        f"#filter-toggle-badge expected '1' after selecting a value; got {badge_text!r}"
    )

    # Assert the match-count chip reflects the narrowed result.
    match_count = page.locator("#filter-match-count")
    match_count_text = match_count.inner_text()
    assert "images match" in match_count_text, (
        f"#filter-match-count expected text containing 'images match'; "
        f"got {match_count_text!r}"
    )

    # Step 6 — close via the offcanvas backdrop.
    # The backdrop is injected by Bootstrap/dbc when the offcanvas is open.
    backdrop = page.locator(".offcanvas-backdrop")
    if backdrop.count() > 0:
        backdrop.first.click()
    else:
        # Fallback: click the toggle button again to close.
        page.locator("#btn-filters-toggle").click()

    # Wait for the offcanvas to close.
    page.wait_for_function(
        """() => {
            const el = document.querySelector('#filter-offcanvas');
            if (!el) return true;
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            return style.display === 'none'
                || style.visibility === 'hidden'
                || rect.width === 0;
        }""",
        timeout=10_000,
    )
    assert not _offcanvas_is_visible(page), (
        "filter offcanvas should be hidden after closing via backdrop"
    )


def test_range_method_filters_picker(page: Page, hub_url: str) -> None:
    """Switching a numeric row to Range (between) actually filters the picker.

    Drives the new per-method controls LIVE so a callback-wiring 500 on
    ``/_dash-update-component`` (which unit tests never exercise) surfaces:

    1. Open the offcanvas from the sticky tab-row Filters button.
    2. Add a row and pick the numeric ``Size_Area`` column.
    3. Switch the Method dropdown to "Range (between)" → assert the
       range-min/range-max inputs render.
    4. Type a min bound that excludes some rows → assert the
       ``#filter-match-count`` chip text changes (the range filter ran).
    5. Assert NO console error and NO failed ``/_dash-update-component``
       request fired during the interaction.
    """
    console_errors: list[str] = []
    page.on(
        "console",
        lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
    )
    failed_updates: list[str] = []

    def _record_failed(response) -> None:
        url = response.url
        if "_dash-update-component" in url and response.status >= 400:
            failed_updates.append(f"{response.status} {url}")

    page.on("response", _record_failed)

    _open_viewer(page, hub_url)
    _open_offcanvas(page)

    # Step 2 — add a row and pick the numeric Size_Area column.
    page.locator("#btn-add-filter-row").click()
    page.wait_for_selector(".filter-row", state="attached", timeout=10_000)
    page.wait_for_timeout(400)

    _open_filter_row_dropdown(page, row_index=0, dropdown_index=0)
    page.wait_for_timeout(200)
    _pick_listbox_option(page, "Size_Area")
    page.wait_for_timeout(800)

    # Step 3 — switch the Method dropdown to "Range (between)". The method
    # dropdown is index 1 inside the row (column dropdown is index 0); for a
    # numeric column the list-mode values dropdown is replaced by the method
    # controls, so index 1 is the method selector.
    _open_filter_row_dropdown(page, row_index=0, dropdown_index=1)
    page.wait_for_timeout(200)
    _pick_listbox_option(page, "Range (between)")
    page.wait_for_timeout(800)

    # The range-min / range-max numeric inputs should now exist.
    range_min = page.locator('input[id*="filter-row-range-min"]')
    range_max = page.locator('input[id*="filter-row-range-max"]')
    range_min.wait_for(state="attached", timeout=8_000)
    range_max.wait_for(state="attached", timeout=8_000)

    # Capture the pre-filter chip text, then apply a min bound and assert the
    # chip text changes. ``Size_Area`` is ``100 + r*10 + c`` (111..123) and is
    # identical across every seeded image file, so a *narrowing* range can't
    # change the per-image count; a min above the max (200) drops every object
    # and unambiguously proves the range predicate ran ("0 images match").
    match_count = page.locator("#filter-match-count")
    before_text = match_count.inner_text()
    assert "images match" in before_text, (
        f"#filter-match-count expected baseline 'images match'; got {before_text!r}"
    )

    range_min.fill("200")
    # Commit the value the way a user tabbing out would, so the dcc.Input
    # debounce / blur fires the Dash callback.
    range_min.press("Tab")
    page.wait_for_timeout(900)  # let the range-sync + filter callbacks resolve

    after_text = match_count.inner_text()
    assert "images match" in after_text, (
        f"#filter-match-count expected text containing 'images match'; "
        f"got {after_text!r}"
    )
    assert after_text != before_text, (
        f"range filter should have changed the match-count chip; "
        f"before={before_text!r} after={after_text!r}"
    )

    # Step 5 — no live callback errors. A range-control wiring bug would show
    # up here as a 500 on /_dash-update-component or a JS console error.
    assert not failed_updates, (
        f"/_dash-update-component returned an error during the range filter "
        f"interaction: {failed_updates}"
    )
    assert not console_errors, (
        f"unexpected browser console errors during the range filter "
        f"interaction: {console_errors}"
    )


def test_filters_button_sticky_after_scroll(page: Page, hub_url: str) -> None:
    """The tab-row Filters button stays visible after scrolling tab content.

    The button lives in ``.results-viewer-tabbar__actions`` (``position:
    sticky``). Scrolling the page must not scroll it out of the viewport.
    The zero-height sticky actions strip must also leave the Bootstrap-small
    button at its natural control height instead of flex-stretching it down.
    """
    _open_viewer(page, hub_url)

    filters_button = page.locator("#btn-filters-toggle")
    expect(filters_button).to_be_visible()
    metrics = filters_button.evaluate(
        """el => {
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            return {
                display: style.display,
                height: rect.height,
                lineHeight: parseFloat(style.lineHeight),
                scrollHeight: el.scrollHeight,
            };
        }"""
    )
    assert metrics["display"] in {"flex", "inline-flex"}
    assert 30 <= metrics["height"] <= 36, (
        "Filters button should keep Bootstrap-small control height; "
        f"got {metrics!r}"
    )
    assert metrics["height"] >= metrics["lineHeight"], (
        "Filters button should not clip its text line-height; "
        f"got {metrics!r}"
    )
    assert metrics["height"] >= metrics["scrollHeight"], (
        "Filters button should not clip its rendered content; "
        f"got {metrics!r}"
    )

    # Scroll the tab content well past a viewport height.
    page.mouse.wheel(0, 2000)
    page.wait_for_timeout(400)

    expect(filters_button).to_be_visible()
