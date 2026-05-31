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

from tests.e2e.gui.conftest import _build_sandbox, _start_live_server

# Module-level marker: skipped on CI via ``-m "not ci_flaky"`` in the
# gui-e2e workflow. Locally these tests pass reliably; on GHA ubuntu-latest
# shared runners the Dash callback chain (open-offcanvas → add-row →
# column-dropdown select → values-dropdown select) stochastically exceeds
# the wait budgets.
pytestmark = pytest.mark.ci_flaky

_OUTPUT_NAME = "FilterOffcanvasOutput"
# Two datasets with distinct values in Metadata_Dataset to drive filtering.
_DS1_IMAGES = ("plate_001.tif", "plate_002.tif")
_DS2_IMAGES = ("plate_003.tif",)
_NROWS, _NCOLS = 2, 3


# ---------------------------------------------------------------------------
# Sandbox seeding
# ---------------------------------------------------------------------------


def _build_master() -> pl.DataFrame:
    """Build a master frame with two distinct Metadata_Dataset values.

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
                            "Metadata_Dataset": dataset,
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


def _seed_viewer_output(sandbox: Path) -> Path:
    """Seed a CLI output dir with master + measurements + overlay PNGs."""
    out = sandbox / "results" / _OUTPUT_NAME
    out.mkdir(parents=True, exist_ok=True)

    master = _build_master()
    master.write_parquet(out / "master_measurements.parquet")
    master.write_parquet(out / "measurements.parquet")

    # Overlay PNGs — seed ds1 overlays so the viewer can resolve them.
    overlays = out / "results" / "ds1" / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _DS1_IMAGES:
        stem = Path(image).stem
        PILImage.new("RGB", (120, 120), (200, 0, 0)).save(overlays / f"{stem}.png")

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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_filter_offcanvas_toggle_and_filtering(page: Page, hub_url: str) -> None:
    """Filter offcanvas boots closed; toggle opens it; filter narrows results.

    Steps exercised:

    1. Open the viewer via the hand-off pattern.
    2. Assert the offcanvas boots CLOSED (zero width / not visible).
    3. Click ``#btn-filters-toggle`` → assert the offcanvas becomes visible.
    4. Click ``#btn-add-filter-row`` → choose ``Metadata_Dataset`` in the
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

    # Step 4 — add a filter row and pick the Metadata_Dataset column.
    page.locator("#btn-add-filter-row").click()

    # Wait for a filter row to appear in the container.
    page.wait_for_selector(".filter-row", state="attached", timeout=10_000)
    page.wait_for_timeout(400)  # let the row fully render its child dropdowns

    # Open the column dropdown (first dcc.Dropdown in the row, index 0)
    # via keyboard — Dash 4 uses Radix UI triggers, keyboard Enter is
    # the most reliable cross-platform opener.
    _open_filter_row_dropdown(page, row_index=0, dropdown_index=0)
    page.wait_for_timeout(200)

    # Click the "Metadata_Dataset" option in the open listbox.
    _pick_listbox_option(page, "Metadata_Dataset")
    page.wait_for_timeout(800)  # let Dash callback write the spec store

    # Assert the badge now shows "1" (one configured filter row).
    badge = page.locator("#filter-toggle-badge")
    badge.wait_for(state="visible", timeout=8_000)
    badge_text = badge.inner_text()
    assert badge_text.strip() == "1", (
        f"#filter-toggle-badge expected '1' after selecting a column; got {badge_text!r}"
    )

    # Step 5 — choose "ds1" in the values dropdown (second dcc.Dropdown,
    # index 1). Wait for its options to populate after the column selection.
    page.wait_for_timeout(400)  # let values dropdown repopulate

    _open_filter_row_dropdown(page, row_index=0, dropdown_index=1)
    page.wait_for_timeout(200)

    _pick_listbox_option(page, "ds1")
    page.wait_for_timeout(800)  # let filter callback resolve image pairs

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
