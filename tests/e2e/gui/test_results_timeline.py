"""Playwright e2e: Results-viewer Timeline tab focus-and-navigate (spec §16.9).

Boots a loaded viewer over a seeded run with overlay PNGs for every
``(dataset, stem)`` and a mirror parquet carrying ``Metadata_ImageNumber``
(Int64 monotonic) + ``Metadata_PlateNum`` so X=ImageNumber, Y=PlateNum yields a
populated matrix. Mirrors ``test_heatmap_tab.py``'s fixture wiring exactly
(function-scoped ``fake_sandbox`` → ``_build_sandbox`` + a heatmap-style
``write_master``/``write_measurements_mirror`` seed + overlays, function-scoped
``live_server``, and a hand-off POST to ``/sandbox/api/viewer/output-root``).

Gated by ``PLAYWRIGHT=1`` (the conftest module-skip). Marked ``ci_flaky``: the
focus-navigate controller + Dash callback chain on a fresh Werkzeug server has a
tight DOM-poll budget that is stochastically slow on GHA shared runners.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import polars as pl
import pytest
from PIL import Image as PILImage
from playwright.sync_api import Page

from tests._output_layout import write_master, write_measurements_mirror
from tests.e2e.gui.conftest import (
    _build_sandbox,
    _start_live_server,
    bind_results_output,
    publish_coherent_terminal_evidence,
)
from phenotypic.schema import EXPERIMENT, IMAGE

# Tight DOM-poll budget on a fresh Werkzeug server: stochastically slow on GHA.
pytestmark = pytest.mark.ci_flaky

_OUTPUT_NAME = "CliOutputExample"
_DATASET = "ds1"
_N_PLATES = 6
_N_TIMES = 12


def _timeline_master_df() -> pl.DataFrame:
    """A 6-plate × 12-image-number matrix big enough that the focus window
    does NOT swallow it (so the bounded-window assertions are meaningful)."""
    rows: list[dict[str, object]] = []
    label = 0
    for plate in range(1, _N_PLATES + 1):
        for img_no in range(1, _N_TIMES + 1):
            label += 1
            rows.append(
                {
                    str(EXPERIMENT.DATASET): _DATASET,
                    str(IMAGE.IMAGE_NAME): f"p{plate}_t{img_no}",
                    "Metadata_ImageNumber": img_no,
                    "Metadata_PlateNum": str(plate),
                    "Object_Label": label,
                    "Size_Area": float(plate * 10 + img_no),
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("Metadata_ImageNumber").cast(pl.Int64)
    )


def _no_time_master_df() -> pl.DataFrame:
    """A frame WITHOUT any eligible time column (only a categorical group), so
    ``has_eligible_time_axis`` is False and the guided empty state renders."""
    rows: list[dict[str, object]] = []
    label = 0
    for plate in range(1, _N_PLATES + 1):
        for rep in range(1, _N_TIMES + 1):
            label += 1
            rows.append(
                {
                    str(EXPERIMENT.DATASET): _DATASET,
                    str(IMAGE.IMAGE_NAME): f"p{plate}_t{rep}",
                    # Categorical group only — String, no numeric/temporal dtype
                    # and no Metadata_Time-like name, so no eligible time axis.
                    "Metadata_PlateNum": str(plate),
                    "Metadata_Replicate": f"r{rep}",
                }
            )
    return pl.DataFrame(rows)


def _seed(sandbox: Path, df: pl.DataFrame) -> Path:
    cli_out = sandbox / "results" / _OUTPUT_NAME
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    (cli_out / "results" / _DATASET / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = cli_out / "deliverables" / "overlays" / _DATASET
    overlays.mkdir(parents=True, exist_ok=True)
    for plate in range(1, _N_PLATES + 1):
        for img_no in range(1, _N_TIMES + 1):
            PILImage.new("RGB", (160, 120), (20, 40, 60)).save(
                overlays / f"p{plate}_t{img_no}.png"
            )
    publish_coherent_terminal_evidence(
        cli_out,
        total_images=_N_PLATES * _N_TIMES,
    )
    return cli_out


@pytest.fixture
def df_factory(request: pytest.FixtureRequest) -> Callable[[], pl.DataFrame]:
    """Allow individual tests to swap the master frame builder (indirect)."""
    return getattr(request, "param", _timeline_master_df)


@pytest.fixture
def fake_sandbox(
    tmp_path: Path, df_factory: Callable[[], pl.DataFrame]
) -> Path:
    sandbox = _build_sandbox(tmp_path)
    _seed(sandbox, df_factory())
    return sandbox


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    yield from _start_live_server(fake_sandbox)


@pytest.fixture
def hub_url(live_server: str) -> str:
    return live_server


def _hand_off_viewer(page: Page, hub_url: str) -> None:
    """POST the seeded output to the viewer-handoff endpoint via the page.

    Polls an accepted asynchronous hand-off to success. If submission or
    publication fails, the shared helper reports the response and terminal
    job payload rather than silently proceeding to an empty viewer.
    """
    bind_results_output(page, hub_url, f"results/{_OUTPUT_NAME}")


def _pick_dropdown(page: Page, dropdown_id: str, label_text: str) -> None:
    """Open a Dash ``dcc.Dropdown`` (Radix button) and click an option."""
    locator = page.locator(f"#{dropdown_id}")
    locator.scroll_into_view_if_needed()
    locator.focus()
    page.keyboard.press("Enter")
    page.wait_for_selector(
        '[role="listbox"] [role="option"]', state="attached", timeout=5_000
    )
    page.locator(
        '[role="listbox"] [role="option"]', has_text=label_text
    ).first.click()


def _open_timeline(page: Page, hub_url: str) -> None:
    _hand_off_viewer(page, hub_url)
    page.goto(hub_url + "/results/")
    page.wait_for_selector("a.nav-link", timeout=15_000)
    page.locator("a.nav-link", has_text="Timeline").first.click()
    page.wait_for_selector(".timeline-cell[data-src]", timeout=15_000)
    # The alphabetical default Y is high-cardinality Metadata_ImageName (one
    # image per row → a sparse diagonal matrix). Pick the plate grouping so the
    # matrix is the dense 6-plate × 12-time-point grid the focus-navigate
    # assertions assume (X stays the only time option, Metadata_ImageNumber).
    _pick_dropdown(page, "timeline-y-dropdown", "Metadata_PlateNum")
    page.wait_for_function(
        "document.querySelectorAll('#timeline-grid .timeline-axis-label--y')"
        ".length === 6"
    )
    page.wait_for_selector(".timeline-cell--focused")


def test_y_and_x_dropdowns_populate(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    # Y offers the high-cardinality plate number (uncapped, spec §16.5);
    # X offers the numeric image number (selectable_time_columns).
    page.wait_for_selector("#timeline-y-dropdown")
    page.wait_for_selector("#timeline-x-dropdown")


def test_focus_starts_on_first_populated_cell(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    assert (
        page.eval_on_selector_all(".timeline-cell--focused", "e => e.length") == 1
    )


def test_arrow_right_moves_focus_and_mounts_neighborhood(
    page: Page, hub_url: str
) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".timeline-viewport")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-col-index') === '1'"
    )
    page.wait_for_function(
        "document.querySelectorAll('#timeline-grid .timeline-cell img').length > 0"
    )


def test_edge_button_down_moves_focus(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click("#timeline-nav-down")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-row-index') === '1'"
    )


def test_far_cell_unmounted_window_is_bounded(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    total = page.eval_on_selector_all(".timeline-cell[data-src]", "e => e.length")
    mounted = page.eval_on_selector_all(
        "#timeline-grid .timeline-cell img", "e => e.length"
    )
    assert 0 < mounted < total


def test_margin_ring_pre_mounted_offscreen(page: Page, hub_url: str) -> None:
    # A SMALL viewport over the dense 6×12 matrix makes the data-focus-margin
    # ring provably extend off-screen at the (0,0) corner focus: only a couple
    # of (populated) columns fit visibly, so the margin ring beyond them is
    # pre-mounted off-screen for instant step-in (mirrors the Browse margin-ring
    # test's deliberately-small viewport).
    _open_timeline(page, hub_url)
    # Configure the timeline at the normal viewport before shrinking it. At the
    # deliberately narrow test width, the sticky tab bar can cover the dropdown
    # menu and prevent this setup helper from selecting the plate grouping.
    page.set_viewport_size({"width": 600, "height": 450})
    page.wait_for_selector(".timeline-cell--focused")
    # Poll until the warm sweep has mounted a margin-ring <img> that sits
    # outside the viewport's visible rectangle (pre-mounted for instant step-in).
    page.wait_for_function(
        """() => {
            const vp = document.querySelector('.timeline-viewport').getBoundingClientRect();
            const imgs = document.querySelectorAll('#timeline-grid .timeline-cell img');
            for (const img of imgs) {
                const r = img.getBoundingClientRect();
                const vis = r.right > vp.left && r.left < vp.right
                    && r.bottom > vp.top && r.top < vp.bottom;
                if (!vis) return true;
            }
            return false;
        }""",
        timeout=15_000,
    )


def test_tab_reentry_reattaches_controller(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    # Leave the tab and come back — the <body> MutationObserver + attach
    # re-fire and re-establish focus (spec §15.7).
    page.locator("a.nav-link", has_text="Plate").first.click()
    page.wait_for_timeout(300)
    page.locator("a.nav-link", has_text="Timeline").first.click()
    page.wait_for_selector(".timeline-cell--focused")
    assert (
        page.eval_on_selector_all(".timeline-cell--focused", "e => e.length") == 1
    )
    # Navigation still works after re-attach.
    page.click(".timeline-viewport")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-col-index') === '1'"
    )


def test_enter_opens_popout(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".timeline-viewport")
    page.keyboard.press("Enter")
    page.wait_for_selector("#timeline-popout-modal.show, .modal.show", timeout=10_000)


def test_repeat_enter_reopens_popout(page: Page, hub_url: str) -> None:
    # OQ-6: opening the SAME focused cell's pop-out twice must re-fire (Dash
    # dedupes an identical bridge value, so Phase 2's bridge-write convention —
    # the `#<nonce>` suffix — must carry through to Results).
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".timeline-viewport")
    page.keyboard.press("Enter")
    page.wait_for_selector(".modal.show", timeout=10_000)
    # Dismiss, then Enter again on the SAME focused cell — must reopen.
    page.keyboard.press("Escape")
    page.wait_for_selector(".modal.show", state="detached", timeout=10_000)
    page.click(".timeline-viewport")
    page.keyboard.press("Enter")
    page.wait_for_selector(".modal.show", timeout=10_000)


def test_hover_reveals_popout_button(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    cell = page.locator(".timeline-cell[data-src]").first
    cell.hover()
    page.wait_for_selector(
        ".timeline-cell:hover .timeline-cell-popout", timeout=5_000
    )


@pytest.mark.parametrize(
    "df_factory",
    [pytest.param(_no_time_master_df, id="no-time-column")],
    indirect=True,
)
def test_empty_state_when_no_time_column(page: Page, hub_url: str) -> None:
    # The mirror has no eligible time column, so has_eligible_time_axis is False
    # and the guided empty state renders (D9).
    _hand_off_viewer(page, hub_url)
    page.goto(hub_url + "/results/")
    page.wait_for_selector("a.nav-link", timeout=15_000)
    page.locator("a.nav-link", has_text="Timeline").first.click()
    page.wait_for_function(
        "() => {"
        "  const el = document.getElementById('timeline-empty-state');"
        "  return el && getComputedStyle(el).display !== 'none';"
        "}",
        timeout=15_000,
    )
