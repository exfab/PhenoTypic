"""Browser-driven E2E smoke test for the error-category radial triage.

One real-browser confirmation (the memory rule: verify Dash callbacks
live, not just unit/integration). Boots the viewer against a seeded
output root, opens a colony tile's radial menu, clicks the ``Debris``
wedge, and asserts the tile flips to a debris category badge and the
durable ``deliverables/errors/debris.parquet`` appears on disk.

Mirrors ``test_qc_tab.py``'s function-scoped sandbox + viewer-handoff
pattern, but seeds **real overlay PNGs** + bbox columns so the colony
grid emits ``<img>`` tiles (each carrying a ``▾`` radial trigger).

Gated by ``PLAYWRIGHT=1`` via the module-level skip in ``conftest.py``.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from PIL import Image as PILImage
from playwright.sync_api import Page, expect

from phenotypic.tools_ import error_category_parquet_path
from tests._output_layout import write_master, write_measurements_mirror
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server

# Module-level marker: skipped on CI via ``-m "not ci_flaky"`` in the
# gui-e2e workflow (see tests/CLAUDE.md). The single test here drives a
# Dash pattern-matched callback chain (radial lazy-populate → wedge mark →
# grid re-render → on-disk parquet write) whose end-to-end budget polls a
# disk-write deadline; on GHA ubuntu-latest shared runners that budget
# stochastically exceeds the Playwright ``wait_for_function`` window.
pytestmark = pytest.mark.ci_flaky


_OUTPUT_NAME = "CliOutputExample"
_DATASET = "ds1"
_IMAGES = ("plate_001.tif", "plate_002.tif")
_NUM_ROWS = 2
_NUM_COLS = 2


def _build_master_df() -> pl.DataFrame:
    """A small two-image, 2x2 grid frame with bbox + centroid columns.

    Bbox columns are required so the colony grid sizes its crops; the
    centroid columns let ``CurationLabels`` fingerprint each object.
    """
    rows: list[dict[str, object]] = []
    label = 0
    for image in _IMAGES:
        for r in range(1, _NUM_ROWS + 1):
            for c in range(1, _NUM_COLS + 1):
                label += 1
                rows.append(
                    {
                        "Metadata_Dataset": _DATASET,
                        "Metadata_ImageFile": image,
                        "Object_Label": label,
                        "Grid_RowNum": r,
                        "Grid_ColNum": c,
                        "Bbox_MinRR": 0,
                        "Bbox_MaxRR": 40,
                        "Bbox_MinCC": 0,
                        "Bbox_MaxCC": 40,
                        "Bbox_CenterRR": 20.0,
                        "Bbox_CenterCC": 20.0,
                        "Size_Area": float(100 + r * 10 + c),
                    }
                )
    return pl.DataFrame(rows)


def _seed_real_output(sandbox: Path) -> Path:
    """Replace the placeholder parquet with a real frame + overlay PNGs.

    The colony grid only emits an ``<img>`` (and thus the per-tile radial
    trigger) when ``OutputRoot.has_overlay`` is True, which is backed by an
    on-disk scan — so every represented image needs a real overlay PNG.
    """
    cli_out = sandbox / "results" / _OUTPUT_NAME
    df = _build_master_df()
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)

    overlays = cli_out / "results" / _DATASET / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _IMAGES:
        stem = Path(image).stem
        PILImage.new("RGB", (64, 64), (180, 120, 60)).save(overlays / f"{stem}.png")
    return cli_out


@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    """Function-scoped sandbox seeded with a real master + overlays."""
    sandbox = _build_sandbox(tmp_path)
    _seed_real_output(sandbox)
    return sandbox


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    """Function-scoped live server bound to the seeded sandbox."""
    yield from _start_live_server(fake_sandbox)


def _hand_off_viewer(page: Page, hub_url: str, output_rel: str) -> None:
    """POST ``output_rel`` to the viewer-handoff endpoint via the page."""
    page.goto(hub_url + "/")
    page.wait_for_load_state("networkidle")
    response = page.evaluate(
        """
        async (path) => {
            const resp = await fetch('/sandbox/api/viewer/output-root', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: path}),
            });
            const body = await resp.text();
            return {status: resp.status, body};
        }
        """,
        output_rel,
    )
    assert response["status"] == 200, (
        f"Viewer hand-off failed: HTTP {response['status']} body={response['body']!r}"
    )


def test_colony_radial_debris_mark_writes_category_parquet(
    page: Page,
    live_server: str,
    fake_sandbox: Path,
) -> None:
    """Open a colony tile's radial, click Debris, assert badge + errors parquet.

    Covers the FEATURES rows: colony radial trigger, the ``debris`` wedge,
    the per-tile category badge, and the live ``deliverables/errors/*`` write.
    """
    hub_url = live_server
    output_dir = fake_sandbox / "results" / _OUTPUT_NAME
    output_rel = f"results/{_OUTPUT_NAME}"

    # Regression guard for a real-browser bug the unit/integration tests missed
    # (caught by a live Playwright-MCP walkthrough): a colony mark changes
    # ``STORE_REMOVED_KEYS``, which fires the QC review ``_render_detail``
    # callback; with zero matched ``qc-worklist-row`` components its wildcard
    # ``.style`` (ALL) output must return a LIST, not ``no_update`` — otherwise
    # Dash raises ``InvalidCallbackReturnValue`` and the POST 500s in the
    # background (the colony assertions still pass, hiding it).
    console_errors: list[str] = []
    page.on(
        "console",
        lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
    )

    _hand_off_viewer(page, hub_url, output_rel)

    # Load the viewer; the colony grid is on the default Plate/Colony surface.
    page.goto(hub_url + "/results/")
    # The colony grid container mounts on viewer boot.
    page.wait_for_selector("#colony-grid-container", state="attached", timeout=20_000)
    # Switch to the Colony tab so its grid is the foreground surface.
    colony_tab = page.locator("a.nav-link", has_text="Colony").first
    if colony_tab.count():
        colony_tab.click()

    # A radial trigger button per tile (neutral ▾).
    trigger = page.locator("button.radial-badge--neutral").first
    expect(trigger).to_be_visible(timeout=20_000)
    trigger.click()

    # The lazy-populate callback fills the popover body with the wedge ring;
    # the debris wedge carries the bare token in its id.
    debris_wedge = page.locator(
        "button.radial-wedge", has_text="debris"
    ).first
    expect(debris_wedge).to_be_visible(timeout=10_000)
    debris_wedge.click()

    # The durable per-category parquet must appear on disk (the live write).
    debris_path = error_category_parquet_path(output_dir, "debris")
    deadline = _time.monotonic() + 15.0
    while _time.monotonic() < deadline and not debris_path.exists():
        page.wait_for_timeout(250)
    assert debris_path.exists(), (
        f"deliverables/errors/debris.parquet was not written within 15s "
        f"(expected at {debris_path})"
    )
    debris = pl.read_parquet(debris_path)
    assert debris.height >= 1

    # The grid re-renders the marked tile with a debris category badge
    # (the neutral ▾ is replaced by a colored badge carrying the token text).
    debris_badge = page.locator(
        "button.radial-badge", has_text="debris"
    ).first
    expect(debris_badge).to_be_visible(timeout=15_000)

    # No background callback 500 fired from the mark (e.g. the QC review
    # ``_render_detail`` wildcard-output regression).
    server_errors = [
        e for e in console_errors if "500" in e or "Callback error" in e
    ]
    assert not server_errors, f"Background callback errors after mark: {server_errors}"
